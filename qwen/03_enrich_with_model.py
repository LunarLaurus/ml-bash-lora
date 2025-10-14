#!/usr/bin/env python3
"""
04_enrich_functions.py

Enrich function-level and file-level JSONL for LoRA training.

- Reads parsed function JSONL (output of parse_code)
- Computes heuristics (LOC, cyclomatic complexity, calls, intent tags, risk notes)
- Integrates dependency graph info (callers/callees, graph distance)
- Groups functions per file to build file-level summaries
- Uses Salesforce/codet5-small for model-based summarization
- Outputs a single enriched JSONL
"""

import json
import sys
import threading
import queue
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

FUNCTIONS_PARSED = Path("data/parsed_functions.jsonl")
GRAPH_FUNCTIONS = Path("data/dep_graph_functions.jsonl")
ENRICHED_OUTPUT = Path("data/enriched_functions.jsonl")
MAX_TOKENS = 512  # approximate token limit for codet5-small
BATCH_SIZE = 16  # batch size for GPU inference

# ------------------------- heuristics -------------------------
CALL_SITE_REGEX = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
CONTROL_KEYWORDS = re.compile(
    r"\b(if|else|for|while|case|switch|goto|&&|\|\||\?|return)\b"
)


def naive_cyclomatic(body: str) -> int:
    return 1 + len(CONTROL_KEYWORDS.findall(body)) if body else 1


def count_loc(body: str) -> int:
    return sum(1 for l in body.splitlines() if l.strip())


def detect_intent_tags(body: str, suffix: str) -> list:
    tags = set()
    b = body.lower()
    if suffix == ".asm":
        tags.add("asm")
    if re.search(r"\b(read|write|scanf|printf|fopen|fread|fwrite)\b", b):
        tags.add("I/O")
    if re.search(r"\b(malloc|calloc|realloc|free|memcpy|memmove|memset)\b", b):
        tags.add("memory")
    if re.search(r"\b(pthread_|mutex|lock|unlock|fork|exec)\b", b):
        tags.add("concurrency")
    if re.search(r"\b(pow|sqrt|sin|cos|log|exp)\b", b):
        tags.add("math")
    if re.search(r"\b(strcmp|strcpy|strncpy|strlen|strcat|memchr)\b", b):
        tags.add("string")
    if re.search(r"\b(md5|sha1|sha256|aes|crypto_)\b", b):
        tags.add("crypto")
    if re.search(r"[&|^~<>]{1,2}", body):
        tags.add("bitops")
    if not tags:
        tags.add("other")
    return sorted(tags)


def naive_risk_notes(body: str, suffix: str) -> str:
    notes = []
    if "malloc" in body or "free" in body or "realloc" in body:
        notes.append("Uses dynamic memory — watch for leaks and double-free.")
    if re.search(r"\b(strcpy|strcat|gets)\b", body):
        notes.append("Potential buffer overflow (unsafe string ops).")
    if "pthread" in body or "mutex" in body:
        notes.append("Concurrency: locking / race conditions possible.")
    if suffix == ".asm":
        notes.append(
            "Assembly code — ABI/calling conventions matter; harder to analyze."
        )
    return " ".join(notes)


# ------------------------- model -------------------------
def get_device():
    return 0 if torch.cuda.is_available() else -1


def load_model():
    device = get_device()
    logging.info("Initializing model load on device %s...", device)

    logging.info("Loading tokenizer for Salesforce/codet5-small...")
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
    logging.info("Tokenizer loaded successfully.")

    logging.info(
        "Loading model weights for Salesforce/codet5-small on device %s...", device
    )
    model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small").to(device)
    logging.info("Model weights loaded successfully.")

    logging.info("Creating pipeline...")
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_new_tokens=MAX_TOKENS,
    )
    logging.info("Pipeline created. Model ready for inference.")
    return pipe


def chunk_text(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """Truncate text to approximately max_tokens words."""
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])


def model_prompt_function(fn_entry: dict) -> str:
    fn = fn_entry.get("function", {})
    suffix = Path(fn_entry.get("file_path", "")).suffix.lower()
    body = chunk_text(fn.get("body") or "")
    signature = fn.get("signature") or ""
    return f"""Summarize the following {('assembly' if suffix=='.asm' else 'C')} function.
Return a single-line JSON with keys: summary, intent_tags (list), risk_notes, change_recipe, confidence.
Function signature:
{signature}
Function body:
{body}"""


def model_prompt_file(file_path: str, fn_entries: list) -> str:
    pseudo_body = "\n\n".join(fn.get("summary", "") for fn in fn_entries)
    return f"""Summarize this file: {file_path}.
Aggregate function summaries and provide:
- summary
- intent_tags
- risk_notes
- change_recipe
Content:
{pseudo_body}"""


def batch_run_model(pipe, prompts: list, batch_size: int = BATCH_SIZE) -> list:
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        outputs = pipe(batch)
        for out in outputs:
            text = out.get("generated_text", "")
            try:
                data = json.loads(text)
            except Exception:
                data = {
                    "summary": text,
                    "intent_tags": [],
                    "risk_notes": "",
                    "change_recipe": "",
                    "confidence": 0.6,
                }
            results.append(data)
    return results


# ------------------------- threading writer -------------------------
_shutdown = threading.Event()


def writer_thread(output_path: Path, q: "queue.Queue[dict]"):
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with tmp_path.open("w", encoding="utf-8") as f:
        while not _shutdown.is_set():
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1
            q.task_done()
    tmp_path.replace(output_path)
    logging.info("Wrote %d entries to %s", written, output_path)


# ------------------------- enrichment -------------------------
def enrich_functions(parsed_path: Path, output_path: Path, dep_graph: dict):
    logging.info("Starting function enrichment...")
    pipe = load_model()

    entries: list[dict] = []
    with parsed_path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                entries.append(json.loads(ln))
    logging.info("Loaded %d parsed functions.", len(entries))

    files_map = {}
    for entry in entries:
        file_path = entry["file_path"]
        files_map.setdefault(file_path, []).append(entry)

    q: "queue.Queue[dict]" = queue.Queue(maxsize=32)
    writer = threading.Thread(target=writer_thread, args=(output_path, q), daemon=True)
    writer.start()

    logging.info("Processing functions with ThreadPoolExecutor...")

    # Prepare prompts and process in batches
    prompts = []
    futures_map = {}
    with ThreadPoolExecutor() as exc:
        for entry in entries:
            fn_prompt = model_prompt_function(entry)
            prompts.append(fn_prompt)
            futures_map[fn_prompt] = entry

        # Process in batches
        for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Function batches"):
            batch_prompts = prompts[i : i + BATCH_SIZE]
            batch_results = batch_run_model(pipe, batch_prompts)
            for j, res in enumerate(batch_results):
                entry = futures_map[batch_prompts[j]]
                entry["summary"] = res.get("summary", "")
                entry["change_recipe"] = res.get("change_recipe", "")
                entry["confidence_score"] = float(res.get("confidence", 0.6))
                entry["generated_at"] = datetime.utcnow().isoformat() + "Z"

                fn = entry.get("function", {})
                body = fn.get("body") or ""
                suffix = Path(entry.get("file_path", "")).suffix.lower()
                entry["loc"] = count_loc(body)
                entry["cyclomatic"] = naive_cyclomatic(body)
                entry["calls"] = CALL_SITE_REGEX.findall(body)
                entry["intent_tags"] = detect_intent_tags(body, suffix)
                entry["risk_notes"] = naive_risk_notes(body, suffix)

                key = entry.get("id")
                graph_info = dep_graph.get(key, {})
                entry["callers"] = graph_info.get("callers", [])
                entry["callees"] = graph_info.get("callees", [])
                entry["graph_distance"] = graph_info.get(
                    "graph_distance", {"to_entry_points": None}
                )

                q.put(entry)

    logging.info("Processing file-level summaries...")
    for file_path, fn_entries in tqdm(
        files_map.items(), total=len(files_map), desc="Files"
    ):
        file_prompt = model_prompt_file(file_path, fn_entries)
        file_res = batch_run_model(pipe, [file_prompt])[0]
        file_entry = {
            "id": f"file:{file_path}",
            "repo": str(Path.cwd().name),
            "file_path": file_path,
            "num_functions": len(fn_entries),
            "loc": sum(fn.get("loc", 0) for fn in fn_entries),
            "cyclomatic": sum(fn.get("cyclomatic", 0) for fn in fn_entries),
            "intent_tags": sorted(
                {tag for fn in fn_entries for tag in fn.get("intent_tags", [])}
            ),
            "risk_notes": " ".join(
                fn.get("risk_notes", "") for fn in fn_entries if fn.get("risk_notes")
            ),
            "summary": file_res.get("summary", ""),
            "change_recipe": file_res.get("change_recipe", ""),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "functions": [fn.get("id") for fn in fn_entries],
        }
        q.put(file_entry)

    q.put(None)
    writer.join()
    logging.info("Enrichment completed successfully.")


# ------------------------- main -------------------------
def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    if len(sys.argv) != 2:
        logging.error("Usage: python3 scripts/04_enrich_functions.py <repo_directory>")
        sys.exit(1)
    repo_dir = Path(sys.argv[1])

    if not FUNCTIONS_PARSED.exists():
        logging.error("Parsed functions file not found: %s", FUNCTIONS_PARSED)
        sys.exit(1)
    if not GRAPH_FUNCTIONS.exists():
        logging.error("Dependency graph file not found: %s", GRAPH_FUNCTIONS)
        sys.exit(1)

    dep_graph = {}
    with GRAPH_FUNCTIONS.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = obj.get("id")
            if key:
                dep_graph[key] = obj

    try:
        enrich_functions(FUNCTIONS_PARSED, ENRICHED_OUTPUT, dep_graph)
    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt received! Shutting down...")
        _shutdown.set()
        # Give writer a chance to exit
        queue_obj = queue.Queue()
        queue_obj.put(None)


if __name__ == "__main__":
    main()
