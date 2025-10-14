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
- Async batching on GPU with full FP32
- Ctrl+C safe
"""

import json
import sys
import threading
import queue
import logging
from pathlib import Path
from datetime import datetime
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import textwrap

# ------------------------- config -------------------------
INPUT_FUNCTIONS_PATH = Path("data/parsed_functions.jsonl")
INPUT_GRAPH_FUNCTIONS_PATH = Path("data/dep_graph_functions.jsonl")
OUTPUT_PATH = Path("data/enriched_functions.jsonl")
MAX_TOKENS = 512
BATCH_SIZE = 32


# ------------------------- etc -------------------------
_input_queue = queue.Queue(maxsize=BATCH_SIZE * 2)
_output_queue = queue.Queue(maxsize=BATCH_SIZE * 2)
_shutdown = threading.Event()

# ------------------------- module-level vars -------------------------
TOKENIZER = None
MODEL = None
PIPELINE = None

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
    return 0 if torch.cuda.is_available() else "cpu"


def load_model():
    global TOKENIZER, MODEL, PIPELINE

    device = get_device()
    logging.info("Initializing model load on device %s...", device)

    logging.info("Loading tokenizer for Salesforce/codet5-small...")
    TOKENIZER = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
    logging.info("Tokenizer loaded successfully.")

    logging.info(
        "Loading model weights for Salesforce/codet5-small on device %s...", device
    )
    MODEL = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small").to(device)
    logging.info("Model weights loaded successfully.")

    logging.info("Creating pipeline...")
    device_index = 0 if torch.cuda.is_available() else -1
    PIPELINE = pipeline(
        "text2text-generation",
        model=MODEL,
        tokenizer=TOKENIZER,
        device=device_index,
        max_new_tokens=MAX_TOKENS,
    )
    logging.info("Pipeline created. Model ready for inference.")


def chunk_text(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """
    Truncate text to fit max_tokens using the global tokenizer.
    """
    tokens = TOKENIZER.encode(text, truncation=True, max_length=max_tokens)
    return TOKENIZER.decode(tokens, skip_special_tokens=True)


def model_prompt_function(fn_entry: dict) -> str:
    fn = fn_entry.get("function", {})
    suffix = Path(fn_entry.get("file_path", "")).suffix.lower()
    body = chunk_text(fn.get("body") or "")
    signature = fn.get("signature") or ""
    return textwrap.dedent(
        f"""\
        Summarize the following {('assembly' if suffix=='.asm' else 'C')} function.
        Return a single-line JSON with keys: summary, intent_tags (list), risk_notes, change_recipe, confidence.
        Function signature:
        {signature}
        Function body:
        {body}
    """
    )


def model_prompt_file(file_path: str, fn_entries: list) -> str:
    pseudo_body = "\n\n".join(fn.get("summary", "") for fn in fn_entries)
    return textwrap.dedent(
        f"""\
        Summarize this file: {file_path}.
        Aggregate function summaries and provide:
        - summary
        - intent_tags
        - risk_notes
        - change_recipe
        Content:
        {pseudo_body}
    """
    )


def batch_run_model(pipe, prompts: list) -> list:
    results = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i : i + BATCH_SIZE]
        logging.info("Sending batch of %d prompts to GPU.", len(batch))
        try:
            outputs = pipe(batch)
        except Exception as e:
            logging.error("Pipeline batch failed: %s", e)
            outputs = [{"generated_text": ""}] * len(batch)
        for out in outputs:
            text = out.get("generated_text", "")
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                logging.warning("Failed to parse model output: %s", text)
                data = {
                    "summary": text,
                    "intent_tags": [],
                    "risk_notes": "",
                    "change_recipe": "",
                    "confidence": 0.6,
                }
            results.append(data)
    return results


# ------------------------- async GPU worker -------------------------
def gpu_worker(pipe):
    """Continuously process prompts from _input_queue in batches."""
    while True:
        if _shutdown.is_set() and _input_queue.empty():
            break
        batch = []
        entries = []
        while len(batch) < BATCH_SIZE:
            try:
                entry, prompt = _input_queue.get(timeout=0.5)
                batch.append(prompt)
                entries.append(entry)
                _input_queue.task_done()
            except queue.Empty:
                break
        if not batch:
            continue

        batch_results = batch_run_model(pipe, batch)
        for entry, res in zip(entries, batch_results):
            entry["summary"] = res.get("summary", "")
            entry["change_recipe"] = res.get("change_recipe", "")
            entry["confidence_score"] = float(res.get("confidence", 0.6))
            entry["generated_at"] = datetime.utcnow().isoformat() + "Z"
            _output_queue.put(entry)
        logging.info("GPU worker processed batch of %d entries.", len(batch))


# ------------------------- writer thread -------------------------
def writer_thread(output_path: Path, q: "queue.Queue[dict]"):
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    while True:
        try:
            item = q.get(timeout=0.5)
        except queue.Empty:
            if _shutdown.is_set() and q.empty():
                break
            continue
        if item is None:
            break
        with tmp_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        written += 1
        q.task_done()
    tmp_path.replace(output_path)
    logging.info("Wrote %d entries to %s", written, output_path)


# ------------------------- main async enrichment -------------------------
def enrich_functions_async(parsed_path: Path, output_path: Path, dep_graph: dict, pipe):
    logging.info("Loading function entries from %s...", parsed_path)
    entries = [
        json.loads(l) for l in parsed_path.read_text(encoding="utf-8").splitlines() if l
    ]
    logging.info("Loaded %d function entries.", len(entries))

    files_map = {}
    for entry in entries:
        file_path = entry["file_path"]
        files_map.setdefault(file_path, []).append(entry)

    gpu_thread = threading.Thread(target=gpu_worker, args=(pipe,), daemon=True)
    gpu_thread.start()

    writer = threading.Thread(
        target=writer_thread, args=(output_path, _output_queue), daemon=True
    )
    writer.start()

    # Fill input queue with per-function prompts and enrich heuristics
    for entry in entries:
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

        prompt = model_prompt_function(entry)
        _input_queue.put((entry, prompt))

    logging.info("All function prompts enqueued. Waiting for GPU worker to finish...")
    _input_queue.join()
    _shutdown.set()
    gpu_thread.join()

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
        _output_queue.put(file_entry)

    _output_queue.put(None)
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

    if not repo_dir.exists():
        logging.error("Repo dir not found: %s", repo_dir)
        sys.exit(1)

    rel_input_func_path = repo_dir / INPUT_FUNCTIONS_PATH
    rel_input_graph_path = repo_dir / INPUT_GRAPH_FUNCTIONS_PATH

    if not rel_input_func_path.exists():
        logging.error("Parsed functions file not found: %s", rel_input_func_path)
        sys.exit(1)
    if not rel_input_graph_path.exists():
        logging.error("Dependency graph file not found: %s", rel_input_graph_path)
        sys.exit(1)

    rel_output_path = repo_dir / OUTPUT_PATH
    dep_graph = {}
    with rel_input_graph_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = obj.get("id")
            if key:
                dep_graph[key] = obj
            else:
                logging.warning("Skipped dep_graph entry without 'id': %s", obj)

    try:
        load_model()
        enrich_functions_async(
            rel_input_func_path, rel_output_path, dep_graph, PIPELINE
        )
    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt received! Shutting down...")
        _shutdown.set()
        _output_queue.put(None)


if __name__ == "__main__":
    main()
