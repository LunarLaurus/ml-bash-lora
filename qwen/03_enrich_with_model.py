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
- Batched model calls (configurable batch size)
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
import re
import textwrap
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ------------------------- config -------------------------
INPUT_FUNCTIONS_PATH = Path("data/parsed_functions.jsonl")
INPUT_GRAPH_FUNCTIONS_PATH = Path("data/dep_graph_functions.jsonl")
OUTPUT_PATH = Path("data/enriched_functions.jsonl")
MAX_TOKENS = 512
DEFAULT_BATCH_SIZE = 16

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


def load_model(max_tokens: int = MAX_TOKENS):
    device = get_device()
    logging.info("Loading Salesforce/codet5-small model on device %s...", device)
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small").to(
        0 if torch.cuda.is_available() else "cpu"
    )
    device_index = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device_index,
        max_new_tokens=max_tokens,
    )
    logging.info("Model loaded and pipeline created.")
    return pipe


def chunk_text(text: str, tokenizer, max_tokens: int = MAX_TOKENS) -> str:
    # Ensure we don't accidentally send excessively long input to pipeline
    try:
        tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
        return tokenizer.decode(tokens, skip_special_tokens=True)
    except Exception:
        # fallback to naive truncation by characters
        return text[: max_tokens * 4]


def model_prompt_function(fn_entry: dict, tokenizer) -> str:
    fn = fn_entry.get("function", {})
    suffix = Path(fn_entry.get("file_path", "")).suffix.lower()
    body = chunk_text(fn.get("body") or "", tokenizer)
    signature = fn.get("signature") or ""
    lang = "assembly" if suffix == ".asm" else "C"
    return textwrap.dedent(
        f"""\
        Summarize the following {lang} function.
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
        Aggregate function summaries and provide a JSON object with:
        - summary
        - intent_tags
        - risk_notes
        - change_recipe
        Content:
        {pseudo_body}
    """
    )


def run_model(pipe, prompts) -> list:
    """
    Run the model on one or more prompts (batched).
    - prompts: single str or iterable of str
    - returns: list of dicts (one per prompt) with keys:
        summary, intent_tags, risk_notes, change_recipe, confidence
    Defensive: handles various pipeline output shapes and non-JSON text.
    """
    # Normalize
    if isinstance(prompts, str):
        prompt_list = [prompts]
    else:
        prompt_list = list(prompts)

    if not prompt_list:
        return []

    try:
        with torch.no_grad():
            outputs = pipe(prompt_list, truncation=True)

        # normalize outputs to list aligned with prompt_list
        if isinstance(outputs, list):
            normalized = outputs
        else:
            # pipeline returned a single item (dict/string) - repeat across prompts
            normalized = [outputs] * len(prompt_list)

        # pad/truncate to exact length
        if len(normalized) < len(prompt_list):
            normalized = list(normalized) + [None] * (
                len(prompt_list) - len(normalized)
            )
        else:
            normalized = normalized[: len(prompt_list)]

        results = []
        for idx, out in enumerate(normalized):
            try:
                text = ""
                if out is None:
                    text = ""
                elif isinstance(out, dict):
                    # typical HF pipeline dict
                    text = out.get("generated_text") or out.get("text") or ""
                elif isinstance(out, str):
                    text = out
                else:
                    # fallback
                    text = str(out)

                parsed = None
                try:
                    parsed = json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    parsed = None

                if isinstance(parsed, dict):
                    parsed.setdefault("summary", "")
                    parsed.setdefault("intent_tags", [])
                    parsed.setdefault("risk_notes", "")
                    parsed.setdefault("change_recipe", "")
                    parsed.setdefault("confidence", 0.6)
                    results.append(parsed)
                else:
                    # fallback: embed raw text as summary
                    results.append(
                        {
                            "summary": text,
                            "intent_tags": [],
                            "risk_notes": "",
                            "change_recipe": "",
                            "confidence": 0.6,
                        }
                    )
            except Exception:
                logging.exception("Failed to normalize model output for index %s", idx)
                results.append(
                    {
                        "summary": "",
                        "intent_tags": [],
                        "risk_notes": "",
                        "change_recipe": "",
                        "confidence": 0.0,
                    }
                )

        return results

    except Exception as e:
        logging.exception("run_model batch failed: %s", e)
        # return an error result for each prompt
        return [
            {
                "summary": "",
                "intent_tags": [],
                "risk_notes": "",
                "change_recipe": "",
                "confidence": 0.0,
            }
            for _ in prompt_list
        ]


# ------------------------- main enrichment -------------------------
def enrich_functions(
    parsed_path: Path,
    output_path: Path,
    dep_graph: dict,
    pipe,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    logging.info("Loading function entries from %s...", parsed_path)
    entries = [
        json.loads(l) for l in parsed_path.read_text(encoding="utf-8").splitlines() if l
    ]
    logging.info("Loaded %d function entries.", len(entries))

    files_map = {}
    for entry in entries:
        file_path = entry["file_path"]
        files_map.setdefault(file_path, []).append(entry)

    enriched_entries = []

    # --- batch containers for functions ---
    func_prompt_entries = []
    func_prompts = []

    def flush_func_batch():
        nonlocal func_prompt_entries, func_prompts, enriched_entries
        if not func_prompts:
            return
        results = run_model(pipe, func_prompts)
        if len(results) < len(func_prompts):
            # pad with failures
            results = results + [
                {
                    "summary": "",
                    "intent_tags": [],
                    "risk_notes": "",
                    "change_recipe": "",
                    "confidence": 0.0,
                }
            ] * (len(func_prompts) - len(results))

        for entry, res in zip(func_prompt_entries, results):
            if not isinstance(res, dict):
                res = {
                    "summary": str(res),
                    "intent_tags": [],
                    "risk_notes": "",
                    "change_recipe": "",
                    "confidence": 0.6,
                }
            entry["summary"] = res.get("summary", "")
            entry["change_recipe"] = res.get("change_recipe", "")
            entry["confidence_score"] = float(res.get("confidence", 0.6))
            entry["generated_at"] = datetime.now(timezone.utc).isoformat()
            enriched_entries.append(entry)

        func_prompt_entries = []
        func_prompts = []

    try:
        for entry in tqdm(entries, desc="Functions"):
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

            prompt = model_prompt_function(entry, pipe.tokenizer)

            func_prompt_entries.append(entry)
            func_prompts.append(prompt)

            if len(func_prompts) >= batch_size:
                flush_func_batch()

        # flush any remaining function-level prompts
        flush_func_batch()

    except KeyboardInterrupt:
        logging.warning(
            "KeyboardInterrupt received during function processing; flushing remaining batches..."
        )
        flush_func_batch()
        raise

    # --- file-level summaries (batched) ---
    file_prompt_items = []  # list of tuples (file_path, fn_entries)
    file_prompts = []

    def flush_file_batch():
        nonlocal file_prompt_items, file_prompts, enriched_entries
        if not file_prompts:
            return
        results = run_model(pipe, file_prompts)
        if len(results) < len(file_prompts):
            results = results + [
                {
                    "summary": "",
                    "intent_tags": [],
                    "risk_notes": "",
                    "change_recipe": "",
                    "confidence": 0.0,
                }
            ] * (len(file_prompts) - len(results))

        for (file_path_i, fn_entries_i), file_res in zip(file_prompt_items, results):
            if not isinstance(file_res, dict):
                file_res = {
                    "summary": str(file_res),
                    "intent_tags": [],
                    "risk_notes": "",
                    "change_recipe": "",
                    "confidence": 0.6,
                }
            file_entry = {
                "id": f"file:{file_path_i}",
                "repo": str(Path.cwd().name),
                "file_path": file_path_i,
                "num_functions": len(fn_entries_i),
                "loc": sum(fn.get("loc", 0) for fn in fn_entries_i),
                "cyclomatic": sum(fn.get("cyclomatic", 0) for fn in fn_entries_i),
                "intent_tags": sorted(
                    {tag for fn in fn_entries_i for tag in fn.get("intent_tags", [])}
                ),
                "risk_notes": " ".join(
                    fn.get("risk_notes", "")
                    for fn in fn_entries_i
                    if fn.get("risk_notes")
                ),
                "summary": file_res.get("summary", ""),
                "change_recipe": file_res.get("change_recipe", ""),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "functions": [fn.get("id") for fn in fn_entries_i],
            }
            enriched_entries.append(file_entry)

        file_prompt_items = []
        file_prompts = []

    try:
        for file_path, fn_entries in tqdm(
            files_map.items(), total=len(files_map), desc="Files"
        ):
            file_prompt = model_prompt_file(file_path, fn_entries)
            file_prompt_items.append((file_path, fn_entries))
            file_prompts.append(file_prompt)

            if len(file_prompts) >= batch_size:
                flush_file_batch()

        # flush remaining file-level prompts
        flush_file_batch()

    except KeyboardInterrupt:
        logging.warning(
            "KeyboardInterrupt received during file processing; flushing remaining file batches..."
        )
        flush_file_batch()
        raise

    # Write all entries to output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for entry in enriched_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logging.info(
        "Enrichment completed. Wrote %d entries to %s",
        len(enriched_entries),
        output_path,
    )


# ------------------------- main -------------------------
def main():
    global MAX_TOKENS
    parser = argparse.ArgumentParser(
        description="Enrich parsed functions into JSONL with model summaries."
    )
    parser.add_argument(
        "repo_dir",
        help="Repository root directory (contains data/parsed_functions.jsonl etc.)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Model batch size for pipeline calls",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help="max_new_tokens for generation",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    repo_dir = Path(args.repo_dir)
    if not repo_dir.exists():
        logging.error("Repo dir not found: %s", repo_dir)
        sys.exit(1)

    rel_input_func_path = repo_dir / INPUT_FUNCTIONS_PATH
    rel_input_graph_path = repo_dir / INPUT_GRAPH_FUNCTIONS_PATH
    rel_output_path = repo_dir / OUTPUT_PATH

    if not rel_input_func_path.exists():
        logging.error("Parsed functions file not found: %s", rel_input_func_path)
        sys.exit(1)
    if not rel_input_graph_path.exists():
        logging.error("Dependency graph file not found: %s", rel_input_graph_path)
        sys.exit(1)

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
        # load model with user-specified max tokens
        MAX_TOKENS = args.max_tokens
        pipe = load_model(max_tokens=MAX_TOKENS)
        enrich_functions(
            rel_input_func_path,
            rel_output_path,
            dep_graph,
            pipe,
            batch_size=args.batch_size,
        )
    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt received! Exiting gracefully.")
    except Exception:
        logging.exception("Fatal error during enrichment.")
        sys.exit(1)


if __name__ == "__main__":
    main()
