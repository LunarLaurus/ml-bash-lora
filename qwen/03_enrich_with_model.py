#!/usr/bin/env python3
"""
04_enrich_functions_improved.py

Improved enrichment script:
- deterministic model.generate usage (batched)
- OOM-aware: halves batch size automatically on OOM and retries
- streaming output with checkpointing (resume)
- stricter JSON-only prompts and robust parsing
- file-level summarizer uses top-N function summaries to avoid huge prompts
"""
import argparse
import json
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
import re
import textwrap
import time
from typing import List, Tuple
from tqdm import tqdm
from ProgressTracker import ProgressTracker
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ------------------------- config -------------------------
INPUT_FUNCTIONS_PATH = Path("data/parsed_functions.jsonl")
INPUT_GRAPH_FUNCTIONS_PATH = Path("data/dep_graph_functions.jsonl")
OUTPUT_PATH = Path("data/enriched_functions.jsonl")
PROCESSED_IDS_PATH = OUTPUT_PATH.with_suffix(".processed_ids")
MAX_TOKENS = 512
DEFAULT_BATCH_SIZE = 16
FILE_SUMMARY_TOP_N = 8  # how many function summaries to include in file prompt

# ------------------------- heuristics -------------------------
CALL_SITE_REGEX = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
CONTROL_KEYWORDS = re.compile(
    r"\b(if|else|for|while|case|switch|goto|&&|\|\||\?|return)\b"
)

tracker = None
trackerTimeStart = None


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


# ------------------------- model utils -------------------------
def get_torch_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_model(
    max_new_tokens: int = MAX_TOKENS,
) -> Tuple[AutoModelForSeq2SeqLM, any, torch.device]:
    device = get_torch_device()
    logging.info("Loading Salesforce/codet5-small on %s", device)
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small", use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")
    model.to(device)
    # note: don't mutate model.config.max_length for generation; we pass max_new_tokens explicitly
    return model, tokenizer, device


def chunk_text(text: str, tokenizer, max_tokens: int = MAX_TOKENS) -> str:
    # Truncate by tokens to avoid huge prompts
    try:
        tok = tokenizer.encode(text, truncation=True, max_length=max_tokens)
        return tokenizer.decode(tok, skip_special_tokens=True)
    except Exception:
        return text[: max_tokens * 4]


def model_prompt_function(fn_entry: dict, tokenizer) -> str:
    fn = fn_entry.get("function", {})
    suffix = Path(fn_entry.get("file_path", "")).suffix.lower()
    body = chunk_text(fn.get("body") or "", tokenizer)
    signature = fn.get("signature") or ""
    lang = "assembly" if suffix == ".asm" else "C"
    # strict instruction to return only JSON
    return textwrap.dedent(
        f"""\
        Summarize the following {lang} function. IMPORTANT: Return only valid JSON (no surrounding text).
        Required keys: summary (string), intent_tags (list of strings), risk_notes (string), change_recipe (string), confidence (float 0-1).

        Function signature:
        {signature}

        Function body:
        {body}

        Return a single JSON object and nothing else.
    """
    )


def model_prompt_file(file_path: str, fn_entries: list) -> str:
    # Use top-N longest summaries to avoid too-large prompts.
    ordered = sorted(fn_entries, key=lambda f: f.get("loc", 0), reverse=True)[
        :FILE_SUMMARY_TOP_N
    ]
    pseudo_body = "\n\n".join(f.get("summary", "") for f in ordered)
    return textwrap.dedent(
        f"""\
        Summarize this file: {file_path}.
        Aggregate the provided function summaries and return only a single JSON object with keys:
        - summary (string)
        - intent_tags (list of strings)
        - risk_notes (string)
        - change_recipe (string)
        - confidence (float 0-1)

        Content (top {FILE_SUMMARY_TOP_N} functions by LOC):
        {pseudo_body}

        Return strictly one JSON object and nothing else.
    """
    )


def extract_json_block(text: str):
    # Try direct JSON load
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fallback: find first {...} block
    m = re.search(r"\{(?:[^{}]|\n|\r)*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    device,
    max_new_tokens: int = MAX_TOKENS,
    num_beams: int = 4,
    do_sample: bool = False,
    retries: int = 2,
    backoff: float = 1.0,
) -> List[dict]:
    """
    Deterministic batched generation using model.generate.
    Returns list of parsed dicts (or fallback dicts).
    """
    # tokenise as batch
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
        device
    )
    for attempt in range(retries + 1):
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    early_stopping=True,
                )
            texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results = []
            for t in texts:
                parsed = extract_json_block(t)
                if isinstance(parsed, dict):
                    parsed.setdefault("summary", "")
                    parsed.setdefault("intent_tags", [])
                    parsed.setdefault("risk_notes", "")
                    parsed.setdefault("change_recipe", "")
                    parsed.setdefault("confidence", 0.6)
                    results.append(parsed)
                else:
                    results.append(
                        {
                            "summary": t.strip(),
                            "intent_tags": [],
                            "risk_notes": "",
                            "change_recipe": "",
                            "confidence": 0.6,
                        }
                    )
            return results
        except RuntimeError as e:
            logging.exception("Generation runtime error (attempt %s): %s", attempt, e)
            if "out of memory" in str(e).lower():
                # bubble up OOM to caller to reduce batch size
                raise
            # otherwise backoff and retry
            time.sleep(backoff * (2**attempt))
    # fallback: empty entries
    return [
        {
            "summary": "",
            "intent_tags": [],
            "risk_notes": "",
            "change_recipe": "",
            "confidence": 0.0,
        }
        for _ in prompts
    ]


# ------------------------- main enrichment -------------------------
def load_processed_ids(path: Path) -> set:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def append_processed_id(path: Path, id_: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(id_ + "\n")


def enrich_functions(
    parsed_path: Path,
    output_path: Path,
    dep_graph: dict,
    model,
    tokenizer,
    device,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_new_tokens: int = MAX_TOKENS,
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

    # streaming output with checkpointing
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = load_processed_ids(PROCESSED_IDS_PATH)
    logging.info("Loaded %d processed ids from checkpoint.", len(processed))

    # function-level batching containers
    func_prompt_entries: List[dict] = []
    func_prompts: List[str] = []
    current_batch_size = max(1, batch_size)

    def flush_func_batch(current_batch_size_local):
        nonlocal func_prompt_entries, func_prompts, processed
        if not func_prompts:
            return
        # try generate, handle OOM by reducing batch size (caller will re-chunk)
        try:
            results = generate_batch(
                model,
                tokenizer,
                func_prompts,
                device,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                do_sample=False,
            )
            batch_latency = time.time() - trackerTimeStart
            tokens_used = sum(len(tokenizer.encode(p)) for p in func_prompts)
            tracker.update(len(func_prompts), tokens_used, batch_latency)

        except RuntimeError as e:
            if "out of memory" in str(e).lower() and current_batch_size_local > 1:
                logging.warning("OOM during generate. Will retry with smaller batches.")
                # re-chunk and retry with half batch size
                half = max(1, current_batch_size_local // 2)
                # process in smaller chunks
                i = 0
                while i < len(func_prompts):
                    chunk_prompts = func_prompts[i : i + half]
                    chunk_entries = func_prompt_entries[i : i + half]
                    # recursive-ish call: attempt to flush smaller chunk
                    try:
                        small_results = generate_batch(
                            model,
                            tokenizer,
                            chunk_prompts,
                            device,
                            max_new_tokens=max_new_tokens,
                            num_beams=4,
                            do_sample=False,
                        )
                        batch_latency = time.time() - trackerTimeStart
                        tokens_used = sum(
                            len(tokenizer.encode(p)) for p in func_prompts
                        )
                        tracker.update(len(func_prompts), tokens_used, batch_latency)
                    except RuntimeError:
                        # if still OOM on batch size 1, re-raise
                        if half == 1:
                            raise
                        # try even smaller in next loop
                        half = max(1, half // 2)
                        continue
                    # write small results
                    for entry, res in zip(chunk_entries, small_results):
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
                        # stream to disk if not processed
                        if entry.get("id") not in processed:
                            with output_path.open("a", encoding="utf-8") as outf:
                                outf.write(json.dumps(entry, ensure_ascii=False) + "\n")
                            append_processed_id(PROCESSED_IDS_PATH, entry.get("id"))
                            processed.add(entry.get("id"))
                    i += len(chunk_prompts)
                func_prompt_entries = []
                func_prompts = []
                return
            else:
                raise

        # normal path: write results
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
            if entry.get("id") not in processed:
                with output_path.open("a", encoding="utf-8") as outf:
                    outf.write(json.dumps(entry, ensure_ascii=False) + "\n")
                append_processed_id(PROCESSED_IDS_PATH, entry.get("id"))
                processed.add(entry.get("id"))

        func_prompt_entries = []
        func_prompts = []

    # process functions
    try:
        for entry in tqdm(entries, desc="Functions"):
            if entry.get("id") in processed:
                continue  # skip already processed
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

            prompt = model_prompt_function(entry, tokenizer)

            func_prompt_entries.append(entry)
            func_prompts.append(prompt)

            if len(func_prompts) >= current_batch_size:
                flush_func_batch(current_batch_size)

        # flush remaining
        flush_func_batch(current_batch_size)

    except KeyboardInterrupt:
        logging.warning(
            "KeyboardInterrupt received during function processing; flushing remaining batches..."
        )
        flush_func_batch(current_batch_size)
        raise

    # --- file-level summaries (batched) ---
    file_prompt_items: List[Tuple[str, list]] = []
    file_prompts: List[str] = []
    try:
        for file_path, fn_entries in tqdm(
            files_map.items(), total=len(files_map), desc="Files"
        ):
            # skip file-level if all functions already processed? We'll still generate file summary if not present
            # build compact prompt
            # ensure function entries have summary (they should after function pass; otherwise include their existing summary)
            file_prompt = model_prompt_file(file_path, fn_entries)
            file_prompt_items.append((file_path, fn_entries))
            file_prompts.append(file_prompt)

            if len(file_prompts) >= batch_size:
                # reuse generate_batch but write file entries to disk
                results = generate_batch(
                    model,
                    tokenizer,
                    file_prompts,
                    device,
                    max_new_tokens=max_new_tokens,
                    num_beams=4,
                    do_sample=False,
                )
                batch_latency = time.time() - trackerTimeStart
                tokens_used = sum(len(tokenizer.encode(p)) for p in file_prompts)
                tracker.update(len(file_prompts), tokens_used, batch_latency)
                for (file_path_i, fn_entries_i), file_res in zip(
                    file_prompt_items, results
                ):
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
                        "cyclomatic": sum(
                            fn.get("cyclomatic", 0) for fn in fn_entries_i
                        ),
                        "intent_tags": sorted(
                            {
                                tag
                                for fn in fn_entries_i
                                for tag in fn.get("intent_tags", [])
                            }
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
                    # write file-level entry (no resume for file entries currently)
                    with output_path.open("a", encoding="utf-8") as outf:
                        outf.write(json.dumps(file_entry, ensure_ascii=False) + "\n")
                file_prompt_items = []
                file_prompts = []

        # flush remaining file-level
        if file_prompts:
            results = generate_batch(
                model,
                tokenizer,
                file_prompts,
                device,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                do_sample=False,
            )
            batch_latency = time.time() - trackerTimeStart
            tokens_used = sum(len(tokenizer.encode(p)) for p in func_prompts)
            tracker.update(len(func_prompts), tokens_used, batch_latency)
            for (file_path_i, fn_entries_i), file_res in zip(
                file_prompt_items, results
            ):
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
                        {
                            tag
                            for fn in fn_entries_i
                            for tag in fn.get("intent_tags", [])
                        }
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
                with output_path.open("a", encoding="utf-8") as outf:
                    outf.write(json.dumps(file_entry, ensure_ascii=False) + "\n")

    except KeyboardInterrupt:
        logging.warning(
            "KeyboardInterrupt received during file processing; flushing remaining file batches..."
        )
        raise

    logging.info("Enrichment completed. Output appended to %s", output_path)


# ------------------------- main -------------------------
def main():
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
        help="Model batch size for generation",
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

    logging.info("Creating progress tracker")
    global tracker
    tracker = ProgressTracker(repo_dir / "stats-enrich.json")
    global trackerTimeStart
    logging.info("Starting progress tracker")
    trackerTimeStart = time.time()
    rel_input_func_path = repo_dir / INPUT_FUNCTIONS_PATH
    rel_input_graph_path = repo_dir / INPUT_GRAPH_FUNCTIONS_PATH
    rel_output_path = repo_dir / OUTPUT_PATH

    if not rel_input_func_path.exists():
        logging.error("Parsed functions file not found: %s", rel_input_func_path)
        sys.exit(1)
    if not rel_input_graph_path.exists():
        logging.error("Dependency graph file not found: %s", rel_input_graph_path)
        sys.exit(1)

    logging.info("Creating dep_graph.")
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

    logging.info("Trying to process input data.")
    try:
        model, tokenizer, device = load_model(max_new_tokens=args.max_tokens)
        logging.info("Enriching function data.")
        enrich_functions(
            rel_input_func_path,
            rel_output_path,
            dep_graph,
            model,
            tokenizer,
            device,
            batch_size=args.batch_size,
            max_new_tokens=args.max_tokens,
        )
        logging.info("Finished Enrichment!")
    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt received! Exiting gracefully.")
    except Exception:
        logging.exception("Fatal error during enrichment.")
        sys.exit(1)


if __name__ == "__main__":
    main()
