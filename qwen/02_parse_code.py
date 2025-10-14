#!/usr/bin/env python3
"""
02_parse_code.py — parse C/.h/.asm files, extract function-level metadata,
enrich entries, and write JSONL.

Key features:
- Default extensions: .c, .h, .asm
- Reuse file hashes from an optional index JSONL (--index)
- Single-pass discovery, thread pool for parse+enrich
- Writer thread streams JSONL to tmp file and atomically moves to final output
- Per-function callees & callers (within-file best-effort)
- Only Python standard library
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
import queue
import re
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Defaults
DEFAULT_EXTENSIONS = [".c", ".h", ".asm"]
INDEX_PATH = Path("data/file_index.jsonl")
OUTPUT_PATH = Path("data/parsed_functions.jsonl")
_TEMP_SUFFIX = ".tmp"

# Globals
_shutdown = threading.Event()
_temp_output_path: Optional[Path] = None


def _signal_handler(signum, frame):
    logging.warning("Received signal %s — requesting shutdown...", signum)
    _shutdown.set()


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def parse_args(argv: Optional[List[str]] = None):
    cpu = os.cpu_count() or 1
    default_workers = max(2, min(32, cpu * 3))
    p = argparse.ArgumentParser(
        description="Parse C/ASM code and produce enriched JSONL per function."
    )
    p.add_argument(
        "repo_directory", type=Path, help="Repository root (assumed); .git ignored."
    )
    p.add_argument(
        "--extensions",
        nargs="+",
        default=DEFAULT_EXTENSIONS,
        help="File extensions to include (default: .c .h .asm).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Worker threads (default auto = {default_workers})",
    )
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args(argv)


# ---------------- helpers ----------------
def _humanize_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def load_index_hashes(index_path: Optional[Path]) -> Dict[str, str]:
    """Load mapping of absolute file paths -> sha256 from a JSONL index file (if present)."""
    mapping: Dict[str, str] = {}
    if not index_path:
        return mapping
    if not index_path.exists():
        logging.warning(
            "Index file %s not found; continuing without reusing hashes.", index_path
        )
        return mapping
    try:
        with index_path.open("r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                fp = obj.get("file_path") or obj.get("repo_relative_path")
                sha = obj.get("sha256") or obj.get("hash")
                if fp and sha:
                    mapping[str(Path(fp).resolve())] = sha
    except Exception as exc:
        logging.warning("Failed reading index file %s: %s", index_path, exc)
    logging.info("Loaded %d hashes from index %s", len(mapping), index_path)
    return mapping


def sha256_for_file(path: Path) -> Optional[str]:
    """Compute SHA256 of file streaming; return hex digest or None on error/interruption."""
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(64 * 1024), b""):
                if _shutdown.is_set():
                    return None
                h.update(chunk)
        return h.hexdigest()
    except (FileNotFoundError, PermissionError) as exc:
        logging.debug("Failed to hash %s: %s", path, exc)
        return None


def discover_files(root: Path, extensions: Iterable[str]) -> List[Path]:
    """Discover files; skip .git. Returns list of Path objects."""
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    candidates: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        if _shutdown.is_set():
            break
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for fn in filenames:
            if Path(fn).suffix.lower() not in exts:
                continue
            candidates.append(Path(dirpath) / fn)
    return candidates


# ---------------- parsing & extraction ----------------
def parse_file_with_treesitter(file_path: Path) -> Dict[str, Optional[str]]:
    """
    Attempt to run tree-sitter parse -p c on the file.
    Returns dict with keys: file_name, file_path, ast (S-exp or None), content (string).
    Non-fatal if tree-sitter missing or fails.
    """
    # Read content (best-effort)
    try:
        content = file_path.read_text(encoding="utf-8", errors="surrogatepass")
    except Exception:
        try:
            content = file_path.read_text(encoding="latin-1", errors="ignore")
        except Exception:
            content = ""
    ast_text = None
    # Only attempt tree-sitter for C files
    if file_path.suffix.lower() in {".c", ".h"}:
        try:
            proc = subprocess.run(
                ["tree-sitter", "parse", "-p", "c", str(file_path)],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            ast_text = proc.stdout
        except Exception as exc:
            logging.debug("tree-sitter parse failed for %s: %s", file_path, exc)
            ast_text = None
    return {
        "file_name": file_path.name,
        "file_path": str(file_path.resolve()),
        "ast": ast_text,
        "content": content,
    }


# C function regex (improved)
_FUNCTION_REGEX = re.compile(
    r"""(?mx)
    ^\s*                                         # line start + optional whitespace
    (?P<ret>(?:[a-zA-Z_][\w\s\*\(\)]*?))\s+     # naive return type (may include pointers)
    (?P<name>[A-Za-z_]\w*)\s*                   # function name
    \(\s*(?P<args>[^)]*)\)\s*                   # argument list
    \{                                           # opening brace of function body
    """
)


CALL_SITE_REGEX = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


def extract_functions_from_c(content: str) -> List[Dict]:
    """
    Extract top-level C functions (naive): find declarations using regex and match braces
    to grab the body. Returns list of dicts with name, signature, start_line, end_line, body.
    """
    functions = []
    for m in _FUNCTION_REGEX.finditer(content):
        try:
            start_line = content[: m.start()].count("\n") + 1
            name = m.group("name")
            signature = f"{m.group('ret').strip()} {name}({m.group('args').strip()})"
            # find matching braces from m.end()-1
            body_start = m.end() - 1
            idx = body_start
            depth = 0
            end_idx = None
            while idx < len(content):
                c = content[idx]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end_idx = idx + 1
                        break
                idx += 1
            body = (
                content[body_start:end_idx]
                if end_idx
                else content[body_start : body_start + 2000]
            )
            end_line = content[: (end_idx or body_start)].count("\n") + 1
            functions.append(
                {
                    "function_name": name,
                    "function_signature": signature,
                    "function_body": body,
                    "function_start_line": start_line,
                    "function_end_line": end_line,
                }
            )
        except Exception:
            continue
    return functions


ASM_LABEL_REGEX = re.compile(
    r"^(?P<label>[A-Za-z_\.][\w\.\$@]*)\s*:\s*(?:\b.*)?$", re.MULTILINE
)


def extract_functions_from_asm(content: str) -> List[Dict]:
    """
    Extract function-like blocks from assembly by locating labels and grouping subsequent lines
    until the next label. Heuristics:
      - A label followed by instructions and/or preceded by .globl/.global/EXPORT likely a function.
      - Provide start/end line and body (best-effort).
    """
    lines = content.splitlines(True)
    candidates: List[Dict] = []
    # Find all labels and their line numbers
    labels = []
    for i, ln in enumerate(lines):
        m = ASM_LABEL_REGEX.match(ln)
        if m:
            labels.append((i, m.group("label")))
    # If no labels, treat whole file as single pseudo-function
    if not labels:
        return [
            {
                "function_name": "<file>",
                "function_signature": "<asm>",
                "function_body": "".join(lines)[:4000],
                "function_start_line": 1,
                "function_end_line": len(lines),
            }
        ]
    # Determine extents
    for idx, (line_no, label) in enumerate(labels):
        start = line_no
        end = labels[idx + 1][0] if idx + 1 < len(labels) else len(lines)
        block = "".join(lines[start:end])
        # Heuristic: check for .globl/.global/.extern/PUBLIC near label or presence of instructions
        prefix_region = "".join(lines[max(0, start - 5) : start + 1])
        if re.search(
            r"\.(globl|global|extern)\b|^\s*PUBLIC\b",
            prefix_region,
            flags=re.IGNORECASE,
        ) or re.search(r"^\s*[A-Za-z]+\s", block, flags=re.MULTILINE):
            fname = label
            functions_block = {
                "function_name": fname,
                "function_signature": f"{fname} (asm)",
                "function_body": block,
                "function_start_line": start + 1,
                "function_end_line": end,
            }
            candidates.append(functions_block)
    # Fallback: if none matched heuristics, include the top labels regardless
    if not candidates:
        for line_no, label in labels:
            start = line_no
            end = next((ln for ln, _ in labels if ln > line_no), len(lines))
            block = "".join(lines[start:end])
            candidates.append(
                {
                    "function_name": label,
                    "function_signature": f"{label} (asm)",
                    "function_body": block,
                    "function_start_line": start + 1,
                    "function_end_line": end,
                }
            )
    return candidates


def extract_functions_from_content(content: str, suffix: str) -> List[Dict]:
    """Dispatch extraction based on suffix (.c/.h vs .asm)."""
    if suffix.lower() in {".c", ".h"}:
        return extract_functions_from_c(content)
    elif suffix.lower() == ".asm":
        return extract_functions_from_asm(content)
    else:
        # generic file fallback
        return [
            {
                "function_name": "<file>",
                "function_signature": "<file>",
                "function_body": content[:2000],
                "function_start_line": 1,
                "function_end_line": content.count("\n") + 1,
            }
        ]


def compute_callees_and_callers(functions: List[Dict]) -> None:
    """
    Populate 'callees' and 'callers' fields for each function dict (in-place),
    by searching for call-site patterns inside function_body. This is best-effort
    and limited to same-file references.
    """
    name_to_idx = {fn["function_name"]: i for i, fn in enumerate(functions)}
    callees_list = [set() for _ in functions]
    callers_list = [set() for _ in functions]

    for i, fn in enumerate(functions):
        body = fn.get("function_body", "") or ""
        for m in CALL_SITE_REGEX.finditer(body):
            callee = m.group(1)
            if callee in name_to_idx and callee != fn["function_name"]:
                callees_list[i].add(callee)
                callers_list[name_to_idx[callee]].add(fn["function_name"])

    for i, fn in enumerate(functions):
        fn["callees"] = sorted(callees_list[i])
        fn["callers"] = sorted(callers_list[i])


def enrich_file(parsed_file: Dict, hash_map: Dict[str, str]) -> List[Dict]:
    """
    Given parsed file (with 'ast' and 'content'), extract functions and enrich each entry.
    """
    results: List[Dict] = []
    file_path = Path(parsed_file["file_path"])
    resolved = str(Path(file_path).resolve())

    sha = hash_map.get(resolved)
    if not sha:
        sha = sha256_for_file(Path(resolved))

    suffix = Path(resolved).suffix.lower()
    content = parsed_file.get("content", "") or ""

    functions = []
    # Prefer robust extraction based on content (AST is kept as context)
    functions = extract_functions_from_content(content, suffix)

    # Derive callees/callers within file
    compute_callees_and_callers(functions)

    for fn in functions:
        entry = {
            "id": f"repo:{resolved}:{fn['function_name']}:{fn['function_start_line']}",
            "repo": str(Path.cwd().name),
            "file_path": resolved,
            "function": {
                "name": fn["function_name"],
                "signature": fn["function_signature"],
                "body": fn.get("function_body", "")[:100_000],
                "start_line": fn.get("function_start_line", 1),
                "end_line": fn.get("function_end_line", 1),
            },
            "includes": [],  # could parse from content
            "callers": fn.get("callers", []),
            "callees": fn.get("callees", []),
            "graph_distance": {"to_entry_points": None},
            "full_file_context": (parsed_file.get("ast") or content)[:8000],
            "hash": sha,
            "summary": "",
            "detailed_description": "",
            "intent_tags": [],
            "risk_notes": "",
            "change_recipe": "",
            "confidence_score": 0.6,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
        results.append(entry)

    return results


# ---------------- writer ----------------
def writer_thread_fn(output: Path, q: "queue.Queue[Optional[dict]]"):
    global _temp_output_path
    tmp = (
        output.with_suffix(output.suffix + _TEMP_SUFFIX)
        if output.suffix
        else Path(str(output) + _TEMP_SUFFIX)
    )
    _temp_output_path = tmp
    tmp.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            while not _shutdown.is_set():
                try:
                    item = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if item is None:
                    break
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1
                q.task_done()
    except Exception:
        logging.exception("Writer thread exception")
        raise
    finally:
        logging.info("Writer finished. Wrote %d entries to %s", written, tmp)


# ---------------- worker ----------------
def parse_and_enrich_file(path: Path, hash_map: Dict[str, str]) -> List[Dict]:
    try:
        parsed = parse_file_with_treesitter(path)
        enriched = enrich_file(parsed, hash_map)
        return enriched
    except Exception:
        logging.exception("Failed to parse/enrich %s", path)
        return []


def _print_progress(files_done: int, files_total: int, funcs_written: int):
    pct = (files_done / files_total) if files_total else 0.0
    bar_width = 40
    filled = int(bar_width * pct)
    bar = "[" + "#" * filled + "-" * (bar_width - filled) + "]"
    sys.stderr.write(
        f"\r{bar} {pct*100:6.2f}% files {files_done}/{files_total} funcs={funcs_written}"
    )
    sys.stderr.flush()


# ---------------- main ----------------
def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    global _temp_output_path

    repo_dir: Path = args.repo_directory
    if not repo_dir.exists() or not repo_dir.is_dir():
        logging.error(
            "repo_directory %s is not a directory or does not exist", repo_dir
        )
        sys.exit(1)

    hash_map = load_index_hashes(INDEX_PATH) if INDEX_PATH else {}

    logging.info("Discovering files under %s ...", repo_dir)
    candidates = discover_files(repo_dir, args.extensions)
    total_files = len(candidates)
    logging.info("Discovered %d files matching %s", total_files, args.extensions)

    if total_files == 0:
        logging.info("No files found; exiting.")
        return

    rel_output_path = repo_dir / OUTPUT_PATH

    q: "queue.Queue[Optional[dict]]" = queue.Queue(maxsize=args.workers * 4)
    writer = threading.Thread(
        target=writer_thread_fn, args=(rel_output_path, q), daemon=True
    )
    writer.start()

    files_done = 0
    funcs_written = 0

    with ThreadPoolExecutor(max_workers=args.workers) as exc:
        futures = {
            exc.submit(parse_and_enrich_file, p, hash_map): p for p in candidates
        }

        for fut in as_completed(futures):
            if _shutdown.is_set():
                break
            src_path = futures[fut]
            try:
                enriched_list = fut.result()
            except Exception:
                enriched_list = []
                logging.exception("Worker failed for %s", src_path)

            for entry in enriched_list:
                try:
                    q.put(entry)
                    funcs_written += 1
                except Exception:
                    logging.exception("Failed to enqueue entry for %s", src_path)

            files_done += 1
            _print_progress(files_done, total_files, funcs_written)

    # finish up
    try:
        q.put(None)
    except Exception:
        pass
    writer.join(timeout=30)

    if _shutdown.is_set():
        logging.error("Interrupted. Partial output (if any) at %s", _temp_output_path)
        raise SystemExit(1)

    if _temp_output_path and Path(_temp_output_path).exists():
        rel_output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(_temp_output_path), str(rel_output_path))
        logging.info(
            "Atomically moved temp %s -> %s", _temp_output_path, rel_output_path
        )
    else:
        logging.error("Temporary output missing; nothing to move.")
        raise SystemExit(1)

    sys.stderr.write("\n")
    logging.info("Done: files=%d funcs=%d \n", total_files, funcs_written)


if __name__ == "__main__":
    main()
