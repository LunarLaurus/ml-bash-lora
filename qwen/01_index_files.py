#!/usr/bin/env python3
"""
01_index_files.py — discover files (by extension), collect metadata → JSONL

Behavior changes per request:
- Only CLI option: --extensions (defaults to .c .h .asm)
- Always computes SHA256 for every file
- Uses only Python standard library
- Dynamically determines worker count
- Assumes repo root: skips .git
- Shows an in-terminal progress bar and logs useful stats
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

# ---------- Configuration / sensible defaults ----------
DEFAULT_EXTENSIONS = [".c", ".h", ".asm"]
OUTPUT_PATH = Path("data/file_index.jsonl")
_TEMP_SUFFIX = ".tmp"
# Determine workers dynamically for IO-bound tasks (conservative cap)
_CPU = os.cpu_count() or 1
DEFAULT_MAX_WORKERS = max(2, min(32, _CPU * 4))

# ---------- Globals for graceful shutdown ----------
_shutdown = threading.Event()
_temp_output_path: Optional[Path] = None


def _signal_handler(signum, frame):
    logging.warning("Received signal %s — requesting shutdown...", signum)
    _shutdown.set()


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Index files into JSONL (always hashes).")
    p.add_argument(
        "repo_directory",
        type=Path,
        help="Root directory (assumed to be a git repo root). .git will be ignored.",
    )
    p.add_argument(
        "--extensions",
        nargs="+",
        default=DEFAULT_EXTENSIONS,
        help="File extensions to include (e.g. .c .h .asm). Default: .c .h .asm",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Number of worker threads (default: auto = {DEFAULT_MAX_WORKERS}).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    return p.parse_args(argv)


def _iter_candidate_paths(root: Path, extensions: Iterable[str]) -> List[Path]:
    """
    Gather and return a list of candidate file paths.
    We do a quick discovery pass to enable a progress bar with a total count.
    Skips any directory named .git (assumes repo root).
    """
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    candidates: List[Path] = []
    total_bytes = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        if _shutdown.is_set():
            break
        # skip git internals and their typical dirs
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for fname in filenames:
            if _shutdown.is_set():
                break
            if Path(fname).suffix.lower() not in exts:
                continue
            p = Path(dirpath) / fname
            try:
                st = p.stat()
            except (FileNotFoundError, PermissionError):
                # file may have disappeared or be unreadable; skip
                continue
            candidates.append(p)
            total_bytes += st.st_size
    return candidates, total_bytes


def _compute_sha256(path: Path) -> Optional[str]:
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                if _shutdown.is_set():
                    return None
                h.update(chunk)
    except (FileNotFoundError, PermissionError) as exc:
        logging.debug("Hash failed for %s: %s", path, exc)
        return None
    return h.hexdigest()


def _gather_file_metadata(path: Path, repo_root: Path) -> Optional[dict]:
    try:
        st = path.stat()
    except (FileNotFoundError, PermissionError) as exc:
        logging.debug("Stat failed for %s: %s", path, exc)
        return None

    try:
        rel = path.relative_to(repo_root)
    except Exception:
        rel = Path(os.path.relpath(path, repo_root))

    sha256 = _compute_sha256(path)
    if sha256 is None and _shutdown.is_set():
        return None

    entry = {
        "file_name": path.name,
        "file_path": str(path.resolve()),
        "file_size": st.st_size,
        "last_modified_time": datetime.fromtimestamp(st.st_mtime).isoformat(),
        "file_creation_time": datetime.fromtimestamp(st.st_ctime).isoformat(),
        "file_extension": path.suffix.lstrip("."),
        "repo_relative_path": str(rel.as_posix()),
        "sha256": sha256,
    }
    return entry


def writer_thread_fn(output: Path, q: "queue.Queue[Optional[dict]]"):
    """
    Consume metadata dicts from queue and write them to a temp JSONL file.
    Producer will enqueue None to signal completion.
    """
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


def _print_progress(
    completed: int, total: int, bytes_done: int, total_bytes: int, width: int = 40
):
    """
    Simple progress bar printed to stderr.
    Shows percent, counts, files/sec (approx), bytes progress.
    """
    pct = (completed / total) if total else 0.0
    filled = int(width * pct)
    bar = "[" + "#" * filled + "-" * (width - filled) + "]"
    percent_text = f"{pct*100:6.2f}%"
    files_text = f"{completed}/{total}"
    bytes_text = f"{_humanize_bytes(bytes_done)}/{_humanize_bytes(total_bytes)}"
    sys.stderr.write(f"\r{bar} {percent_text} files {files_text} bytes {bytes_text}")
    sys.stderr.flush()


def _humanize_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def index_files(
    repo_directory: Path, output_path: Path, extensions: Iterable[str], workers: int
) -> int:
    global _temp_output_path  # ← moved here (top of function)

    start = time.time()
    logging.info("Scanning for files under %s", repo_directory)
    candidates, total_bytes = _iter_candidate_paths(repo_directory, extensions)
    total = len(candidates)
    logging.info(
        "Discovered %d files matching %s (total bytes ~ %s)",
        total,
        list(extensions),
        _humanize_bytes(total_bytes),
    )

    if total == 0:
        logging.info("No files found. Exiting.")
        return 0

    q: "queue.Queue[Optional[dict]]" = queue.Queue(maxsize=workers * 4)
    writer = threading.Thread(
        target=writer_thread_fn, args=(output_path, q), daemon=True
    )
    writer.start()

    indexed = 0
    failures = 0
    bytes_done = 0

    futures = []
    fut_to_size = {}

    with ThreadPoolExecutor(max_workers=workers) as exc:
        for p in candidates:
            if _shutdown.is_set():
                break
            fut = exc.submit(_gather_file_metadata, p, repo_directory)
            futures.append(fut)
            try:
                fut_to_size[fut] = p.stat().st_size
            except Exception:
                fut_to_size[fut] = 0

        for fut in as_completed(futures):
            if _shutdown.is_set():
                break
            try:
                result = fut.result()
            except Exception:
                logging.exception("Worker raised exception")
                result = None

            size = fut_to_size.get(fut, 0)
            if result:
                q.put(result)
                indexed += 1
                bytes_done += size
            else:
                failures += 1
                bytes_done += size

            _print_progress(indexed + failures, total, bytes_done, total_bytes)

    try:
        q.put(None)
    except Exception:
        pass

    writer.join(timeout=30)

    elapsed = time.time() - start
    if _shutdown.is_set():
        logging.error("Interrupted. Partial output (if any) at %s", _temp_output_path)
        raise SystemExit(1)

    # atomic move
    if _temp_output_path and Path(_temp_output_path).exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(_temp_output_path), str(output_path))
        logging.info("Atomically moved %s -> %s", _temp_output_path, output_path)
    else:
        logging.error("Temporary output missing; nothing to move.")
        raise SystemExit(1)

    sys.stderr.write("\n")
    logging.info(
        "Indexing complete: %d indexed, %d failures, elapsed %.2fs",
        indexed,
        failures,
        elapsed,
    )
    logging.info(
        "Throughput: %.2f files/sec, %.2f MB/sec",
        (indexed / elapsed) if elapsed > 0 else 0.0,
        (total_bytes / elapsed / (1024 * 1024)) if elapsed > 0 else 0.0,
    )

    return indexed


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    repo_dir: Path = args.repo_directory
    if not repo_dir.exists() or not repo_dir.is_dir():
        logging.error(
            "repo_directory %s is not a directory or does not exist", repo_dir
        )
        sys.exit(1)

    rel_output_path = repo_dir / OUTPUT_PATH

    try:
        count = index_files(repo_dir, rel_output_path, args.extensions, args.workers)
        print(f"Indexed {count} files. Saved to {rel_output_path}")
    except Exception as exc:
        logging.exception("Indexing failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
