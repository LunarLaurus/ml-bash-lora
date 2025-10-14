#!/usr/bin/env python3
"""
02b_build_dependency_graphs.py

Reads a JSONL of enriched function entries (default: data/parsed_functions.jsonl),
builds function- and file-level dependency graphs, computes basic metrics (in/out degree,
SCC/cycles), and writes two JSONL outputs:
  - data/dep_graph_functions.jsonl
  - data/dep_graph_files.jsonl

Only uses Python standard library.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Defaults
INPUT_PATH = Path("data/parsed_functions.jsonl")
OUTPUT_PATH_FUNCS = Path("data/dep_graph_functions.jsonl")
OUTPUT_PATH_FILES = Path("data/dep_graph_files.jsonl")
_TEMP_SUFFIX = ".tmp"

# graceful shutdown
_shutdown = threading.Event()
_temp_files: List[Path] = []


def _signal_handler(signum, frame):
    logging.warning("Received signal %s — requesting shutdown...", signum)
    _shutdown.set()


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


@dataclass
class FuncNode:
    id: str
    repo: str
    file_path: str
    name: str
    start_line: int
    end_line: int
    callers: List[str]
    callees: List[str]
    calls_external: List[str]
    called_by_external: List[str]
    in_degree: int = 0
    out_degree: int = 0
    scc_id: Optional[int] = None
    in_cycle: bool = False


@dataclass
class FileNode:
    file_path: str
    repo: str
    calls_to_files: List[str]
    called_by_files: List[str]
    in_degree: int = 0
    out_degree: int = 0
    scc_id: Optional[int] = None
    in_cycle: bool = False


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(
        description="Build dependency graphs from parsed function JSONL"
    )
    p.add_argument(
        "repo_directory", type=Path, help="Repository root (used for resolving)."
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _atomic_writer(path: Path, items: Iterable[dict]) -> None:
    tmp = (
        path.with_suffix(path.suffix + _TEMP_SUFFIX)
        if path.suffix
        else Path(str(path) + _TEMP_SUFFIX)
    )
    _temp_files.append(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(str(tmp), str(path))
    _temp_files.remove(tmp)


def load_parsed_functions(parsed_path: Path) -> List[dict]:
    """
    Load all parsed function entries from the JSONL file.
    Each line expected to be a JSON object with at least:
      - id
      - file_path
      - function.name
      - function.start_line
      - function.end_line
      - callers (optional)
      - callees (optional)
    Returns list of dicts.
    """
    res: List[dict] = []
    if not parsed_path.exists():
        raise FileNotFoundError(f"Parsed functions file not found: {parsed_path}")
    with parsed_path.open("r", encoding="utf-8") as fh:
        for ln in fh:
            if _shutdown.is_set():
                break
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                res.append(obj)
            except Exception:
                logging.debug(
                    "Skipping invalid JSON line in %s: %.200s", parsed_path, ln
                )
    return res


def build_mappings(
    parsed_entries: List[dict],
) -> Tuple[Dict[str, dict], Dict[Tuple[str, str], List[str]]]:
    """
    Build:
      - func_by_id: mapping function id -> full entry dict
      - functions_by_name: mapping (file_path, function_name) -> list of function ids (usually 1)
    """
    func_by_id: Dict[str, dict] = {}
    functions_by_name: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for e in parsed_entries:
        fid = e.get("id")
        if not fid:
            # if no id, build one from file_path/name/start_line
            file_path = e.get("file_path") or e.get("file")
            fn = (e.get("function") or {}).get("name") or "<unknown>"
            sl = (e.get("function") or {}).get("start_line") or 0
            fid = f"repo:{file_path}:{fn}:{sl}"
            e["id"] = fid
        func_by_id[fid] = e

        file_path = str(Path(e.get("file_path") or "").resolve())
        fn_name = (e.get("function") or {}).get("name") or "<unknown>"
        functions_by_name[(file_path, fn_name)].append(fid)

    return func_by_id, functions_by_name


def link_functions(
    func_by_id: Dict[str, dict],
    functions_by_name: Dict[Tuple[str, str], List[str]],
) -> Tuple[Dict[str, FuncNode], Dict[str, Set[str]]]:
    """
    Construct function-level graph edges by resolving callees/callers.
    Returns:
      - nodes: mapping func_id -> FuncNode (with lists of callers/callees and external lists)
      - file_calls: mapping file_path -> set of files it calls
    """
    nodes: Dict[str, FuncNode] = {}
    file_calls: Dict[str, Set[str]] = defaultdict(set)

    # Build initial nodes with empty relationships
    for fid, e in func_by_id.items():
        fn = e.get("function", {})
        nodes[fid] = FuncNode(
            id=fid,
            repo=e.get("repo", ""),
            file_path=str(Path(e.get("file_path", "")).resolve()),
            name=fn.get("name", "<unknown>"),
            start_line=fn.get("start_line", 0) or 0,
            end_line=fn.get("end_line", 0) or 0,
            callers=list(e.get("callers", []) or []),
            callees=list(e.get("callees", []) or []),
            calls_external=[],
            called_by_external=[],
        )

    # Resolve callee names: entries may contain callee names not full ids.
    # Strategy:
    #  - If callee looks like a full id (contains ':' and a path) and present in func_by_id -> link directly.
    #  - Else, search for functions with the same name across all files (prefer same file first).
    #  - If unresolved, treat as external symbol.
    # Build a map name->func ids for faster lookup (across files)
    name_to_ids: Dict[str, List[str]] = defaultdict(list)
    for fid, node in nodes.items():
        name_to_ids[node.name].append(fid)

    # Now resolve edges
    for fid, node in list(nodes.items()):
        resolved_callees: List[str] = []
        external_callees: List[str] = []
        for callee in node.callees:
            if not callee:
                continue
            # direct id reference?
            if callee in func_by_id:
                resolved_callees.append(callee)
                continue
            # maybe fully-qualified like "repo:/abs/path:func:line"
            if ":" in callee and callee in func_by_id:
                resolved_callees.append(callee)
                continue
            # prefer functions in same file
            same_file_key = (node.file_path, callee)
            if same_file_key in functions_by_name:
                resolved_callees.extend(functions_by_name[same_file_key])
                continue
            # otherwise, any function with that name
            if callee in name_to_ids:
                resolved_callees.extend(name_to_ids[callee])
                continue
            # unresolved -> external symbol
            external_callees.append(callee)

        # dedupe
        resolved_callees = list(dict.fromkeys(resolved_callees))
        external_callees = list(dict.fromkeys(external_callees))

        node.callees = resolved_callees
        node.calls_external = external_callees

    # Build caller lists (reverse edges)
    for fid, node in nodes.items():
        for callee_id in node.callees:
            if callee_id in nodes:
                nodes[callee_id].callers.append(fid)

    # Normalize callers lists and compute degrees
    for fid, node in nodes.items():
        node.callers = list(dict.fromkeys(node.callers))
        node.in_degree = len(node.callers)
        node.out_degree = len(node.callees) + len(node.calls_external)
        # update file-level calls
        src_file = node.file_path
        for callee_id in node.callees:
            tgt = nodes.get(callee_id)
            if tgt:
                file_calls[src_file].add(tgt.file_path)

    return nodes, file_calls


# ---------------- SCC / cycle detection ----------------
def _kosaraju_scc(adj: Dict[str, List[str]]) -> Dict[str, int]:
    """
    Kosaraju's algorithm to compute strongly connected components.
    Returns mapping node -> scc_id (int starting at 0).
    """
    # first pass: order by finish time on original graph
    visited: Set[str] = set()
    order: List[str] = []

    def dfs(u: str):
        visited.add(u)
        for v in adj.get(u, ()):
            if v not in visited:
                dfs(v)
        order.append(u)

    for u in adj:
        if u not in visited:
            dfs(u)

    # transpose graph
    t_adj: Dict[str, List[str]] = {}
    for u in adj:
        for v in adj.get(u, ()):
            t_adj.setdefault(v, []).append(u)
    for u in adj:
        t_adj.setdefault(u, [])

    # second pass: discover components in reverse finish order
    comp_id = 0
    scc_map: Dict[str, int] = {}
    visited.clear()

    def dfs2(u: str, cid: int):
        visited.add(u)
        scc_map[u] = cid
        for v in t_adj.get(u, ()):
            if v not in visited:
                dfs2(v, cid)

    for u in reversed(order):
        if u not in visited:
            dfs2(u, comp_id)
            comp_id += 1

    return scc_map


# ---------------- file-level graph assembly ----------------
def build_file_nodes(file_calls: Dict[str, Set[str]]) -> Dict[str, FileNode]:
    """
    Build file-level nodes and compute degrees.
    file_calls: src_file -> set of tgt_files
    """
    nodes: Dict[str, FileNode] = {}
    all_files = set(file_calls.keys()) | {
        t for vals in file_calls.values() for t in vals
    }
    for f in all_files:
        nodes[f] = FileNode(
            file_path=f,
            repo=str(Path.cwd().name),
            calls_to_files=[],
            called_by_files=[],
        )
    for src, targets in file_calls.items():
        for tgt in targets:
            nodes[src].calls_to_files.append(tgt)
            nodes[tgt].called_by_files.append(src)
    # dedupe & degrees
    for f, n in nodes.items():
        n.calls_to_files = sorted(dict.fromkeys(n.calls_to_files))
        n.called_by_files = sorted(dict.fromkeys(n.called_by_files))
        n.out_degree = len(n.calls_to_files)
        n.in_degree = len(n.called_by_files)
    return nodes


# ---------------- main builder ----------------
def build_and_write_graphs(
    repo_dir: Path, parsed_path: Path, out_funcs: Path, out_files: Path
) -> None:
    start = time.time()
    entries = load_parsed_functions(parsed_path)
    logging.info("Loaded %d parsed function entries.", len(entries))

    func_by_id, functions_by_name = build_mappings(entries)
    logging.info(
        "Built mappings: %d functions, %d unique (file,name) keys",
        len(func_by_id),
        len(functions_by_name),
    )

    nodes, file_calls = link_functions(func_by_id, functions_by_name)
    logging.info(
        "Linked functions -> nodes=%d file_call_srcs=%d", len(nodes), len(file_calls)
    )

    # build adjacency for SCC detection (function-level)
    adj: Dict[str, List[str]] = {fid: list(node.callees) for fid, node in nodes.items()}
    # ensure all nodes present
    for fid in nodes:
        adj.setdefault(fid, [])

    scc_map = _kosaraju_scc(adj)
    logging.info("Computed %d SCCs", len(set(scc_map.values())))

    # annotate nodes with scc id and whether in cycle (component size > 1)
    comp_sizes: Dict[int, int] = defaultdict(int)
    for v, cid in scc_map.items():
        comp_sizes[cid] += 1
    for fid, node in nodes.items():
        cid = scc_map.get(fid)
        node.scc_id = cid
        node.in_cycle = cid is not None and comp_sizes.get(cid, 0) > 1

    # file-level nodes + SCCs
    file_nodes = build_file_nodes(file_calls)
    # file adjacency for SCC detection
    f_adj: Dict[str, List[str]] = {
        f: list(n.calls_to_files) for f, n in file_nodes.items()
    }
    for f in file_nodes:
        f_adj.setdefault(f, [])
    f_scc_map = _kosaraju_scc(f_adj)
    f_comp_sizes: Dict[int, int] = defaultdict(int)
    for v, cid in f_scc_map.items():
        f_comp_sizes[cid] += 1
    for fp, fn in file_nodes.items():
        cid = f_scc_map.get(fp)
        fn.scc_id = cid
        fn.in_cycle = cid is not None and f_comp_sizes.get(cid, 0) > 1

    # Prepare function objects for output
    func_items = []
    for fid, node in nodes.items():
        func_items.append(
            {
                "id": node.id,
                "repo": node.repo,
                "file_path": node.file_path,
                "function_name": node.name,
                "start_line": node.start_line,
                "end_line": node.end_line,
                "callers": node.callers,
                "callees": node.callees,
                "calls_external": node.calls_external,
                "called_by_external": node.called_by_external,
                "in_degree": node.in_degree,
                "out_degree": node.out_degree,
                "scc_id": node.scc_id,
                "in_cycle": node.in_cycle,
            }
        )

    file_items = []
    for fp, fn in file_nodes.items():
        file_items.append(
            {
                "file_path": fn.file_path,
                "repo": fn.repo,
                "calls_to_files": fn.calls_to_files,
                "called_by_files": fn.called_by_files,
                "in_degree": fn.in_degree,
                "out_degree": fn.out_degree,
                "scc_id": fn.scc_id,
                "in_cycle": fn.in_cycle,
            }
        )

    # atomic writes
    logging.info("Writing %d function items to %s", len(func_items), out_funcs)
    _atomic_writer(out_funcs, func_items)
    logging.info("Writing %d file items to %s", len(file_items), out_files)
    _atomic_writer(out_files, file_items)

    elapsed = time.time() - start
    logging.info(
        "Done: functions=%d files=%d elapsed=%.2fs",
        len(func_items),
        len(file_items),
        elapsed,
    )


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
    rel_output_path_funcs = repo_dir / OUTPUT_PATH_FUNCS
    rel_output_path_files = repo_dir / OUTPUT_PATH_FILES
    rel_input_path = repo_dir / INPUT_PATH

    try:
        build_and_write_graphs(
            repo_dir, rel_input_path, rel_output_path_funcs, rel_output_path_files
        )
    except KeyboardInterrupt:
        logging.error("Interrupted.")
        sys.exit(1)
    except Exception:
        logging.exception("Failed to build graphs")
        sys.exit(1)


if __name__ == "__main__":
    main()
