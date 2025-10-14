#!/usr/bin/env python3
"""
04_link_headers.py

Links header files (.h) to their corresponding implementations using parsed function
entries and dependency graph information from prior scripts.

Inputs:
    - repo_directory: repository root
    - parsed_functions.jsonl: output from 02_parse_code.py
    - dep_graph_functions.jsonl: output from 02b_build_dependency_graphs.py

Outputs:
    - linked_functions.jsonl: mapping of header files to implementation files
"""

import json
import os
import logging
from pathlib import Path
import sys
from typing import List, Dict

INPUT_GRAPH_FUNCTIONS_PATH = Path("data/dep_graph_functions.jsonl")
INPUT_FUNCTIONS_PATH = Path("data/enriched_functions.jsonl")
OUTPUT_PATH = Path("data/linked_functions.jsonl")


# ------------------ helper ------------------
def find_implementations(header_ast: dict, repo_directory: Path) -> List[Dict]:
    """
    Placeholder: find functions in repo that implement this header.
    Currently, we assume function names match those declared in header AST.
    """
    impls = []
    for root, dirs, files in os.walk(repo_directory):
        for file in files:
            if file.endswith((".c", ".cpp")):
                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        code = f.read()
                        # naive: check if any function name in header AST is present in code
                        for fn in header_ast.get("functions", []):
                            if fn.get("name") and fn["name"] in code:
                                impls.append(
                                    {"file_name": file, "file_path": str(file_path)}
                                )
                except Exception:
                    continue
    return impls


# ------------------ main linking ------------------
def link_headers(
    repo_directory: Path,
    parsed_functions_path: Path,
    dep_graph_path: Path,
    output_path: Path,
):
    linked_functions = []

    # Load parsed functions (enriched with dependency graph info)
    parsed_entries = []
    with parsed_functions_path.open("r", encoding="utf-8") as f:
        for line in f:
            parsed_entries.append(json.loads(line))

    # Optionally load dependency graph if needed for advanced linking
    dep_graph = {}
    if dep_graph_path.exists():
        with dep_graph_path.open("r", encoding="utf-8") as f:
            dep_graph = json.load(f)

    # Map file_path -> parsed function entries
    file_to_funcs = {}
    for entry in parsed_entries:
        file_to_funcs.setdefault(entry["file_path"], []).append(entry)

    # Traverse headers in repo
    for root, dirs, files in os.walk(repo_directory):
        for file in files:
            if file.endswith(".h"):
                header_path = Path(root) / file
                funcs_in_header = file_to_funcs.get(str(header_path), [])

                # Skip if no functions found
                if not funcs_in_header:
                    continue

                for func_entry in funcs_in_header:
                    ast = func_entry.get("ast", {})
                    implementations = find_implementations(ast, repo_directory)
                    for impl in implementations:
                        linked_functions.append(
                            {
                                "header_file": file,
                                "header_path": str(header_path),
                                "implementation_file": impl["file_name"],
                                "implementation_path": impl["file_path"],
                                "function_id": func_entry.get("id"),
                            }
                        )

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for entry in linked_functions:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[INFO] Linked {len(linked_functions)} header-function implementations.")
    return linked_functions


# ------------------ entry point ------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/04_link_headers.py <repo_directory>")
        sys.exit(1)

    repo_dir = Path(sys.argv[1])

    if not repo_dir.exists():
        print("Repo dir not found: " + repo_dir)
        sys.exit(1)

    rel_output_path = repo_dir / OUTPUT_PATH
    rel_input_func_path = repo_dir / INPUT_FUNCTIONS_PATH
    rel_input_graph_path = repo_dir / INPUT_GRAPH_FUNCTIONS_PATH

    if not rel_input_func_path.exists():
        logging.error("Parsed functions file not found: %s", rel_input_func_path)
        sys.exit(1)
    if not rel_input_graph_path.exists():
        logging.error("Dependency graph file not found: %s", rel_input_graph_path)
        sys.exit(1)

    link_headers(repo_dir, rel_input_func_path, rel_input_graph_path, rel_output_path)
