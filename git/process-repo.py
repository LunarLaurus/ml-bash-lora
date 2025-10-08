#!/usr/bin/env python3
"""
Modern process-repo.py for extracting code units (C/ASM/other) from Pokémon decomp repos.

This script:
 - Uses modern py-tree-sitter language packages when available (e.g. `tree-sitter-c`)
 - Falls back to helpful instructions if required modern packages are not present
 - Extracts C units (functions/structs/enums/declarations) via tree-sitter
 - Extracts ASM labeled blocks via regex
 - Emits a JSONL dataset usable for LoRA / fine-tuning

Requirements (modern path):
    pip install tree-sitter tree-sitter-c tqdm

Notes:
 - For C parsing we prefer the PyPI language package `tree-sitter-c`. This is the modern/recommended approach.
 - If you want to support additional languages via PyPI, install their respective `tree-sitter-<lang>` packages.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
from typing import List, Tuple, Optional

from tqdm import tqdm

# -------------------------------
# Try to import py-tree-sitter core
# -------------------------------
try:
    from tree_sitter import Language, Parser
except Exception as e:
    print("[ERROR] Missing or incompatible 'tree_sitter' package:", e)
    print("Install with: pip install tree-sitter")
    sys.exit(1)


# -------------------------------
# Helpers to load the C language (modern)
# -------------------------------
def try_load_c_language() -> Tuple[Optional[object], Optional[str]]:
    """
    Attempt to obtain a C language object in the modern style.

    Returns (language_obj, hint_string).
    - language_obj: object usable with Parser (or None if not found)
    - hint_string: user-facing hint if we couldn't load from packages
    """
    # 1) Most modern & recommended: language-specific Python package (py-tree-sitter style)
    #    e.g. `pip install tree-sitter-c` -> import name: `tree_sitter_c` (exposes `.language()`).
    candidates = [
        "tree_sitter_c",  # common naming for the pip package tree-sitter-c
        "tree_sitter_c_lang",  # some packages use alternate names (rare)
        "tree_sitter_languages",  # package that exposes get_language/get_parser
        "tree_sitter_language_pack",  # alternative package names
    ]

    for cname in candidates:
        try:
            mod = importlib.import_module(cname)
        except Exception:
            continue

        # Most language modules expose a `language()` function that returns a language handle
        if hasattr(mod, "language") and callable(getattr(mod, "language")):
            try:
                lang_handle = mod.language()
                # The py-tree-sitter README shows using Language(lang_handle)
                # but many language modules already expose the underlying language object.
                # Return the handle; the caller will try to construct Parser appropriately.
                return lang_handle, None
            except Exception:
                # if calling language() failed, continue searching
                continue

        # Some helpers expose `get_language(name)` / `get_parser(name)`
        if hasattr(mod, "get_language") and callable(getattr(mod, "get_language")):
            try:
                lang_handle = mod.get_language("c")
                return lang_handle, None
            except Exception:
                continue

    # 2) If we got here, the modern pip language package for C wasn't found.
    hint = (
        "Install a modern Python tree-sitter C grammar package and retry:\n"
        "  pip install tree-sitter-c\n\n"
        "Also make sure `tree-sitter` core is installed:\n"
        "  pip install tree-sitter\n\n"
        "After installing, rerun this script."
    )
    return None, hint


def make_parser_from_lang(lang_handle) -> Parser:
    """
    Create a Parser configured with lang_handle.
    Handles small API differences in py-tree-sitter versions.
    """
    # Some language handles are already 'Language' instances that can be passed directly
    # to Parser(language) or parser.set_language(language). We'll try a few patterns.
    # If lang_handle is already a tree_sitter.Language, this will work.
    try:
        # Preferred: construct Parser with language
        parser = Parser(lang_handle)
        return parser
    except Exception:
        pass

    # Try Parser() then set_language (older API)
    try:
        parser = Parser()
        parser.set_language(lang_handle)
        return parser
    except Exception:
        pass

    # As last resort, try wrapping the handle into Language(...) (some packages require this)
    try:
        wrapped = Language(lang_handle)
        parser = Parser(wrapped)
        return parser
    except Exception as e:
        raise RuntimeError(f"Failed to create Parser from language handle: {e}")


# -------------------------------
# Extraction functions
# -------------------------------
def read_file_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    except Exception as e:
        print(f"[WARNING] Could not read {path}: {e}")
        return None


def extract_c_units_tree_sitter(
    file_path: str, parser: Parser, context_lines: int = 8
) -> List[dict]:
    """
    Use a tree-sitter Parser to extract C units (functions, structs, enums, declarations).
    """
    text = read_file_text(file_path)
    if text is None:
        return []

    src_bytes = text.encode("utf8")
    try:
        tree = parser.parse(src_bytes)
    except Exception as e:
        print(f"[WARNING] tree-sitter parse failed for {file_path}: {e}")
        return []

    root = tree.root_node
    samples = []
    lines = text.splitlines()

    def node_text(node):
        return src_bytes[node.start_byte : node.end_byte].decode(
            "utf8", errors="ignore"
        )

    def get_node_name(node):
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return None
        try:
            return src_bytes[name_node.start_byte : name_node.end_byte].decode(
                "utf8", errors="ignore"
            )
        except Exception:
            return None

    def context_for_node(node):
        # Use start_point row to pick surrounding lines for context
        start_row = node.start_point[0]  # zero-based
        from_row = max(0, start_row - context_lines)
        to_row = min(len(lines), start_row + context_lines)
        return "\n".join(lines[from_row:to_row])

    def walk(node):
        if node.type in (
            "function_definition",
            "struct_specifier",
            "enum_specifier",
            "declaration",
        ):
            samples.append(
                {
                    "type": node.type,
                    "name": get_node_name(node),
                    "text": context_for_node(node) + "\n" + node_text(node),
                }
            )
        # walk children
        for ch in node.children:
            walk(ch)

    walk(root)
    return samples


def extract_asm_units(file_path: str) -> List[dict]:
    """
    Simple regex-based ASM labeled-block extractor. Keeps previous behavior.
    """
    text = read_file_text(file_path)
    if text is None:
        return []
    blocks = re.split(r"^([A-Za-z0-9_]+:)", text, flags=re.MULTILINE)
    if len(blocks) <= 1:
        return [
            {"type": "asm_block", "name": os.path.basename(file_path), "text": text}
        ]
    samples = []
    for i in range(1, len(blocks), 2):
        label = blocks[i].rstrip(":").strip()
        body = blocks[i + 1] if (i + 1) < len(blocks) else ""
        samples.append({"type": "asm_block", "name": label, "text": body})
    return samples


def extract_units_from_file(path: str, c_parser: Optional[Parser]) -> List[dict]:
    if path.endswith((".c", ".h")):
        if c_parser:
            return extract_c_units_tree_sitter(path, c_parser)
        else:
            # No parser available: return whole file as fallback (still useful)
            text = read_file_text(path)
            return (
                [{"type": "full_file", "name": os.path.basename(path), "text": text}]
                if text is not None
                else []
            )
    elif path.endswith((".s", ".asm")):
        return extract_asm_units(path)
    else:
        text = read_file_text(path)
        return (
            [{"type": "full_file", "name": os.path.basename(path), "text": text}]
            if text is not None
            else []
        )


def walk_repo(repo_path: str, c_parser: Optional[Parser]) -> List[dict]:
    samples = []
    for root, dirs, files in tqdm(list(os.walk(repo_path)), desc="Walking repo"):
        for fname in files:
            full = os.path.join(root, fname)
            try:
                units = extract_units_from_file(full, c_parser)
            except Exception as e:
                print(f"[WARNING] Failed to extract {full}: {e}")
                units = []
            for u in units:
                u["source_file"] = os.path.relpath(full, repo_path)
                samples.append(u)
    return samples


def save_jsonl(samples: List[dict], out: str) -> None:
    with open(out, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[INFO] Saved {len(samples)} units to {out}")


# -------------------------------
# CLI
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract Pokémon repo code units to JSONL (modern py-tree-sitter)"
    )
    parser.add_argument("repo", help="Path to repo")
    parser.add_argument("--out", default="dataset.jsonl", help="Output JSONL file")
    parser.add_argument(
        "--no-c-parser",
        action="store_true",
        help="Skip tree-sitter C parsing (useful if not installed)",
    )
    args = parser.parse_args()

    # Load modern C language (preferred)
    c_parser = None
    if not args.no_c_parser:
        lang_handle, hint = try_load_c_language()
        if lang_handle is None:
            print("[WARN] Could not find a modern tree-sitter C language package.")
            print(hint)
            print(
                "[WARN] Falling back to whole-file extraction for C files. (Use --no-c-parser to silence.)"
            )
            c_parser = None
        else:
            try:
                # Wrap into Parser appropriately
                c_parser = make_parser_from_lang(lang_handle)
            except Exception as e:
                print(f"[ERROR] Failed to make parser from language handle: {e}")
                print(
                    "Hint: ensure you installed the language package (e.g. `pip install tree-sitter-c`) and that versions are compatible with `tree-sitter`."
                )
                sys.exit(1)

    print(f"[INFO] Extracting code dataset from '{args.repo}' into '{args.out}'...")
    samples = walk_repo(args.repo, c_parser)
    save_jsonl(samples, args.out)


if __name__ == "__main__":
    main()
