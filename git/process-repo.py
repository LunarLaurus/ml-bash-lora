#!/usr/bin/env python3
"""
Foolproof code dataset extractor for Pokémon decomp repos (Gen I–IV)

Extracts:
- C code: functions, structs, enums, globals
- ASM code: labeled blocks
- Full files (optional)
Outputs JSONL for LoRA fine-tuning

Optional: generate description with a code model (commented out)
"""

import os
import json
import sys
import re

# -------------------------------
# Check dependencies
# -------------------------------
missing = []

try:
    from tree_sitter import Language, Parser
except ImportError:
    missing.append("tree_sitter")
try:
    from tqdm import tqdm
except ImportError:
    missing.append("tqdm")

if missing:
    print(f"[ERROR] Missing dependencies: {', '.join(missing)}")
    print("Install with: pip install " + " ".join(missing))
    sys.exit(1)

# -------------------------------
# Build C parser
# -------------------------------
C_LANG_SO = "build/my-languages.so"
if not os.path.exists(C_LANG_SO):
    print("[INFO] Building C parser for Tree-sitter...")
    Language.build_library(
        C_LANG_SO,
        ['tree-sitter-c']
    )
C_LANGUAGE = Language(C_LANG_SO, 'c')
C_PARSER = Parser()
C_PARSER.set_language(C_LANGUAGE)

# -------------------------------
# Extraction functions
# -------------------------------
def extract_c_units(file_path, context_lines=10):
    """Extract functions, structs, enums, globals from a C file"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        code = f.read()
        lines = code.splitlines()
    tree = C_PARSER.parse(bytes(code, 'utf8'))
    root = tree.root_node
    samples = []

    def walk(node):
        # C unit types
        if node.type in ("function_definition", "struct_specifier", "enum_specifier", "declaration"):
            start, end = node.start_byte, node.end_byte
            unit_code = code[start:end]
            # prepend file context
            ctx = "\n".join(lines[:context_lines])
            samples.append({
                "type": node.type,
                "name": node.child_by_field_name("name").text.decode("utf-8") if node.child_by_field_name("name") else None,
                "text": ctx + "\n" + unit_code
            })
        for child in node.children:
            walk(child)

    walk(root)
    return samples

def extract_asm_units(file_path):
    """Extract labeled blocks from assembly"""
    samples = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        code = f.read()
    # naive regex for labels
    blocks = re.split(r"^([A-Za-z0-9_]+:)", code, flags=re.MULTILINE)
    if len(blocks) <= 1:
        return [{"type": "asm_block", "name": os.path.basename(file_path), "text": code}]
    for i in range(1, len(blocks), 2):
        label = blocks[i].rstrip(":").strip()
        body = blocks[i + 1] if i + 1 < len(blocks) else ""
        samples.append({
            "type": "asm_block",
            "name": label,
            "text": body
        })
    return samples

def extract_units_from_file(file_path):
    if file_path.endswith((".c", ".h")):
        return extract_c_units(file_path)
    elif file_path.endswith((".s", ".asm")):
        return extract_asm_units(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return [{"type": "full_file", "name": os.path.basename(file_path), "text": f.read()}]

def walk_repo(repo_path):
    """Walk repo and extract units from all files"""
    samples = []
    for root, dirs, files in os.walk(repo_path):
        for f in files:
            path = os.path.join(root, f)
            try:
                units = extract_units_from_file(path)
                for u in units:
                    u["source_file"] = os.path.relpath(path, repo_path)
                    samples.append(u)
            except Exception as e:
                print(f"[WARNING] Failed to parse {path}: {e}")
    return samples

def save_jsonl(samples, out_file):
    with open(out_file, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[INFO] Saved {len(samples)} units to {out_file}")

# -------------------------------
# Optional: generate descriptions (disabled by default)
# -------------------------------
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# model_name = "codellama/7b-hf"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
# for sample in samples:
#     prompt = f"Describe what the following code does:\n\n{sample['text']}\n\nDescription:"
#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#     output = model.generate(**inputs, max_length=128)
#     sample["description"] = tokenizer.decode(output[0], skip_special_tokens=True)

# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract Pokémon repo code units to JSONL")
    parser.add_argument("repo", help="Path to repo")
    parser.add_argument("--out", default="dataset.jsonl", help="Output JSONL file")
    args = parser.parse_args()

    samples = walk_repo(args.repo)
    save_jsonl(samples, args.out)
