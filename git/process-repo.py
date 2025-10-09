#!/usr/bin/env python3
"""
Modern process-repo.py for extracting code units (C/ASM/other) from Pokémon decomp reposwith optional assist model integration

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
# (imports remain unchanged from your previous script)
from __future__ import annotations
import argparse, hashlib, importlib, json, os, re, subprocess, sys, time, math
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from tqdm import tqdm

try:
    from tree_sitter import Language, Parser
except Exception as e:
    print("[ERROR] Missing 'tree_sitter' package:", e)
    sys.exit(1)


# ------------------------- utilities -------------------------
def sha1_short(s: str, length: int = 12) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:length]


def read_file_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    except Exception as e:
        print(f"[WARNING] Could not read {path}: {e}")
        return None


# (Tree-sitter, extraction, chunking, dedupe/filter, git_diff_pairs, save_jsonl, walk_repo_collect functions unchanged)


# ------------------------- assist model helpers -------------------------
def load_assist_model(model_name: str, device: str = None):
    """Load small causal LM + tokenizer for assistive processing."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except Exception as e:
        print(f"[ERROR] transformers not installed or import failed: {e}")
        return None, None, None
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading assist model {model_name} on {device} ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    try:
        model.to(device)
    except Exception:
        pass
    return tok, model, device


def generate_with_model(
    tokenizer,
    model,
    prompts: List[str],
    device="cpu",
    max_new_tokens=128,
    batch_size=4,
    do_sample=False,
    temperature=0.0,
    stop_token: Optional[str] = None,
):
    """Batch generate outputs for a list of prompts."""
    import torch

    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True,
                num_beams=1 if not do_sample else 4,
            )
        for j, g in enumerate(gen):
            txt = tokenizer.decode(g, skip_special_tokens=True)
            prompt = batch_prompts[j].strip()
            out = txt[len(prompt) :].strip() if txt.startswith(prompt) else txt.strip()
            if stop_token and stop_token in out:
                out = out.split(stop_token)[0].strip()
            outputs.append(out)
        time.sleep(0.01)
    return outputs


# ------------------------- process chunks + assist integration -------------------------
def process_and_chunk(
    samples,
    args,
    tokenizer=None,
    assist_tok=None,
    assist_model=None,
    assist_device=None,
):
    processed = []
    assist_prompts = []
    assist_rec_indices = []
    assist_done = 0

    def flush_assist_queue(force_all=False):
        nonlocal assist_prompts, assist_rec_indices, assist_done, processed
        if not assist_prompts:
            return
        if args.assist_limit > 0:
            remaining_allowed = args.assist_limit - assist_done
            if remaining_allowed <= 0:
                assist_prompts, assist_rec_indices = [], []
                return
            if not force_all and len(assist_prompts) > remaining_allowed:
                assist_prompts = assist_prompts[:remaining_allowed]
                assist_rec_indices = assist_rec_indices[:remaining_allowed]
        if not assist_prompts:
            return
        print(f"[INFO] Generating {len(assist_prompts)} assist outputs ...")
        try:
            gen_outs = generate_with_model(
                assist_tok,
                assist_model,
                assist_prompts,
                device=assist_device,
                max_new_tokens=args.assist_max_tokens,
                batch_size=args.assist_batch,
                do_sample=False,
            )
        except Exception as e:
            print(f"[WARN] Assist model generation failed: {e}")
            assist_prompts, assist_rec_indices = [], []
            return
        for idx_in_queue, out_text in enumerate(gen_outs):
            rec_idx = assist_rec_indices[idx_in_queue]
            base_rec = processed[rec_idx]
            if args.assist_mode == "summarize":
                instruction = "Summarize the purpose of this code:"
            else:
                instruction = "Suggest a short, safe improvement or minimal patch:"
            input_payload = (
                f"{instruction}\n\n### Code:\n{base_rec.get('text','')}\n\n### Output:"
            )
            seqrec = {
                "type": "seq2seq_pair",
                "name": base_rec.get("name"),
                "text": None,
                "source_file": base_rec.get("source_file"),
                "start_line": base_rec.get("start_line"),
                "end_line": base_rec.get("end_line"),
                "language": base_rec.get("language"),
                "sha1": sha1_short((base_rec.get("text", "") + out_text)[:10000]),
                "kind": "seq2seq",
                "input": input_payload,
                "target": out_text or "",
            }
            processed.append(seqrec)
            assist_done += 1
        assist_prompts, assist_rec_indices = [], []

    use_token_chunking = bool(args.token_chunking and tokenizer)
    flush_threshold = max(1, args.assist_batch * 4)

    for unit in tqdm(samples, desc="Chunking units + assist"):
        text = (unit.get("text") or "").strip()
        if not text:
            continue

        chunks = []
        if use_token_chunking:
            try:
                chunks = token_aware_chunk(
                    text, tokenizer, args.chunk_tokens, args.stride_tokens
                )
            except Exception:
                use_token_chunking = False
        if not use_token_chunking:
            chunks = chunk_by_lines(text, args.chunk_lines, args.stride_lines)
        for c in chunks:
            if use_token_chunking:
                chunk_text, s_tok, e_tok = c
                rec = {
                    "type": "chunk",
                    "name": unit.get("name"),
                    "text": chunk_text,
                    "source_file": unit.get("source_file"),
                    "start_line": unit.get("start_line"),
                    "end_line": unit.get("end_line"),
                    "language": unit.get("language"),
                    "sha1": None,
                    "kind": "causal",
                    "token_start": int(s_tok),
                    "token_end": int(e_tok),
                }
            else:
                chunk_text, s_line, e_line = c
                rec = {
                    "type": "chunk",
                    "name": unit.get("name"),
                    "text": chunk_text,
                    "source_file": unit.get("source_file"),
                    "start_line": int(unit.get("start_line", 1) + (s_line - 1)),
                    "end_line": int(unit.get("start_line", 1) + (e_line - 1)),
                    "language": unit.get("language"),
                    "sha1": None,
                    "kind": "causal",
                }
            processed.append(rec)
            cur_idx = len(processed) - 1

            if (
                assist_tok
                and assist_model
                and (args.assist_limit == 0 or assist_done < args.assist_limit)
            ):
                prompt = (
                    f"Summarize this code:\n{chunk_text}\n"
                    if args.assist_mode == "summarize"
                    else f"Suggest a patch for this code:\n{chunk_text}\n"
                )
                assist_prompts.append(prompt)
                assist_rec_indices.append(cur_idx)
                if len(assist_prompts) >= flush_threshold:
                    flush_assist_queue()
    flush_assist_queue(force_all=True)

    for r in processed:
        if not r.get("sha1"):
            r["sha1"] = sha1_short((r.get("text") or r.get("input") or "")[:10000])
    print(
        f"[INFO] Chunking complete. {len(processed)} records including assist outputs."
    )
    return processed


# ------------------------- main -------------------------
def main():
    args = parse_args()
    repo = args.repo
    if not os.path.isdir(repo):
        print(f"[ERROR] repo not found: {repo}")
        sys.exit(1)

    # Load C parser
    c_parser = None
    if not args.no_c_parser:
        lang_handle, hint = try_load_c_language()
        if lang_handle:
            c_parser = make_parser_from_lang(lang_handle)

    # Ask user about assist model if not specified
    if not args.assist_model:
        answer = (
            input(
                "Do you want to use an assist model to generate summaries/patches? [y/N]: "
            )
            .strip()
            .lower()
        )
        if answer == "y":
            args.assist_model = "google/flan-t5-small"
            print(f"[INFO] Using default assist model: {args.assist_model}")

    assist_tok = assist_model = assist_device = None
    if args.assist_model:
        assist_tok, assist_model, assist_device = load_assist_model(args.assist_model)
        if not assist_tok:
            assist_tok = assist_model = assist_device = None

    # Tokenizer for token-aware chunking
    tokenizer = None
    if args.token_chunking:
        if not args.tokenizer:
            print("[ERROR] --token-chunking requires --tokenizer")
            sys.exit(1)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Walk repo
    print(f"[INFO] Extracting units from {repo} ...")
    samples = walk_repo_collect(repo, c_parser)

    # Optional seq2seq from git
    seq_pairs = []
    if args.make_seq2seq_from_git:
        print("[INFO] Extracting seq2seq pairs from git history ...")
        seq_pairs = git_diff_pairs(repo)
        if seq_pairs:
            save_jsonl(seq_pairs, args.out_seq)

    # Process chunks + assist
    processed = process_and_chunk(
        samples, args, tokenizer, assist_tok, assist_model, assist_device
    )

    if args.dedupe:
        processed = dedupe_and_filter(processed, args.min_chars, args.min_lines)
    else:
        for r in processed:
            if not r.get("sha1"):
                r["sha1"] = sample_hash(r)

    # Shuffle & validation
    if args.shuffle:
        import random

        random.shuffle(processed)
    val_records = []
    if args.val_frac > 0.0:
        nval = max(1, int(len(processed) * args.val_frac))
        val_records = processed[:nval]
        train_records = processed[nval:]
    else:
        train_records = processed
    if args.max_records > 0:
        train_records = train_records[: args.max_records]

    # Save
    save_jsonl(train_records, args.out)
    if args.val_out and val_records:
        save_jsonl(val_records, args.val_out)
    if args.make_seq2seq_from_git and seq_pairs:
        print(f"[INFO] Generated {len(seq_pairs)} git seq2seq pairs to {args.out_seq}")


if __name__ == "__main__":
    main()
