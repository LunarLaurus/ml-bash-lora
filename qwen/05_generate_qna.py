#!/usr/bin/env python3
"""
05_generate_qna.py

Generate multi-question Q&A pairs from enriched function and file summaries for LoRA training.

Inputs:
    - data/enriched_functions.jsonl (output of Step 04)
    - optional: data/linked_functions.jsonl (header -> impl mapping)

Outputs:
    - data/qna_train.jsonl
"""

import json
from pathlib import Path
import sys

ENRICHED_FUNCTIONS = Path("data/enriched_functions.jsonl")
LINKED_FUNCTIONS = Path("data/linked_functions.jsonl")
QNA_OUTPUT = Path("data/qna_train.jsonl")


def load_jsonl(path: Path):
    if not path.exists():
        return []
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def generate_qna(entries, linked_entries=None):
    """
    Returns a list of Q&A dictionaries.
    Each entry may generate multiple Qs (summary, risk, intent, change recipe).
    """
    qnas = []
    linked_map = {}
    if linked_entries:
        # map function_id -> list of headers
        for le in linked_entries:
            linked_map.setdefault(le["function_id"], []).append(le["header_file"])

    for entry in entries:
        file_path = entry.get("file_path", "<unknown file>")
        fn_id = entry.get("id", "")
        fn_name = entry.get("function", {}).get("name", "<unknown>")

        if not fn_id.startswith("file:"):
            # Function-level Q&A
            base_question = f"Function '{fn_name}' in file '{file_path}'"
            qnas.append(
                {
                    "question": f"What does {base_question} do?",
                    "answer": entry.get("summary", ""),
                    "level": "function",
                }
            )
            if entry.get("risk_notes"):
                qnas.append(
                    {
                        "question": f"What are the potential risks of {base_question}?",
                        "answer": entry["risk_notes"],
                        "level": "function",
                    }
                )
            if entry.get("intent_tags"):
                qnas.append(
                    {
                        "question": f"What intent tags apply to {base_question}?",
                        "answer": ", ".join(entry["intent_tags"]),
                        "level": "function",
                    }
                )
            if entry.get("change_recipe"):
                qnas.append(
                    {
                        "question": f"What changes might be needed for {base_question}?",
                        "answer": entry["change_recipe"],
                        "level": "function",
                    }
                )
            if fn_id in linked_map:
                headers = linked_map[fn_id]
                qnas.append(
                    {
                        "question": f"Which headers declare {base_question}?",
                        "answer": ", ".join(headers),
                        "level": "function",
                    }
                )

        else:
            # File-level Q&A
            base_question = f"File '{file_path}'"
            qnas.append(
                {
                    "question": f"What is the overall purpose of {base_question}?",
                    "answer": entry.get("summary", ""),
                    "level": "file",
                }
            )
            if entry.get("risk_notes"):
                qnas.append(
                    {
                        "question": f"What are the potential risks in {base_question}?",
                        "answer": entry["risk_notes"],
                        "level": "file",
                    }
                )
            if entry.get("intent_tags"):
                qnas.append(
                    {
                        "question": f"What intent tags apply to {base_question}?",
                        "answer": ", ".join(entry["intent_tags"]),
                        "level": "file",
                    }
                )
            if entry.get("change_recipe"):
                qnas.append(
                    {
                        "question": f"What changes might be needed in {base_question}?",
                        "answer": entry["change_recipe"],
                        "level": "file",
                    }
                )

    return qnas


def write_qna(qnas, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for q in qnas:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"[INFO] Wrote {len(qnas)} Q&A entries to {output_path}")


# ------------------ entry point ------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/04_link_headers.py <repo_directory>")
        sys.exit(1)

    repo_dir = Path(sys.argv[1])
    enriched_entries = load_jsonl(ENRICHED_FUNCTIONS)
    linked_entries = load_jsonl(LINKED_FUNCTIONS)
    qnas = generate_qna(enriched_entries, linked_entries)

    if not repo_dir.exists():
        print("Repo dir not found: " + repo_dir)
        sys.exit(1)
    if not ENRICHED_FUNCTIONS.exists():
        print("Enriched functions file not found: " + ENRICHED_FUNCTIONS)
        sys.exit(1)
    if not LINKED_FUNCTIONS.exists():
        print("Linked functions file not found: " + LINKED_FUNCTIONS)
        sys.exit(1)

    rel_output_path = repo_dir / QNA_OUTPUT

    write_qna(qnas, rel_output_path)
