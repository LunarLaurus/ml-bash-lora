#!/usr/bin/env python3
"""
06_train_lora.py

Train a LoRA-adapted Seq2Seq model using all prior pipeline data:
- parsed functions
- function-level dependency graphs
- file-level graphs
- enriched summaries
- header-function links
- generated Q&A

Outputs:
- ./results: LoRA-fine-tuned model
"""

import json
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

# ----------------------- pipeline outputs -----------------------
DEFAULT_PARSED = Path("data/parsed_functions.jsonl")
OUT_FUNCS = Path("data/dep_graph_functions.jsonl")
OUT_FILES = Path("data/dep_graph_files.jsonl")
ENRICHED_OUTPUT = Path("data/enriched_functions.jsonl")
LINKED_OUTPUT = Path("data/linked_functions.jsonl")
QNA_OUTPUT = Path("data/qna_train.jsonl")


# ----------------------- data loading -------------------------
def load_jsonl(path: Path):
    data = []
    if not path.exists():
        print(f"[WARN] File not found: {path}")
        return data
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    data.append(json.loads(ln))
                except Exception:
                    continue
    return data


def build_training_dataset():
    """
    Merge all available info into training examples.
    Input: question/answer pairs + enriched summaries + headers + graph info
    Output: Dataset for seq2seq training
    """
    dataset_entries = []

    # 1. Use Q&A directly
    qna_entries = load_jsonl(QNA_OUTPUT)
    for q in qna_entries:
        dataset_entries.append(
            {"input_text": q["question"], "target_text": q["answer"]}
        )

    # 2. Function-level enrichment
    enriched = load_jsonl(ENRICHED_OUTPUT)
    func_entries = [e for e in enriched if not str(e.get("id", "")).startswith("file:")]
    for fn in func_entries:
        context_parts = [
            f"Function: {fn.get('function', {}).get('name','<unknown>')}",
            f"File: {fn.get('file_path','<unknown>')}",
            f"LOC: {fn.get('loc',0)}, Cyclomatic: {fn.get('cyclomatic',0)}",
            f"Intent tags: {','.join(fn.get('intent_tags',[]))}",
            f"Risk notes: {fn.get('risk_notes','')}",
            f"Callers: {','.join(fn.get('callers',[]))}",
            f"Callees: {','.join(fn.get('callees',[]))}",
        ]
        context = " | ".join(context_parts)
        answer = fn.get("summary", "")
        dataset_entries.append({"input_text": context, "target_text": answer})

    # 3. File-level summaries
    file_entries = [e for e in enriched if str(e.get("id", "")).startswith("file:")]
    for f in file_entries:
        context = f"File: {f.get('file_path','<unknown>')} | LOC: {f.get('loc',0)} | Functions: {len(f.get('functions',[]))} | Intent: {','.join(f.get('intent_tags',[]))}"
        answer = f.get("summary", "")
        dataset_entries.append({"input_text": context, "target_text": answer})

    # 4. Header links
    header_links = load_jsonl(LINKED_OUTPUT)
    for link in header_links:
        context = f"Header: {link.get('header_file','')} | Implementation: {link.get('implementation_file','')} | Function ID: {link.get('function_id','')}"
        answer = f"Linking header {link.get('header_file','')} to implementation {link.get('implementation_file','')}."
        dataset_entries.append({"input_text": context, "target_text": answer})

    return Dataset.from_list(dataset_entries)


# ----------------------- LoRA training -------------------------
def train_lora(model_name="Salesforce/codet5-small", epochs=3, batch_size=4, lr=1e-4):
    dataset = build_training_dataset()
    print(f"[INFO] Training dataset size: {len(dataset)} examples")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Tokenize
    def tokenize_fn(ex):
        model_inputs = tokenizer(
            ex["input_text"], truncation=True, padding="max_length", max_length=256
        )
        labels = tokenizer(
            ex["target_text"], truncation=True, padding="max_length", max_length=256
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(tokenize_fn, batched=True)

    # LoRA config
    peft_config = LoraConfig(
        task_type="SEQ_2_SEQ_LM",
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="no",
        save_total_limit=2,
        fp16=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("[INFO] Starting LoRA training...")
    trainer.train()
    print("[INFO] LoRA training complete. Model saved in ./results")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    repo_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    train_lora()
