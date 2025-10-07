#!/usr/bin/env python3
"""
Fine-tune a 7B code model using LoRA on a JSONL dataset extracted from a repo
"""

import argparse
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning on a code JSONL dataset")
    parser.add_argument("dataset", help="Path to extracted JSONL dataset")
    parser.add_argument("--output_dir", default="./lora_repo", help="Output folder for LoRA adapter")
    parser.add_argument("--base_model", default="codellama/7b-hf", help="Base code model")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()

    print(f"[INFO] Loading dataset from {args.dataset}...")
    dataset = load_dataset("json", data_files=args.dataset, split="train")

    print(f"[INFO] Loading tokenizer and base model ({args.base_model})...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map="auto", load_in_8bit=True
    )
    model = prepare_model_for_int8_training(model)

    print("[INFO] Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    print("[INFO] Tokenizing dataset...")
    def tokenize(ex):
        return tokenizer(ex["text"], truncation=True, max_length=args.max_length)
    tokenized_dataset = dataset.map(tokenize, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids"])

    print("[INFO] Starting LoRA fine-tuning...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        learning_rate=1e-4,
        fp16=True,
        save_total_limit=2,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()
    print(f"[INFO] Saving LoRA adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    print("[INFO] LoRA fine-tuning complete!")

if __name__ == "__main__":
    main()
