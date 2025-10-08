#!/usr/bin/env python3
"""
Fine-tune a 7B code model using LoRA on a JSONL dataset extracted from a repo
"""

import argparse
import logging
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
import torch


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    setup_logger()

    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning on a code JSONL dataset"
    )
    parser.add_argument("dataset", help="Path to extracted JSONL dataset")
    parser.add_argument(
        "--output_dir", default="./lora_repo", help="Output folder for LoRA adapter"
    )
    parser.add_argument(
        "--base_model", default="codellama/3b-hf", help="Base code model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size per device"
    )
    parser.add_argument(
        "--grad_accum", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_length", type=int, default=1024, help="Max sequence length (tokens)"
    )
    parser.add_argument(
        "--num_proc", type=int, default=8, help="CPU cores for tokenization"
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Preview dataset stats without training"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.dataset):
        logging.error(f"Dataset file '{args.dataset}' does not exist.")
        return

    logging.info(f"Loading dataset from {args.dataset}...")
    try:
        dataset = load_dataset("json", data_files=args.dataset, split="train")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    logging.info(f"Dataset loaded: {len(dataset)} samples")
    if args.dry_run:
        logging.info("Dry run: exiting after dataset preview")
        return

    logging.info(f"Loading tokenizer and base model ({args.base_model})...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map="auto", load_in_8bit=True
    )
    model = prepare_model_for_int8_training(model)

    logging.info("Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(
        f"Total parameters: {total_params:,}, trainable parameters (LoRA): {trainable_params:,}"
    )

    logging.info("Tokenizing dataset...")

    def tokenize(ex):
        return tokenizer(ex["text"], truncation=True, max_length=args.max_length)

    tokenized_dataset = dataset.map(tokenize, batched=True, num_proc=args.num_proc)
    tokenized_dataset.set_format(type="torch", columns=["input_ids"])

    logging.info("Preparing training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=1e-4,
        fp16=True,
        save_total_limit=2,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )
    logging.info(f"Training arguments: {training_args}")

    logging.info("Starting LoRA fine-tuning...")
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        args=training_args,
    )

    try:
        trainer.train()
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return

    logging.info(f"Saving LoRA adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    logging.info("LoRA fine-tuning complete!")


if __name__ == "__main__":
    main()
