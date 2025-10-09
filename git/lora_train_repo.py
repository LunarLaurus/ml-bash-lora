#!/usr/bin/env python3
"""
Fine-tune a 7B-8B code model using LoRA on a JSONL dataset extracted from a repo.
Supports interactive model selection and per-model output directories.
"""

import argparse
import logging
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from tqdm.auto import tqdm

print(f"[LoRA] torch={torch.__version__}, CUDA available={torch.cuda.is_available()}")
import peft

print(f"[LoRA] peft={peft.__version__}")


# Recommended models (add more as you like)
RECOMMENDED_MODELS = [
    "meta-llama/CodeLlama-7b-hf",
    "glaiveai/glaive-coder-7b",
    "open-r1/OlympicCoder-7B",
    "microsoft/NextCoder-7B",
    "google/codegemma-7b",
    "aixcoder-7b",
]


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class GPUUsageCallback(TrainerCallback):
    """Logs GPU memory usage during training steps."""

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            logging.info(
                f"[GPU] Step {state.global_step} - Memory Allocated: {allocated:.1f} MB, Reserved: {reserved:.1f} MB"
            )


def select_model(interactive=True, default_model=None):
    """
    If interactive, prompt user to select a model from RECOMMENDED_MODELS.
    Otherwise return default_model or raise if missing.
    """
    if default_model:
        logging.info(f"Using model from argument: {default_model}")
        return default_model

    if not interactive:
        raise ValueError("No model provided and interactive mode disabled")

    print("\nSelect a model to fine-tune:")
    for i, m in enumerate(RECOMMENDED_MODELS, 1):
        print(f"{i}) {m}")
    while True:
        choice = input(f"Enter number (1-{len(RECOMMENDED_MODELS)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(RECOMMENDED_MODELS):
            selected = RECOMMENDED_MODELS[int(choice) - 1]
            logging.info(f"Selected model: {selected}")
            return selected
        print("Invalid choice, try again.")


def main():
    setup_logger()

    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning on code JSONL dataset"
    )
    parser.add_argument("dataset", help="Path to extracted JSONL dataset")
    parser.add_argument(
        "--output_dir",
        default="./lora_repo",
        help="Base output folder for LoRA adapters",
    )
    parser.add_argument("--base_model", help="Base code model to fine-tune")
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

    base_model = select_model(
        interactive=(args.base_model is None), default_model=args.base_model
    )

    # Unique per-model output directory
    model_safe_name = base_model.replace("/", "_")
    output_dir = os.path.join(args.output_dir, model_safe_name)
    os.makedirs(output_dir, exist_ok=True)

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

    logging.info(f"Loading tokenizer and base model ({base_model})...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", load_in_8bit=True
    )
    model = prepare_model_for_kbit_training(model)

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

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(
        f"Total parameters: {total_params:,}, trainable (LoRA): {trainable_params:,}"
    )

    logging.info("Tokenizing dataset...")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_length)

    # Show tqdm progress bar for mapping
    tokenized_dataset = dataset.map(
        tokenize, batched=True, num_proc=args.num_proc, desc="Tokenizing", disable=False
    )
    tokenized_dataset.set_format(type="torch", columns=["input_ids"])

    logging.info("Preparing training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
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
        callbacks=[GPUUsageCallback],
    )

    try:
        trainer.train()
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return

    logging.info(f"Saving LoRA adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    logging.info("LoRA fine-tuning complete!")


if __name__ == "__main__":
    main()
