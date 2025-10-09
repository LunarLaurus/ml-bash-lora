#!/usr/bin/env python3
"""
Fine-tune a 7B-8B code model using LoRA on a JSONL dataset extracted from a repo.
Supports interactive model selection, per-model output directories, and automatic Hugging Face login.
"""
import argparse
import logging
import os
import subprocess
from pathlib import Path

import torch
from tqdm.auto import tqdm

from datasets import load_dataset, Dataset
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
import peft

print(f"[LoRA] torch={torch.__version__}, CUDA available={torch.cuda.is_available()}")
print(f"[LoRA] peft={peft.__version__}")

# Recommended models (can expand)
RECOMMENDED_MODELS = [
    "meta-llama/CodeLlama-7b-hf",
    "glaiveai/glaive-coder-7b",
    "open-r1/OlympicCoder-7B",
    "microsoft/NextCoder-7B",
    "google/codegemma-7b",
    "aixcoder-7b",
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    "yasserrmd/Coder-GRPO-3B",
    "Novora/CodeClassifier-v1-Tiny",
]

HF_TOKEN_FILE = Path.home() / ".huggingface/token"


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_hf_token():
    """
    Returns Hugging Face token to use for model downloads.
    Priority:
      1) Huggingface CLI login token (if available)
      2) Local token file (~/.huggingface/token)
      3) Prompt user for token and save it
    """
    # 1️⃣ Try CLI login
    try:
        result = subprocess.run(
            ["huggingface-cli", "whoami"], capture_output=True, text=True, check=True
        )
        logging.info(f"Hugging Face CLI authenticated as: {result.stdout.strip()}")
        return None  # CLI manages token automatically
    except (FileNotFoundError, subprocess.CalledProcessError):
        logging.info(
            "HF CLI not found or not logged in, falling back to token file/prompt"
        )

    # 2️⃣ Try token file
    if HF_TOKEN_FILE.exists():
        token = HF_TOKEN_FILE.read_text().strip()
        if token:
            logging.info(f"Using Hugging Face token from {HF_TOKEN_FILE}")
            return token

    # 3️⃣ Prompt user
    token = input(
        "Enter your Hugging Face access token (gated/private models): "
    ).strip()
    HF_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    HF_TOKEN_FILE.write_text(token)
    logging.info(f"Token saved to {HF_TOKEN_FILE}")
    return token


class GPUUsageCallback(TrainerCallback):
    """Displays GPU usage alongside a live progress bar."""

    def __init__(self):
        super().__init__()
        self.pbar = None

    def on_train_begin(self, args, state, control, **kwargs):
        # Initialize tqdm for training steps
        self.pbar = tqdm(total=state.max_steps, desc="Training", unit="step")

    def on_step_end(self, args, state, control, **kwargs):
        self.pbar.update(1)
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            self.pbar.set_postfix(
                {
                    "GPU Alloc(MB)": f"{allocated:.0f}",
                    "GPU Reserved(MB)": f"{reserved:.0f}",
                }
            )

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar is not None:
            self.pbar.close()


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


def get_tokenized_dataset(dataset, tokenizer, output_dir, max_length, num_proc):
    cache_file = Path(output_dir) / "tokenized_dataset.arrow"

    if cache_file.exists():
        logging.info(f"Loading tokenized dataset from cache: {cache_file}")
        from datasets import load_from_disk

        tokenized_dataset = load_from_disk(cache_file)
    else:
        logging.info("Tokenizing dataset...")

        def tokenize(examples):
            return tokenizer(examples["text"], truncation=True, max_length=max_length)

        tokenized_dataset = dataset.map(
            tokenize, batched=True, num_proc=num_proc, desc="Tokenizing", disable=False
        )
        tokenized_dataset.set_format(type="torch", columns=["input_ids"])
        tokenized_dataset.save_to_disk(cache_file)
        logging.info(f"Tokenized dataset cached at: {cache_file}")

    return tokenized_dataset


def main():
    setup_logger()
    hf_token = get_hf_token()

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
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", load_in_8bit=True, use_auth_token=hf_token
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

    tokenized_dataset = get_tokenized_dataset(
        dataset,
        tokenizer,
        output_dir,
        max_length=args.max_length,
        num_proc=args.num_proc,
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
