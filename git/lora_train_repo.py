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
    BitsAndBytesConfig,
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


def configure_lora(interactive=True):
    """
    Interactive configuration for LoRA fine-tuning.
    Returns a dictionary with configuration values:
      - bnb_config (BitsAndBytesConfig or None)
      - torch_dtype (torch dtype or None)
      - training_args (dict)
      - lora_args (dict)
      - checkpoint_args (dict)
    """
    cfg = {}

    # ---------------------------
    # Quantization / Precision
    # ---------------------------
    if interactive:
        print("\nSelect quantization / precision (default=8-bit):")
        print("1) 8-bit (with CPU offload)")
        print("2) 4-bit (with CPU offload)")
        print("3) FP16")
        print("4) BF16")
        print("5) FP32 (no quantization)")
        choice = input("Enter 1-5 [default=1]: ").strip() or "1"
    else:
        choice = "1"  # default 8-bit

    # Initialize bnb_config and torch_dtype
    bnb_config = None
    torch_dtype = None

    if choice == "1":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True
        )
    elif choice == "2":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif choice == "3":
        torch_dtype = torch.float16
    elif choice == "4":
        torch_dtype = torch.bfloat16
    elif choice == "5":
        torch_dtype = None

    cfg["bnb_config"] = bnb_config
    cfg["torch_dtype"] = torch_dtype
    logging.info(f"Quantization/precision selected: {choice}")

    # ---------------------------
    # Memory / device map
    # ---------------------------
    max_mem_input = input(
        "\nMax GPU memory per device (e.g., '80%' or '8GB') [default=80%]: "
    ).strip()
    cfg["max_memory"] = {"0": max_mem_input or "80%"}

    # ---------------------------
    # Training hyperparameters
    # ---------------------------
    print("\nTraining hyperparameters (press Enter to keep defaults)")
    batch_size = input("Batch size per device [default=1]: ").strip()
    grad_accum = input("Gradient accumulation steps [default=4]: ").strip()
    max_length = input("Max sequence length (tokens) [default=1024]: ").strip()
    epochs = input("Number of training epochs [default=3]: ").strip()
    learning_rate = input("Learning rate [default=1e-4]: ").strip()

    cfg["training_args"] = {
        "per_device_train_batch_size": int(batch_size) if batch_size else 1,
        "gradient_accumulation_steps": int(grad_accum) if grad_accum else 4,
        "max_length": int(max_length) if max_length else 1024,
        "num_train_epochs": int(epochs) if epochs else 3,
        "learning_rate": float(learning_rate) if learning_rate else 1e-4,
        "fp16": torch_dtype == torch.float16,
        "bf16": torch_dtype == torch.bfloat16,
    }

    # ---------------------------
    # LoRA hyperparameters
    # ---------------------------
    print("\nLoRA hyperparameters (press Enter to keep defaults)")
    r = input("LoRA rank r [default=16]: ").strip()
    alpha = input("LoRA alpha [default=32]: ").strip()
    dropout = input("LoRA dropout [default=0.1]: ").strip()
    target_modules = input(
        "Target modules (comma-separated) [default=q_proj,v_proj]: "
    ).strip()

    cfg["lora_args"] = {
        "r": int(r) if r else 16,
        "lora_alpha": int(alpha) if alpha else 32,
        "lora_dropout": float(dropout) if dropout else 0.1,
        "target_modules": (
            [m.strip() for m in target_modules.split(",")]
            if target_modules
            else ["q_proj", "v_proj"]
        ),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    # ---------------------------
    # Checkpoint / saving options
    # ---------------------------
    print("\nCheckpointing options (press Enter to keep defaults)")
    save_strategy = input("Save strategy (epoch/steps) [default=epoch]: ").strip()
    save_total_limit = input("Number of checkpoints to keep [default=2]: ").strip()

    cfg["checkpoint_args"] = {
        "save_strategy": save_strategy or "epoch",
        "save_total_limit": int(save_total_limit) if save_total_limit else 2,
    }

    return cfg


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


def load_dataset_from_file(dataset_path):
    if not os.path.isfile(dataset_path):
        logging.error(f"Dataset file '{dataset_path}' does not exist.")
        return None

    logging.info(f"Loading dataset from {dataset_path}...")
    try:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return None

    logging.info(f"Dataset loaded: {len(dataset)} samples")
    return dataset


def prepare_tokenizer_and_model(base_model, hf_token, bnb_config=None):
    logging.info(f"Loading tokenizer and base model ({base_model})...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        quantization_config=bnb_config,
        use_auth_token=hf_token,
    )
    model = prepare_model_for_kbit_training(model)
    return tokenizer, model


def configure_lora_model(model, r=16, alpha=32, dropout=0.1, target_modules=None):
    target_modules = target_modules or ["q_proj", "v_proj"]
    logging.info("Configuring LoRA...")
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(
        f"Total parameters: {total_params:,}, trainable (LoRA): {trainable_params:,}"
    )
    return model


def prepare_training_args(
    output_dir,
    batch_size=1,
    grad_accum=4,
    epochs=3,
    learning_rate=1e-4,
    max_length=1024,
):
    logging.info("Preparing training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        fp16=True,
        save_total_limit=2,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )
    logging.info(f"Training arguments: {training_args}")
    return training_args


def main():
    setup_logger()
    hf_token = get_hf_token()

    # -------------------------
    # Argument parsing
    # -------------------------
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
        "--dry_run", action="store_true", help="Preview dataset stats without training"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Enable interactive configuration"
    )
    args = parser.parse_args()

    # -------------------------
    # Dataset loading
    # -------------------------
    dataset = load_dataset_from_file(args.dataset)
    if dataset is None or args.dry_run:
        logging.info("Dry run: exiting after dataset preview")
        return

    # -------------------------
    # Model selection
    # -------------------------
    base_model = select_model(
        interactive=(args.base_model is None), default_model=args.base_model
    )

    # -------------------------
    # Output directory
    # -------------------------
    model_safe_name = base_model.replace("/", "_")
    output_dir = os.path.join(args.output_dir, model_safe_name)
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # LoRA & training configuration
    # -------------------------
    cfg = configure_lora(interactive=args.interactive)

    bnb_config = cfg["bnb_config"]
    torch_dtype = cfg["torch_dtype"]
    training_args_cfg = cfg["training_args"]
    lora_args_cfg = cfg["lora_args"]
    checkpoint_args_cfg = cfg["checkpoint_args"]

    # -------------------------
    # Tokenizer & model loading
    # -------------------------
    tokenizer, model = prepare_tokenizer_and_model(base_model, hf_token, bnb_config)
    model = configure_lora_model(model, **lora_args_cfg)

    # -------------------------
    # Tokenize dataset
    # -------------------------
    tokenized_dataset = get_tokenized_dataset(
        dataset,
        tokenizer,
        output_dir,
        max_length=training_args_cfg.get("max_length", 1024),
        num_proc=8,
    )
    tokenized_dataset.set_format(type="torch", columns=["input_ids"])

    # -------------------------
    # TrainingArguments setup
    # -------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=training_args_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=training_args_cfg["gradient_accumulation_steps"],
        num_train_epochs=training_args_cfg["num_train_epochs"],
        learning_rate=training_args_cfg["learning_rate"],
        fp16=training_args_cfg.get("fp16", False),
        bf16=training_args_cfg.get("bf16", False),
        save_strategy=checkpoint_args_cfg.get("save_strategy", "epoch"),
        save_total_limit=checkpoint_args_cfg.get("save_total_limit", 2),
        logging_steps=10,
        report_to="none",
    )

    # -------------------------
    # Trainer initialization & training
    # -------------------------
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[GPUUsageCallback],
    )

    logging.info("Starting LoRA fine-tuning...")
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
