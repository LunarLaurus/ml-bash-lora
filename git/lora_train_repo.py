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

LEARN_RATE_SUGGESTED = {
    "small": ["1e-4", "5e-5", "2e-4"],
    "medium": ["2e-4", "3e-4", "5e-4"],
    "large": ["5e-4", "1e-3"],
}

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


# Quantization / Precision
def configure_quantization(interactive=True):
    """
    Returns (bnb_config, torch_dtype) or (None, None) if skipped.
    """
    if not interactive:
        return (
            BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True
            ),
            None,
        )

    print("\nSelect quantization / precision (press Enter to skip):")
    options = {
        "1": "8-bit (with CPU offload)",
        "2": "4-bit (with CPU offload)",
        "3": "FP16",
        "4": "BF16",
        "5": "FP32 (no quantization)",
    }
    for k, v in options.items():
        print(f"{k}) {v}")
    logging.info("Select model precision / quantization. Default: 8-Bit.")
    choice = input("Enter 1-5 [skip]: ").strip()
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

    logging.info(f"Quantization/precision selected: {choice or 'skipped'}")
    return bnb_config, torch_dtype


# Memory / Device Map
def configure_memory(interactive=True):
    """
    Returns a dictionary suitable for max_memory, or None if skipped.
    """
    if not interactive:
        return None
    logging.info(
        "Set the maximum GPU memory per device. Example: '80%' or '8GB'. Default: skip for auto."
    )
    val = input("Max GPU memory per device [skip]: ").strip()
    return {"0": val} if val else None


# Training hyperparameters
def configure_training_args(interactive=True):
    """
    Returns a dictionary of training args or None if skipped.
    """
    defaults = {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_length": 1024,
        "num_train_epochs": 3,
        "learning_rate": "2e-4",
    }
    if not interactive:
        return defaults.copy()

    logging.info("Training hyperparameters (press Enter to skip any)")
    logging.info(
        "Batch size per device: Number of samples processed simultaneously on each GPU."
    )
    per_device_train_batch_size = input(
        f"Batch size per device (default {defaults['per_device_train_batch_size']}): "
    ).strip()

    logging.info(
        "Gradient accumulation steps: Number of steps to accumulate gradients before updating weights. Useful for simulating larger batch sizes."
    )
    gradient_accumulation_steps = input(
        f"Gradient accumulation steps (default {defaults['gradient_accumulation_steps']}): "
    ).strip()

    logging.info(
        "Max sequence length (tokens): Maximum number of tokens per training sample. Longer sequences use more memory."
    )
    max_length = input(
        f"Max sequence length in tokens (default {defaults['max_length']}): "
    ).strip()

    logging.info(
        "Number of training epochs: How many times the model will iterate over the entire dataset."
    )
    num_train_epochs = input(
        f"Number of training epochs (default {defaults['num_train_epochs']}): "
    ).strip()

    logging.info(
        "Learning rate: Step size for optimizer updates. Too high can destabilize training, too low slows convergence."
    )

    # Prompt user for model size
    logging.info("Enter model size:\n   small >= 3B\n   medium ~= 7B\n   large >= 13B")
    model_size = input("Size (default medium): ").strip()

    # Suggest learning rates
    suggested_lrs = LEARN_RATE_SUGGESTED.get(model_size, ["2e-4"])  # fallback default
    logging.info(
        "Typical learning rate choices depending on model size and fine-tuning strategy:"
    )

    logging.info("LoRA fine-tuning small model (1–3B): 1e-4, 5e-5, 2e-4")
    logging.info("LoRA fine-tuning medium model (7B): 2e-4, 3e-4, 5e-4")
    logging.info("LoRA fine-tuning large model (13B+): 5e-4, 1e-3 (rarely higher)")
    logging.info("Full model fine-tuning small: 1e-5 – 5e-5")
    logging.info("Full model fine-tuning large: 1e-5 – 2e-5")
    logging.info("Extremely stable training / long sequences: 5e-5 or lower")
    logging.info("Aggressive quick adaptation: 3e-4 – 5e-4")

    logging.info(
        f"Recommended learning rates for model size {model_size}: {', '.join(suggested_lrs)}"
    )
    logging.info(
        "Lower LR = safer, slower training; higher LR = faster, more aggressive adaptation."
    )

    # Ask user to pick one or type custom
    learning_rate = input(
        f"Enter learning rate (default {defaults['learning_rate']} / suggested {suggested_lrs[0]}): "
    ).strip()
    learning_rate = float(learning_rate) if learning_rate else float(suggested_lrs[0])

    args = {
        "per_device_train_batch_size": (
            int(per_device_train_batch_size)
            if per_device_train_batch_size
            else defaults["per_device_train_batch_size"]
        ),
        "gradient_accumulation_steps": (
            int(gradient_accumulation_steps)
            if gradient_accumulation_steps
            else defaults["gradient_accumulation_steps"]
        ),
        "max_length": int(max_length) if max_length else defaults["max_length"],
        "num_train_epochs": (
            int(num_train_epochs) if num_train_epochs else defaults["num_train_epochs"]
        ),
        "learning_rate": (
            float(learning_rate) if learning_rate else defaults["learning_rate"]
        ),
    }

    return args


# LoRA hyperparameters
def configure_lora_args(interactive=True):
    """
    Returns a dictionary of LoRA args or None if skipped.
    """
    defaults = {
        "use_rslora": False,
        "use_dora": False,
        "init_lora_weights": True,
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
        "task_type": "SEQ_2_SEQ_LM",
    }
    if not interactive:
        return defaults.copy()

    logging.info("LoRA hyperparameters (press Enter to skip any)")

    logging.info(
        "use_rslora: If True, uses Rank-Stabilized LoRA. "
        "This sets the adapter scaling factor to lora_alpha / sqrt(r), which often improves stability. "
    )
    use_rslora = (
        input(f"Use Rank-Stabilized LoRA? (default {defaults['use_rslora']}): ")
        .strip()
        .lower()
    )
    use_rslora = True if use_rslora in ["true", "t", "yes", "y"] else False

    logging.info(
        "use_dora: Enable Weight-Decomposed Low-Rank Adaptation (DoRA). "
        "DoRA decomposes weight updates into magnitude and direction for better performance at low ranks. "
        "Only supports linear and Conv2D layers. Introduces higher training overhead."
    )
    use_dora = input(f"Enable DoRA? (default {defaults['use_dora']}): ").strip().lower()
    use_dora = True if use_dora in ["true", "t", "yes", "y"] else False

    # init_lora_weights
    logging.info(
        "init_lora_weights: How to initialize LoRA adapter weights. Options: "
        "   True (default, reference implementation, B weights=0), "
        "   False (random, not a no-op), 'gaussian', 'eva', 'olora', 'pissa', 'pissa_niter_[num]', 'corda', 'loftq', 'orthogonal'."
    )
    init_lora_weights = input(
        f"Initialize LoRA weights (default {defaults['init_lora_weights']}): "
    ).strip()

    if init_lora_weights == "":
        init_lora_weights = True
    elif init_lora_weights.lower() in ["true", "t", "yes", "y"]:
        init_lora_weights = True
    elif init_lora_weights.lower() in ["false", "f", "no", "n"]:
        init_lora_weights = False

    logging.info(
        "LoRA rank r: Dimensionality of the low-rank decomposition for LoRA. Higher r = more capacity, more memory."
    )
    r = input(f"LoRA rank r (default {defaults['r']}): ").strip()

    logging.info(
        "LoRA alpha: Scaling factor for LoRA updates. Higher alpha increases adaptation strength."
    )
    alpha = input(f"LoRA alpha (default {defaults['lora_alpha']}): ").strip()

    logging.info(
        "LoRA dropout: Dropout probability applied to LoRA layers. Helps regularize training."
    )
    dropout = input(f"LoRA dropout (default {defaults['lora_dropout']}): ").strip()

    logging.info(
        "Target modules: Comma-separated list of model modules to apply LoRA to (e.g., 'q_proj,v_proj')."
    )
    target_modules = input(
        f"Target modules (comma-separated, default {','.join(defaults['target_modules'])}): "
    ).strip()

    logging.info(
        "Bias: Bias type for LoRA. Can be none, all or lora_only.\nIf all or lora_only, the corresponding biases will be updated during training. \nBe aware that this means that, even when disabling the adapters, the model will not produce the same output as the base model would have without adaptation."
    )
    bias = input(f"Bias (default {defaults['bias']}): ").strip()

    logging.info(
        "Overview of types of tasks supported by PEFT:\n\
            SEQ_CLS: Text classification. \n\
            SEQ_2_SEQ_LM: Sequence-to-sequence language modeling. \n\
            CAUSAL_LM: Causal language modeling. \n\
            TOKEN_CLS: Token classification. \n\
            QUESTION_ANS: Question answering. \n\
            FEATURE_EXTRACTION: Feature extraction, Provides the hidden states for downstream tasks. "
    )
    task_type = input(f"Task Type (default {defaults['task_type']}): ").strip()

    # Apply inputs or fallback to defaults
    args = {
        "r": int(r) if r else defaults["r"],
        "lora_alpha": int(alpha) if alpha else defaults["lora_alpha"],
        "lora_dropout": float(dropout) if dropout else defaults["lora_dropout"],
        "target_modules": (
            [m.strip() for m in target_modules.split(",")]
            if target_modules
            else defaults["target_modules"]
        ),
        "bias": bias if bias else defaults["bias"],
        "task_type": task_type if task_type else defaults["task_type"],
        "use_rslora": use_rslora if use_rslora else defaults["use_rslora"],
        "use_dora": use_dora if use_dora else defaults["use_dora"],
        "init_lora_weights": (
            init_lora_weights if init_lora_weights else defaults["init_lora_weights"]
        ),
    }

    return args


# Checkpoint / saving options
def configure_checkpoint_args(interactive=True):
    """
    Returns a dictionary of checkpoint args or None if skipped.
    """
    defaults = {"save_strategy": "best", "save_total_limit": 5}
    if not interactive:
        return defaults.copy()

    logging.info("Checkpointing options (press Enter to skip any)")
    logging.info(
        f"The checkpoint save strategy to adopt during training. Possible values are:\n\
            no: No save is done during training. \n\
            epoch: Save is done at the end of each epoch. \n\
            steps: Save is done every save_steps. \n\
            best: Save is done whenever a new best_metric is achieved. \n\
            (default {defaults['save_strategy']})"
    )
    save_strategy = input(
        f"Save strategy (default {defaults['save_strategy']}): "
    ).strip()

    logging.info(
        "Number of checkpoints to keep: Limits how many past checkpoints are retained to save disk space."
    )
    save_total_limit = input(
        f"Number of checkpoints to keep (default {defaults['save_total_limit']}): "
    ).strip()

    args = {
        "save_strategy": save_strategy if save_strategy else defaults["save_strategy"],
        "save_total_limit": (
            save_total_limit if save_total_limit else defaults["save_total_limit"]
        ),
    }

    return args


def configure_lora(interactive=True):
    """
    Wrapper that calls modular LoRA configuration functions.
    Returns a dictionary with keys:
      - bnb_config
      - torch_dtype
      - max_memory
      - training_args
      - lora_args
      - checkpoint_args
    Any value may be None if skipped.
    """

    cfg = {}

    # Quantization / precision
    bnb_config, torch_dtype = configure_quantization(interactive=interactive)
    cfg["bnb_config"] = bnb_config
    cfg["torch_dtype"] = torch_dtype

    # Memory / device map
    cfg["max_memory"] = configure_memory(interactive=interactive)

    # Training hyperparameters
    cfg["training_args"] = configure_training_args(interactive=interactive)

    # LoRA hyperparameters
    cfg["lora_args"] = configure_lora_args(interactive=interactive)

    # Checkpoint / saving options
    cfg["checkpoint_args"] = configure_checkpoint_args(interactive=interactive)

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
            tokenize, batched=True, num_proc=num_proc, desc="Tokenizing"
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
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        quantization_config=bnb_config,
        token=hf_token,
    )
    model = prepare_model_for_kbit_training(model)
    return tokenizer, model


def configure_lora_model(model, cfg):
    """
    Configure LoRA adapters on a model using a config dictionary.
    Uses defaults if keys are missing.
    """

    lora_args_cfg = cfg["lora_args"]

    use_rslora = lora_args_cfg.get("use_rslora")
    use_dora = lora_args_cfg.get("use_dora")
    init_lora_weights = lora_args_cfg.get("init_lora_weights")
    r = lora_args_cfg.get("r")
    lora_alpha = lora_args_cfg.get("lora_alpha")
    lora_dropout = lora_args_cfg.get("lora_dropout")
    target_modules = lora_args_cfg.get("target_modules", ["q_proj", "v_proj"])
    bias = lora_args_cfg.get("bias")
    task_type = lora_args_cfg.get("task_type")

    logging.info("Configuring LoRA...")
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type,
        use_rslora=use_rslora,
        use_dora=use_dora,
        init_lora_weights=init_lora_weights,
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
    checkpoint_args_cfg = cfg["checkpoint_args"]

    # -------------------------
    # Tokenizer & model loading
    # -------------------------
    tokenizer, model = prepare_tokenizer_and_model(base_model, hf_token, bnb_config)
    model = configure_lora_model(model, cfg)

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
        callbacks=[GPUUsageCallback()],
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
