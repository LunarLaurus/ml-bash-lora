import os
from pathlib import Path
import json
from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from peft import PeftConfig, PeftModel, LoraConfig


def train_lora(repo_directory):
    # Load dataset from jsonl file
    with open("train_legacy.jsonl", "r") as f:
        data = [json.loads(line) for line in f]

    dataset = DatasetDict(
        {"train": load_dataset("json", data_files={"train": "train_legacy.jsonl"})}
    )

    # Load model and tokenizer
    base_model_name_or_path = "Qwen/Qwen-7B"
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name_or_path)
    tokenizer = model.config.tokenizer

    # Set up LoRA configuration
    peft_config = LoraConfig(
        task_type="SEQ_2_SEQ_LM",
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
    )

    model = PeftModel.from_pretrained(model, peft_config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
    )

    # Train the model
    trainer.train()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python3 scripts/06_train_lora.py <repo_directory>")
        sys.exit(1)

    repo_directory = sys.argv[1]
    train_lora(repo_directory)
