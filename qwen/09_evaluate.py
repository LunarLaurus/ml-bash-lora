import os
from pathlib import Path
import json
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


def evaluate(repo_directory):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B")
    model = AutoModelForSeq2SeqLM.from_pretrained("Qwen/Qwen-7B")

    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

    # Load test dataset from jsonl file
    with open("test_data.jsonl", "r") as f:
        test_data = [json.loads(line) for line in f]

    predictions = []
    ground_truths = []

    for data in test_data:
        question = data["question"]
        answer = data["answer"]

        inputs = tokenizer(
            question, return_tensors="pt", max_length=512, truncation=True
        )
        outputs = model.generate(**inputs, max_length=100)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        predictions.append(prediction)
        ground_truths.append(answer)

    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    accuracy = accuracy_score(ground_truths, predictions)
    f1 = f1_score(ground_truths, predictions, average="weighted")
    precision = precision_score(ground_truths, predictions, average="weighted")
    recall = recall_score(ground_truths, predictions, average="weighted")

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python3 scripts/09_evaluate.py <repo_directory>")
        sys.exit(1)

    repo_directory = sys.argv[1]
    evaluate(repo_directory)
