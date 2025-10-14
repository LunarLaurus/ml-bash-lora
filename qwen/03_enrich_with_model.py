import os
from pathlib import Path
import json
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


def enrich_with_model(repo_directory):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B")
    model = AutoModelForSeq2SeqLM.from_pretrained("Qwen/Qwen-7B")

    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    classifier = pipeline(
        "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    enriched_functions = []

    # Traverse through the directory and find all .c files
    for root, dirs, files in os.walk(repo_directory):
        for file in files:
            if file.endswith(".c"):
                file_path = Path(root) / file

                # Load parsed AST from jsonl
                with open("parsed_functions.jsonl", "r") as f:
                    for line in f:
                        data = json.loads(line)
                        if data["file_path"] == str(file_path):
                            ast = data["ast"]

                            # Placeholder for actual enrichment logic
                            summary = summarizer(ast, max_length=100)[0]["summary_text"]
                            intent = classifier(ast)[0]["label"]
                            complexity = calculate_complexity(
                                ast
                            )  # Implement this function

                            enriched_data = {
                                "file_name": file,
                                "file_path": str(file_path),
                                "ast": ast,
                                "summary": summary,
                                "intent": intent,
                                "complexity": complexity,
                            }

                            enriched_functions.append(enriched_data)

    # Save the list of enriched functions to a jsonl file
    with open("enriched_parsed.jsonl", "w") as f:
        for entry in enriched_functions:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python3 scripts/03_enrich_with_model.py <repo_directory>")
        sys.exit(1)

    repo_directory = sys.argv[1]
    enrich_with_model(repo_directory)
