import os
from pathlib import Path
import json


def generate_qna(repo_directory):
    qnas = []

    # Traverse through the directory and find all .c files
    for root, dirs, files in os.walk(repo_directory):
        for file in files:
            if file.endswith(".c"):
                file_path = Path(root) / file

                # Load enriched parsed data from jsonl
                with open("enriched_parsed.jsonl", "r") as f:
                    for line in f:
                        data = json.loads(line)
                        if data["file_path"] == str(file_path):
                            summary = data["summary"]
                            intent = data["intent"]

                            # Placeholder for actual Q&A generation logic
                            qna_data = {
                                "question": f"What is the purpose of the function in {file}?",
                                "answer": summary,
                                "intent_label": intent,
                            }

                            qnas.append(qna_data)

    # Save the list of Q&A pairs to a jsonl file
    with open("train_legacy.jsonl", "w") as f:
        for entry in qnas:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python3 scripts/05_generate_qna.py <repo_directory>")
        sys.exit(1)

    repo_directory = sys.argv[1]
    generate_qna(repo_directory)
