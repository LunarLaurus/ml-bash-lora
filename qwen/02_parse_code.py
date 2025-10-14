import os
from pathlib import Path
import json
import subprocess


def parse_code(repo_directory):
    parsed_functions = []

    # Traverse through the directory and find all .c files
    for root, dirs, files in os.walk(repo_directory):
        for file in files:
            if file.endswith(".c"):
                file_path = Path(root) / file

                # Run Tree-sitter-C to parse AST
                command = f"tree-sitter parse -p c {file_path}"
                result = subprocess.run(command.split(), capture_output=True, text=True)

                if result.returncode != 0:
                    print(f"Error parsing {file_path}: {result.stderr}")
                    continue

                parsed_data = {
                    "file_name": file,
                    "file_path": str(file_path),
                    "ast": result.stdout,
                }

                parsed_functions.append(parsed_data)

    # Save the list of parsed functions to a jsonl file
    with open("parsed_functions.jsonl", "w") as f:
        for entry in parsed_functions:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python3 scripts/02_parse_code.py <repo_directory>")
        sys.exit(1)

    repo_directory = sys.argv[1]
    parse_code(repo_directory)
