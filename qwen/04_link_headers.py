import os
from pathlib import Path
import json


def link_headers(repo_directory):
    linked_functions = []

    # Traverse through the directory and find all .h files
    for root, dirs, files in os.walk(repo_directory):
        for file in files:
            if file.endswith(".h"):
                header_path = Path(root) / file

                # Load parsed AST from jsonl
                with open("parsed_functions.jsonl", "r") as f:
                    for line in f:
                        data = json.loads(line)
                        if data["file_path"] == str(header_path):
                            ast = data["ast"]

                            # Placeholder for actual linking logic
                            implementations = find_implementations(
                                ast, repo_directory
                            )  # Implement this function

                            for implementation in implementations:
                                linked_data = {
                                    "header_file": file,
                                    "header_path": str(header_path),
                                    "implementation_file": implementation["file_name"],
                                    "implementation_path": implementation["file_path"],
                                }

                                linked_functions.append(linked_data)

    # Save the list of linked functions to a jsonl file
    with open("linked_functions.jsonl", "w") as f:
        for entry in linked_functions:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python3 scripts/04_link_headers.py <repo_directory>")
        sys.exit(1)

    repo_directory = sys.argv[1]
    link_headers(repo_directory)
