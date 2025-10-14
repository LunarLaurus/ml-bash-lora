import os
from pathlib import Path
import json


def index_files(repo_directory):
    file_list = []

    # Traverse through the directory and find all .c and .h files
    for root, dirs, files in os.walk(repo_directory):
        for file in files:
            if file.endswith(".c") or file.endswith(".h"):
                file_path = Path(root) / file
                file_size = file_path.stat().st_size
                last_modified_time = file_path.stat().st_mtime

                file_info = {
                    "file_name": file,
                    "file_path": str(file_path),
                    "file_size": file_size,
                    "last_modified_time": last_modified_time,
                }

                file_list.append(file_info)

    # Save the list of files to a jsonl file
    with open("file_index.jsonl", "w") as f:
        for entry in file_list:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python3 scripts/01_index_files.py <repo_directory>")
        sys.exit(1)

    repo_directory = sys.argv[1]
    index_files(repo_directory)
