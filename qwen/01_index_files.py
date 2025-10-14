# 01_index_files.py — discover .c/.h, collect file metadata → file_index.jsonl
import os
from pathlib import Path
import json
import sys
from datetime import datetime


def index_files(repo_directory, output_path="data/file_index.jsonl"):
    """
    Traverse a directory to find all .c and .h files, collect metadata, and save to a JSONL file.
    """
    file_list = []

    try:
        # Traverse the directory tree
        for root, dirs, files in os.walk(repo_directory):
            for file in files:
                if file.endswith(".c") or file.endswith(".h"):
                    file_path = Path(root) / file
                    file_size = file_path.stat().st_size
                    last_modified_time = datetime.fromtimestamp(
                        file_path.stat().st_mtime
                    )
                    file_creation_time = datetime.fromtimestamp(
                        file_path.stat().st_ctime
                    )

                    file_info = {
                        "file_name": file,
                        "file_path": str(file_path),
                        "file_size": file_size,
                        "last_modified_time": last_modified_time.isoformat(),
                        "file_creation_time": file_creation_time.isoformat(),
                        "file_extension": file.split(".")[-1],
                        "repo_relative_path": str(
                            file_path.relative_to(repo_directory)
                        ),
                    }

                    file_list.append(file_info)

        # Save to JSONL file
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in file_list:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Indexed {len(file_list)} files. Saved to {output_path}")

    except Exception as e:
        print(f"Error indexing files: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/01_index_files.py <repo_directory>")
        sys.exit(1)

    repo_directory = sys.argv[1]
    output_path = os.path.join("data", "file_index.jsonl")  # Default output path

    index_files(repo_directory, output_path)
