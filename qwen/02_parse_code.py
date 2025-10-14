import os
from pathlib import Path
import json
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib


def generate_hash(file_path):
    """Generate SHA-256 hash for a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def parse_file(file_path):
    """Parse a single file using Tree-sitter and extract structured data."""
    try:
        result = subprocess.run(
            ["tree-sitter", "parse", "-p", "c", str(file_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        ast_data = {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "ast": result.stdout,
        }
        return ast_data
    except subprocess.CalledProcessError as e:
        print(f"Error parsing {file_path}: {e.stderr}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Unexpected error parsing {file_path}: {e}", file=sys.stderr)
        return None


def enrich_function(parsed_data):
    """Enrich the parsed function with additional metadata."""
    enriched_data = {
        "id": f"repo:/path/to/file.c:{parsed_data['function_name']}",
        "repo": "repo",
        "file_path": parsed_data["file_path"],
        "function": {
            "name": parsed_data["function_name"],
            "signature": parsed_data["function_signature"],
            "body": parsed_data["function_body"],
            "start_line": parsed_data["function_start_line"],
            "end_line": parsed_data["function_end_line"],
        },
        "includes": parsed_data.get("includes", []),
        "callers": parsed_data.get("callers", []),
        "callees": parsed_data.get("callees", []),
        "graph_distance": {"to_entry_points": 3},
        "full_file_context": "...",
        "hash": generate_hash(Path(parsed_data["file_path"])),
        "summary": "Function summary here",
        "detailed_description": "Detailed description of the function.",
        "intent_tags": ["I/O"],
        "risk_notes": "Side effects and concurrency issues.",
        "change_recipe": "How to safely modify the function.",
        "confidence_score": 0.8,
    }
    return enriched_data


def parse_code(repo_directory, output_path="data/parsed_functions.jsonl"):
    """Parse .c and .h files using Tree-sitter, extract AST data, and save to JSONL."""
    parsed_functions = []

    # Traverse the directory tree
    for root, dirs, files in os.walk(repo_directory):
        for file in files:
            if file.endswith(".c") or file.endswith(".h"):
                file_path = Path(root) / file
                try:
                    ast_data = parse_file(file_path)
                    if ast_data:
                        parsed_functions.append(ast_data)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}", file=sys.stderr)

    # Enrich functions
    enriched_functions = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(enrich_function, func) for func in parsed_functions]
        for future in as_completed(futures):
            enriched_function = future.result()
            if enriched_function:
                enriched_functions.append(enriched_function)

    # Save to JSONL file
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in enriched_functions:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Parsed {len(enriched_functions)} files. Saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/02_parse_code.py <repo_directory>")
        sys.exit(1)

    repo_directory = Path(sys.argv[1])
    output_path = Path("data", "parsed_functions.jsonl")  # Default output path

    parse_code(repo_directory, output_path)
