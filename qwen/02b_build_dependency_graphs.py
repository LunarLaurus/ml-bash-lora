import os
from pathlib import Path
import json
import networkx as nx


def build_dependency_graphs(repo_directory):
    G = nx.DiGraph()

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

                            # Parse the AST and extract function calls
                            # This is a placeholder for actual parsing logic
                            functions_called = find_functions_called(
                                ast
                            )  # Implement this function

                            for func in functions_called:
                                G.add_node(func)
                                G.add_edge(func, file_path.stem)

    # Save the dependency graph to a json file
    with open("dependency_graphs.json", "w") as f:
        nx.write_json(G, f)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python3 scripts/02b_build_dependency_graphs.py <repo_directory>")
        sys.exit(1)

    repo_directory = sys.argv[1]
    build_dependency_graphs(repo_directory)
