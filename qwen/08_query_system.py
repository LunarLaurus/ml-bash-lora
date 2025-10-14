import os
from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np


def query_system(repo_directory):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B")
    model = AutoModelForSeq2SeqLM.from_pretrained("Qwen/Qwen-7B")

    # Load FAISS index and embeddings
    index = faiss.read_index("code_embeddings.index")
    with open("code_embeddings.jsonl", "r") as f:
        data = [json.loads(line) for line in f]

    embeddings = np.array([item["embedding"] for item in data], dtype=np.float32)

    # Search the index
    queries = ["What is the purpose of the function in file1.c?"]
    query_embeddings = []

    for query in queries:
        inputs = tokenizer(query, return_tensors="pt", max_length=512, truncation=True)
        outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        query_embeddings.append(emb.flatten())

    D, I = index.search(np.array(query_embeddings, dtype=np.float32), k=3)

    # Retrieve the top hits
    for i, query in enumerate(queries):
        print(f"Query: {query}")
        for j, idx in enumerate(I[i]):
            distance = D[i][j]
            file_id = data[idx]["file_id"]
            print(f"  Top {j+1} hit: {file_id}, Distance: {distance}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python3 scripts/08_query_system.py <repo_directory>")
        sys.exit(1)

    repo_directory = sys.argv[1]
    query_system(repo_directory)
