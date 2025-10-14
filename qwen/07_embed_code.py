import os
from pathlib import Path
import json
import faiss
from transformers import AutoTokenizer, AutoModel


def embed_code(repo_directory):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B")
    model = AutoModel.from_pretrained("Qwen/Qwen-7B")

    embeddings = []
    file_ids = []

    # Load enriched parsed data from jsonl
    with open("enriched_parsed.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            file_path = Path(data["file_path"])
            code = file_path.read_text()

            inputs = tokenizer(
                code, return_tensors="pt", max_length=512, truncation=True
            )
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

            embeddings.append(emb.flatten())
            file_ids.append(str(file_path))

    # Save the list of embeddings to a jsonl file
    with open("code_embeddings.jsonl", "w") as f:
        for i, embedding in enumerate(embeddings):
            f.write(
                json.dumps({"file_id": file_ids[i], "embedding": embedding.tolist()})
                + "\n"
            )

    # Create an FAISS index and add embeddings
    index = faiss.IndexFlatL2(len(embedding))
    index.add(np.array(embeddings, dtype=np.float32))

    # Save the index to a binary file
    faiss.write_index(index, "code_embeddings.index")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python3 scripts/07_embed_code.py <repo_directory>")
        sys.exit(1)

    repo_directory = sys.argv[1]
    embed_code(repo_directory)
