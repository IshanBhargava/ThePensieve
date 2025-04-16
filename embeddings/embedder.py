import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_files = {
        'HP': "data/chunks/harry_potter_chunks.json",
        'HL': "data/chunks/hogwarts_legacy_chunks.json",
        'FB': "data/chunks/fantastic_beasts_chunks.json"
    }

all_chunks = []
embedded_chunks = []

for chunkFile in chunk_files.values():
    with open (chunkFile, 'r', encoding='utf-8') as f:
        chunk = json.load(f)
        all_chunks.extend(chunk)

for chunk in all_chunks:
    embedding = model.encode(chunk['text'])
    embedded_chunks.append({
        "id": chunk['chunk_id'],
        "embedding": embedding,
        "metadata": {
            "source": chunk['source'],
            "text": chunk['text'],
        }
    })

# Save the embedded chunks to a pkl file
with open('data/embeddings/embedded_chunks.pkl', 'wb') as f:
    pickle.dump(embedded_chunks, f)

print("✅ Embeddings saved to data/chunks/embedded_chunks.pkl")