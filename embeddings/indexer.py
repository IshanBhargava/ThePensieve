# embeddings/indexer.py (optimized for logging, .env, batch control)
import os
import pickle
import logging
from dotenv import load_dotenv
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec

# ─── Config ─────────────────────────────────────────────────────────────
load_dotenv()
pinecone_api_key = os.getenv("PINECONE")
index_name = os.getenv("PINECONE_INDEX", "pensieve-index")
embedding_dim = int(os.getenv("EMBEDDING_DIM", 1536))
batch_size = int(os.getenv("UPSERT_BATCH_SIZE", 100))
region = os.getenv("PINECONE_REGION", "us-east-1")
cloud = os.getenv("PINECONE_CLOUD", "aws")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)

pc = Pinecone(api_key=pinecone_api_key)

# ─── Main Runner ────────────────────────────────────────────────────────
def run_indexer(pkl_path: str = "data/embeddings/embedded_chunks.pkl"):
    with open(pkl_path, "rb") as f:
        embedded_chunks = pickle.load(f)

    existing_indexes = [i['name'] for i in pc.list_indexes()]
    if index_name not in existing_indexes:
        logging.info(f"Creating new Pinecone index '{index_name}' with dimension {embedding_dim}")
        pc.create_index(
            name=index_name,
            dimension=embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region)
        )
    else:
        logging.info(f"Using existing index: {index_name}")

    index = pc.Index(index_name)

    logging.info(f"Upserting {len(embedded_chunks)} vectors in batches of {batch_size}...")
    for i in range(0, len(embedded_chunks), batch_size):
        batch = embedded_chunks[i:i + batch_size]
        vectors = [
            {
                "id": item["id"],
                "values": item["embedding"],
                "metadata": item["metadata"]
            }
            for item in batch
        ]
        index.upsert(vectors=vectors)
        logging.info(f"✅ Uploaded batch {i // batch_size + 1} of {len(embedded_chunks) // batch_size + 1}")

    logging.info("🎉 Indexing complete.")


# ─── Entrypoint ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_indexer()
