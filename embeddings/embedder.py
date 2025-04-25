import os
import json
import pickle
import time
import logging
from pathlib import Path
from typing import List, Dict
import openai
from dotenv import load_dotenv
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

# ─── Configuration ────────────────────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 100))
OUTPUT_DIR = Path("data/embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)

# ─── Helper: retryable embedding call ─────────────────────────────────────────
@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=10, max=60),
    retry=retry_if_exception_type(openai.error.RateLimitError)
)
def embed_batch(texts: List[str]) -> List[float]:
    """Call OpenAI API and retry on rate limits."""
    response = openai.Embedding.create(input=texts, model=MODEL)
    return [datum["embedding"] for datum in response["data"]]

# ─── Core: process & embed ────────────────────────────────────────────────────
def run_embedding(
    chunk_files: Dict[str, str],
    output_path: str
) -> None:
    """
    Read chunk JSON files, generate embeddings in batches,
    and save results to a pickle file.
    """
    embedded_chunks = []
    total_chunks = 0

    # Stream through each file so we don’t OOM
    for file_label, filepath in chunk_files.items():
        logging.info(f"Loading chunks from {filepath!r}")
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        total_chunks += len(chunks)

        # Process in sub‐batches per file
        for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc=file_label):
            batch = chunks[i : i + BATCH_SIZE]
            texts = [c["text"] for c in batch]
            try:
                embeddings = embed_batch(texts)
            except Exception as e:
                logging.error(f"Batch {i}-{i+len(batch)} failed: {e!r}")
                raise

            for c, embed in zip(batch, embeddings):
                embedded_chunks.append({
                    "id": c["chunk_id"],
                    "embedding": embed,
                    "metadata": {
                        "source": c["source"],
                        "text": c["text"]
                    }
                })

            # Optional: checkpoint to disk every N batches
            if len(embedded_chunks) % (10 * BATCH_SIZE) == 0:
                with open(output_path, "wb") as ckpt:
                    pickle.dump(embedded_chunks, ckpt)
                logging.info(f"Checkpointed {len(embedded_chunks)} embeddings so far")

    logging.info(f"🔄 Generated embeddings for {total_chunks} chunks using '{MODEL}'")
    with open(output_path, "wb") as f:
        pickle.dump(embedded_chunks, f)
    logging.info(f"✅ Saved all {len(embedded_chunks)} embeddings to {output_path}")

# ─── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    chunk_files = {
        "HP": "data/chunks/harry_potter_chunks.json",
        "HL": "data/chunks/hogwarts_legacy_chunks.json",
        "FB": "data/chunks/fantastic_beasts_chunks.json",
    }
    output_path = OUTPUT_DIR / "embedded_chunks.pkl"
    run_embedding(chunk_files, str(output_path))


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

# import os
# import json
# import pickle
# import time
# import openai
# from dotenv import load_dotenv
# from tqdm import tqdm

# load_dotenv()

# openai.api_key = os.getenv("OPENAI_API_KEY")
# model  = os.getenv("OPENAI_EMBEDDING_MODEL")
# BATCH_SIZE = 100

# def run_embedding(chunk_files: dict, output_path: str):
#     all_chunks = []
#     for chunkfile in chunk_files.values():
#         with open(chunkfile, 'r', encoding='utf-8') as f:
#             chunks = json.load(f)
#             all_chunks.extend(chunks)
    
#     texts = [chunk["text"] for chunk in all_chunks]

#     print(f"🔄 Generating embeddings for {len(texts)} chunks using the '{model}' model...")
#     embedded_chunks = []
    
#     for i in tqdm(range(0, len(texts), BATCH_SIZE)):
#         batch_texts = texts[i:i + BATCH_SIZE]
#         batch_chunks = all_chunks[i:i + BATCH_SIZE]

#         try:
#             response = openai.Embedding.create(input=batch_texts, model=model)
#             for j, item in enumerate(batch_chunks):
#                 embedded_chunks.append({
#                     "id": item["chunk_id"],
#                     "embedding": response["data"][j]["embedding"],
#                     "metadata": {
#                         "source": item["source"],
#                         "text": item["text"]
#                     }
#                 })
#         except openai.error.RateLimitError:
#             print("⚠️ Rate limit hit. Waiting 60 seconds before retrying...")
#             time.sleep(60)
#             continue
    
#     #Save to pkl
#     with open(output_path, "wb") as f:
#         pickle.dump(embedded_chunks, f)
#     print(f"✅ Saved embeddings to {output_path}")

# if __name__ == "__main__":
#     chunk_files = {
#         'HP': "data/chunks/harry_potter_chunks.json",
#         'HL': "data/chunks/hogwarts_legacy_chunks.json",
#         'FB': "data/chunks/fantastic_beasts_chunks.json"
#     }

#     output_path = "data/embeddings/embedded_chunks.pkl"
#     run_embedding(chunk_files, output_path)