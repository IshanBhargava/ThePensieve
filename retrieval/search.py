# import os
# from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer
# from pinecone import Pinecone

# load_dotenv()

# pc = Pinecone(api_key=os.getenv("PINECONE"))

# index = pc.Index(os.getenv("PINECONE_INDEX"))
# model = SentenceTransformer(os.getenv("MODEL_NAME"))

# def query_pensieve(user_query: str, top_k: int = 5):
#     # Step 1: Embed the user's question
#     query_embedding = model.encode(user_query).tolist()

#     # Step 2: Query Pinecone
#     result = index.query(
#         vector=query_embedding,
#         top_k=top_k,
#         include_metadata=True
#     )

#     return result

# if __name__ == "__main__":
#     res = query_pensieve("thestrals", 100)

#     # Write res to a txt file
#     with open("retrieval_results.txt", "w") as f:
#         f.write(str(res))
#     print("✅ Results written to retrieval_results.txt")

# retrieval/search.py (OpenAI-based query embedding + Pinecone search)
import os
import openai
import logging
from dotenv import load_dotenv
from pinecone import Pinecone
from typing import List

# ─── Config ─────────────────────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE")
index_name = os.getenv("PINECONE_INDEX", "pensieve-index")
model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
top_k = int(os.getenv("TOP_K", 5))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

# ─── Core Query Logic ───────────────────────────────────────────────────
def run_query(user_query: str, top_k: int = top_k) -> List[dict]:
    logging.info(f"🔍 Querying Pinecone with: '{user_query}'")

    # Get embedding for query
    response = openai.Embedding.create(input=[user_query], model=model)
    query_vector = response["data"][0]["embedding"]

    # Search Pinecone
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    # for i, match in enumerate(results["matches"], 1):
    #     logging.info(f"#{i} (score: {match['score']:.4f})\n{match['metadata']['text'][:300]}\n")

    return results["matches"]


# ─── Entrypoint ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_query("Who is Harry Potter?")
