import os
import pickle
from pinecone import Pinecone, ServerlessSpec
import dotenv

dotenv.load_dotenv('.env')

index_name = "pensieve-index"
dimension = 384

with open("data/embeddings/embedded_chunks.pkl", "rb") as f:
    embedded_chunks = pickle.load(f)

pc = Pinecone(api_key=os.getenv("PINECONE"))

if index_name not in pc.list_indexes():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region = "us-east-1"
        )
    )
    print(f"✅ Created index '{index_name}'")
else:
    print(f"✅ Index '{index_name}' already exists")

index = pc.Index(index_name)

# Upsert in batches
batch_size = 100
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
    print(f"✅ Uploaded batch {i // batch_size + 1} of {len(embedded_chunks) // batch_size + 1}")






