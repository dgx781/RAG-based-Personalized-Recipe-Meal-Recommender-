# endee_client.py

from endee import Endee, Precision
import uuid

# -----------------------------
# CONFIG
# -----------------------------
INDEX_NAME = "recipes"
DIMENSION = 384

# -----------------------------
# CLIENT INIT
# -----------------------------
client = Endee()


# -----------------------------
# CREATE INDEX
# -----------------------------
def create_index():
    client.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        space_type="cosine",
        precision=Precision.INT8
    )
    print("✅ Index created")


def reset_index():
    try:
        client.delete_index(name=INDEX_NAME)
        print("🗑️ Old index deleted")
    except:
        pass

    create_index()


# -----------------------------
# GET INDEX
# -----------------------------
def get_index():
    return client.get_index(name=INDEX_NAME)


# -----------------------------
# UPSERT (EMBEDDING INSIDE)
# -----------------------------
def upsert_vectors(chunks, embedding_model, batch_size=500):
    index = get_index()

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        texts = [doc.page_content for doc in batch]
        embeddings = embedding_model.embed_documents(texts)

        data = [
            {
                "id": str(uuid.uuid4()),
                "vector": emb,
                "meta": {"text": text}
            }
            for emb, text in zip(embeddings, texts)
        ]

        index.upsert(data)
        print(f"✅ Batch {i//batch_size + 1} inserted")


# -----------------------------
# QUERY
# -----------------------------
def query_endee(query_text, embedding_model, top_k=5):
    index = get_index()

    query_vector = embedding_model.embed_query(query_text)

    results = index.query(
        vector=query_vector,
        top_k=top_k,
    )

    return [
        {
            "text": r["meta"]["text"],
            "score": r.get("score")
        }
        for r in results
    ]