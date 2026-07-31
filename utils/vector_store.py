import os
import faiss
import pickle
import numpy as np


# Folder where vector database will be stored
VECTOR_DB = "vector_db"

INDEX_PATH = os.path.join(VECTOR_DB, "faiss.index")
CHUNKS_PATH = os.path.join(VECTOR_DB, "chunks.pkl")


def save_index(embeddings, chunks):
    """
    Save FAISS index and text chunks
    """

    os.makedirs(VECTOR_DB, exist_ok=True)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Vector database saved successfully!")


def load_index():
    """
    Load FAISS index and chunks
    """

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_PATH}. "
            "Please run rebuild_vector_db.py first."
        )

    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"Chunks file not found at {CHUNKS_PATH}. "
            "Please run rebuild_vector_db.py first."
        )

    index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks