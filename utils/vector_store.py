import faiss
import pickle
import numpy as np

def save_index(embeddings, chunks):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    faiss.write_index(index, "vector_store/legal.index")

    with open("vector_store/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Vector Store Saved")


def load_index():
    index = faiss.read_index("vector_store/legal.index")

    with open("vector_store/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    return index, chunks