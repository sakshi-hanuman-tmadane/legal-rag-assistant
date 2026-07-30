import re
import numpy as np
from utils.vector_store import load_index
from utils.embeddings import model

# Load FAISS index and chunks
index, chunks = load_index()


def search(query, top_k=5):

    query_lower = query.lower()

    # ============================
    # Exact Article Search
    # ============================
    article_match = re.search(r"article\s+(\d+[A-Za-z]?)", query_lower)

    if article_match:

        article = article_match.group(1)

        exact_results = []

        for chunk in chunks:

            text = chunk["text"]

            # Skip Table of Contents page
            if "Right to Freedom" in text:
                continue

            # Match headings like:
            # 21. Protection of life...
            # 19. Protection of...
            if re.search(rf"\b{article}\.\s", text):

                exact_results.append(chunk)

        if exact_results:
            return exact_results[:top_k]

    # ============================
    # Semantic Search (FAISS)
    # ============================

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for idx in indices[0]:
        if idx != -1:
            results.append(chunks[idx])

    return results