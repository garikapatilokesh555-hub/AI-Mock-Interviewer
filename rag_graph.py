import numpy as np
from embeddings import model  # reuse the same SentenceTransformer model

# In-memory store
_documents = []
_embeddings = []


def store_vectors(chunks):
    """Encode chunks and store them in memory."""
    global _documents, _embeddings
    _documents = chunks
    _embeddings = model.encode(chunks)  # shape: (n_chunks, embedding_dim)


def retrieve_docs(query, n_results=20):
    """Return top-n_results chunks most similar to the query."""
    if not _documents:
        return []

    query_embedding = model.encode([query])[0]  # shape: (embedding_dim,)

    # Cosine similarity
    norms = np.linalg.norm(_embeddings, axis=1) * np.linalg.norm(query_embedding)
    norms = np.where(norms == 0, 1e-10, norms)  # avoid division by zero
    similarities = np.dot(_embeddings, query_embedding) / norms

    top_indices = np.argsort(similarities)[::-1][:n_results]
    return [_documents[i] for i in top_indices]