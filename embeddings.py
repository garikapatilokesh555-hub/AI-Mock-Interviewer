from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def create_chunks(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings