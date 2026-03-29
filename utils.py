from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def load_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """
    Very simple character-based chunking.
    Later we can improve this to sentence-based chunking.
    """
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk.strip())
        start += chunk_size
    return [c for c in chunks if c]


class Retriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.chunk_embeddings = None

    def fit(self, chunks: list[str]):
        self.chunks = chunks
        self.chunk_embeddings = self.model.encode(chunks)

    def retrieve(self, question: str, top_k: int = 1):
        question_embedding = self.model.encode([question])
        scores = cosine_similarity(question_embedding, self.chunk_embeddings)[0]

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(scores[idx])
            })
        return results