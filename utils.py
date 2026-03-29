import re
from io import BytesIO
from typing import List, Dict

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text: str) -> str:
    """
    Basic cleanup for extracted text.
    Keeps it simple and interview-friendly.
    """
    if not text:
        return ""

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Fix simple hyphenation across line breaks / extraction artifacts
    text = re.sub(r"-\s+", "", text)

    return text.strip()


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract text from a PDF uploaded through Streamlit.
    """
    pdf_bytes = uploaded_file.read()
    reader = PdfReader(BytesIO(pdf_bytes))

    pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)

    return clean_text("\n".join(pages))


def extract_text_from_txt(uploaded_file) -> str:
    """
    Read plain text file uploaded through Streamlit.
    """
    text = uploaded_file.read().decode("utf-8", errors="ignore")
    return clean_text(text)


def split_into_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitting.
    Not perfect, but fine for a 1-2 day project.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, max_words: int = 120, overlap_sentences: int = 1) -> List[str]:
    """
    Sentence-aware chunking.
    Better than raw character splitting because it avoids cutting
    sentences in the middle too often.
    """
    sentences = split_into_sentences(text)

    chunks = []
    current_chunk = []
    current_word_count = 0

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_words = sentence.split()

        if current_word_count + len(sentence_words) <= max_words:
            current_chunk.append(sentence)
            current_word_count += len(sentence_words)
            i += 1
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))

                # overlap by last 1 sentence by default
                overlap = current_chunk[-overlap_sentences:] if len(current_chunk) >= overlap_sentences else current_chunk
                current_chunk = overlap[:]
                current_word_count = sum(len(s.split()) for s in current_chunk)
            else:
                # if one sentence alone is too long, force add it
                chunks.append(sentence)
                i += 1
                current_chunk = []
                current_word_count = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [c.strip() for c in chunks if c.strip()]


class Retriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks: List[str] = []
        self.embeddings = None

    def fit(self, chunks: List[str]) -> None:
        self.chunks = chunks
        self.embeddings = self.model.encode(chunks, convert_to_numpy=True)

    def retrieve(self, question: str, top_k: int = 3) -> List[Dict]:
        question_embedding = self.model.encode([question], convert_to_numpy=True)
        scores = cosine_similarity(question_embedding, self.embeddings)[0]

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(scores[idx]),
                "index": int(idx)
            })
        return results