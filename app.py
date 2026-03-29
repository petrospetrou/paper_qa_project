import streamlit as st
from transformers import pipeline

from utils import (
    extract_text_from_pdf,
    extract_text_from_txt,
    chunk_text,
    Retriever,
)

st.set_page_config(page_title="Research Paper QA Assistant", layout="wide")

st.title("Document-Grounded Research Paper QA Assistant")
st.write(
    "Upload a PDF or TXT file, then ask questions. "
    "The system retrieves relevant chunks and answers from the document."
)

# Sidebar settings
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Number of retrieved chunks", min_value=1, max_value=5, value=3)
max_words = st.sidebar.slider("Chunk size (words)", min_value=60, max_value=200, value=120)
qa_threshold = st.sidebar.slider(
    "Minimum QA confidence", min_value=0.0, max_value=1.0, value=0.20, step=0.05
)

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])


@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


@st.cache_resource
def load_retriever():
    return Retriever(model_name="all-MiniLM-L6-v2")


def is_list_question(question: str) -> bool:
    q = question.lower()
    keywords = [
        "list",
        "applications",
        "uses",
        "examples",
        "types",
        "give me",
        "what are the applications",
        "what are the uses",
    ]
    return any(keyword in q for keyword in keywords)


def extract_list_items_from_chunks(retrieved_chunks):
    """
    Very lightweight heuristic for list-style questions.
    Good enough for a 1-2 day project.
    """
    text = " ".join([item["chunk"] for item in retrieved_chunks]).lower()

    candidate_items = [
        "machine translation",
        "text summarization",
        "question answering",
        "sentiment analysis",
        "computer vision",
        "speech processing",
    ]

    found_items = []
    for item in candidate_items:
        if item in text:
            found_items.append(item.title())

    return found_items


if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()

    with st.spinner("Reading document..."):
        if file_type == "pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif file_type == "txt":
            text = extract_text_from_txt(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()

    if not text.strip():
        st.error("No text could be extracted from this file.")
        st.stop()

    chunks = chunk_text(text, max_words=max_words, overlap_sentences=1)

    st.success("Document loaded successfully.")
    st.write(f"**Extracted text length:** {len(text)} characters")
    st.write(f"**Number of chunks:** {len(chunks)}")

    with st.expander("Preview first 1000 characters of extracted text"):
        st.write(text[:1000] + ("..." if len(text) > 1000 else ""))

    with st.spinner("Building retriever..."):
        retriever = load_retriever()
        retriever.fit(chunks)

    qa_pipeline = load_qa_pipeline()

    question = st.text_input("Ask a question about the document:")

    if question:
        with st.spinner("Retrieving evidence and answering..."):
            retrieved = retriever.retrieve(question, top_k=top_k)
            combined_context = "\n\n".join([item["chunk"] for item in retrieved])

            list_question = is_list_question(question)

            if not list_question:
                result = qa_pipeline(question=question, context=combined_context)

        st.subheader("Answer")

        if list_question:
            found_items = extract_list_items_from_chunks(retrieved)

            if found_items:
                st.write("Relevant information from the document:")
                for item in found_items:
                    st.write(f"- {item}")
            else:
                st.write("Relevant information from the document:")
                st.write(retrieved[0]["chunk"])

            st.info(
                "This question looks like a list-type question, so the system returns "
                "retrieved information instead of a single extracted span."
            )

            col1, col2 = st.columns(2)
            col1.metric("Answer confidence", "N/A")
            col2.metric("Top retrieval score", f"{retrieved[0]['score']:.3f}")

        else:
            if result["score"] < qa_threshold:
                st.warning(
                    f"The system is not confident enough to answer reliably "
                    f"(confidence={result['score']:.3f})."
                )
            else:
                st.write(result["answer"])

            col1, col2 = st.columns(2)
            col1.metric("Answer confidence", f"{result['score']:.3f}")
            col2.metric("Top retrieval score", f"{retrieved[0]['score']:.3f}")

        st.subheader("Retrieved evidence")
        for i, item in enumerate(retrieved, start=1):
            with st.expander(f"Chunk {i} | retrieval score = {item['score']:.3f}"):
                st.write(item["chunk"])

else:
    st.info("Upload a document to begin.")