# Document-Grounded Research Paper QA Assistant

A lightweight NLP project that demonstrates how to build a **document-grounded question answering system** using transformer models.

Users can upload a **PDF or TXT file**, ask questions, and receive answers based on **retrieved evidence from the document**.

---

## Features

- Upload **PDF or TXT documents**
- Semantic retrieval using **sentence-transformers**
- Question answering using a **pretrained transformer (DistilBERT)**
- Displays **retrieved evidence chunks**
- Handles **low-confidence answers**
- Special handling for **list-type questions**

---

## How It Works

The system follows a **retrieval + QA pipeline**:

1. **Document ingestion**
   - Extract text from PDF or TXT files

2. **Text chunking**
   - Split document into overlapping chunks (sentence-aware)

3. **Embedding**
   - Convert chunks into vector representations using:
   - `all-MiniLM-L6-v2`

4. **Retrieval**
   - Convert the user question into an embedding
   - Retrieve top-k most relevant chunks using cosine similarity

5. **Question Answering**
   - Use a pretrained QA model:
   - `distilbert-base-cased-distilled-squad`
   - Extract answer from retrieved context

6. **List-question handling**
   - Detect questions like:
     - “applications”, “uses”, “examples”
   - Return structured results instead of a single span

---

## Tech Stack

- Python
- Streamlit
- Hugging Face Transformers
- Sentence Transformers
- Scikit-learn
- PyPDF

---

## Installation

```bash
pip install -r requirements.txt