from transformers import pipeline
from utils import load_text, chunk_text, Retriever


def main():
    # 1. Load document
    text = load_text("sample_paper.txt")

    # 2. Split into chunks
    chunks = chunk_text(text, chunk_size=300)

    print("\nDocument loaded.")
    print(f"Number of chunks: {len(chunks)}")

    # 3. Build retriever
    retriever = Retriever()
    retriever.fit(chunks)

    # 4. Load QA model
    print("Loading QA model...")
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    print("\nAsk questions about the document.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() == "exit":
            break

        # 5. Retrieve best chunk
        retrieved = retriever.retrieve(question, top_k=1)
        best_chunk = retrieved[0]["chunk"]
        retrieval_score = retrieved[0]["score"]

        # 6. Ask QA model
        result = qa_pipeline(question=question, context=best_chunk)

        print("\n--- RESULT ---")
        print("Answer:", result["answer"])
        print("Answer confidence:", round(result["score"], 4))
        print("Retrieval score:", round(retrieval_score, 4))
        print("\nSource chunk:")
        print(best_chunk)
        print("--------------\n")


if __name__ == "__main__":
    main()