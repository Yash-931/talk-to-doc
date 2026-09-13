from . import RAG
from pathlib import Path

PDF_PATH = Path(__file__).resolve().parents[2] / "ai-product-engineer-guide.pdf"


def get_user_answer():
    rag = RAG()
    rag.ingest(str(PDF_PATH), chunk_size=50, overlap=5)

    while True:
        user_input = input("User: ")

        if user_input == "/exit":
            break

        response = rag.ask(user_input)
        print("Agent: ", end="")
        print(response.text or "")


if __name__ == "__main__":
    get_user_answer()
