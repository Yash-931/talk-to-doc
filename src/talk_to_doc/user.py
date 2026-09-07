from . import rag_pipeline

def get_user_answer():
    while(True):
        user_input = input("User: ")

        if user_input == "/exit":
            break

        response = rag_pipeline(path="ai-product-engineer-guide.pdf", chunk_size=100, overlap=10, query=user_input)
        print("Agent: ", end="")
        print((response or "").strip())

if __name__ == "__main__":
    get_user_answer()

