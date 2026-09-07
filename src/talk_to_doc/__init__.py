from .data_loader import parse_pdf, Sentence
from .chunker import create_chunks
from .embedder import generate_embeddings, EmbeddingCollection, generate_query_embedding
from .db import store_embeddings, query_db
from .ai import generate_llm_response


def rag_ingestion(path: str, chunk_size: int, overlap: int):
    sentences: list[Sentence] = parse_pdf(path)

    chunks = create_chunks(sentences, chunk_size, overlap)

    embedding_collection: EmbeddingCollection = generate_embeddings(chunks)

    store_embeddings(embeddings=embedding_collection.embeddings, chunk_texts=embedding_collection.chunk_texts, metadata=embedding_collection.metadata)

def rag_retrieval(query: str):
    query_vector = generate_query_embedding(query)

    results = query_db(query_vector)

    response = generate_llm_response(query, results)

    print(response.text)



if __name__ == "__main__":
    # rag_ingestion("ai-product-engineer-guide.pdf", 500, 10)

    rag_retrieval("What needs to be done in phase 2")