from .data_loader import parse_pdf, Sentence
from .chunker import create_chunks
from .embedder import generate_embeddings, EmbeddingCollection, generate_query_embedding
from .db import store_embeddings, query_db
from .ai import generate_llm_response


# TODOS: Implement the hybrid search (BM25 + semantic search)
def rag_ingestion(path: str, chunk_size: int, overlap: int):
    sentences: list[Sentence] = parse_pdf(path)

    chunks = create_chunks(sentences, chunk_size, overlap)

    embedding_collection: EmbeddingCollection = generate_embeddings(chunks)

    store_embeddings(
        embeddings=embedding_collection.embeddings,
        chunk_texts=embedding_collection.chunk_texts,
        metadata=embedding_collection.metadata,
    )
    return


def rag_retrieval(query: str):
    query_vector = generate_query_embedding(query)

    results = query_db(query_vector)

    response = generate_llm_response(query, results)

    return response


def rag_pipeline(path: str, chunk_size: int, overlap: int, query: str):
    response = rag_retrieval(query)
    return response.text
