from .data_loader import parse_pdf, Sentence
from .chunker import create_chunks
from .embedder import generate_embeddings, EmbeddingCollection
from .db import store_embeddings


def rag_pipeline(path: str, chunk_size: int, overlap: int):
    sentences: list[Sentence] = parse_pdf(path)

    chunks = create_chunks(sentences, chunk_size, overlap)

    embedding_collection: EmbeddingCollection = generate_embeddings(chunks)

    store_embeddings(embeddings=embedding_collection.embeddings, chunk_texts=embedding_collection.chunk_texts, metadata=embedding_collection.metadata)


if __name__ == "__main__":
    rag_pipeline("ai-product-engineer-guide.pdf", 500, 10)