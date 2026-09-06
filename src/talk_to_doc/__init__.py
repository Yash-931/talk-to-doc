from .data_loader import parse_pdf, Sentence
from .chunker import create_chunks
from .embedder import generate_embeddings

def rag_pipeline(path: str, chunk_size: int, overlap: int):
    sentences: list[Sentence] = parse_pdf(path)

    chunks = create_chunks(sentences, chunk_size, overlap)

    generate_embeddings(chunks)
    return