from .data_loader import parse_pdf, Sentence
from .chunker import create_chunks
from .embedder import generate_embeddings, EmbeddingCollection, generate_query_embedding
from .db import store_embeddings, query_db, index_bm25, hybrid_search
from .ai import generate_llm_response


class RAG:

    def __init__(self):
        self.bm25_index = None

    def ingest(self, path, chunk_size, overlap):

        sentences = parse_pdf(path)
        chunks = create_chunks(sentences, chunk_size, overlap)

        embedding_collection: EmbeddingCollection = generate_embeddings(chunks)

        store_embeddings(
            embeddings=embedding_collection.embeddings,
            chunk_texts=embedding_collection.chunk_texts,
            metadata=embedding_collection.metadata,
        )

        self.bm25_index = index_bm25(
            embedding_collection.chunk_texts, embedding_collection.metadata
        )

    def retrieve(self, query: str):
        if self.bm25_index is None:
            raise RuntimeError("Call ingest() before retrieving documents.")

        query_vector = generate_query_embedding(query)

        semantic_results = query_db(query_vector)

        tokenized_query = query.lower().split()
        return hybrid_search(semantic_results, self.bm25_index, tokenized_query)

    def ask(self, query):

        results = self.retrieve(query)

        return generate_llm_response(query, results)


def main():
    rag_ingest = RAG()
    print("Ingestion...")
    rag_ingest.ingest("ai-product-engineer-guide.pdf", 50, 5)


if __name__ == "__main__":
    main()
