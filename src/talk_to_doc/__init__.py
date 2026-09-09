from .data_loader import parse_pdf, Sentence
from .chunker import create_chunks
from .embedder import generate_embeddings, EmbeddingCollection, generate_query_embedding
from .db import store_embeddings, query_db, index_bm25, hybrid_search
from .ai import generate_llm_response


# TODOS: Implement the hybrid search (BM25 + semantic search)
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

        self.bm25_index = index_bm25(embedding_collection.chunk_texts)

    def retrieve(self, query: str):

        query_vector = generate_query_embedding(query)

        semantic_results = query_db(query_vector)

        tokenized_query = query.lower().split()
        bm25_results = self.bm25_index.get_scores(tokenized_query)

        return hybrid_search(semantic_results, bm25_results)

    def ask(self, query):

        results = self.retrieve(query)

        return generate_llm_response(query, results)
