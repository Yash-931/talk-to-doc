import chromadb
from chromadb import types as chroma_types
from rank_bm25 import BM25Okapi
from dataclasses import dataclass

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="documents")


@dataclass
class BM25Index:
    index: BM25Okapi
    ids: list[str]
    chunk_texts: list[str]
    metadata: list[chroma_types.Metadata]


def store_embeddings(
    embeddings: list[list[float]],
    chunk_texts: list[str],
    metadata: list[chroma_types.Metadata],
):
    collection.upsert(
        ids=[f"chunk_{i}" for i in range(len(chunk_texts))],
        documents=chunk_texts,
        embeddings=embeddings,
        metadatas=metadata,
    )

    return


def get_db_data():

    results = collection.get(include=["documents", "metadatas", "embeddings"])

    for id, document, metadata, embedding in zip(
        results["ids"],
        results["documents"],
        results["metadatas"],
        results["embeddings"],
    ):
        print("=" * 50)
        print("ID:", id)
        print("Pages:", metadata["page_nums"])
        print("Document:")
        print(document)
        print("Embedding: ", embedding)

    return


def query_db(query_embedding: list[float]):
    results = collection.query(query_embeddings=query_embedding, n_results=10)
    return results


def index_bm25(
    chunk_texts: list[str], metadata: list[chroma_types.Metadata]
) -> BM25Index:
    tokenized_chunks = [chunk.lower().split() for chunk in chunk_texts]
    return BM25Index(
        index=BM25Okapi(tokenized_chunks),
        ids=[f"chunk_{i}" for i in range(len(chunk_texts))],
        chunk_texts=chunk_texts,
        metadata=metadata,
    )


def hybrid_search(
    semantic_results: chroma_types.QueryResult,
    bm25_index: BM25Index,
    query_tokens: list[str],
    k: int = 3,
) -> chroma_types.QueryResult:
    rrf_constant = 60
    candidates = {}

    semantic_ids = semantic_results["ids"][0]
    semantic_documents = semantic_results["documents"][0]
    semantic_metadata = semantic_results["metadatas"][0]

    for rank, (chunk_id, document, metadata) in enumerate(
        zip(semantic_ids, semantic_documents, semantic_metadata), start=1
    ):
        candidates[chunk_id] = {
            "score": 1 / (rrf_constant + rank),
            "document": document,
            "metadata": metadata,
        }

    bm25_scores = bm25_index.index.get_scores(query_tokens)
    lexical_ranking = sorted(
        enumerate(bm25_scores), key=lambda item: item[1], reverse=True
    )

    for rank, (chunk_index, _) in enumerate(lexical_ranking, start=1):
        chunk_id = bm25_index.ids[chunk_index]
        candidate = candidates.setdefault(
            chunk_id,
            {
                "score": 0,
                "document": bm25_index.chunk_texts[chunk_index],
                "metadata": bm25_index.metadata[chunk_index],
            },
        )
        candidate["score"] += 1 / (rrf_constant + rank)

    ranked_candidates = sorted(
        candidates.items(), key=lambda item: item[1]["score"], reverse=True
    )[:k]

    return {
        "ids": [[chunk_id for chunk_id, _ in ranked_candidates]],
        "documents": [[candidate["document"] for _, candidate in ranked_candidates]],
        "metadatas": [[candidate["metadata"] for _, candidate in ranked_candidates]],
    }
