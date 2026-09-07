import chromadb
from chromadb import types as chroma_types

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="documents")


def store_embeddings(
    embeddings: list[list[float]],
    chunk_texts: list[str],
    metadata: list[chroma_types.Metadata],
):
    collection.add(
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
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    return results