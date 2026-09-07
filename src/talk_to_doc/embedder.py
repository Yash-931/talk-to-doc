from google import genai
from .data_loader import Sentence
import os
from dotenv import load_dotenv
from google.genai import types as genai_types
from chromadb import types as chroma_types
from pydantic import BaseModel

load_dotenv()


class EmbeddingCollection(BaseModel):
    embeddings: list[list[float]]
    chunk_texts: list[str]
    metadata: list[chroma_types.Metadata]


client = genai.Client(
    vertexai=True, project=os.getenv("GCP_PROJECT"), location="us-central1"
)


def generate_embeddings(chunks: list[list[Sentence]]):
    chunk_texts: list[str] = []
    metadata = []

    for chunk in chunks:
        chunk_text = " ".join(sentence.text for sentence in chunk)

        pages = list(dict.fromkeys(sentence.page_num for sentence in chunk))

        chunk_texts.append(chunk_text)
        metadata.append({"page_nums": ",".join(map(str, pages))})

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk_texts,
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )

    embeddings = [embedding.values for embedding in response.embeddings]

    return EmbeddingCollection(
        embeddings=embeddings, chunk_texts=chunk_texts, metadata=metadata
    )


def generate_query_embedding(query: str):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )

    embedded_query = response.embeddings[0].values
    return embedded_query
