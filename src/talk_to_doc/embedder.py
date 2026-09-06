from google import genai
from .data_loader import Sentence
import os
from dotenv import load_dotenv
from google.genai import types


load_dotenv()

def generate_embeddings(chunks: list[list[Sentence]]):
    client = genai.Client(
        vertexai=True, project=os.getenv("GCP_PROJECT"), location="us-central1"
    )

    chunk_texts: list[str] = []

    for chunk in chunks:
        for sentence in chunk:
            chunk_texts.append(sentence.text)

    response = client.models.embed_content(
        model='gemini-embedding-001',
        contents=chunk_texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT"
        )
    )
