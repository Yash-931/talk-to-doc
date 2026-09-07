from google import genai
import os
from dotenv import load_dotenv
from chromadb.api import types as chroma_types

load_dotenv()

client = genai.Client(
    vertexai=True, project=os.getenv("GCP_PROJECT"), location="us-central1"
)


def generate_llm_response(query: str, query_result: chroma_types.QueryResult):

    documents = query_result["documents"][0]
    metadatas = query_result["metadatas"][0]

    context_parts = []

    for document, metadata in zip(documents, metadatas):
        context_parts.append(f"Page(s): {metadata["page_nums"]}\n" f"{document}")

    context = "\n\n".join(context_parts)

    prompt = f"""
Answer the user's question using the provided context. Only use this information and if you cannot answer the query based on this, say I can't answer instead of guessing

Context:
{context}

User question:
{query}"""

    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=[prompt]
    )

    return response
