import tiktoken
from .data_loader import Sentence, parse_pdf


def tokenize_sentences(pdf_sentences: list[Sentence]):
    encoding = tiktoken.encoding_for_model("gpt-4o")

    for sentence in pdf_sentences:
        sentence_tokens = encoding.encode(sentence.text)
        sentence.tokens = len(sentence_tokens)


def create_chunks(pdf_sentences: list[Sentence], chunk_size: int = 100, overlap: int = 10):
    curr_tokens = 0
    chunks = []
    current_chunk: list[Sentence] = []

    tokenize_sentences(pdf_sentences)

    for sentence in pdf_sentences:
        sen_tokens = sentence.tokens

        if curr_tokens + sen_tokens > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
            overlap_tokens = 0
            overlap_chunk = []

            for s in reversed(current_chunk):
                if s.tokens + overlap_tokens > overlap:
                    break
                overlap_chunk.insert(0, s)
                overlap_tokens += s.tokens

            curr_tokens = overlap_tokens
            current_chunk = overlap_chunk

        current_chunk.append(sentence)
        curr_tokens += sentence.tokens

    if len(current_chunk) > 0:
        chunks.append(current_chunk)

    return chunks
