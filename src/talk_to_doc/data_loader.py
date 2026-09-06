import pdfplumber
import nltk
from pydantic import BaseModel
from typing import Optional

nltk.download('punkt')
nltk.download('punkt_tab')

class Sentence(BaseModel):
    page_num: int
    text: str
    tokens: Optional[int] = None

def parse_pdf(path: str) -> list[Sentence]:
    pdf_sentences: list[Sentence] = []
    with pdfplumber.open(path) as pdf:
        for i,page in enumerate(pdf.pages):
            text = page.extract_text()

            if text:
                sentences = nltk.tokenize.sent_tokenize(text)

                for sentence in sentences:
                    clean_sentence = sentence.replace('\n', ' ').strip()
                    pdf_sentences.append(Sentence(page_num=i,text=clean_sentence))

    return pdf_sentences
