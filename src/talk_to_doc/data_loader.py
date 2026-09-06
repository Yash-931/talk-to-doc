import pdfplumber

def parse_pdf(path: str):
    pdf_content: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            pdf_content.append(text)

    return pdf_content
