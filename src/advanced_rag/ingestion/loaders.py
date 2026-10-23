\"\"\"Document-loading helpers.\"\"\"

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(path: str | Path) -> list[Document]:
    \"\"\"Load a PDF and return page-level LangChain documents.\"\"\"
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    return PyPDFLoader(str(pdf_path)).load()
