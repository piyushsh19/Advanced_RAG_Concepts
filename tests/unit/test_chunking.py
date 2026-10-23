import pytest
from langchain_core.documents import Document

from advanced_rag.ingestion.chunking import split_documents


def test_split_documents_preserves_source_metadata() -> None:
    docs = [Document(page_content="hello world " * 100, metadata={"source": "sample.pdf"})]
    chunks = split_documents(docs, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "sample.pdf" for chunk in chunks)


def test_invalid_chunk_overlap_raises() -> None:
    with pytest.raises(ValueError):
        split_documents([Document(page_content="hello")], chunk_size=100, chunk_overlap=100)
