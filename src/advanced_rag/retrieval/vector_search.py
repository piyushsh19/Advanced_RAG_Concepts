\"\"\"Simple in-memory FAISS index for learning experiments.\"\"\"

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from advanced_rag.config import settings


def build_faiss_index(documents: list[Document]) -> FAISS:
    \"\"\"Embed documents and build an in-memory FAISS index.\"\"\"
    if not documents:
        raise ValueError("At least one document chunk is required")
    if not settings.openai_api_key:
        raise RuntimeError("Set OPENAI_API_KEY in your .env file")

    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )
    return FAISS.from_documents(documents, embeddings)


def make_retriever(index: FAISS, k: int = 4):
    \"\"\"Return a similarity retriever for the top-k chunks.\"\"\"
    if k < 1:
        raise ValueError("k must be at least 1")
    return index.as_retriever(search_type="similarity", search_kwargs={"k": k})
