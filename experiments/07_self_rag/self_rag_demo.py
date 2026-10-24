"""
Self-RAG: Retrieval decision + direct generation or document retrieval.

Expected PDF files (relative to this script):
    documents/Company_Policies.pdf
    documents/Company_Profile.pdf
    documents/Product_and_Pricing.pdf

Environment:
    OPENAI_API_KEY=your_api_key

Install dependencies:
    pip install langchain langchain-community langchain-openai langchain-text-splitters langgraph pydantic python-dotenv faiss-cpu pypdf
"""

from pathlib import Path
from typing import List, Literal, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DOCUMENT_DIR = BASE_DIR / "documents"
PDF_FILES = [
    DOCUMENT_DIR / "Company_Policies.pdf",
    DOCUMENT_DIR / "Company_Profile.pdf",
    DOCUMENT_DIR / "Product_and_Pricing.pdf",
]

EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 150
RETRIEVAL_K = 4


# ---------------------------------------------------------------------------
# Load and index documents
# ---------------------------------------------------------------------------

def load_documents() -> List[Document]:
    """Load all configured PDFs, with a clear error if any file is missing."""
    missing = [str(path) for path in PDF_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing PDF file(s):\n- "
            + "\n- ".join(missing)
            + f"\n\nPlace the PDFs inside: {DOCUMENT_DIR}"
        )

    documents: List[Document] = []
    for path in PDF_FILES:
        documents.extend(PyPDFLoader(str(path)).load())

    if not documents:
        raise ValueError("No text was extracted from the configured PDFs.")

    return documents


def build_retriever():
    """Split the source documents and build a FAISS retriever."""
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("Document splitting produced no chunks.")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})


# ---------------------------------------------------------------------------
# LangGraph state and structured retrieval decision
# ---------------------------------------------------------------------------

class State(TypedDict):
    question: str
    need_retrieval: bool
    docs: List[Document]
    relev
    answer: str


class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        ...,
        description=(
            "True if external documents are needed to answer reliably, "
            "otherwise False."
        ),
    )


def build_graph(retriever):
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    decide_retrieval_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You decide whether document retrieval is needed.\n"
                "Return a structured boolean matching the schema.\n"
                "Guidelines:\n"
                "- Choose True when the answer needs company-specific facts, "
                "citations, policies, product details, pricing, or information "
                "that may not be in general model knowledge.\n"
                "- Choose False for general explanations, definitions, or "
                "reasoning that does not need the supplied documents.\n"
                "- If unsure, choose True.",
            ),
            ("human", "Question: {question}"),
        ]
    )
    structured_decider = llm.with_structured_output(RetrieveDecision)

    direct_generation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the question using only your general knowledge.\n"
                "Do not assume access to external documents.\n"
                "If you are unsure or the answer requires specific sources, "
                "say: 'I don't know based on my general knowledge.'",
            ),
            ("human", "{question}"),
        ]
    )

    retrieved_generation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's question using only the context below.\n"
                "If the context does not contain the answer, say that the "
                "provided documents do not contain enough information.\n"
                "Do not invent facts. Mention the source filename/page when "
                "that metadata is available.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    def decide_retrieval(state: State):
        decision = structured_decider.invoke(
            decide_retrieval_prompt.format_messages(
                question=state["question"]
            )
        )
        return {"need_retrieval": decision.should_retrieve}

    def generate_direct(state: State):
        response = llm.invoke(
            direct_generation_prompt.format_messages(
                question=state["question"]
            )
        )
        return {"answer": response.content, "docs": []}

    def retrieve(state: State):
        documents = retriever.invoke(state["question"])
        return {"docs": documents}

    def generate_from_docs(state: State):
        context_parts = []
        for doc in state.get("docs", []):
            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get("page")
            page_label = f", page {page + 1}" if isinstance(page, int) else ""
            context_parts.append(
                f"[Source: {Path(source).name}{page_label}]\n{doc.page_content}"
            )

        context = "\n\n---\n\n".join(context_parts)
        response = llm.invoke(
            retrieved_generation_prompt.format_messages(
                question=state["question"],
                context=context or "No relevant documents were retrieved.",
            )
        )
        return {"answer": response.content}

    def route_after_decide(
        state: State,
    ) -> Literal["generate_direct", "retrieve"]:
        if state["need_retrieval"]:
            return "retrieve"
        return "generate_direct"

    graph = StateGraph(State)
    graph.add_node("decide_retrieval", decide_retrieval)
    graph.add_node("generate_direct", generate_direct)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_from_docs", generate_from_docs)

    graph.add_edge(START, "decide_retrieval")
    graph.add_conditional_edges(
        "decide_retrieval",
        route_after_decide,
        {
            "generate_direct": "generate_direct",
            "retrieve": "retrieve",
        },
    )
    graph.add_edge("generate_direct", END)
    graph.add_edge("retrieve", "generate_from_docs")
    graph.add_edge("generate_from_docs", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Run the Self-RAG application
# ---------------------------------------------------------------------------

def main():
    print("Building document retriever...")
    retriever = build_retriever()

    print("Compiling Self-RAG graph...")
    app = build_graph(retriever)

    print("\nSelf-RAG is ready. Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        initial_state: State = {
            "question": question,
            "need_retrieval": False,
            "docs": [],
            "answer": "",
        }

        try:
            result = app.invoke(initial_state)
        except Exception as exc:
            print(f"\nError while answering: {exc}\n")
            continue

        print(f"\nRetrieval required: {result['need_retrieval']}")
        if result.get("docs"):
            print(f"Documents retrieved: {len(result['docs'])}")
        print(f"\nAnswer:\n{result['answer']}\n")


if __name__ == "__main__":
    main()
