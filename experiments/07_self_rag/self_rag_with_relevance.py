"""
Self-RAG: retrieval decision, document retrieval, relevance filtering,
and answer generation in one Python file.

Expected PDFs beside this script in documents/:
    Company_Policies.pdf
    Company_Profile.pdf
    Product_and_Pricing.pdf

.env:
    OPENAI_API_KEY=your_openai_api_key

Install:
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
# Load PDFs and build the FAISS retriever
# ---------------------------------------------------------------------------

def load_documents() -> List[Document]:
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
# LangGraph state and structured decision schemas
# ---------------------------------------------------------------------------

class State(TypedDict):
    question: str
    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    answer: str


class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        ...,
        description=(
            "True if external documents are needed to answer reliably; "
            "False for general questions answerable without those documents."
        ),
    )


class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        ...,
        description=(
            "True if the document contains useful information for answering "
            "the question; otherwise False."
        ),
    )


# ---------------------------------------------------------------------------
# Build Self-RAG graph
# ---------------------------------------------------------------------------

def build_graph(retriever):
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    decide_retrieval_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Decide whether document retrieval is needed.\n"
                "Choose True for company-specific facts, policies, pricing, "
                "product details, citations, or information that may not be "
                "in general model knowledge.\n"
                "Choose False for general explanations and definitions that "
                "do not require the supplied documents.\n"
                "If unsure, choose True.",
            ),
            ("human", "Question: {question}"),
        ]
    )
    structured_decider = llm.with_structured_output(RetrieveDecision)

    direct_generation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer using general knowledge only. Do not claim to have "
                "checked company documents. If unsure, say you do not know.",
            ),
            ("human", "{question}"),
        ]
    )

    is_relevant_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Judge whether the document is relevant to answering the "
                "question. Return the requested structured Boolean. A "
                "document is relevant only if it contains useful information "
                "for answering; a related keyword alone is not enough.",
            ),
            ("human", "Question:\n{question}\n\nDocument:\n{document}"),
        ]
    )
    relevance_llm = llm.with_structured_output(RelevanceDecision)

    context_generation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the question using only the context below. If the "
                "context does not contain the answer, say the supplied "
                "documents do not contain enough information. Do not invent "
                "facts. When available, cite source filename and page number.\n\n"
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
        return {
            "answer": response.content,
            "docs": [],
            "relevant_docs": [],
        }

    def retrieve(state: State):
        return {"docs": retriever.invoke(state["question"])}

    def is_relevant(state: State):
        """Evaluate each retrieved chunk and retain relevant chunks only."""
        relevant_docs: List[Document] = []

        for doc in state.get("docs", []):
            decision: RelevanceDecision = relevance_llm.invoke(
                is_relevant_prompt.format_messages(
                    question=state["question"],
                    document=doc.page_content,
                )
            )
            if decision.is_relevant:
                relevant_docs.append(doc)

        return {"relevant_docs": relevant_docs}

    def generate_from_context(state: State):
        """Generate a grounded answer using only relevance-filtered chunks."""
        context_parts = []

        for doc in state.get("relevant_docs", []):
            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get("page")
            page_label = f", page {page + 1}" if isinstance(page, int) else ""
            context_parts.append(
                f"[Source: {Path(source).name}{page_label}]\n{doc.page_content}"
            )

        context = "\n\n---\n\n".join(context_parts)
        if not context:
            context = (
                "No retrieved chunks were judged relevant. The supplied "
                "documents do not provide enough verified information."
            )

        response = llm.invoke(
            context_generation_prompt.format_messages(
                question=state["question"],
                context=context,
            )
        )
        return {"answer": response.content}

    def route_after_decide(
        state: State,
    ) -> Literal["generate_direct", "retrieve"]:
        return "retrieve" if state["need_retrieval"] else "generate_direct"

    graph = StateGraph(State)
    graph.add_node("decide_retrieval", decide_retrieval)
    graph.add_node("generate_direct", generate_direct)
    graph.add_node("retrieve", retrieve)
    graph.add_node("is_relevant", is_relevant)
    graph.add_node("generate_from_context", generate_from_context)

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
    graph.add_edge("retrieve", "is_relevant")
    graph.add_edge("is_relevant", "generate_from_context")
    graph.add_edge("generate_from_context", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Interactive command-line application
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
            print("Goodbye!")
            break
        if not question:
            continue

        initial_state: State = {
            "question": question,
            "need_retrieval": False,
            "docs": [],
            "relevant_docs": [],
            "answer": "",
        }

        try:
            result = app.invoke(initial_state)
        except Exception as exc:
            print(f"\nError while answering: {exc}\n")
            continue

        print(f"\nRetrieval required: {result['need_retrieval']}")
        print(f"Chunks retrieved: {len(result.get('docs', []))}")
        print(f"Relevant chunks: {len(result.get('relevant_docs', []))}")
        print(f"\nAnswer:\n{result['answer']}\n")


if __name__ == "__main__":
    main()
