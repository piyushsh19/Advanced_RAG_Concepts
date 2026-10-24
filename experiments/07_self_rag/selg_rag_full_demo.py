"""
Self-RAG: retrieval decision, retrieval, relevance filtering, and answer
support verification in one Python file.

Expected PDFs relative to this script:
    documents/Company_Policies.pdf
    documents/Company_Profile.pdf
    documents/Product_and_Pricing.pdf

Create a .env file beside this script:
    OPENAI_API_KEY=your_openai_api_key

Install dependencies:
    pip install langchain langchain-community langchain-openai langchain-text-splitters langgraph pydantic python-dotenv faiss-cpu pypdf

Run:
    python self_rag_with_relevance.py
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
    """Load the configured PDF files and report missing files clearly."""
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
    """Split documents, embed chunks, and create a FAISS retriever."""
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
# Graph state and structured-output schemas
# ---------------------------------------------------------------------------

class State(TypedDict):
    question: str
    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str
    issup: str
    evidence: List[str]


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


class IsSUPDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"] = Field(
        ...,
        description=(
            "Whether the generated answer is supported by the supplied context."
        ),
    )
    evidence: List[str] = Field(
        default_factory=list,
        description=(
            "Up to three short direct quotations from the context that support "
            "the answer's supported claims. Return an empty list if none."
        ),
    )


# ---------------------------------------------------------------------------
# Build the Self-RAG graph
# ---------------------------------------------------------------------------

def build_graph(retriever):
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    # 1. Decide whether this question requires document retrieval.
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

    # 2. Direct answer path for general questions.
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

    # 3. Judge whether each retrieved chunk is relevant.
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

    # 4. Generate an answer from relevant documents only.
    context_generation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the question using only the context below. If the "
                "context does not contain the answer, say the supplied "
                "documents do not contain enough information. Do not invent "
                "facts. Cite source filename and page number when available.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    # 5. Verify that the answer is supported by the context.
    issup_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are verifying whether the ANSWER is supported by the CONTEXT.\n"
                "Return the structured output with fields issup and evidence.\n"
                "issup must be one of: fully_supported, partially_supported, "
                "no_support.\n\n"
                "How to decide:\n"
                "- fully_supported: Every meaningful factual claim is explicitly "
                "supported by the context, and the answer does not add "
                "qualitative or interpretive claims absent from the context.\n"
                "- partially_supported: The core facts are supported, but the "
                "answer includes at least one unsupported detail, inference, "
                "or interpretive phrase.\n"
                "- no_support: The answer is mostly unrelated to the question "
                "or its claims are not supported by the context.\n\n"
                "Be strict. Do not use outside knowledge. Evidence must contain "
                "up to three short, direct quotations copied from the context "
                "that support the supported parts of the answer. If nothing "
                "supports the answer, return an empty evidence list.",
            ),
            (
                "human",
                "Question:\n{question}\n\n"
                "Answer:\n{answer}\n\n"
                "Context:\n{context}",
            ),
        ]
    )
    issup_llm = llm.with_structured_output(IsSUPDecision)

    # -----------------------------------------------------------------------
    # Node functions
    # -----------------------------------------------------------------------

    def decide_retrieval(state: State):
        decision: RetrieveDecision = structured_decider.invoke(
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
        # Direct answers do not have retrieved context to verify against.
        return {
            "answer": response.content,
            "docs": [],
            "relevant_docs": [],
            "context": "",
            "issup": "not_checked",
            "evidence": [],
        }

    def retrieve(state: State):
        documents = retriever.invoke(state["question"])
        return {"docs": documents}

    def is_relevant(state: State):
        """Retain only retrieved chunks that help answer the question."""
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

    def route_after_relevance(
        state: State,
    ) -> Literal["generate_from_context", "no_answer_found"]:
        if state.get("relevant_docs"):
            return "generate_from_context"
        return "no_answer_found"

    def no_answer_found(state: State):
        """Stop cleanly when retrieval returns no relevant chunks."""
        return {
            "context": "",
            "answer": (
                "I couldn't find relevant information in the supplied company "
                "documents to answer this question."
            ),
            "issup": "no_support",
            "evidence": [],
        }

    def generate_from_context(state: State):
        """Generate a grounded answer and preserve the context for verification."""
        context_parts = []

        for doc in state.get("relevant_docs", []):
            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get("page")
            page_label = f", page {page + 1}" if isinstance(page, int) else ""
            context_parts.append(
                f"[Source: {Path(source).name}{page_label}]\n{doc.page_content}"
            )

        context = "\n\n---\n\n".join(context_parts)
        response = llm.invoke(
            context_generation_prompt.format_messages(
                question=state["question"],
                context=context,
            )
        )
        return {"context": context, "answer": response.content}

    def is_sup(state: State):
        """Check whether generated claims are supported by retrieved context."""
        decision: IsSUPDecision = issup_llm.invoke(
            issup_prompt.format_messages(
                question=state["question"],
                answer=state.get("answer", ""),
                context=state.get("context", ""),
            )
        )
        return {
            "issup": decision.issup,
            "evidence": decision.evidence,
        }

    def route_after_decide(
        state: State,
    ) -> Literal["generate_direct", "retrieve"]:
        return "retrieve" if state["need_retrieval"] else "generate_direct"

    # -----------------------------------------------------------------------
    # Graph structure
    # -----------------------------------------------------------------------

    graph = StateGraph(State)

    graph.add_node("decide_retrieval", decide_retrieval)
    graph.add_node("generate_direct", generate_direct)
    graph.add_node("retrieve", retrieve)
    graph.add_node("is_relevant", is_relevant)
    graph.add_node("no_answer_found", no_answer_found)
    graph.add_node("generate_from_context", generate_from_context)
    graph.add_node("is_sup", is_sup)

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
    graph.add_conditional_edges(
        "is_relevant",
        route_after_relevance,
        {
            "generate_from_context": "generate_from_context",
            "no_answer_found": "no_answer_found",
        },
    )
    graph.add_edge("no_answer_found", END)
    graph.add_edge("generate_from_context", "is_sup")
    graph.add_edge("is_sup", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Interactive command-line application
# --------------------------------------------------------------------------------------

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
            "context": "",
            "answer": "",
            "issup": "",
            "evidence": [],
        }

        try:
            result = app.invoke(initial_state)
        except Exception as exc:
            print(f"\nError while answering: {exc}\n")
            continue

        print(f"\nneed_retrieval: {result.get('need_retrieval')}")
        print(f"#docs: {len(result.get('docs', []))}")
        print(f"#relevant_docs: {len(result.get('relevant_docs', []))}")
        print(f"issup: {result.get('issup')}")
        print(f"evidence: {result.get('evidence', [])}")
        print(f"answer: {result.get('answer')}\n")


if __name__ == "__main__":
    main()
