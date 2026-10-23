from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END


# Load environment variables from .env (for example, OPENAI_API_KEY).
load_dotenv()


# 1. Load PDF documents
docs = (
    PyPDFLoader("./documents/book1.pdf").load()
    + PyPDFLoader("./documents/book2.pdf").load()
    + PyPDFLoader("./documents/book3.pdf").load()
)

print(f"Loaded document pages: {len(docs)}")


# 2. Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150,
)
chunks = text_splitter.split_documents(docs)


# 3. Clean extracted text to avoid UnicodeEncodeError caused by
# unusual surrogate characters sometimes found in PDF extraction.
for doc in chunks:
    doc.page_content = (
        doc.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")
    )

print(f"Number of chunks: {len(chunks)}")


# 4. Create embeddings and a FAISS vector store.
# A fresh in-memory vector store is created each time this script runs.
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = FAISS.from_documents(chunks, embeddings)


# 5. Create a retriever that returns the four most similar chunks.
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)


# 6. Configure the chat model and prompt.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer only from the context. If not in context, say you don't know.",
        ),
        (
            "human",
            "Question: {question}\n\nContext:\n{context}",
        ),
    ]
)


# 7. Define the graph state.
class State(TypedDict):
    question: str
    docs: List[Document]
    answer: str


# 8. Retrieve relevant documents.
def retrieve(state: State) -> dict:
    question = state["question"]
    return {"docs": retriever.invoke(question)}


# 9. Generate an answer from the retrieved context.
def generate(state: State) -> dict:
    context = "\n\n".join(doc.page_content for doc in state["docs"])

    response = (prompt | llm).invoke(
        {
            "question": state["question"],
            "context": context,
        }
    )

    return {"answer": response.content}


# 10. Build and compile the LangGraph workflow.
graph = StateGraph(State)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()


# 11. Ask a question.
if __name__ == "__main__":
    question = "Explain the bias-variance tradeoff."

    result = app.invoke(
        {
            "question": question,
            "docs": [],
            "answer": "",
        }
    )

    print("\nAnswer:\n")
    print(result["answer"])
