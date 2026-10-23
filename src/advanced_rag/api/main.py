\"\"\"Minimal FastAPI health endpoint; add a RAG endpoint after pipeline tests.\"\"\"

from fastapi import FastAPI

app = FastAPI(
    title="Advanced RAG Concepts API",
    version="0.1.0",
    description="Learning API for advanced RAG experiments.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
