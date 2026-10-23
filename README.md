# Advanced RAG Concepts

A hands-on learning repository that progresses from foundational Retrieval-Augmented Generation (RAG) to advanced retrieval strategies and production-oriented RAG systems.

## Learning goals

- Understand ingestion, chunking, embeddings, vector search, retrieval, prompting, and generation.
- Compare retrieval strategies with repeatable experiments.
- Implement hybrid search, reranking, query rewriting, Corrective RAG (CRAG), Self-RAG, Graph RAG, and agentic RAG.
- Evaluate retrieval and answer quality with explicit datasets and metrics.
- Expose a RAG pipeline through FastAPI and learn production concerns such as observability, latency, cost, security, and deployment.

## Repository layout

```text
src/advanced_rag/     Reusable application code
experiments/          Focused, documented learning experiments
tests/                Unit and integration tests
docs/                 Architecture, roadmap, and glossary
data/                 Local datasets (not committed by default)
scripts/              Developer utilities
```

## Quick start

Recommended: Python 3.11 or 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Add your API key to `.env`:

```env
OPENAI_API_KEY=replace_me
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Do not commit `.env`, API keys, private PDFs, or private datasets.

## First experiment

Place PDFs you are allowed to process under `data/raw/`, then run:

```bash
python scripts/ingest.py --help
```

The initial scaffold deliberately does not ingest or upload any documents automatically. Implement and test each stage as you progress through the roadmap.

## Learning sequence

1. Basic RAG and document ingestion
2. Chunking strategies and metadata
3. Embeddings and vector similarity
4. Hybrid search (dense + lexical)
5. Reranking
6. Query rewriting, decomposition, and multi-query retrieval
7. Context compression and filtering
8. Corrective RAG (CRAG)
9. Self-RAG and reflection patterns
10. Graph RAG
11. Agentic RAG and tool use
12. Evaluation, tracing, and regression testing
13. FastAPI serving, deployment, and production hardening

See [`docs/learning-roadmap.md`](docs/learning-roadmap.md).

## Principles

- Prefer measurable improvements over complexity for its own sake.
- Keep experiments reproducible and record model names, parameters, dataset versions, and costs.
- Separate retrieval evaluation from answer-generation evaluation.
- Never treat retrieved document text as trusted instructions.
- Pin dependencies and add tests before production use.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
