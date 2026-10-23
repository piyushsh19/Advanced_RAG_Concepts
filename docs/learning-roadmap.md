# Advanced RAG Learning Roadmap

For each module, write down the hypothesis, baseline, dataset, configuration, metrics, observed results, failure cases, and lessons learned.

## Phase 1 — Foundations
- Basic RAG: load documents, split them, embed chunks, retrieve top-k, generate grounded answers.
- Chunking: fixed-size, recursive, structure-aware, parent-child, and overlap experiments.
- Metadata: source, page, section, timestamps, and access controls.
- Embeddings and vector similarity; understand cosine similarity and distance metrics.

## Phase 2 — Better retrieval
- Sparse retrieval (BM25) and dense retrieval.
- Hybrid search and rank fusion.
- Reranking with cross-encoders or model-based rerankers.
- Query rewriting, multi-query retrieval, decomposition, and HyDE.
- Metadata filtering, contextual compression, and parent-document retrieval.

## Phase 3 — Advanced RAG patterns
- CRAG: assess retrieval quality and take a corrective action when evidence is weak.
- Self-RAG: use reflection/critique signals to decide whether retrieval or generation needs correction.
- Graph RAG: build and query entity/relationship structures; compare with vector-only retrieval.
- Agentic RAG: let a controlled agent select tools or retrieval steps, with limits and traceability.

## Phase 4 — Evaluation and reliability
- Create a small, versioned golden dataset of questions, expected facts, and source references.
- Retrieval metrics: Recall@k, Precision@k, MRR, and nDCG where appropriate.
- Answer metrics: faithfulness/groundedness, relevance, correctness, citation quality.
- Evaluate latency, token usage, cost, and failure rate.
- Add regression tests and inspect failures rather than relying on a single aggregate score.

## Phase 5 — Production
- FastAPI request/response schemas, validation, and error handling.
- Tracing, structured logs, monitoring, rate limits, and retries.
- Prompt-injection defenses, access-aware retrieval, PII handling, and secrets management.
- Docker, CI, deployment, load testing, and rollback strategy.

## Suggested deliverable per experiment
1. `README.md`: question, motivation, prerequisites, run instructions.
2. Implementation code with type hints.
3. Unit tests for key logic.
4. A baseline-versus-improved comparison.
5. A short `results.md` documenting trade-offs and limitations.
