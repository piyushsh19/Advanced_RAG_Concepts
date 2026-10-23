# Architecture

## Intended request flow

```text
Documents -> Load -> Normalize -> Chunk + metadata -> Embed -> Index
                                                        |
User question -> Validate -> Retrieve -> (Filter / Rerank / Correct)
                                      -> Build grounded prompt -> Generate
                                      -> Validate output / citations -> Response
```

## Module boundaries

- `ingestion/`: document loading and chunking.
- `retrieval/`: retrieval algorithms and ranking.
- `pipelines/`: orchestration patterns such as basic RAG, CRAG, Self-RAG, and agentic RAG.
- `evaluation/`: metrics and evaluation harnesses.
- `api/`: HTTP boundary only; keep business logic in pipelines.

Treat retrieved content as untrusted input. Enforce authorization filters before returning documents, and do not rely on the LLM alone for access control.
