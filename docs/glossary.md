# RAG Glossary

- **RAG:** Retrieval-Augmented Generation; retrieves external context before generating a response.
- **Chunk:** A segment of a source document indexed for retrieval.
- **Embedding:** A vector representation used to compare semantic similarity.
- **Vector store:** A system that indexes vectors and associated metadata.
- **Hybrid search:** Combining lexical retrieval (such as BM25) with dense vector retrieval.
- **Reranking:** Reordering retrieved candidates using a more precise relevance model.
- **CRAG:** Corrective RAG; evaluates retrieval quality and applies corrective retrieval actions when needed.
- **Self-RAG:** A pattern using reflection signals to assess retrieval and generation decisions.
- **Graph RAG:** Uses entity and relationship graphs to support retrieval and reasoning.
- **Agentic RAG:** Uses an agent/workflow to decide among retrieval or other tools.
- **Recall@k:** Fraction of relevant items found among the top k results.
- **Faithfulness:** Whether generated claims are supported by the provided evidence.
