\"\"\"Corrective RAG (CRAG) learning module.

Suggested implementation steps:
1. Retrieve an initial candidate set.
2. Score/grade whether the retrieved evidence is useful for the question.
3. If evidence is strong, generate from it.
4. If evidence is weak, rewrite the query and/or search a trusted fallback source.
5. Generate with the corrected evidence and log which path was taken.

Do not claim a production-ready CRAG implementation until grading, fallback
behavior, failure modes, and evaluation are implemented and tested.
\"\"\"
