# Advanced RAG Pipeline: Knowledge Correction and Refinement

The provided diagram illustrates an advanced Retrieval-Augmented Generation (RAG) architecture designed to evaluate, refine, and correct retrieved information before passing it to a final generator model. This prevents the model from hallucinating based on irrelevant or incorrect retrieved documents. 

Here is the step-by-step breakdown of the concept:

## Step 1: Retrieval
*   **Input:** The system receives a user query $x$ (e.g., "Who was the screenwriter for *Death of a Batman*?").
*   **Action:** A standard retriever fetches an initial set of documents from a local database or corpus, resulting in Retrieved Documents ($d_1$, $d_2$).

## Step 2: Knowledge Correction (The Evaluator)
Instead of blindly trusting the retrieved documents, the system uses a **Retrieval Evaluator** (usually a specialized neural network or LLM).
*   **Action:** The evaluator asks a crucial question: "Are these retrieved documents actually correct and relevant to the query $x$?"
*   **Routing:** Based on its assessment, it routes the process down one of three distinct paths: **Correct**, **Ambiguous**, or **Incorrect**.

## Step 3: Processing based on Evaluation
Depending on the evaluator's classification, the system takes specific actions:

### Path A: Correct $\rightarrow$ Knowledge Refinement
If the documents are deemed mostly correct, the system optimizes them to reduce noise.
1.  **Decompose (Chunking):** The documents ($d_1$, $d_2$) are broken down into smaller "strips" or chunks of text (e.g., 900 characters each, as noted in the handwriting).
2.  **Filter:** An AI filter removes strips that are irrelevant to the specific query.
3.  **Recompose:** The remaining, highly relevant strips are stitched back together into a refined internal knowledge payload, denoted as $k_{in}$.

### Path C: Incorrect $\rightarrow$ Knowledge Searching
If the retrieved documents do not contain the answer, the system discards them and looks externally.
1.  **Rewrite:** The original query $x$ is rewritten into an optimized search engine query $q$ (e.g., "Death of a Batman; screenwriter; Wikipedia").
2.  **Web Search:** The system performs a live web search, retrieving external knowledge strips ($k_1$, $k_2$, ..., $k_n$).
3.  **Select:** The most relevant external strips are selected to form the external knowledge payload, $k_{ex}$.

### Path B: Ambiguous
If the evaluator is unsure or if the retrieved documents only partially answer the query, the system uses a hybrid approach. It performs both **Knowledge Refinement** on the existing documents to get $k_{in}$, AND **Knowledge Searching** to gather missing context, resulting in $k_{ex}$.

## Step 4: Generation
In the final step, the original query $x$ is combined with the carefully curated knowledge and sent to the Generator (the final LLM) to produce the answer.
*   **If Correct:** Generator input is query $x$ + internal knowledge $k_{in}$.
*   **If Ambiguous:** Generator input is query $x$ + internal knowledge $k_{in}$ + external knowledge $k_{ex}$.
*   **If Incorrect:** Generator input is query $x$ + external knowledge $k_{ex}$.

## Summary
By introducing an "Evaluator" and dynamic routing, this architecture ensures that the final generator is only fed concise, highly relevant, and fact-checked information, significantly improving the accuracy of the final response.