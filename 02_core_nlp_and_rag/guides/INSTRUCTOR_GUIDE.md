# Instructor Guide: Workshop 2 (Core NLP & RAG) 🎓

This guide is designed to help you deliver a comprehensive, 2-hour session on tokenization, attention, vector search, and RAG pipelines.

---

## ⏱️ Timeline & Agenda

| Time | Topic | Format | Instructor Focus |
|------|-------|--------|------------------|
| **00:00 - 00:10** | Recap & Welcome | Talk | Quick review of W1 (Optimization/Generalization). |
| **00:10 - 00:35** | Tokenization & Embeddings | Interactive Notebook | Explain sub-word tokens and cosine similarity. Compute similarity. |
| **00:35 - 00:50** | Self-Attention Mechanism | Talk / Diagram | Explain Q, K, V vectors. Show multi-head scaling. |
| **00:50 - 01:10** | Chunking Strategies & Vector DBs | Notebook / Discussion | Show tradeoffs of chunk sizes & overlaps. Explain HNSW & IVF indexing. |
| **01:10 - 01:25** | Lexical, Vector & Hybrid Search | Interactive Notebook | Reciprocal Rank Fusion (RRF) and Reranking logic. |
| **01:25 - 01:50** | Building an In-Memory RAG | Hands-on Coding | Guide students through building a simple RAG pipeline from scratch. |
| **01:50 - 02:00** | Q&A & Wrap-up | Discussion | Next-step pathways. |

---

## 💡 Key Teaching Hooks

- **The Index Analogy (Embeddings):** Explain that embeddings are like GPS coordinates for meaning. "King" and "Queen" share very similar latitudes and longitudes, while "Banana" is on another continent.
- **The Search Analogy (Attention Q, K, V):**
  - **Query (Q):** What I'm searching for (e.g., "subject").
  - **Key (K):** The label/tags on other words (e.g., "noun", "verb").
  - **Value (V):** The actual informational content of the word.
- **The Grounding Rule (RAG):** RAG is like an open-book exam. Instead of memorizing all information, the model retrieves the relevant page and references it to write the answer.

---

## 🙋 Q&A Prep & Common Student Roadblocks

### Q: Why do we need sub-word tokenization instead of character-level or word-level?
*   **Answer:** Word-level tokenization results in a massive vocabulary size (millions of words) and fails on unseen words (Out-Of-Vocabulary). Character-level tokenization has a tiny vocabulary (letters/symbols) but makes sequences extremely long, degrading the model's capacity to learn long-range patterns. Sub-word tokenization (BPE/WordPiece) finds the optimal balance, breaking rare words into common fragments while keeping standard words intact.

### Q: Why is HNSW (Hierarchical Navigable Small World) preferred for vector search?
*   **Answer:** Exact nearest neighbor search requires comparing the query vector against every single vector in the database ($O(N)$ complexity), which is too slow. HNSW builds a hierarchical graph (like skip-lists) where higher layers let you skip large portions of the graph and lower layers let you do fine-grained search, achieving $O(\log N)$ search time.

---

## 🎯 Conceptual Quiz Answers (Notebook Checkpoints)

### Quiz 1: Cosine Similarity
- *Question:* What does a cosine similarity of 0 mean?
- *Answer:* The two vectors are orthogonal (independent/unrelated in meaning).

### Quiz 2: Reranking
- *Question:* Why retrieve 100 documents first and rerank down to 5, instead of running the reranker on all 1,000,000 documents?
- *Answer:* Speed. Vector retrieval (Bi-encoders) is extremely fast but slightly less precise. Rerankers (Cross-encoders) are highly precise but very slow because they concatenate query + doc. Running the reranker on all documents would take minutes; running it on the top 100 takes milliseconds.
