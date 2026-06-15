# Workshop 2: Core NLP & Retrieval-Augmented Generation (RAG) 🔍

> **Part of the AI Unboxed Workshop Series.** Learn the core mechanics of Natural Language Processing (NLP) that enable Large Language Models (LLMs) to read language, represent meaning mathematically, and connect to external databases for grounded, real-time responses.

[![Open in Colab](https://img.shields.io/badge/Open_in_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/imaakanksha/AI-Unboxed-Teachable-Machine-Codelab/blob/main/02_core_nlp_and_rag/Workshop_2_Core_NLP_and_RAG.ipynb)

---

## 🎯 Learning Objectives

By the end of this workshop, you will be able to:
- 💡 Describe how raw text is broken down into **sub-word tokens** and model vocabulary constraints.
- 💡 Explain **word, sentence, and document embeddings** and compute cosine similarity.
- 💡 Explain the Transformer's **self-attention mechanism** (Queries, Keys, and Values).
- 💡 Formulate a **chunking strategy** balancing chunk size, overlap, and document structure.
- 💡 Differentiate between **lexical, vector, and hybrid search** systems.
- 💡 Assemble a **complete, in-memory RAG pipeline** using standard Python libraries.

---

## 🗺️ Curriculum Core

This workshop covers Modules 6 through 14, 21, and 22 of the AI Unboxed curriculum:

1. **Tokenization:** Sub-word tokenization algorithms (BPE, WordPiece, SentencePiece).
2. **Embeddings:** High-dimensional vector spaces and meaning computation.
3. **Attention Mechanism:** Multi-head self-attention equations and context modeling.
4. **Chunking Strategies:** Fixed-size, sentence, paragraph, semantic, and recursive splitting.
5. **Vector Databases:** Indexing algorithms (HNSW, IVF, PQ) and metadata filtering.
6. **Search & Retrieval:** Lexical (BM25) vs. Vector search, Hybrid search, and Reranking.
7. **RAG Architecture:** Integrating retrieval index with LLM generation prompt templates.
8. **Advanced RAG:** Knowledge ingestion and retrieval optimization.

---

## 🚀 Quick Start

### Option A: Run in Google Colab (Recommended)
Simply click the **Open in Colab** badge at the top of this page to run the workshop on a free cloud GPU environment. No installation required.

### Option B: Run Locally
If you prefer running the notebook on your local machine, ensure you have Python 3.10+ installed and run:

1. Navigate to this workshop directory:
   ```bash
   cd 02_core_nlp_and_rag
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1   # Windows
   source .venv/bin/activate    # macOS/Linux
   ```
3. Install Jupyter and dependencies:
   ```bash
   pip install jupyter numpy
   ```
4. Start the Jupyter Notebook interface:
   ```bash
   jupyter notebook
   ```
5. Open `Workshop_2_Core_NLP_and_RAG.ipynb` and run the cells.

---

## 👨‍🏫 Instructor Materials
Teaching this workshop? Check out the [Instructor Guide](./guides/INSTRUCTOR_GUIDE.md) and [Additional Resources](./guides/RESOURCES.md) in the `guides/` folder.
