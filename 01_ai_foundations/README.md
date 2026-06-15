# Workshop 1: Foundations of AI 🧠

> **Part of the AI Unboxed Workshop Series.** Learn the mechanics of how artificial intelligence processes data, makes decisions, and generalizes to the real world.

[![Open in Colab](https://img.shields.io/badge/Open_in_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/imaakanksha/AI-Unboxed-Teachable-Machine-Codelab/blob/main/01_ai_foundations/Workshop_1_AI_Foundations.ipynb)

---

## 🎯 Learning Objectives

By the end of this workshop, you will be able to:
- 💡 Explain **how AI models optimize parameters** using Gradient Descent and Backpropagation instead of "thinking".
- 💡 Deconstruct why modern Large Language Models (LLMs) **exhibit emergent behaviors** and navigate the "stochastic parrot" debate.
- 💡 Identify the structural reasons behind **AI hallucinations** and apply practical mitigation strategies.
- 💡 Diagnoses training errors like **Overfitting** and **Underfitting** and manage the Bias-Variance tradeoff.
- 💡 Analyze how **systemic bias** creeps into machine learning pipelines and evaluate mitigation techniques.

---

## 🗺️ Curriculum Core

This workshop covers Modules 1 through 5 of the AI Unboxed curriculum:

1. **How AI Actually Learns:** The iterative numerical optimization loop (Forward Pass, Loss Functions, Backpropagation, Optimizers).
2. **Why AI Feels Intelligent:** Parameters, emergence, RLHF/fine-tuning, and statistical next-token prediction.
3. **Hallucinations:** Why generative models make up facts confidently, types of hallucinations, and mitigation tools.
4. **Overfitting vs Underfitting:** High bias vs high variance, learning curves, and regularization techniques (dropout, early stopping).
5. **AI Bias:** Systemic bias in data collection, historical bias, and real-world case studies (COMPAS, Amazon hiring).

---

## 🚀 Quick Start

### Option A: Run in Google Colab (Recommended)
Simply click the **Open in Colab** badge at the top of this page to run the workshop on a free cloud GPU environment. No installation required.

### Option B: Run Locally
If you prefer running the notebook on your local machine, ensure you have Python 3.10+ installed and run:

1. Navigate to this workshop directory:
   ```bash
   cd 01_ai_foundations
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1   # Windows
   source .venv/bin/activate    # macOS/Linux
   ```
3. Install Jupyter:
   ```bash
   pip install jupyter
   ```
4. Start the Jupyter Notebook interface:
   ```bash
   jupyter notebook
   ```
5. Open `Workshop_1_AI_Foundations.ipynb` and run the cells.

---

## 👨‍🏫 Instructor Materials
Teaching this workshop? Check out the [Instructor Guide](./guides/INSTRUCTOR_GUIDE.md) and [Additional Resources](./guides/RESOURCES.md) in the `guides/` folder.
