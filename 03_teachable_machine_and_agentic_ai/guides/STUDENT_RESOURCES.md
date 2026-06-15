# Student Resources: Advanced Agentic AI & Trustworthiness 📚

This document compiles advanced reading materials, design patterns, and case studies covering Explainable AI (XAI), Prompt Engineering, and Agentic Architectures (including MCP and Multi-Agent Orchestration).

---

## 🔍 1. Explainable AI (XAI)
Explainable AI focuses on opening the "black box" of neural networks to understand why a model made a specific prediction.
- **Saliency Maps:** Visual representations highlight which pixels contributed most to a classification. For example, in our Laptop vs. Mobile classifier, a saliency map might highlight the keyboard or screen edges.
- **SHAP (SHapley Additive exPlanations):** A game-theoretic approach to explain the output of any machine learning model by computing feature attribution values.
- **LIME (Local Interpretable Model-agnostic Explanations):** Approximates the complex model locally with a simple, interpretable linear model around the prediction point.

---

## 📈 2. Advanced Evaluation Metrics
Beyond standard accuracy, production models require careful statistical evaluation:
- **Precision:** True Positives / (True Positives + False Positives). Best when false alarms are costly.
- **Recall (Sensitivity):** True Positives / (True Positives + False Negatives). Critical when missing a positive case is costly (e.g., medical diagnoses).
- **F1-Score:** Harmonic mean of Precision and Recall. The standard metric for class-imbalanced datasets.
- **Confusion Matrix:** An $N \times N$ grid detailing actual vs. predicted classifications to pinpoint exactly where models confuse labels.

---

## 📝 3. Prompt Engineering
Prompt Engineering is the practice of structuring text inputs to steer foundation models toward high-quality, structured outputs.
- **Chain of Thought (CoT):** Encouraging the model to generate its step-by-step reasoning process before outputting the final answer (drastically reduces logical errors).
- **Few-Shot Prompting:** Providing a few demonstration examples in the prompt to align output formatting and expectations.
- **System Instructions:** Defining role constraints, behavior safety guardrails, and context limitations at the system prompt level.

---

## 🤖 4. AI Agents & ReAct Pattern
AI Agents are autonomous systems that use foundation models to reason, plan, and invoke external tools (APIs, databases, browsers) to execute complex objectives.
- **ReAct (Reasoning + Action) Loop:** The core loop behind single-agent planning:
  ```
  Thought ➔ Action (Tool Call) ➔ Observation (Tool Output) ➔ Repeat ➔ Final Answer
  ```
- **Tool Calling:** The LLM does not run code directly; instead, it outputs structured JSON indicating which tool to run and with what arguments (e.g., calling our `teachable-tm predict` script to analyze a photo).

---

## 🌐 5. Multi-Agent Systems & MCP
When a task is too complex for a single agent, we deploy multi-agent systems where agents take on specialized roles (e.g., Researcher, Coder, Critic) and collaborate.
- **Model Context Protocol (MCP):** An open standard protocol designed to connect Large Language Models directly to data sources, tools, and environments in a secure, standardized manner (similar to how USB standardizes hardware connections).
- **Orchestration:** Frameworks like AutoGen, CrewAI, or LangGraph manage state transitions, memory routing, and agent collaboration loops.
