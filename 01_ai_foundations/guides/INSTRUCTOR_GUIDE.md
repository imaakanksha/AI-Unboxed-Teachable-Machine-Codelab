# Instructor Guide: Workshop 1 (Foundations of AI) 🎓

This guide is designed to help you deliver a high-impact, engaging 90-minute session on AI foundations.

---

## ⏱️ Timeline & Agenda

| Time | Topic | Format | Instructor Focus |
|------|-------|--------|------------------|
| **00:00 - 00:10** | Welcome & Introduction | Slide / Talk | Hook the audience, overview the series. |
| **00:10 - 00:30** | How AI Learns (Optimization) | Interactive Notebook | Explain Gradient Descent & Backpropagation. Run linear demo. |
| **00:30 - 00:45** | Why AI Feels Intelligent | Conceptual / Discussion | P parameters, Emergent behaviors, Stochastic parrot debate. |
| **00:45 - 01:00** | Hallucinations | Interactive Notebook | Show hallucinations, discuss temperature, RAG as solution. |
| **01:00 - 01:15** | Overfitting & Underfitting | Live Coding / Notebook | Demonstrate learning curves, dropout, early stopping. |
| **01:15 - 01:30** | AI Bias & Wrap-up | Discussion / Q&A | Historical bias, COMPAS case study, wrap-up. |

---

## 💡 Key Teaching Hooks

- **The Mixer Analogy (How AI Learns):** Explain that training a neural network is not like teaching a child; it is like adjusting 100 million dials on a music mixer console. We turn the dials until the song sounds right.
- **The Next-Token Illusion (Why AI Feels Intelligent):** Show that when a model outputs "Paris" for "The capital of France is ___", it didn't look up a database. It simply predicted the statistically most probable next word fragment.
- **The "Confident Liar" (Hallucinations):** Emphasize that models are trained to be fluent, not truthful. A model doesn't know what it doesn't know.

---

## 🙋 Q&A Prep & Common Student Roadblocks

### Q: Why does raising the temperature make the model hallucinate more?
*   **Answer:** Temperature controls the probability distribution of next tokens. High temperature flattens the distribution, giving less probable words (which are often incorrect or creative) a higher chance of being selected. Low temperature (near 0) makes the model select the single most likely token every time.

### Q: How do we fix bias if it is already in the training data?
*   **Answer:** There are three stages of mitigation:
    1. **Pre-processing:** Re-weighting or augmenting training data (e.g., adding representative samples).
    2. **In-processing:** Modifying the loss function during training to penalize disparate outcomes.
    3. **Post-processing:** Adjusting output classification thresholds for different demographic groups (e.g., using equalized odds).

---

## 🎯 Conceptual Quiz Answers (Notebook Checkpoints)

### Quiz 1: Gradient Descent
- *Question:* What happens if the learning rate is too large?
- *Answer:* The optimizer can overshoot the minimum and diverge, causing the loss to increase or fluctuate wildly.

### Quiz 2: Overfitting
- *Question:* If training accuracy is 99% and validation accuracy is 65%, what is the model experiencing?
- *Answer:* Overfitting (high variance). The model has memorized training noise instead of learning generalizable patterns.
