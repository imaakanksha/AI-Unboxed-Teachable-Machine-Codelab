# Instructor Guide: Workshop 3 (Teachable Machine & Agentic AI) 🎓

This guide is designed to help you deliver an interactive 2-hour session that bridges local model deployment to advanced agentic architecture.

---

## ⏱️ Timeline & Agenda

| Time | Topic | Format | Instructor Focus |
|------|-------|--------|------------------|
| **00:00 - 00:10** | Welcome & Overview | Slide / Talk | Connect RAG principles to perception in agents. |
| **00:10 - 00:30** | Modular Python Packaging | Talk / Demo | Explain project layouts (`src/` structure, `pyproject.toml`, editable installs). |
| **00:30 - 00:50** | Solving Compatibility (Patching) | Live Coding | Show the Keras 3 groups error and how the patcher changes the model metadata binary representation. |
| **00:50 - 01:10** | Model Evaluation | Hands-on Demo | Run the dataset evaluator. Explain F1-Score and Confusion Matrix. |
| **01:10 - 01:30** | Connecting to Agent Loops | Talk / Conceptual | Explain how this classifier acts as a tool/perception component in a ReAct loop. |
| **01:30 - 01:50** | Interactive Exercises & Debug | Live Lab | Guide students through pytest, adding a test case, or resolving import errors. |
| **01:50 - 02:00** | Q&A & Wrap-up | Discussion | Final remarks, feedback collection. |

---

## 💡 Key Teaching Hooks

- **The perception Hook:** Ask students: *"How does an autonomous AI agent see?"* Introduce the idea that our image classifier acts as the "eyes" (perception layer) of an agent. The agent calls this script as a Tool.
- **The Binary Hack (Patcher):** Explain that we don't need to retrain legacy models when versions break. Instead, we can inspect their HDF5 metadata headers, identify offending layers (like DepthwiseConv2D group configs), and modify them directly using Python's `h5py` library.
- **The Evaluation Grid:** Highlight that "Accuracy" is a dangerous metric for imbalanced datasets. Use the F1-score to illustrate how a model that fails on one rare category can still report high accuracy.

---

## 🔧 Troubleshooting Common Errors

### 1. `ModuleNotFoundError: No module named 'teachable_machine'`
*   **Cause:** The package was not installed in editable mode, or the virtual environment is not activated.
*   **Fix:** Run `pip install -e .` from the root of this folder, and ensure your terminal prompt shows the `(.venv)` prefix.

### 2. `TypeError: DepthwiseConv2D got unexpected keyword argument 'groups'`
*   **Cause:** Running a model exported from Teachable Machine (compiled with Keras 2) in Keras 3.
*   **Fix:** Run `teachable-tm patch model/keras_model.h5` to rewrite the model parameters.
