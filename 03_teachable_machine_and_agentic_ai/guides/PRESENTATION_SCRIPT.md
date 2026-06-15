# Presentation Script: Workshop 3 🎤

This script provides minute-by-minute speaking guidelines, live-coding cues, and slides outline for presenting Workshop 3.

---

## ⏱️ Section 1: Introduction (0 - 10 min)

### Slide 1: Welcome to Workshop 3
*   **Visual:** Workshop title, badges, links.
*   **Talk Track:**
    > "Welcome back, everyone! In Workshops 1 and 2, we covered the theory of optimization, NLP, and RAG. Today, we're transitioning to software engineering and agentic deployment. We're going to package a custom machine learning model, fix its legacy metadata, test it, and show how it bridges to the world of autonomous agents. Let's get started!"

---

## ⏱️ Section 2: Modular Packaging (10 - 30 min)

### Slide 2: Moving from Scripts to Packages
*   **Visual:** Directory comparison tree showing a flat script vs. clean package structure (`src/`, `tests/`, `pyproject.toml`).
*   **Talk Track:**
    > "When you prototype in notebooks, it's easy to keep all files in one folder. But to deploy this in a production microservice or share it with other developers, we must package it. Notice our project structure. We keep our core code in `src/teachable_machine/` and register an executable entrypoint called `teachable-tm` inside `pyproject.toml` so users can run it directly from their command line."

---

## ⏱️ Section 3: Live Demo: The Keras 3 Patch (30 - 50 min)

### Action: Run Terminal
*   **Visual:** Open VS Code terminal.
*   **Talk Track:**
    > "If I try to load this legacy model in a modern environment with TensorFlow 2.15+, it crashes. It throws a `TypeError: DepthwiseConv2D got unexpected keyword argument 'groups'`. Watch as I run our patching tool..."
*   **Demo Command:**
    ```bash
    teachable-tm patch model/keras_model.h5
    ```
*   **Talk Track:**
    > "The patcher went in, opened the HDF5 binary structure, removed the deprecated 'groups' configuration, and saved it back. Now, if I run `python model_test.py`... look! It classifies the image successfully! No retraining required."

---

## ⏱️ Section 4: Dataset Evaluation (50 - 70 min)

### Slide 3: Metrics That Matter
*   **Visual:** Classification performance table (Precision, Recall, F1-Score) and a Confusion Matrix grid.
*   **Talk Track:**
    > "How do we know if our model is good? We run a verification pipeline over a test dataset folder. Let's run the evaluate command. It outputs a markdown report showcasing our precision, recall, and a confusion matrix indicating where our laptop vs. mobile model is making mistakes. This is the first step toward validating a model for production."

---

## ⏱️ Section 5: Wrap-up & Q&A (70 - 90 min)

### Slide 4: Summary & Next Steps
*   **Visual:** Checklist of what was accomplished (packaged tool, patched compatibility, evaluated model).
*   **Talk Track:**
    > "That wraps up our live presentation. You now have a complete, production-grade image classification pipeline. Thank you for your time, and I am happy to open the floor to any questions!"
