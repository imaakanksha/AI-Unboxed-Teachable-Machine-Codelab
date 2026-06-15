# Workshop 3: Teachable Machine & Agentic AI 🤖

> **Part of the AI Unboxed Workshop Series.** Take a local Teachable Machine model prototype, apply compatibility patches, deploy an inference engine, run comprehensive validation metrics, and bridge the classifier to autonomous AI agent patterns.

---

## 🎯 Learning Objectives

By the end of this workshop, you will be able to:
- 💡 Structure and package an inference application as a **distributable Python package**.
- 💡 Resolve **Keras 3 compatibility errors** in legacy model structures using binary metadata patching.
- 💡 Evaluate classification models using **Precision, Recall, F1-score, and Confusion Matrices**.
- 💡 Implement a **rich CLI tool** (`teachable-tm`) with status spinner animations.
- 💡 Trace how classifiers act as the **vision perception system** in agentic loops.

---

## 📁 Repository structure

```
03_teachable_machine_and_agentic_ai/
├── src/
│   └── teachable_machine/       # Core package code
│       ├── __init__.py          # Package entrypoint
│       ├── classifier.py        # Keras inference engine
│       ├── patcher.py           # Keras 2 -> Keras 3 compatibility patcher
│       ├── evaluator.py         # Precision/Recall & Confusion Matrix generator
│       ├── cli.py               # Rich CLI application Console
│       ├── exceptions.py        # Structured exceptions
│       └── utils.py             # Helper tools
├── tests/                       # Pytest test suite
├── model/                       # Exported model & label files
├── test_images/                 # Inference verification images
├── dataset/                     # Class evaluation folders (Laptop/Mobile)
├── pyproject.toml               # Python setup package metadata
└── model_test.py                # Backward-compatible check wrapper
```

---

## 🚀 Installation & Setup

Ensure you have Python 3.10+ installed.

### 1. Create a Virtual Environment
Using `uv` (Recommended for speed):
```bash
uv venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell
source .venv/bin/activate    # Linux/macOS
```
Using standard `venv`:
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell
```

### 2. Install Package in Editable Mode
```bash
# Editable install
pip install -e .

# Or with dev dependencies for running tests
pip install -e .[dev]
```

---

## ⚡ Quick Start & CLI Reference: `teachable-tm`

Once installed, the unified command-line tool `teachable-tm` is available in your shell.

### 1. Predict a Single Image
Run predictions with rich visual feedback:
```bash
teachable-tm predict test_images/test.jpg
```
Customize model and label paths if needed:
```bash
teachable-tm predict test_images/test.jpg --model model/keras_model.h5 --labels model/labels.txt
```

### 2. Apply Legacy Metadata Patch
Teachable Machine exports model formats built on older Keras configurations. In modern Keras 3 / TensorFlow setups, running them directly throws depthwise layer keyword arguments.

Patch your model file safely:
```bash
teachable-tm patch model/keras_model.h5
```
*Note: This automatically creates a backup at `keras_model.h5.bak`.*

### 3. Evaluate Dataset Performance
Generate metrics across a validation directory:
```bash
teachable-tm evaluate dataset/ --output report.md
```

---

## 🐍 Library API Reference

You can import the module directly into your projects:
```python
from teachable_machine import ImageClassifier, TeachableException

try:
    classifier = ImageClassifier(model_path="model/keras_model.h5", label_path="model/labels.txt")
    result = classifier.predict("test_images/test.jpg")
    print(f"Pred: {result.class_name} (Conf: {result.confidence:.4f})")
except TeachableException as e:
    print(f"Error: {e}")
```

---

## 🧪 Running Unit & Integration Tests

Ensure model loading, patching, and calculations function correctly:
```bash
pytest
```

---

## 👨‍🏫 Instructor Materials
Teaching this workshop? Check out the [Instructor Guide](./guides/INSTRUCTOR_GUIDE.md), [Presentation Script](./guides/PRESENTATION_SCRIPT.md), and [Student Resources](./guides/STUDENT_RESOURCES.md) in the `guides/` folder.
