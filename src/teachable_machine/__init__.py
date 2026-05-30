"""Teachable Machine Inference, Patching, and Evaluation Library.

A professional-grade, type-hinted package to load, patch, execute, and
evaluate Google Teachable Machine exported Keras models.
"""

__version__ = "0.1.0"

from teachable_machine.classifier import ImageClassifier, PredictionResult
from teachable_machine.evaluator import DatasetEvaluator
from teachable_machine.exceptions import (
    ImagePreprocessingError,
    LabelLoadError,
    ModelLoadError,
    TeachableException,
)
from teachable_machine.patcher import ModelPatcher

__all__ = [
    "ImageClassifier",
    "PredictionResult",
    "ModelPatcher",
    "DatasetEvaluator",
    "TeachableException",
    "ModelLoadError",
    "LabelLoadError",
    "ImagePreprocessingError",
]
