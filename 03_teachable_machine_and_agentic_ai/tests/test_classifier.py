import os
import numpy as np
import pytest
from teachable_machine import ImageClassifier, PredictionResult
from teachable_machine.exceptions import ModelLoadError, LabelLoadError

# Setup paths relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "keras_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.txt")
TEST_IMAGE_PATH = os.path.join(BASE_DIR, "test_images", "test.jpg")


def test_classifier_initialization():
    classifier = ImageClassifier(MODEL_PATH, LABELS_PATH)
    assert classifier.labels[0] == "Laptop"
    assert classifier.labels[1] == "Mobile"
    assert classifier.target_height == 224
    assert classifier.target_width == 224


def test_classifier_nonexistent_model():
    with pytest.raises(ModelLoadError):
        ImageClassifier("nonexistent_model.h5", LABELS_PATH)


def test_classifier_nonexistent_labels():
    with pytest.raises(LabelLoadError):
        ImageClassifier(MODEL_PATH, "nonexistent_labels.txt")


def test_image_preprocessing():
    classifier = ImageClassifier(MODEL_PATH, LABELS_PATH)
    processed = classifier.preprocess_image(TEST_IMAGE_PATH)
    assert isinstance(processed, np.ndarray)
    assert processed.shape == (1, 224, 224, 3)
    assert processed.min() >= -1.0
    assert processed.max() <= 1.0


def test_prediction():
    classifier = ImageClassifier(MODEL_PATH, LABELS_PATH)
    result = classifier.predict(TEST_IMAGE_PATH)
    assert isinstance(result, PredictionResult)
    assert result.class_name in ["Laptop", "Mobile"]
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.probabilities) == 2
    assert "Laptop" in result.probabilities
    assert "Mobile" in result.probabilities
