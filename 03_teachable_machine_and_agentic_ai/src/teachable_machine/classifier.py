"""Core classifier engine for running Teachable Machine models."""

import os
from dataclasses import dataclass
from typing import Dict, Union

import numpy as np
from PIL import Image, ImageOps

# Import keras/tensorflow lazily or normally.
# Since tensorflow takes a while to load, we can import it when the class is initialized
# or keep standard import. Standard import is fine as this library is focused on inference.
try:
    from keras.models import load_model
except ImportError:
    try:
        from tensorflow.keras.models import load_model  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Keras/TensorFlow is required to use the teachable_machine library. "
            "Install it via 'pip install tensorflow'."
        ) from e

from teachable_machine.exceptions import (
    ImagePreprocessingError,
    ModelLoadError,
)
from teachable_machine.utils import load_labels


@dataclass
class PredictionResult:
    """Dataclass holding structured prediction results."""

    class_index: int
    class_name: str
    confidence: float
    probabilities: Dict[str, float]

    def __str__(self) -> str:
        return f"PredictionResult(class_name='{self.class_name}', confidence={self.confidence:.4f})"


class ImageClassifier:
    """Class to load and execute Teachable Machine Keras models for image classification."""

    def __init__(self, model_path: Union[str, os.PathLike], label_path: Union[str, os.PathLike]):
        """Initializes the classifier with the specified model and labels.

        Args:
            model_path: Path to the Keras .h5 model file.
            label_path: Path to the labels.txt file.

        Raises:
            ModelLoadError: If the model file cannot be loaded.
            LabelLoadError: If the labels file cannot be parsed.
        """
        self.model_path = os.path.abspath(model_path)
        self.label_path = os.path.abspath(label_path)

        # Load labels
        self.labels = load_labels(self.label_path)

        # Load keras model
        if not os.path.exists(self.model_path):
            raise ModelLoadError(f"Model file not found at: {self.model_path}")

        try:
            # compile=False avoids trying to load optimizer/loss states which aren't needed for inference
            self.model = load_model(self.model_path, compile=False)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load Keras model at {self.model_path}. "
                f"It may be corrupt or require patching for Keras 3 compatibility. "
                f"Error: {e}"
            ) from e

        # Resolve expected input shape from model
        try:
            # model.input_shape is usually (None, height, width, channels)
            shape = self.model.input_shape
            self.target_height = shape[1] if shape[1] is not None else 224
            self.target_width = shape[2] if shape[2] is not None else 224
            self.channels = shape[3] if shape[3] is not None else 3
        except Exception:
            # Fallback to standard MobileNet shape
            self.target_height = 224
            self.target_width = 224
            self.channels = 3

    def preprocess_image(self, image_source: Union[str, os.PathLike, Image.Image]) -> np.ndarray:
        """Loads, resizes, crops, and normalizes an image to match model input requirements.

        Args:
            image_source: Path to the image file or a PIL Image object.

        Returns:
            A normalized numpy array of shape (1, height, width, channels).

        Raises:
            ImagePreprocessingError: If image loading or processing fails.
        """
        try:
            # 1. Load image if path is given
            if isinstance(image_source, (str, os.PathLike)):
                if not os.path.exists(image_source):
                    raise FileNotFoundError(f"Image file not found: {image_source}")
                pil_image = Image.open(image_source)
            elif isinstance(image_source, Image.Image):
                pil_image = image_source
            else:
                raise TypeError("image_source must be a path or a PIL Image object.")

            # 2. Ensure RGB mode (handles PNG alpha channel or grayscale)
            pil_image = pil_image.convert("RGB")

            # 3. Resize and crop from center (matches Teachable Machine logic)
            target_size = (self.target_width, self.target_height)
            resized_image = ImageOps.fit(pil_image, target_size, Image.Resampling.LANCZOS)

            # 4. Convert to float32 numpy array
            image_array = np.asarray(resized_image, dtype=np.float32)

            # 5. Normalize from [0, 255] to [-1.0, 1.0]
            normalized = (image_array / 127.5) - 1.0

            # 6. Expand dimensions to shape (1, height, width, channels)
            batch_data = np.expand_dims(normalized, axis=0)

            return batch_data

        except Exception as e:
            raise ImagePreprocessingError(f"Image preprocessing failed: {e}") from e

    def predict(self, image_source: Union[str, os.PathLike, Image.Image]) -> PredictionResult:
        """Executes model inference on the provided image and returns structured predictions.

        Args:
            image_source: Path to the image file or a PIL Image object.

        Returns:
            A PredictionResult containing classification details and class probabilities.

        Raises:
            ImagePreprocessingError: If image preprocessing fails.
            TeachableException: If prediction execution fails.
        """
        # Preprocess
        input_data = self.preprocess_image(image_source)

        try:
            # Run inference quietly without Keras progress bar (verbose=0)
            predictions = self.model.predict(input_data, verbose=0)
        except Exception as e:
            raise RuntimeError(f"Prediction failed during model execution: {e}") from e

        # Extract prediction details
        # predictions shape is (1, num_classes)
        prediction_vector = predictions[0]
        best_index = int(np.argmax(prediction_vector))
        best_confidence = float(prediction_vector[best_index])
        best_class = self.labels.get(best_index, f"Class_{best_index}")

        # Construct probabilities dictionary
        probabilities: Dict[str, float] = {}
        for idx, score in enumerate(prediction_vector):
            class_name = self.labels.get(idx, f"Class_{idx}")
            probabilities[class_name] = float(score)

        return PredictionResult(
            class_index=best_index,
            class_name=best_class,
            confidence=best_confidence,
            probabilities=probabilities,
        )
