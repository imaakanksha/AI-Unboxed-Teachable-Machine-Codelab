#!/usr/bin/env python
"""Simple demonstration script for Teachable Machine model inference.

This script uses the `teachable_machine` package under the hood to perform
image loading, preprocessing, and classification. It maintains strict backwards
compatibility with the original codelab outputs.
"""

import sys
from teachable_machine import ImageClassifier, TeachableException


def main():
    # File path configurations
    model_path = "model/keras_model.h5"
    labels_path = "model/labels.txt"
    image_path = "test_images/test.jpg"

    try:
        # Initialize the professional classifier engine
        classifier = ImageClassifier(model_path, labels_path)

        # Run inference
        result = classifier.predict(image_path)

        # Print output format matching the original script
        print("Class:", result.class_name)
        print("Confidence Score:", result.confidence)

    except TeachableException as e:
        print(f"Inference execution failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected system error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
