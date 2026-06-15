"""Dataset evaluation module for calculating classification metrics."""

import os
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from tabulate import tabulate

from teachable_machine.classifier import ImageClassifier


@dataclass
class ClassMetrics:
    """Dataclass holding precision, recall, and f1-score for a single class."""

    precision: float
    recall: float
    f1_score: float
    support: int


@dataclass
class EvaluationReport:
    """Dataclass holding aggregate and class-level evaluation results."""

    accuracy: float
    class_metrics: Dict[str, ClassMetrics]
    confusion_matrix: Dict[str, Dict[str, int]]
    total_processed: int
    total_failed: int
    all_classes: List[str]

    def render_text(self) -> str:
        """Renders the evaluation report as a clean plain-text string."""
        lines = []
        lines.append("=" * 60)
        lines.append("              DATASET EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Accuracy: {self.accuracy * 100:.2f}%")
        lines.append(f"Total Processed: {self.total_processed}")
        lines.append(f"Total Failures:  {self.total_failed}")
        lines.append("")

        # Render Class-level Metrics Table
        table_data = []
        for name in self.all_classes:
            metrics = self.class_metrics[name]
            table_data.append(
                [
                    name,
                    f"{metrics.precision:.4f}",
                    f"{metrics.recall:.4f}",
                    f"{metrics.f1_score:.4f}",
                    metrics.support,
                ]
            )

        headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
        lines.append("Class Performance:")
        lines.append(tabulate(table_data, headers=headers, tablefmt="simple"))
        lines.append("")

        # Render Confusion Matrix
        lines.append("Confusion Matrix (Row: Actual, Column: Predicted):")
        matrix_headers = [""] + self.all_classes
        matrix_rows = []
        for actual in self.all_classes:
            row = [actual]
            for predicted in self.all_classes:
                row.append(self.confusion_matrix[actual].get(predicted, 0))
            matrix_rows.append(row)

        lines.append(tabulate(matrix_rows, headers=matrix_headers, tablefmt="simple"))
        lines.append("=" * 60)
        return "\n".join(lines)

    def render_markdown(self) -> str:
        """Renders the evaluation report as a clean Markdown string."""
        lines = []
        lines.append("# Dataset Evaluation Report")
        lines.append("")
        lines.append(f"- **Accuracy:** {self.accuracy * 100:.2f}%")
        lines.append(f"- **Total Processed:** {self.total_processed}")
        lines.append(f"- **Failed Images:** {self.total_failed}")
        lines.append("")
        lines.append("## Class Performance")
        lines.append("")

        table_data = []
        for name in self.all_classes:
            metrics = self.class_metrics[name]
            table_data.append(
                [
                    name,
                    f"{metrics.precision:.4f}",
                    f"{metrics.recall:.4f}",
                    f"{metrics.f1_score:.4f}",
                    metrics.support,
                ]
            )
        headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
        lines.append(tabulate(table_data, headers=headers, tablefmt="github"))
        lines.append("")
        lines.append("## Confusion Matrix")
        lines.append("")
        lines.append("> **Row:** Actual class, **Column:** Predicted class.")
        lines.append("")

        matrix_headers = ["Actual \\ Predicted"] + self.all_classes
        matrix_rows = []
        for actual in self.all_classes:
            row = [actual]
            for predicted in self.all_classes:
                row.append(self.confusion_matrix[actual].get(predicted, 0))
            matrix_rows.append(row)

        lines.append(tabulate(matrix_rows, headers=matrix_headers, tablefmt="github"))
        return "\n".join(lines)


class DatasetEvaluator:
    """Evaluates an ImageClassifier against a structured dataset directory."""

    def __init__(self, classifier: ImageClassifier):
        """Initializes the evaluator with a pre-loaded ImageClassifier.

        Args:
            classifier: An instance of ImageClassifier.
        """
        self.classifier = classifier
        # Extract clean labels from the classifier
        self.classes = list(self.classifier.labels.values())

    def _resolve_ground_truth(self, dir_name: str) -> str:
        """Attempts to match a subdirectory name to the classifier's labels.

        Matches case-insensitively and looks for sub-string matches.
        """
        clean_dir = "".join(c for c in dir_name if c.isalnum()).lower()
        for label in self.classes:
            clean_label = "".join(c for c in label if c.isalnum()).lower()
            if (
                clean_dir == clean_label
                or clean_label in clean_dir
                or clean_dir in clean_label
            ):
                return label
        return dir_name

    def evaluate(self, dataset_path: str) -> EvaluationReport:
        """Evaluates the classifier against images in the given dataset directory.

        Expects the directory to contain subdirectories named after classes.

        Args:
            dataset_path: Path to the root dataset directory.

        Returns:
            An EvaluationReport containing accuracy, precision, recall, and confusion matrix.
        """
        abs_dataset_path = os.path.abspath(dataset_path)
        if not os.path.isdir(abs_dataset_path):
            raise NotADirectoryError(f"Dataset path is not a directory: {abs_dataset_path}")

        # Supported image extensions
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        # Initialize confusion matrix and tracking counters
        all_labels = set(self.classes)
        confusion_matrix: Dict[str, Dict[str, int]] = {}

        # Add classes to confusion matrix to handle unseen classes in evaluation dataset
        for actual in all_labels:
            confusion_matrix[actual] = {predicted: 0 for predicted in all_labels}

        correct_count = 0
        total_processed = 0
        total_failed = 0

        # Scan directory contents
        for item in os.listdir(abs_dataset_path):
            sub_dir = os.path.join(abs_dataset_path, item)
            if not os.path.isdir(sub_dir):
                continue

            actual_label = self._resolve_ground_truth(item)
            if actual_label not in all_labels:
                # Dynamically add unexpected actual labels if present
                all_labels.add(actual_label)
                confusion_matrix[actual_label] = {p: 0 for p in all_labels}
                for act in confusion_matrix:
                    confusion_matrix[act][actual_label] = 0

            # List and process images in this subdirectory
            for filename in os.listdir(sub_dir):
                ext = os.path.splitext(filename)[1].lower()
                if ext not in extensions:
                    continue

                img_path = os.path.join(sub_dir, filename)
                try:
                    res = self.classifier.predict(img_path)
                    pred_label = res.class_name

                    # Ensure prediction label is in the confusion matrix structure
                    if pred_label not in confusion_matrix[actual_label]:
                        # Handle case where classifier predicted a class not previously recorded
                        all_labels.add(pred_label)
                        for act in confusion_matrix:
                            confusion_matrix[act][pred_label] = 0
                        confusion_matrix[actual_label][pred_label] = 0

                    confusion_matrix[actual_label][pred_label] += 1
                    total_processed += 1

                    if pred_label == actual_label:
                        correct_count += 1

                except Exception:
                    total_failed += 1

        if total_processed == 0:
            raise ValueError(f"No valid image files found for evaluation under {abs_dataset_path}")

        # Calculate metrics for each class
        class_metrics: Dict[str, ClassMetrics] = {}
        sorted_classes = sorted(list(all_labels))

        for target in sorted_classes:
            # 1. Support (Actual samples for this class)
            support = sum(confusion_matrix[target].values())

            # 2. True Positives (TP)
            tp = confusion_matrix[target].get(target, 0)

            # 3. False Positives (FP): Sum of predictions of `target` when actual was not `target`
            fp = sum(
                confusion_matrix[act].get(target, 0)
                for act in sorted_classes
                if act != target
            )

            # 4. False Negatives (FN): Actual was `target` but prediction was not `target`
            fn = sum(
                val for pred, val in confusion_matrix[target].items() if pred != target
            )

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            class_metrics[target] = ClassMetrics(
                precision=precision, recall=recall, f1_score=f1_score, support=support
            )

        accuracy = correct_count / total_processed

        return EvaluationReport(
            accuracy=accuracy,
            class_metrics=class_metrics,
            confusion_matrix=confusion_matrix,
            total_processed=total_processed,
            total_failed=total_failed,
            all_classes=sorted_classes,
        )
