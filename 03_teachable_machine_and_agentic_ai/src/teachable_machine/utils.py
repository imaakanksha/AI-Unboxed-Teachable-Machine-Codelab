"""Utility functions for the teachable_machine package."""

import os
from typing import Dict, List, Union
from teachable_machine.exceptions import LabelLoadError


def load_labels(source: Union[str, os.PathLike, List[str]]) -> Dict[int, str]:
    """Loads and parses class labels.

    Supports:
    - Path to a text file containing labels (one per line).
    - List of label strings.

    Parses lines in the format "index ClassName" (e.g., "0 Laptop") or
    plain "ClassName" (where sequential 0-based indices are assigned).

    Args:
        source: File path (str or PathLike) or list of strings.

    Returns:
        A dictionary mapping integer indices to class name strings.

    Raises:
        LabelLoadError: If the source cannot be loaded or is invalid.
    """
    lines: List[str] = []

    if isinstance(source, (str, os.PathLike)):
        if not os.path.exists(source):
            raise LabelLoadError(f"Label file not found at: {source}")
        try:
            with open(source, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
        except Exception as e:
            raise LabelLoadError(f"Failed to read label file {source}: {e}") from e
    elif isinstance(source, list):
        lines = [line.strip() for line in source if line.strip()]
    else:
        raise LabelLoadError("Label source must be a file path or a list of strings.")

    if not lines:
        raise LabelLoadError("Label source contains no valid classes.")

    labels: Dict[int, str] = {}
    for fallback_idx, line in enumerate(lines):
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            idx = int(parts[0])
            name = parts[1]
            labels[idx] = name
        else:
            # Fallback if no digit prefix exists
            labels[fallback_idx] = line

    return labels
