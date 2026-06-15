"""Custom exception definitions for the teachable_machine package."""

class TeachableException(Exception):
    """Base exception class for all errors in the teachable_machine package."""
    pass


class ModelLoadError(TeachableException):
    """Raised when the Keras model file cannot be loaded or is corrupt."""
    pass


class LabelLoadError(TeachableException):
    """Raised when the labels file is missing, empty, or improperly formatted."""
    pass


class ImagePreprocessingError(TeachableException):
    """Raised when an input image cannot be loaded, resized, or normalized."""
    pass
