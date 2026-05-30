"""Metadata patcher utility for Keras models exported from Teachable Machine."""

import json
import os
import shutil
from typing import Any, Union


class ModelPatcher:
    """Utility class to patch legacy Teachable Machine Keras models to make them Keras 3 compatible."""

    @staticmethod
    def patch(file_path: Union[str, os.PathLike], create_backup: bool = True) -> bool:
        """Patches H5 model metadata for Keras 3 compatibility.

        Legacy Teachable Machine exports include 'groups' in DepthwiseConv2D configs,
        which Keras 3 rejects, and nested output_layers formatting that crashes loader.

        Args:
            file_path: Path to the Keras .h5 model file.
            create_backup: If True, creates a .bak file before modifying the model.

        Returns:
            True if patching was applied, False if no changes were needed.

        Raises:
            FileNotFoundError: If the model file does not exist.
            ValueError: If the file is not a valid H5 or does not contain model configuration.
        """
        import h5py

        file_path_str = os.path.abspath(file_path)
        if not os.path.exists(file_path_str):
            raise FileNotFoundError(f"Model file not found at: {file_path_str}")

        # Optional backup
        if create_backup:
            backup_path = f"{file_path_str}.bak"
            shutil.copy2(file_path_str, backup_path)

        modified = False
        try:
            with h5py.File(file_path_str, "r+") as f:
                if "model_config" not in f.attrs:
                    raise ValueError(
                        f"H5 file {file_path_str} is missing 'model_config' attribute. "
                        f"It might not be a valid Keras model file."
                    )

                # Decode config
                config_raw = f.attrs["model_config"]
                if isinstance(config_raw, bytes):
                    config_str = config_raw.decode("utf-8")
                else:
                    config_str = config_raw

                config = json.loads(config_str)

                # Track changes to see if we actually modified anything
                original_str = json.dumps(config, sort_keys=True)

                # 1. Clean DepthwiseConv2D groups parameter
                ModelPatcher._clean_depthwise_conv2d(config)

                # 2. Clean Doubly-Nested output_layers
                ModelPatcher._clean_output_layers(config)

                updated_str = json.dumps(config, sort_keys=True)

                if original_str != updated_str:
                    f.attrs["model_config"] = json.dumps(config)
                    modified = True

        except Exception as e:
            # Restore backup if failed
            if create_backup and os.path.exists(f"{file_path_str}.bak"):
                shutil.copy2(f"{file_path_str}.bak", file_path_str)
            raise ValueError(f"Failed to patch model metadata: {e}") from e

        return modified

    @staticmethod
    def _clean_depthwise_conv2d(config_dict: Any) -> None:
        """Recursively traverses the config dictionary and removes the 'groups' parameter

        from DepthwiseConv2D layer configurations.
        """
        if isinstance(config_dict, dict):
            if (
                config_dict.get("class_name") == "DepthwiseConv2D"
                and "groups" in config_dict.get("config", {})
            ):
                config_dict["config"].pop("groups", None)
            for val in config_dict.values():
                ModelPatcher._clean_depthwise_conv2d(val)
        elif isinstance(config_dict, list):
            for item in config_dict:
                ModelPatcher._clean_depthwise_conv2d(item)

    @staticmethod
    def _clean_output_layers(config_dict: Any) -> None:
        """Resolves nested output_layers formatting that crashes Keras 3 loader."""
        if not isinstance(config_dict, dict):
            return

        # Check in sequential/functional model configs
        config_body = config_dict.get("config")
        if isinstance(config_body, dict) and "output_layers" in config_body:
            out_layers = config_body["output_layers"]
            # Doubly nested: [[["layer_name", 0, 0]]] -> needs to be [["layer_name", 0, 0]]
            if (
                isinstance(out_layers, list)
                and len(out_layers) == 1
                and isinstance(out_layers[0], list)
                and len(out_layers[0]) == 1
                and isinstance(out_layers[0][0], list)
            ):
                config_body["output_layers"] = out_layers[0]
