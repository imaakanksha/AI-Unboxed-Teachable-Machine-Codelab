import json
import os
import tempfile
import h5py
import pytest
from teachable_machine.patcher import ModelPatcher


def test_patcher_logic():
    # Create a temporary H5 file
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_name = tmp.name

    try:
        # Create a fake model config containing Keras 2 style metadata
        mock_config = {
            "class_name": "Sequential",
            "config": {
                "layers": [
                    {
                        "class_name": "DepthwiseConv2D",
                        "config": {
                            "name": "depthwise_conv",
                            "groups": 1,  # Legacy parameter that must be removed
                            "kernel_size": [3, 3],
                        },
                    }
                ],
                "output_layers": [[["dense", 0, 0]]],  # Legacy doubly nested list
            },
        }

        # Write to H5 file
        with h5py.File(tmp_name, "w") as f:
            f.attrs["model_config"] = json.dumps(mock_config)

        # Patch the file
        modified = ModelPatcher.patch(tmp_name, create_backup=False)
        assert modified is True

        # Verify the modifications
        with h5py.File(tmp_name, "r") as f:
            patched_config = json.loads(f.attrs["model_config"])

        # Verify 'groups' parameter was removed
        layers = patched_config["config"]["layers"]
        assert "groups" not in layers[0]["config"]

        # Verify output_layers was flattened
        output_layers = patched_config["config"]["output_layers"]
        assert output_layers == [["dense", 0, 0]]

    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
