from contextlib import nullcontext as does_not_raise

from importlib import resources
from importlib.abc import Traversable
import pytest
from torch import Tensor

import plenoptic as po


@pytest.mark.parametrize(
    "item_name, expectation",
    [
        ("color_wheel", does_not_raise()),
        ("xyz", pytest.raises(AssertionError, match="Expected exactly one file for xyz, but found 2")),
        ("xyzw", pytest.raises(AssertionError, match=f"Expected exactly one file for xyzw, but found 0"))
    ]
)
def test_data_get_path(item_name, expectation):
    """Test the retrieval of file paths with varying expectations."""
    fh = resources.files("plenoptic.data")
    # ensures that the files gets deleted, even if an exception gets hit in
    # the try block.
    try:
        with expectation:
            # Create files with specific extensions
            for ext in ["abc", "abcd"]:
                file_path = fh / f"xyz.{ext}"
                file_path.touch(exist_ok=True)
            po.data.data_utils.get_path(item_name)

    finally:
        # Delete the files created for the test
        for ext in ["abc", "abcd"]:
            path = fh / f"xyz.{ext}"
            if path.exists():
                path.unlink()  # This deletes the file


@pytest.mark.parametrize("item_name", ["color_wheel", "flowers", "curie"])
def test_data_get_path_type(item_name):
    """Test that the returned path object is an instance of Traversable."""
    assert isinstance(po.data.data_utils.get_path(item_name), Traversable)


@pytest.mark.parametrize(
    "item_name", ["color_wheel", "flowers", "curie"]
)
def test_data_get_type(item_name):
    """Test that the retrieved data is of type Tensor."""
    img = po.data.data_utils.get(item_name)
    assert isinstance(img, Tensor)


@pytest.mark.parametrize(
    "item_name, img_shape",
    [
        ("color_wheel", (1, 3, 600, 600)),
        ("flowers", (1, 3, 512, 512)),
        ("curie", (1, 1, 256, 256))
    ]
)
def test_data_get_shape(item_name, img_shape):
    """Check if the shape of the retrieved image matches the expected dimensions."""
    img = po.data.data_utils.get(item_name)
    assert all(shp == img_shape[i] for i, shp in enumerate(img.shape))
