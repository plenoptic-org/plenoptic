import pytest
from torch import Tensor

import plenoptic as po


@pytest.mark.parametrize("item_name", [img for img in dir(po.data)
                                       if img not in ['fetch_data', 'DOWNLOADABLE_FILES']])
def test_data_module(item_name):
    """Test that data module works."""
    assert isinstance(eval(f"po.data.{item_name}()"), Tensor)


@pytest.mark.parametrize(
    "item_name, img_shape",
    [
        ("color_wheel", (1, 3, 600, 600)),
        ("parrot", (1, 3, 254, 266)),
        ("curie", (1, 1, 256, 256)),
        ("einstein", (1, 1, 256, 256)),
        ("reptile_skin", (1, 1, 256, 256)),
    ]
)
def test_data_get_shape(item_name, img_shape):
    """Check if the shape of the retrieved image matches the expected dimensions."""
    img = eval(f"po.data.{item_name}()")
    assert all(shp == img_shape[i] for i, shp in enumerate(img.shape))
