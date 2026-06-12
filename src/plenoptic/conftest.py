"""
Configuration for pytest to apply to all doctests.

Realized this was the solution when reading docs about doctest_namespace fixture
(https://docs.pytest.org/en/stable/how-to/doctest.html#doctest-namespace), which says
that: "Note that like the normal conftest.py, the fixtures are discovered in the
directory tree conftest is in. Meaning that if you put your doctest with your source
code, the relevant conftest.py needs to be in the same directory tree. Fixtures will not
be discovered in a sibling directory tree!" Thus, this has to live here in order to
affect doctests.
"""

import matplotlib.pyplot as plt
import pytest
import timm
from torchvision import models


# following https://github.com/scverse/scanpy/issues/1662
@pytest.fixture(autouse=True)
def close_figures_on_teardown():
    """
    Close figures when exiting from doctests.

    Note that this won't close between tests in a given docstring, but it will close
    between docstrings.
    """  # numpydoc ignore=YD01
    yield
    plt.close("all")


@pytest.fixture(autouse=True, scope="session")
def download_torchvision():
    """
    Pre-download torchvision models for use in doctests.

    Whether torchvision outputs a message about "Downloading" depends on whether it
    already exists in the cache (so, probably for local development and never for CI)
    and where that message goes depends on the torch version (stderr for torch<2.7,
    stdout after). This makes writing the doctests so they're successful across contexts
    difficult. Thus, this fixture makes sure they exist in the cache before anything
    else happens.
    """
    models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1, progress=False)
    models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1, progress=False)


@pytest.fixture(autouse=True, scope="session")
def download_timm():
    """
    Pre-download timm models for use in doctests.

    Similar potential problem to torchvision.
    """
    timm.create_model("timm/resnet50.tv_in1k", pretrained=True)
