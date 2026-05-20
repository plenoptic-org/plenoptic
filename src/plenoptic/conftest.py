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
