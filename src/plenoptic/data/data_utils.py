from importlib import resources
from importlib.abc import Traversable


from ..tools.data import load_images


def get_path(item_name: str) -> Traversable:
    """
    Retrieve the filename that matches the given item name with any extension.

    Parameters
    ----------
    item_name
        The name of the item to find the file for, without specifying the file extension.

    Returns
    -------
    :
        The filename matching the `item_name` with its extension.

    Raises
    ------
    AssertionError
        If no files or more than one file match the `item_name`.

    Notes
    -----
    This function uses glob to search for files in the current directory matching the `item_name`.
    It is assumed that there is only one file matching the name regardless of its extension.
    """
    fhs = [
        file
        for file in resources.files("plenoptic.data").iterdir()
        if file.stem == item_name
    ]
    assert (
        len(fhs) == 1
    ), f"Expected exactly one file for {item_name}, but found {len(fhs)}."
    return fhs[0]


def get(*item_names: str, as_gray: None | bool = None):
    """Load an image based on the item name from the package's data resources.

    Parameters
    ----------
    item_names :
        The names of the items to load, without specifying the file extension.
    as_gray :
        Whether to load in the image(s) as grayscale or not. If None, will make
        best guess based on file extension.

    Returns
    -------
    The loaded image object. The exact return type depends on the `load_images` function implementation.

    Notes
    -----
    This function first retrieves the full filename using `get_filename` and then loads the image
    using `load_images` from the `tools.data` module. It supports loading images as grayscale if
    they have a `.pgm` extension.

    """
    paths = [get_path(name) for name in item_names]
    if as_gray is None:
        as_gray = all(path.suffix == ".pgm" for path in paths)
    return load_images(paths, as_gray=as_gray)
