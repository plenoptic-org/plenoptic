from importlib import resources
from importlib.abc import Traversable
import requests
import os
import tarfile
import math
import tqdm
from typing import Union
import pathlib

from ..tools.data import load_images


# If you add anything here, remember to update the docstring in osf_download!
OSF_URL = {'plenoptic-test-files.tar.gz': 'q9kn8', 'ssim_images.tar.gz': 'j65tw',
           'ssim_analysis.mat': 'ndtc7', 'msssim_images.tar.gz': '5fuba', 'MAD_results.tar.gz': 'jwcsr',
           'portilla_simoncelli_matlab_test_vectors.tar.gz': 'qtn5y',
           'portilla_simoncelli_test_vectors.tar.gz': '8r2gq',
           'portilla_simoncelli_images.tar.gz':'eqr3t',
           'portilla_simoncelli_synthesize.npz': 'a7p9r',
           'portilla_simoncelli_synthesize_torch_v1.12.0.npz': 'gbv8e',
           'portilla_simoncelli_synthesize_gpu.npz': 'tn4y8',
           'portilla_simoncelli_scales.npz': 'xhwv3'}


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
    fhs = [file for file in resources.files('plenoptic.data').iterdir() if file.stem == item_name]
    assert len(fhs) == 1, f"Expected exactly one file for {item_name}, but found {len(fhs)}."
    return fhs[0]


def get(item_name: str):
    """
    Load an image based on the item name from the package's data resources.

    Parameters
    ----------
    item_name : str
        The name of the item to load, without specifying the file extension.

    Returns
    -------
    The loaded image object. The exact return type depends on the `load_images` function implementation.

    Notes
    -----
    This function first retrieves the full filename using `get_filename` and then loads the image
    using `load_images` from the `tools.data` module. It supports loading images as grayscale if
    they have a `.pgm` extension.
    """
    with get_path(item_name) as path:
        ext = path.suffix
        return load_images(str(path), as_gray=ext == ".pgm")


def download(url: str, destination: Union[str, pathlib.Path]) -> str:
    r"""Download file from url.

    Downloads file found at `url` to `destination`, extracts and deletes the
    the .tar.gz file (if applicable), and returns the path.

    Parameters
    ----------
    url :
        Url of the file to download
    destination :
        Where to download the file.

    Returns
    -------
    destination :
        The path to the downloaded directory or file.

    """
    destination = pathlib.Path(destination)
    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024*1024
    wrote = 0
    with open(destination, 'wb') as f:
        for data in tqdm.tqdm(r.iter_content(block_size), unit='MB',
                              unit_scale=True,
                              total=math.ceil(total_size//block_size)):
            wrote += len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        raise Exception(f"Error downloading {destination.name} from {url}!")
    if destination.name.endswith('.tar.gz'):
        with tarfile.open(destination) as f:
            f.extractall(os.path.dirname(destination))
        os.remove(destination)
    return str(destination).replace('.tar.gz', '')


def osf_download(filename: str, destination_dir: Union[str, pathlib.Path] = '.') -> str:
    r"""Download file from plenoptic OSF page.

    From the OSF project at https://osf.io/ts37w/.

    Downloads the specified file, extracts and deletes the the .tar.gz file (if
    applicable), and returns the path.

    Parameters
    ----------
    filename : {'plenoptic-test-files.tar.gz', 'ssim_images.tar.gz',
                'ssim_analysis.mat', 'msssim_images.tar.gz',
                'MAD_results.tar.gz',
                'portilla_simoncelli_images.tar.gz',
                'portilla_simoncelli_matlab_test_vectors.tar.gz',
                'portilla_simoncelli_test_vectors.tar.gz',
                'portilla_simoncelli_synthesize.npz',
                'portilla_simoncelli_synthesize_torch_v1.12.0.npz',
                'portilla_simoncelli_synthesize_gpu.npz'}
        Which file to download.
    destination_dir :
        Directory to download the file into.

    Returns
    -------
    destination :
        The path to the downloaded directory or file.

    """
    destination = pathlib.Path(destination_dir) / pathlib.Path(filename)
    non_tar_destination = pathlib.Path(destination_dir) / pathlib.Path(filename.replace('.tar.gz', ''))
    if not os.path.exists(non_tar_destination):
        print(f"{non_tar_destination} not found, downloading now...")
        url = f"https://osf.io/{OSF_URL[filename]}/download"
        return download(url, destination)
    else:
        return str(non_tar_destination)
