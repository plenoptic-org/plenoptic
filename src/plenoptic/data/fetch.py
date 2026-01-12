"""
Fetch data using pooch.

This is inspired by scipy's datasets module.
"""  # numpydoc ignore=EX01

import pathlib
import sys

from tqdm.auto import tqdm

__all__ = ["DOWNLOADABLE_FILES", "fetch_data"]


def __dir__() -> list[str]:
    return __all__


REGISTRY = {
    "plenoptic-test-files.tar.gz": "a6b8e03ecc8d7e40c505c88e6c767af5da670478d3bebb4e13a9d08ee4f39ae8",  # noqa: E501
    "ssim_images.tar.gz": "19c1955921a3c37d30c88724fd5a13bdbc9620c9e7dfaeaa3ff835283d2bb42e",  # noqa: E501
    "ssim_analysis.mat": "921d324783f06d1a2e6f1ce154c7ba9204f91c569772936991311ff299597f24",  # noqa: E501
    "msssim_images.tar.gz": "a01273c95c231ba9e860dfc48f2ac8044ac3db13ad7061739c29ea5f9f20382c",  # noqa: E501
    "MAD_results.tar.gz": "29794ed7dc14626f115b9e4173bff88884cb356378a1d4f1f6cd940dd5b31dbe",  # noqa: E501
    "portilla_simoncelli_matlab_test_vectors.tar.gz": "83087d4d9808a3935b8eb4197624bbae19007189cd0d786527084c98b0b0ab81",  # noqa: E501
    "portilla_simoncelli_test_vectors.tar.gz": "d67787620a0cf13addbe4588ec05b276105ff1fad46e72f8c58d99f184386dfb",  # noqa: E501
    "portilla_simoncelli_images.tar.gz": "4d3228fbb51de45b4fc81eba590d20f5861a54d9e46766c8431ab08326e80827",  # noqa: E501
    "portilla_simoncelli_synthesize.npz": "9c304580cd60e0275a2ef663187eccb71f2e6a31883c88acf4c6a699f4854c80",  # noqa: E501
    "portilla_simoncelli_synthesize_torch_v1.12.0.npz": "5a76ef223bac641c9d48a0b7f49b3ce0a05c12a48e96cd309866b1e7d5e4473f",  # noqa: E501
    "portilla_simoncelli_synthesize_gpu.npz": "324efc2a6c54382aae414d361c099394227b56cd24460eebab2532f70728c3ee",  # noqa: E501
    "portilla_simoncelli_scales.npz": "eae2db6bd5db7d37c28d8f8320c4dd4fa5ab38294f5be22f8cf69e5cd5e4936a",  # noqa: E501
    "sample_images.tar.gz": "0ba6fe668a61e9f3cb52032da740fbcf32399ffcc142ddb14380a8e404409bf5",  # noqa: E501
    "test_images.tar.gz": "eaf35f5f6136e2d51e513f00202a11188a85cae8c6f44141fb9666de25ae9554",  # noqa: E501
    "tid2013.tar.gz": "bc486ac749b6cfca8dc5f5340b04b9bb01ab24149a5f3a712f13e9d0489dcde0",  # noqa: E501
    "portilla_simoncelli_test_vectors_refactor.tar.gz": "b72661836e5830c1473b8a2292075a8e9c1aca00faf97cc6809ec28f19d3f9ce",  # noqa: E501
    "portilla_simoncelli_synthesize_torch_v1.12.0_ps-refactor.npz": "9525844b71cf81509b86ed9677172745353588c6bb54e4de8000d695598afa47",  # noqa: E501
    "portilla_simoncelli_synthesize_gpu_ps-refactor.npz": "9fbb490f1548133f6aa49c54832130cf70f8dc6546af59688ead17f62ab94e61",  # noqa: E501
    "portilla_simoncelli_scales_ps-refactor.npz": "ce11d85e6bcf5fad1b819c36dac584c3e933706a0ee423ea1c76ffe0daccbae5",  # noqa: E501
    "portilla_simoncelli_synthesize_torch_v1.12.0_ps-refactor-2.npz": "ffd967543d58a03df390008c35878791590520624aa0e5e5a26ad3f877345ab4",  # noqa: E501
    "example_eigendistortion.pt": "87080836713e8efe1e7ff29538099e82a26b8700080e1bc1d30f00de1a54b2f5",  # noqa: E501
    "load_image_test.tar.gz": "8a2b92dc0d442695c45b1e908ef0a04cae35c5f21b774a93b9fc6b675423b526",  # noqa: E501
    "berardino_onoff.pt": "2174a40005489b9c94acc91213b2f6d57a75f262caf118cb1980658eadbfd047",  # noqa: E501
    "berardino_vgg16.pt": "5e0d10f4a367244879cd8a61c453992370ab801db1f66e10caa1ee2ecfab8ca4",  # noqa: E501
    "ps_regression.tar.gz": "7520ed2dbcb2647ac814a02e436e1f1e41fb06ca2534561c83f9e76193a12108",  # noqa: E501
}

OSF_TEMPLATE = "https://osf.io/download/{}"
# these are all from the OSF project at https://osf.io/ts37w/.
REGISTRY_URLS = {
    "plenoptic-test-files.tar.gz": OSF_TEMPLATE.format("q9kn8"),
    "ssim_images.tar.gz": OSF_TEMPLATE.format("j65tw"),
    "ssim_analysis.mat": OSF_TEMPLATE.format("ndtc7"),
    "msssim_images.tar.gz": OSF_TEMPLATE.format("5fuba"),
    "MAD_results.tar.gz": OSF_TEMPLATE.format("jwcsr"),
    "portilla_simoncelli_matlab_test_vectors.tar.gz": OSF_TEMPLATE.format("qtn5y"),
    "portilla_simoncelli_test_vectors.tar.gz": OSF_TEMPLATE.format("8r2gq"),
    "portilla_simoncelli_images.tar.gz": OSF_TEMPLATE.format("eqr3t"),
    "portilla_simoncelli_synthesize.npz": OSF_TEMPLATE.format("a7p9r"),
    "portilla_simoncelli_synthesize_torch_v1.12.0.npz": OSF_TEMPLATE.format("gbv8e"),
    "portilla_simoncelli_synthesize_gpu.npz": OSF_TEMPLATE.format("tn4y8"),
    "portilla_simoncelli_scales.npz": OSF_TEMPLATE.format("xhwv3"),
    "sample_images.tar.gz": OSF_TEMPLATE.format("6drmy"),
    "test_images.tar.gz": OSF_TEMPLATE.format("au3b8"),
    "tid2013.tar.gz": OSF_TEMPLATE.format("uscgv"),
    "portilla_simoncelli_test_vectors_refactor.tar.gz": OSF_TEMPLATE.format("ca7qt"),
    "portilla_simoncelli_synthesize_torch_v1.12.0_ps-refactor.npz": OSF_TEMPLATE.format(
        "vmwzd"
    ),
    "portilla_simoncelli_synthesize_gpu_ps-refactor.npz": OSF_TEMPLATE.format("mqs6y"),
    "portilla_simoncelli_scales_ps-refactor.npz": OSF_TEMPLATE.format("nvpr4"),
    "portilla_simoncelli_synthesize_torch_v1.12.0_ps-refactor-2.npz": OSF_TEMPLATE.format(  # noqa: E501
        "en8du"
    ),
    "example_eigendistortion.pt": OSF_TEMPLATE.format("gwhz2"),
    "load_image_test.tar.gz": OSF_TEMPLATE.format("avpzq"),
    "berardino_onoff.pt": OSF_TEMPLATE.format("uqfa8"),
    "berardino_vgg16.pt": OSF_TEMPLATE.format("6r87b"),
    "ps_regression.tar.gz": OSF_TEMPLATE.format("7t4fj/?revision=11"),
}

#: List of files that can be downloaded using :func:`fetch_data`
DOWNLOADABLE_FILES = list(REGISTRY_URLS.keys())

try:
    import pooch
except ImportError:
    pooch = None
    retriever = None
else:
    retriever = pooch.create(
        # Use the default cache folder for the operating system
        # Pooch uses appdirs (https://github.com/ActiveState/appdirs) to
        # select an appropriate directory for the cache on each platform.
        path=pooch.os_cache("plenoptic"),
        base_url="",
        urls=REGISTRY_URLS,
        registry=REGISTRY,
        retry_if_failed=2,
        allow_updates="POOCH_ALLOW_UPDATES",
        env="PLENOPTIC_CACHE_DIR",
    )


def _find_shared_directory(paths: list[pathlib.Path]) -> pathlib.Path:
    """
    Find directory shared by all paths.

    Helper function for when downloading tar archives.

    Parameters
    ----------
    paths
        List of paths to check.

    Returns
    -------
    shared_dir
        Most recent common ancestor.
    """  # numpydoc ignore=EX01
    for dir in paths[0].parents:
        if all([dir in p.parents for p in paths]):
            break
    return dir


def fetch_data(dataset_name: str) -> pathlib.Path:
    """
    Download data, using pooch. These are largely used for testing.

    To view list of downloadable files, look at :const:`DOWNLOADABLE_FILES`.

    This checks whether the data already exists and is unchanged and downloads
    again, if necessary. If dataset_name ends in .tar.gz, this also
    decompresses and extracts the archive, returning the Path to the resulting
    directory. Else, it just returns the Path to the downloaded file.

    Parameters
    ----------
    dataset_name
        Name of the dataset to download.

    Returns
    -------
    path
        Path of the downloaded dataset.

    Raises
    ------
    ImportError
        If ``pooch`` isn't installed.

    Examples
    --------
    .. plot::

      >>> import plenoptic as po
      >>> from plenoptic.data import fetch
      >>> path = fetch.fetch_data("portilla_simoncelli_images.tar.gz")
      >>> len(list(path.glob("*")))
      38
      >>> img = po.load_images(path / "fig3b.jpg")
      >>> po.imshow(img)
      <PyrFigure size ...>
    """
    if retriever is None:
        raise ImportError(
            "Missing optional dependency 'pooch'."
            " Please use pip or "
            "conda to install 'pooch'."
        )
    processor = pooch.Untar() if dataset_name.endswith(".tar.gz") else None
    use_ascii = bool(sys.platform == "win32")
    fname = retriever.fetch(
        dataset_name,
        progressbar=tqdm(
            total=1,
            ncols=79,
            unit_scale=True,
            delay=1e-5,
            leave=True,
            unit="B",
            ascii=use_ascii,
        ),
        processor=processor,
    )
    if dataset_name.endswith(".tar.gz"):
        fname = _find_shared_directory([pathlib.Path(f) for f in fname])
    else:
        fname = pathlib.Path(fname)
    return fname
