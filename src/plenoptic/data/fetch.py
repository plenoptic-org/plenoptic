#!/usr/bin/env python3
"""Fetch data using pooch.

This is inspired by scipy's datasets module.
"""

REGISTRY = {
    'plenoptic-test-files.tar.gz': 'a6b8e03ecc8d7e40c505c88e6c767af5da670478d3bebb4e13a9d08ee4f39ae8',
    'ssim_images.tar.gz': '19c1955921a3c37d30c88724fd5a13bdbc9620c9e7dfaeaa3ff835283d2bb42e',
    'ssim_analysis.mat': '921d324783f06d1a2e6f1ce154c7ba9204f91c569772936991311ff299597f24',
    'msssim_images.tar.gz': 'a01273c95c231ba9e860dfc48f2ac8044ac3db13ad7061739c29ea5f9f20382c',
    'MAD_results.tar.gz': '29794ed7dc14626f115b9e4173bff88884cb356378a1d4f1f6cd940dd5b31dbe',
    'portilla_simoncelli_matlab_test_vectors.tar.gz': '83087d4d9808a3935b8eb4197624bbae19007189cd0d786527084c98b0b0ab81',
    'portilla_simoncelli_test_vectors.tar.gz': 'd67787620a0cf13addbe4588ec05b276105ff1fad46e72f8c58d99f184386dfb',
    'portilla_simoncelli_images.tar.gz': '4d3228fbb51de45b4fc81eba590d20f5861a54d9e46766c8431ab08326e80827',
    'portilla_simoncelli_synthesize.npz': '9c304580cd60e0275a2ef663187eccb71f2e6a31883c88acf4c6a699f4854c80',
    'portilla_simoncelli_synthesize_torch_v1.12.0.npz': '5a76ef223bac641c9d48a0b7f49b3ce0a05c12a48e96cd309866b1e7d5e4473f',
    'portilla_simoncelli_synthesize_gpu.npz': '324efc2a6c54382aae414d361c099394227b56cd24460eebab2532f70728c3ee',
    'portilla_simoncelli_scales.npz': 'eae2db6bd5db7d37c28d8f8320c4dd4fa5ab38294f5be22f8cf69e5cd5e4936a',
    'sample_images.tar.gz': '0ba6fe668a61e9f3cb52032da740fbcf32399ffcc142ddb14380a8e404409bf5',
    'test_images.tar.gz': 'eaf35f5f6136e2d51e513f00202a11188a85cae8c6f44141fb9666de25ae9554',
    'tid2013.tar.gz': 'bc486ac749b6cfca8dc5f5340b04b9bb01ab24149a5f3a712f13e9d0489dcde0',
}

OSF_TEMPLATE = "https://osf.io/{}/download"
# these are all from the OSF project at https://osf.io/ts37w/.
REGISTRY_URLS = {
    'plenoptic-test-files.tar.gz': OSF_TEMPLATE.format('q9kn8'),
    'ssim_images.tar.gz': OSF_TEMPLATE.format('j65tw'),
    'ssim_analysis.mat': OSF_TEMPLATE.format('ndtc7'),
    'msssim_images.tar.gz': OSF_TEMPLATE.format('5fuba'),
    'MAD_results.tar.gz': OSF_TEMPLATE.format('jwcsr'),
    'portilla_simoncelli_matlab_test_vectors.tar.gz': OSF_TEMPLATE.format('qtn5y'),
    'portilla_simoncelli_test_vectors.tar.gz': OSF_TEMPLATE.format('8r2gq'),
    'portilla_simoncelli_images.tar.gz': OSF_TEMPLATE.format('eqr3t'),
    'portilla_simoncelli_synthesize.npz': OSF_TEMPLATE.format('a7p9r'),
    'portilla_simoncelli_synthesize_torch_v1.12.0.npz': OSF_TEMPLATE.format('gbv8e'),
    'portilla_simoncelli_synthesize_gpu.npz': OSF_TEMPLATE.format('tn4y8'),
    'portilla_simoncelli_scales.npz': OSF_TEMPLATE.format('xhwv3'),
    'sample_images.tar.gz': OSF_TEMPLATE.format('6drmy'),
    'test_images.tar.gz': OSF_TEMPLATE.format('au3b8'),
    'tid2013.tar.gz': OSF_TEMPLATE.format('uscgv'),
}
DOWNLOADABLE_FILES = list(REGISTRY.keys())

import pathlib
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
        path=pooch.os_cache('plenoptic'),
        base_url="",
        urls=REGISTRY_URLS,
        registry=REGISTRY,
    )

def fetch_data(dataset_name: str) -> pathlib.Path:
    """Download data, using pooch. These are largely used for testing.

    To view list of downloadable files, look at `DOWNLOADABLE_FILES`.

    This checks whether the data already exists and is unchanged and downloads
    again, if necessary. If dataset_name ends in .tar.gz, this also
    decompresses and extracts the archive, returning the Path to the resulting
    directory. Else, it just returns the Path to the downloaded file.

    """
    if retriever is None:
        raise ImportError("Missing optional dependency 'pooch'."
                          " Please use pip or "
                          "conda to install 'pooch'.")
    if dataset_name.endswith('.tar.gz'):
        processor = pooch.Untar()
    else:
        processor = None
    fname = retriever.fetch(dataset_name,
                            progressbar=True,
                            processor=processor)
    if dataset_name.endswith('.tar.gz'):
        fname = pathlib.Path(fname[0]).parent
    else:
        fname = pathlib.Path(fname)
    return fname
