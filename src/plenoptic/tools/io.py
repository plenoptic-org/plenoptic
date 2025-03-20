#!/usr/bin/env python3

import importlib

import torch


def examine_saved_synthesis(file_path: str, map_location: str | None = None):
    """Examine saved synthesis object.

    This is used for debugging, it will print out information about the versions used,
    names of the callable attributes, shapes of tensor attributes, etc.

    Parameters
    ----------
    file_path :
        The path to load the synthesis object from
    map_location :
        map_location argument to pass to ``torch.load``. If you save
        stuff that was being run on a GPU and are loading onto a
        CPU, you'll need this to make sure everything lines up
        properly. This should be structured like the str you would
        pass to ``torch.device``
    """
    load_dict = torch.load(file_path, map_location=map_location, weights_only=True)
    metadata = load_dict.pop("save_metadata")
    print("Metadata\n--------")
    print(
        f"plenoptic version : {metadata['plenoptic_version']} "
        f"(installed: {importlib.metadata.version('plenoptic')})"
    )
    print(
        f"torch version     : {metadata['torch_version']} "
        f"(installed: {importlib.metadata.version('torch')})"
    )
    print(f"Saved object type : {metadata['synthesis_object']}")
    print("\nCallables attributes\n--------------------")
    callables = [
        (k, v)
        for k, v in load_dict.items()
        if isinstance(v, tuple) and (isinstance(v[0], str) or v[0] is None)
    ]
    pad_len = max([len(k[1:] if k.startswith("_") else k) for k, v in callables]) + 1
    for k, v in callables:
        display_k = k[1:] if k.startswith("_") else k
        load_dict.pop(k)
        # then this is state_dict attribute
        if len(v) == 2:
            print(f"{display_k:<{pad_len}}: {v[0]}")
        # then this is an io attribute
        else:
            print(
                f"{display_k:<{pad_len}}: {v[0]}, "
                f"{[load_dict[t].shape for t in v[1]]} -> {v[2].shape}"
            )
    print("\nTensor attributes\n-----------------")
    tensors = [(k, v) for k, v in load_dict.items() if isinstance(v, torch.Tensor)]
    pad_len = max([len(k[1:] if k.startswith("_") else k) for k, v in tensors]) + 1
    for k, v in tensors:
        display_k = k[1:] if k.startswith("_") else k
        load_dict.pop(k)
        print(f"{display_k:<{pad_len}}: {v.dtype}, shape {v.shape}")
    print("\nOther attributes\n----------------")
    pad_len = max([len(k[1:] if k.startswith("_") else k) for k in load_dict]) + 1
    for k, v in load_dict.items():
        display_k = k[1:] if k.startswith("_") else k
        if hasattr(v, "__len__") and not isinstance(v, str):
            print(f"{display_k:<{pad_len}}: {type(v)} with length {len(v)}")
        else:
            print(f"{display_k:<{pad_len}}: {v}")
