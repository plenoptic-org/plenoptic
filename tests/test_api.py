#!/usr/bin/env python3

import inspect

import pytest

import plenoptic
from conftest import OLD_API
from plenoptic import _api_change

UPDATED_API = _api_change.API_CHANGE
UPDATED_API.update(_api_change.SYNTH_PLOT_FUNCS)
UPDATED_API.update(_api_change.PLOT_FUNCS)


def test_dunder_module():
    # test that all objects have __module__ that match the way they're called
    for mod in dir(plenoptic):
        obj = eval(f"plenoptic.{mod}")
        if inspect.isclass(obj) and obj.__module__ != "plenoptic":
            raise ValueError(
                f"{mod} module should be plenoptic but got {obj.__module__}"
            )
        if inspect.ismodule(obj):
            for mod2 in dir(obj):
                obj2 = eval(f"plenoptic.{mod}.{mod2}")
                if inspect.isclass(obj2) and obj2.__module__ != f"plenoptic.{mod}":
                    raise ValueError(
                        f"{mod2} module should be plenoptic.{mod} but got "
                        f"{obj2.__module__}"
                    )


def test_api_nesting():
    # that our API only has a single level of nesting
    for mod in dir(plenoptic):
        obj = eval(f"plenoptic.{mod}")
        if inspect.ismodule(obj):
            for mod2 in dir(obj):
                obj2 = eval(f"plenoptic.{mod}.{mod2}")
                if inspect.ismodule(obj2):
                    raise ValueError("Only should have one level of nesting!")


@pytest.mark.parametrize("old_func, new_func", UPDATED_API.items())
def test_api_change(old_func, new_func):
    # test that all none of the old ways work and that all the new ones do
    if "tools" not in old_func and any(
        [old_func.endswith(fname) for fname in ["imshow", "animshow", "pyrshow"]]
    ):
        match = r".*was moved in plenoptic"
    elif "fetch." in old_func or "metric." in old_func:
        match = r"No .* attribute .*"
    else:
        match = r".* not available from plenoptic.*"
    with pytest.raises(AttributeError, match=match):
        eval(old_func)
    eval(new_func)


def test_deprecated():
    # test funcs that are only in old api are all in DEPRECATED
    old_funcs = set(OLD_API)
    new_funcs = set(UPDATED_API.keys())
    for k in old_funcs - new_funcs:
        if k not in _api_change.DEPRECATED:
            try:
                eval(k)
            except AttributeError:
                raise ValueError(f"{k} is found only in old API but not in DEPRECATED!")


def test_api_change_deprecated():
    # test funcs that are in DEPRECATED are in old but not new api
    for k in _api_change.DEPRECATED:
        if k in UPDATED_API:
            raise ValueError(f"{k} is supposed to be deprecated but is in new API!")
        if k not in OLD_API:
            raise ValueError(f"{k} is supposed to be deprecated but is not in old API!")


def test_new_api():
    # test that current API is correctly described by _api_change
    for mod in dir(plenoptic):
        mod_name = f"plenoptic.{mod}"
        obj = eval(mod_name)
        if not inspect.ismodule(obj):
            if mod_name not in UPDATED_API.values() and mod_name not in OLD_API:
                raise ValueError(f"{mod_name} not found in api change or old api!")
        else:
            for mod2 in dir(obj):
                mod2_name = f"plenoptic.{mod}.{mod2}"
                if mod2_name not in UPDATED_API.values() and mod2_name not in OLD_API:
                    raise ValueError(f"{mod2_name} not found in api change or old api!")


# get old api and check:
# - that everything in current API is either in old api or in _api_change
# - that everything in new and not old is in _api_change
