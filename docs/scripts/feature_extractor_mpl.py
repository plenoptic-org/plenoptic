"""Just put this in a separate function for matplotlib sphinxext."""

import feature_extractor
import matplotlib.pyplot as plt


def layer2():
    # need this for matplotlib's sphinxext, which needs a function that doesn't have any
    # arguments
    feature_extractor.main("layer2")


def layer2_stats():
    orig, met, corr = feature_extractor.get_stats("layer2")
    fig = plt.figure()
    txt = (
        f"Target image category: {orig}\n"
        f"Metamer category: {met}\n"
        f"Pearson correlation: {corr}\n"
    )
    fig.text(0, 0, txt)


def layer3():
    # need this for matplotlib's sphinxext, which needs a function that doesn't have any
    # arguments
    feature_extractor.main("layer3")


def layer3_stats():
    orig, met, corr = feature_extractor.get_stats("layer3")
    fig = plt.figure()
    txt = (
        f"Target image category: {orig}\n"
        f"Metamer category: {met}\n"
        f"Pearson correlation: {corr}\n"
    )
    fig.text(0, 0, txt)


def layer4():
    # need this for matplotlib's sphinxext, which needs a function that doesn't have any
    # arguments
    feature_extractor.main("layer4")


def layer4_stats():
    orig, met, corr = feature_extractor.get_stats("layer4")
    fig = plt.figure()
    txt = (
        f"Target image category: {orig}\n"
        f"Metamer category: {met}\n"
        f"Pearson correlation: {corr}\n"
    )
    fig.text(0, 0, txt)
