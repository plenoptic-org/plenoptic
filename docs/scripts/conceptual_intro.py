import json

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

# set the style
plt.style.use("fivethirtyeight")
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["savefig.edgecolor"] = "white"
plt.rcParams["savefig.bbox"] = "tight"

with open("conceptual_intro_data.json") as f:
    data = json.load(f)
    # 2-deg fundamentals (energy linear, 5nm stepsize) from Stockman & Sharpe 2000,
    # downloaded from http://www.cvrl.org/cones.htm
    CONES = data["CONES"]
    # from math tools homework, interpolated to have same number of entries as
    # CONES
    PRIMARIES = np.array(data["PRIMARIES"])

# for matrix multiplication
CONES_MATRIX = np.stack([CONES["S"], CONES["M"], CONES["L"]])
CONES_MATRIX[np.isnan(CONES_MATRIX)] = 0

RANDOM_LIGHT = np.random.random(len(PRIMARIES))


def cones():
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(CONES["nm"], CONES["S"], c="b")
    axes[0].plot(CONES["nm"], CONES["M"], c="g")
    axes[0].plot(CONES["nm"], CONES["L"], c="r")
    axes[0].set(
        xlabel="Wavelength (nm)",
        ylabel="Cone sensitivity (arbitrary units)",
        title="Cone sensitivity curves",
    )
    random_light_weights = CONES_MATRIX @ RANDOM_LIGHT
    x = np.arange(len(random_light_weights))
    axes[1].bar(x, random_light_weights, width=0.4, color=["b", "g", "r"])
    axes[1].set(
        xlabel="Cone class",
        ylabel="Cone response (arbitrary units)",
        xticks=x,
        xticklabels=["S", "M", "L"],
        title="Cone responses to random light",
    )


def primaries():
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(CONES["nm"], RANDOM_LIGHT)
    for p, c in zip(PRIMARIES.T, ["g", "b", "r"]):
        axes[1].plot(CONES["nm"], p, c=c)
    axes[0].set(
        xlabel="Wavelength (nm)",
        ylabel="Energy (arbitrary units)",
        title="Random light",
    )
    axes[1].set(xlabel="Wavelength (nm)", title="Primaries")


def matched_light():
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    cones_to_primaries = np.linalg.inv(CONES_MATRIX @ PRIMARIES)
    matched_light = PRIMARIES @ cones_to_primaries @ CONES_MATRIX @ RANDOM_LIGHT
    axes[0].plot(CONES["nm"], RANDOM_LIGHT, label="Random light")
    axes[0].plot(CONES["nm"], matched_light, "--", label="Matched light")
    axes[0].set(
        xlabel="Wavelength (nm)",
        ylabel="Energy (arbitrary units)",
        title="Random and matched light",
    )
    axes[0].legend(loc="upper left")
    random_light_weights = CONES_MATRIX @ RANDOM_LIGHT
    matched_light_weights = CONES_MATRIX @ matched_light
    x = np.arange(len(random_light_weights))
    # from https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
    width = 0.4
    styles = [
        {
            "color": ["b", "g", "r"],
            "edgecolor": ["b", "g", "r"],
            "linewidth": plt.rcParams["lines.linewidth"],
        },
        {
            "color": "w",
            "edgecolor": ["b", "g", "r"],
            "linestyle": "--",
            "linewidth": plt.rcParams["lines.linewidth"],
        },
    ]

    labels = ["Random light", "Matched light"]
    weights = [random_light_weights, matched_light_weights]

    for multiplier, (name, wts, sty) in enumerate(zip(labels, weights, styles)):
        offset = width * multiplier
        axes[1].bar(x + offset, wts, label=name, width=width, **sty)

    axes[1].set(
        xlabel="Cone class",
        ylabel="Cone response (arbitrary units)",
        xticks=x + width / 2,
        xticklabels=["S", "M", "L"],
        title="Cone responses",
    )
