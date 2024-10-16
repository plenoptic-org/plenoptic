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

# 2-deg fundamentals (energy linear, 5nm stepsize) from Stockman & Sharpe 2000,
# downloaded from http://www.cvrl.org/cones.htm
CONES = {
    "nm": [
        400,
        405,
        410,
        415,
        420,
        425,
        430,
        435,
        440,
        445,
        450,
        455,
        460,
        465,
        470,
        475,
        480,
        485,
        490,
        495,
        500,
        505,
        510,
        515,
        520,
        525,
        530,
        535,
        540,
        545,
        550,
        555,
        560,
        565,
        570,
        575,
        580,
        585,
        590,
        595,
        600,
        605,
        610,
        615,
        620,
        625,
        630,
        635,
        640,
        645,
        650,
        655,
        660,
        665,
        670,
        675,
        680,
        685,
        690,
        695,
        700,
    ],
    "L": [
        0.00240836,
        0.00483339,
        0.00872127,
        0.0133837,
        0.018448,
        0.0229317,
        0.0281877,
        0.0341054,
        0.0402563,
        0.044938,
        0.0498639,
        0.0553418,
        0.0647164,
        0.0806894,
        0.0994755,
        0.118802,
        0.140145,
        0.163952,
        0.191556,
        0.232926,
        0.288959,
        0.359716,
        0.443683,
        0.536494,
        0.628561,
        0.70472,
        0.77063,
        0.825711,
        0.881011,
        0.919067,
        0.940198,
        0.965733,
        0.981445,
        0.994486,
        0.999993,
        0.99231,
        0.969429,
        0.955602,
        0.927673,
        0.885969,
        0.833982,
        0.775103,
        0.705713,
        0.630773,
        0.554224,
        0.479941,
        0.400711,
        0.327864,
        0.265784,
        0.213284,
        0.165141,
        0.124749,
        0.0930085,
        0.06851,
        0.0498661,
        0.0358233,
        0.025379,
        0.0177201,
        0.0121701,
        0.0084717,
        0.00589749,
    ],
    "M": [
        0.00226991,
        0.0047001,
        0.00879369,
        0.0145277,
        0.0216649,
        0.0295714,
        0.0394566,
        0.0518199,
        0.0647782,
        0.0758812,
        0.0870524,
        0.0981934,
        0.116272,
        0.144541,
        0.175893,
        0.205398,
        0.235754,
        0.268063,
        0.30363,
        0.357061,
        0.427764,
        0.515587,
        0.61552,
        0.719154,
        0.81661,
        0.88555,
        0.935687,
        0.968858,
        0.995217,
        0.997193,
        0.977193,
        0.956583,
        0.91775,
        0.873205,
        0.813509,
        0.740291,
        0.653274,
        0.572597,
        0.492599,
        0.411246,
        0.334429,
        0.264872,
        0.205273,
        0.156243,
        0.116641,
        0.0855872,
        0.062112,
        0.0444879,
        0.0314282,
        0.0218037,
        0.015448,
        0.010712,
        0.00730255,
        0.00497179,
        0.00343667,
        0.00237617,
        0.00163734,
        0.00112128,
        0.000761051,
        0.000525457,
        0.000365317,
    ],
    "S": [
        0.0566498,
        0.122451,
        0.233008,
        0.381363,
        0.543618,
        0.674474,
        0.802555,
        0.903573,
        0.99102,
        0.991515,
        0.955393,
        0.86024,
        0.786704,
        0.738268,
        0.646359,
        0.516411,
        0.390333,
        0.290322,
        0.211867,
        0.160526,
        0.122839,
        0.0888965,
        0.060821,
        0.0428123,
        0.0292033,
        0.0193912,
        0.0126013,
        0.00809453,
        0.005089,
        0.00316893,
        0.00195896,
        0.00120277,
        0.000740174,
        0.000455979,
        0.0002818,
        0.000175039,
        0.000109454,
        6.89991e-05,
        4.39024e-05,
        2.82228e-05,
        1.83459e-05,
        1.20667e-05,
        8.03488e-06,
        5.41843e-06,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ],
}

# for matrix multiplication
CONES_MATRIX = np.stack([CONES["S"], CONES["M"], CONES["L"]])
CONES_MATRIX[np.isnan(CONES_MATRIX)] = 0

# from math tools homework, interpolated to have same number of entries as
# CONES
PRIMARIES = np.array(
    [
        [3.72665317e-06, 4.57833362e-01, 4.39369336e-02],
        [1.11955611e-05, 5.32182011e-01, 2.75229651e-02],
        [1.86644691e-05, 6.06530660e-01, 1.11089965e-02],
        [5.11564081e-05, 6.80685131e-01, 6.64824381e-03],
        [8.36483472e-05, 7.54839602e-01, 2.18749112e-03],
        [2.09555488e-04, 8.18668252e-01, 1.26147687e-03],
        [3.35462628e-04, 8.82496903e-01, 3.35462628e-04],
        [7.69661309e-04, 9.25865069e-01, 1.87763963e-04],
        [1.20385999e-03, 9.69233234e-01, 4.00652974e-05],
        [2.53489006e-03, 9.84616617e-01, 2.18959753e-05],
        [3.86592014e-03, 1.00000000e00, 3.72665317e-06],
        [7.48745832e-03, 9.84616617e-01, 1.99830551e-06],
        [1.11089965e-02, 9.69233234e-01, 2.69957850e-07],
        [1.98372487e-02, 9.25865069e-01, 1.42593915e-07],
        [2.85655008e-02, 8.82496903e-01, 1.52299797e-08],
        [4.71470147e-02, 8.18668252e-01, 7.94956915e-09],
        [6.57285286e-02, 7.54839602e-01, 6.69158609e-10],
        [1.00531906e-01, 6.80685131e-01, 3.46027979e-10],
        [1.35335283e-01, 6.06530660e-01, 2.28973485e-11],
        [1.92343746e-01, 5.32182011e-01, 1.17537711e-11],
        [2.49352209e-01, 4.57833362e-01, 6.10193668e-13],
        [3.30232250e-01, 3.91242915e-01, 6.10193668e-13],
        [4.11112291e-01, 3.24652467e-01, 6.10193668e-13],
        [5.08821476e-01, 2.70458817e-01, 1.17537711e-11],
        [6.06530660e-01, 2.16265167e-01, 2.28973485e-11],
        [7.03634032e-01, 1.75800225e-01, 3.46027979e-10],
        [8.00737403e-01, 1.35335283e-01, 6.69158609e-10],
        [8.73348436e-01, 1.07447396e-01, 7.94956915e-09],
        [9.45959469e-01, 7.95595087e-02, 1.52299797e-08],
        [9.72979734e-01, 6.17482212e-02, 1.42593915e-07],
        [1.00000000e00, 4.39369336e-02, 2.69957850e-07],
        [9.72979734e-01, 3.33655572e-02, 1.99830551e-06],
        [9.45959469e-01, 2.27941809e-02, 3.72665317e-06],
        [8.73348436e-01, 1.69515887e-02, 2.18959753e-05],
        [8.00737403e-01, 1.11089965e-02, 4.00652974e-05],
        [7.03634032e-01, 8.09753286e-03, 1.87763963e-04],
        [6.06530660e-01, 5.08606923e-03, 3.35462628e-04],
        [5.08821476e-01, 3.63678018e-03, 1.26147687e-03],
        [4.11112291e-01, 2.18749112e-03, 2.18749112e-03],
        [3.30232250e-01, 1.53565871e-03, 6.64824381e-03],
        [2.49352209e-01, 8.83826307e-04, 1.11089965e-02],
        [1.92343746e-01, 8.83826307e-04, 2.75229650e-02],
        [1.35335283e-01, 8.83826307e-04, 4.39369336e-02],
        [1.00531906e-01, 1.53565871e-03, 8.96361083e-02],
        [6.57285286e-02, 2.18749112e-03, 1.35335283e-01],
        [4.71470147e-02, 3.63678018e-03, 2.29993875e-01],
        [2.85655008e-02, 5.08606923e-03, 3.24652467e-01],
        [1.98372487e-02, 8.09753286e-03, 4.65591563e-01],
        [1.11089965e-02, 1.11089965e-02, 6.06530660e-01],
        [7.48745832e-03, 1.69515887e-02, 7.44513782e-01],
        [3.86592014e-03, 2.27941809e-02, 8.82496903e-01],
        [2.53489006e-03, 3.33655572e-02, 9.41248452e-01],
        [1.20385999e-03, 4.39369336e-02, 1.00000000e00],
        [7.69661309e-04, 6.17482212e-02, 9.41248452e-01],
        [3.35462628e-04, 7.95595087e-02, 8.82496903e-01],
        [2.09555488e-04, 1.07447396e-01, 7.44513782e-01],
        [8.36483472e-05, 1.35335283e-01, 6.06530660e-01],
        [5.11564081e-05, 1.75800225e-01, 4.65591563e-01],
        [1.86644691e-05, 2.16265167e-01, 3.24652467e-01],
        [1.11955611e-05, 2.70458817e-01, 2.29993875e-01],
        [3.72665317e-06, 3.24652467e-01, 1.35335283e-01],
    ]
)

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
    multiplier = 0
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

    for i, (name, wts, sty) in enumerate(zip(labels, weights, styles)):
        offset = width * (multiplier + i)
        axes[1].bar(x + offset, wts, label=name, width=width, **sty)

    axes[1].set(
        xlabel="Cone class",
        ylabel="Cone response (arbitrary units)",
        xticks=x + width / 2,
        xticklabels=["S", "M", "L"],
        title="Cone responses",
    )
