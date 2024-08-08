"""tools to deal with data from outside plenoptic

For example, pre-existing synthesized images

"""

import os.path as op

import imageio
import matplotlib.lines as lines
import numpy as np
import pyrtools as pt
import scipy.io as sio
from ..data import fetch_data


def plot_MAD_results(
    original_image,
    noise_levels=None,
    results_dir=None,
    ssim_images_dir=None,
    zoom=3,
    vrange="indep1",
    **kwargs,
):
    r"""plot original MAD results, provided by Zhou Wang

    Plot the results of original MAD Competition, as provided in .mat
    files. The figure created shows the results for one reference image
    and multiple noise levels. The reference image is plotted on the
    first row, followed by a separate row for each noise level, which
    will show the initial (noisy) image and the four synthesized images,
    with their respective losses for the two metrics (MSE and SSIM).

    We also return a DataFrame that contains the losses, noise levels,
    and original image name for each plotted noise level.

    This code can probably be adapted to other uses, but requires that
    all images are the same size and assumes they're all 64 x 64 pixels.

    Parameters
    ----------
    original_image : {samp1, samp2, samp3, samp4, samp5, samp6, samp7,
                      samp8, samp9, samp10}
        which of the sample images to plot
    noise_levels : list or None, optional
        which noise levels to plot. if None, will plot all. If a list,
        elements must be 2**i where i is in [1, 10]
    results_dir : None or str, optional
        path to the results directory containing the results.mat files. If
        None, we call `po.data.fetch_data` to download (requires optional
        dependency pooch).
    ssim_images_dir : None or str, optional
        path to the directory containing the .tif images used in SSIM paper. If
        None, we call `po.data.fetch_data` to download (requires optional
        dependency pooch).
    zoom : int, optional
        amount to zoom each image, passed to pyrtools.imshow
    vrange : str, optional
        in addition to the values accepted by pyrtools.imshow, we also
        accept 'row0/1/2/3', which is the same as 'auto0/1/2/3', except
        that we do it on a per-row basis (all images with same noise
        level)
    kwargs :
        passed to pyrtools.imshow. Note that we call imshow separately
        on each image and so any argument that relies on imshow having
        access to all images will probably not work as expected

    Returns
    -------
    fig : pyrtools.tools.display.Figure
        figure containing the images
    results : dict
        dictionary containing the errors for each noise level. To
        convert to a well-structured pandas DataFrame, run
        ``pd.DataFrame(results).T``

    """
    if results_dir is None:
        results_dir = str(fetch_data("MAD_results.tar.gz"))
    if ssim_images_dir is None:
        ssim_images_dir = str(fetch_data("ssim_images.tar.gz"))
    img_path = op.join(op.expanduser(ssim_images_dir), f"{original_image}.tif")
    orig_img = imageio.imread(img_path)
    blanks = np.ones((*orig_img.shape, 4))
    if noise_levels is None:
        noise_levels = [2**i for i in range(1, 11)]
    results = {}
    images = np.dstack([orig_img, blanks])
    titles = ["Original image"] + 4 * [None]
    super_titles = 5 * [None]
    keys = [
        "im_init",
        "im_fixmse_maxssim",
        "im_fixmse_minssim",
        "im_fixssim_minmse",
        "im_fixssim_maxmse",
    ]
    for l in noise_levels:
        mat = sio.loadmat(
            op.join(
                op.expanduser(results_dir),
                f"{original_image}_L{l}_results.mat",
            ),
            squeeze_me=True,
        )
        # remove these metadata keys
        [mat.pop(k) for k in ["__header__", "__version__", "__globals__"]]
        key_titles = [
            f"Noise level: {l}",
            f"Best SSIM: {mat['maxssim']:.05f}",
            f"Worst SSIM: {mat['minssim']:.05f}",
            f"Best MSE: {mat['minmse']:.05f}",
            f"Worst MSE: {mat['maxmse']:.05f}",
        ]
        key_super_titles = [
            None,
            f"Fix MSE: {mat['FIX_MSE']:.0f}",
            None,
            f"Fix SSIM: {mat['FIX_SSIM']:.05f}",
            None,
        ]
        for k, t, s in zip(keys, key_titles, key_super_titles):
            images = np.dstack([images, mat.pop(k)])
            titles.append(t)
            super_titles.append(s)
        # this then just contains the loss information
        mat.update({"noise_level": l, "original_image": original_image})
        results[f"L{l}"] = mat
    images = images.transpose((2, 0, 1))
    if vrange.startswith("row"):
        vrange_list = []
        for i in range(len(images) // 5):
            vr, cmap = pt.tools.display.colormap_range(
                images[5 * i : 5 * (i + 1)], vrange.replace("row", "auto")
            )
            vrange_list.extend(vr)
    else:
        vrange_list, cmap = pt.tools.display.colormap_range(images, vrange)
    # this is a bit of hack to do the same thing imshow does, but with
    # slightly more space dedicated to the title
    fig = pt.tools.display.make_figure(
        len(images) // 5,
        5,
        [zoom * i + 1 for i in images.shape[-2:]],
        vert_pct=0.75,
    )
    for img, ax, t, vr, s in zip(
        images, fig.axes, titles, vrange_list, super_titles
    ):
        # these are the blanks
        if (img == 1).all():
            continue
        pt.imshow(
            img, ax=ax, title=t, zoom=zoom, vrange=vr, cmap=cmap, **kwargs
        )
        if s is not None:
            font = {
                k.replace("_", ""): v
                for k, v in ax.title.get_font_properties().__dict__.items()
            }
            # these are the acceptable keys for the fontdict below
            font = {
                k: v
                for k, v in font.items()
                if k in ["family", "color", "weight", "size", "style"]
            }
            # for some reason, this (with passing the transform) is
            # different (and looks better) than using ax.text. We also
            # slightly adjust the placement of the text to account for
            # different zoom levels (we also have 10 pixels between the
            # rows and columns, which correspond to a different)
            img_size = ax.bbox.size
            fig.text(
                1 + (5 / img_size[0]),
                (1 / 0.75),
                s,
                fontdict=font,
                transform=ax.transAxes,
                ha="center",
                va="top",
            )
            # linewidth of 1.5 looks good with bbox of 192, 192
            linewidth = np.max([1.5 * np.mean(img_size / 192), 1])
            line = lines.Line2D(
                2 * [0 - ((5 + linewidth / 2) / img_size[0])],
                [0, (1 / 0.75)],
                transform=ax.transAxes,
                figure=fig,
                linewidth=linewidth,
            )
            fig.lines.append(line)
    return fig, results
