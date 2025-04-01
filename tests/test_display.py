# necessary to avoid issues with animate:
# https://github.com/matplotlib/matplotlib/issues/10287/
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyrtools as pt
import pytest
import torch

import plenoptic as po
from conftest import DEVICE, IMG_DIR

# use the html backend, so we don't need to have ffmpeg
mpl.rcParams["animation.writer"] = "html"
mpl.use("agg")


class TestDisplay:
    def test_update_plot_line(self):
        x = np.linspace(0, 100)
        y1 = np.random.rand(*x.shape)
        y2 = np.random.rand(*x.shape)
        fig, ax = plt.subplots(1, 1)
        ax.plot(x, y1, "-o", label="hi")
        po.tools.update_plot(ax, torch.as_tensor(y2).reshape(1, 1, len(x)))
        assert len(ax.lines) == 1, "Too many lines were plotted!"
        _, ax_y = ax.lines[0].get_data()
        if not np.allclose(ax_y, y2):
            raise Exception("Didn't update line correctly!")
        plt.close("all")

    @pytest.mark.parametrize("how", ["dict", "tensor"])
    def test_update_plot_line_multi_axes(self, how):
        x = np.linspace(0, 100)
        y1 = np.random.rand(*x.shape)
        y2 = np.random.rand(2, *y1.shape)
        if how == "tensor":
            y2 = torch.as_tensor(y2).reshape(1, 2, *y1.shape)
        elif how == "dict":
            y2 = {i: torch.as_tensor(y2[i]).reshape(1, 1, *y1.shape) for i in range(2)}
        fig, axes = plt.subplots(1, 2)
        for ax in axes:
            ax.plot(x, y1, "-o", label="hi")
        po.tools.update_plot(axes, y2)
        for i, ax in enumerate(axes):
            assert len(ax.lines) == 1, "Too many lines were plotted!"
            _, ax_y = ax.lines[0].get_data()
            y_check = y2[0, i] if how == "tensor" else y2[i]

            if not np.allclose(ax_y, y_check):
                raise Exception("Didn't update line correctly!")
        plt.close("all")

    @pytest.mark.parametrize("how", ["dict-single", "dict-multi", "tensor"])
    def test_update_plot_line_multi_channel(self, how):
        n_data = 1 if how == "dict-single" else 2

        x = np.linspace(0, 100)
        y1 = np.random.rand(2, *x.shape)
        y2 = np.random.rand(n_data, *x.shape)
        if how == "tensor":
            y2 = torch.as_tensor(y2).reshape(1, 2, len(x))
        elif how == "dict-multi":
            y2 = {i: torch.as_tensor(y2[i]).reshape(1, 1, len(x)) for i in range(2)}
        elif how == "dict-single":
            y2 = {0: torch.as_tensor(y2[0]).reshape(1, 1, len(x))}
        fig, ax = plt.subplots(1, 1)
        for i in range(2):
            ax.plot(x, y1[i], label=i)
        po.tools.update_plot(ax, y2)
        assert len(ax.lines) == 2, "Incorrect number of lines were plotted!"
        for i in range(2):
            _, ax_y = ax.lines[i].get_data()
            if how == "tensor":
                y_check = y2[0, i]
            elif how == "dict-multi":
                y_check = y2[i]
            elif how == "dict-single":
                y_check = {0: y2[0], 1: y1[1]}[i]
            if not np.allclose(ax_y, y_check):
                raise Exception("Didn't update line correctly!")
        plt.close("all")

    def test_update_plot_stem(self):
        x = np.linspace(0, 100)
        y1 = np.random.rand(*x.shape)
        y2 = np.random.rand(*x.shape)
        fig, ax = plt.subplots(1, 1)
        ax.stem(x, y1, "-o", label="hi")
        po.tools.update_plot(ax, torch.as_tensor(y2).reshape(1, 1, len(x)))
        assert len(ax.containers) == 1, "Too many stems were plotted!"
        ax_y = ax.containers[0].markerline.get_ydata()
        if not np.allclose(ax_y, y2):
            raise Exception("Didn't update stems correctly!")
        plt.close("all")

    @pytest.mark.parametrize("how", ["dict", "tensor"])
    def test_update_plot_stem_multi_axes(self, how):
        x = np.linspace(0, 100)
        y1 = np.random.rand(*x.shape)
        y2 = np.random.rand(2, *y1.shape)
        if how == "tensor":
            y2 = torch.as_tensor(y2).reshape(1, 2, *y1.shape)
        elif how == "dict":
            y2 = {i: torch.as_tensor(y2[i]).reshape(1, 1, *y1.shape) for i in range(2)}
        fig, axes = plt.subplots(1, 2)
        for ax in axes:
            ax.stem(x, y1, label="hi")
        po.tools.update_plot(axes, y2)
        for i, ax in enumerate(axes):
            assert len(ax.containers) == 1, "Too many stems were plotted!"
            ax_y = ax.containers[0].markerline.get_ydata()
            y_check = y2[0, i] if how == "tensor" else y2[i]

            if not np.allclose(ax_y, y_check):
                raise Exception("Didn't update stem correctly!")
        plt.close("all")

    @pytest.mark.parametrize("how", ["dict-single", "dict-multi", "tensor"])
    def test_update_plot_stem_multi_channel(self, how):
        n_data = 1 if how == "dict-single" else 2

        x = np.linspace(0, 100)
        y1 = np.random.rand(2, *x.shape)
        y2 = np.random.rand(n_data, *x.shape)
        if how == "tensor":
            y2 = torch.as_tensor(y2).reshape(1, 2, len(x))
        elif how == "dict-multi":
            y2 = {i: torch.as_tensor(y2[i]).reshape(1, 1, len(x)) for i in range(2)}
        elif how == "dict-single":
            y2 = {0: torch.as_tensor(y2[0]).reshape(1, 1, len(x))}
        fig, ax = plt.subplots(1, 1)
        for i in range(2):
            ax.stem(x, y1[i], label=str(i))
        po.tools.update_plot(ax, y2)
        assert len(ax.containers) == 2, "Incorrect number of stems were plotted!"
        for i in range(2):
            ax_y = ax.containers[i].markerline.get_ydata()
            if how == "tensor":
                y_check = y2[0, i]
            elif how == "dict-multi":
                y_check = y2[i]
            elif how == "dict-single":
                y_check = {0: y2[0], 1: y1[1]}[i]
            if not np.allclose(ax_y, y_check):
                raise Exception("Didn't update stem correctly!")
        plt.close("all")

    def test_update_plot_image(self):
        y1 = np.random.rand(1, 1, 100, 100)
        y2 = np.random.rand(*y1.shape)
        fig = pt.imshow(y1.squeeze())
        ax = fig.axes[0]
        po.tools.update_plot(ax, torch.as_tensor(y2))
        assert len(ax.images) == 1, "Too many images were plotted!"
        ax_y = ax.images[0].get_array().data
        if not np.allclose(ax_y, y2):
            raise Exception("Didn't update image correctly!")
        plt.close("all")

    @pytest.mark.parametrize("how", ["dict", "tensor"])
    def test_update_plot_image_multi_axes(self, how):
        y1 = np.random.rand(1, 2, 100, 100)
        y2 = np.random.rand(1, 2, 100, 100)
        if how == "tensor":
            y2 = torch.as_tensor(y2)
        elif how == "dict":
            y2 = {
                i: torch.as_tensor(y2[0, i]).reshape(1, 1, 100, 100) for i in range(2)
            }
        fig = pt.imshow([y for y in y1.squeeze()])
        po.tools.update_plot(fig.axes, y2)
        for i, ax in enumerate(fig.axes):
            assert len(ax.images) == 1, "Too many lines were plotted!"
            ax_y = ax.images[0].get_array().data

            y_check = y2[0, i] if how == "tensor" else y2[i]

            if not np.allclose(ax_y, y_check):
                raise Exception("Didn't update image correctly!")
        plt.close("all")

    def test_update_plot_scatter(self):
        x1 = np.random.rand(100)
        x2 = np.random.rand(100)
        y1 = np.random.rand(*x1.shape)
        y2 = np.random.rand(*x2.shape)
        fig, ax = plt.subplots(1, 1)
        ax.scatter(x1, y1)
        data = torch.stack((torch.as_tensor(x2), torch.as_tensor(y2)), -1).reshape(
            1, 1, len(x2), 2
        )
        po.tools.update_plot(ax, data)
        assert len(ax.collections) == 1, "Too many scatter plots created"
        ax_data = ax.collections[0].get_offsets()
        if not np.allclose(ax_data, data):
            raise Exception("Didn't update points of the scatter plot correctly!")
        plt.close("all")

    @pytest.mark.parametrize("how", ["dict", "tensor"])
    def test_update_plot_scatter_multi_axes(self, how):
        x1 = np.random.rand(100)
        x2 = np.random.rand(2, 100)
        y1 = np.random.rand(*x1.shape)
        y2 = np.random.rand(2, *y1.shape)
        if how == "tensor":
            data = torch.stack((torch.as_tensor(x2), torch.as_tensor(y2)), -1).reshape(
                1, 2, len(x1), 2
            )
        elif how == "dict":
            data = {
                i: torch.stack(
                    (torch.as_tensor(x2[i]), torch.as_tensor(y2[i])), -1
                ).reshape(1, 1, len(x1), 2)
                for i in range(2)
            }
        fig, axes = plt.subplots(1, 2)
        for ax in axes:
            ax.scatter(x1, y1)
        po.tools.update_plot(axes, data)
        for i, ax in enumerate(axes):
            assert len(ax.collections) == 1, "Too many scatter plots created"
            ax_data = ax.collections[0].get_offsets()
            data_check = data[0, i] if how == "tensor" else data[i]

            if not np.allclose(ax_data, data_check):
                raise Exception("Didn't update points of the scatter plot correctly!")
        plt.close("all")

    @pytest.mark.parametrize("how", ["dict-single", "dict-multi", "tensor"])
    def test_update_plot_scatter_multi_channel(self, how):
        n_data = 1 if how == "dict-single" else 2
        x1 = np.random.rand(2, 100)
        x2 = np.random.rand(n_data, 100)
        y1 = np.random.rand(*x1.shape)
        y2 = np.random.rand(*x2.shape)
        if how == "tensor":
            data = torch.stack((torch.as_tensor(x2), torch.as_tensor(y2)), -1).reshape(
                1, 2, x1.shape[-1], 2
            )
        elif how == "dict-multi":
            data = {
                i: torch.stack(
                    (torch.as_tensor(x2[i]), torch.as_tensor(y2[i])), -1
                ).reshape(1, 1, x1.shape[-1], 2)
                for i in range(2)
            }
        elif how == "dict-single":
            data = {
                0: torch.stack(
                    (torch.as_tensor(x2[0]), torch.as_tensor(y2[0])), -1
                ).reshape(1, 1, x1.shape[-1], 2)
            }
        fig, ax = plt.subplots(1, 1)
        for i in range(2):
            ax.scatter(x1[i], y1[i], label=i)
        po.tools.update_plot(ax, data)
        assert len(ax.collections) == 2, "Incorrect number of scatter plots created"
        for i in range(2):
            ax_data = ax.collections[i].get_offsets()
            if how == "tensor":
                data_check = data[0, i]
            elif how == "dict-multi":
                data_check = data[i]
            elif how == "dict-single":
                tmp = torch.stack((torch.as_tensor(x1), torch.as_tensor(y1)), -1)
                data_check = {0: data[0], 1: tmp[1]}[i]
            if not np.allclose(ax_data, data_check):
                raise Exception("Didn't update points of the scatter plot correctly!")

    def test_update_plot_mixed_multi_axes(self):
        x1 = np.linspace(0, 1, 100)
        x2 = np.random.rand(2, 100)
        y1 = np.random.rand(*x1.shape)
        y2 = np.random.rand(*x2.shape)
        data = {
            0: torch.stack(
                (torch.as_tensor(x2[0]), torch.as_tensor(y2[0])), -1
            ).reshape(1, 1, x2.shape[-1], 2)
        }
        data[1] = torch.as_tensor(y2[1]).reshape(1, 1, x2.shape[-1])
        fig, axes = plt.subplots(1, 2)
        for i, ax in enumerate(axes):
            if i == 0:
                ax.scatter(x1, y1)
            else:
                ax.plot(x1, y1)
        po.tools.update_plot(axes, data)
        for i, ax in enumerate(axes):
            if i == 0:
                assert len(ax.collections) == 1, "Too many scatter plots created"
                assert len(ax.lines) == 0, "Too many lines created"
                ax_data = ax.collections[0].get_offsets()
            else:
                assert len(ax.collections) == 0, "Too many scatter plots created"
                assert len(ax.lines) == 1, "Too many lines created"
                _, ax_data = ax.lines[0].get_data()
            if not np.allclose(ax_data, data[i]):
                raise Exception("Didn't update points of the scatter plot correctly!")
        plt.close("all")

    @pytest.mark.parametrize("as_rgb", [True, False])
    @pytest.mark.parametrize("channel_idx", [None, 0, [0, 1]])
    @pytest.mark.parametrize("batch_idx", [None, 0, [0, 1]])
    @pytest.mark.parametrize("is_complex", [False, "logpolar", "rectangular", "polar"])
    @pytest.mark.parametrize("mini_im", [True, False])
    # test the edge cases where we try to plot a tensor that's (b, c, 1, w) or
    # (b, c, h, 1)
    @pytest.mark.parametrize("one_dim", [False, "h", "w"])
    def test_imshow(self, as_rgb, channel_idx, batch_idx, is_complex, mini_im, one_dim):
        fails = False
        if one_dim == "h":
            im_shape = [2, 4, 1, 5]
        elif one_dim == "w":
            im_shape = [2, 4, 5, 1]
        else:
            im_shape = [2, 4, 5, 5]
        if is_complex:
            dtype = torch.complex64
            # this is 4 (the four channels) * 2 (the two batches) * 2 (the two
            # complex components)
            n_axes = 16
        else:
            dtype = torch.float32
            # this is 4 (the four channels) * 2 (the two batches)
            n_axes = 8
        im = torch.rand(im_shape, dtype=dtype)
        if mini_im:
            # n_axes here follows the same logic as above
            if is_complex:
                n_axes += 16
            else:
                n_axes += 8
            # same number of batches and channels, then double the height and
            # width
            shape = im_shape[:2] + [i * 2 for i in im_shape[-2:]]
            im = [im, torch.rand(shape, dtype=dtype)]
        if not is_complex:
            # need to change this to one of the acceptable strings
            is_complex = "rectangular"
        if batch_idx is None and channel_idx is None and not as_rgb:
            # then we'd have a 4d array we want to plot in grayscale -- don't
            # know how to do that
            fails = True
        else:
            if batch_idx is not None:
                # then we're only plotting one of the two batches
                n_axes /= 2
            if channel_idx is not None:
                # then we're only plotting one of the four channels
                n_axes /= 4
                # if channel_idx is not None, then we don't have all the
                # channels necessary for plotting RGB, so this will fail
                if as_rgb:
                    fails = True
            # when channel_idx=0, as_rgb does nothing, so don't want to
            # double-count
            elif as_rgb:
                # if we're plotting as_rgb, the four channels just specify
                # RGBA, so we only have one image for them
                n_axes /= 4
        if isinstance(batch_idx, list) or isinstance(channel_idx, list):
            # neither of these are supported
            fails = True
        if not fails:
            fig = po.imshow(
                im,
                as_rgb=as_rgb,
                channel_idx=channel_idx,
                batch_idx=batch_idx,
                plot_complex=is_complex,
            )
            assert len(fig.axes) == n_axes, (
                f"Created {len(fig.axes)} axes, but expected {n_axes}!"
                " Probably plotting color as grayscale or vice versa"
            )
            plt.close("all")
        if fails:
            with pytest.raises(Exception):
                po.imshow(
                    im,
                    as_rgb=as_rgb,
                    channel_idx=channel_idx,
                    batch_idx=batch_idx,
                    plot_complex=is_complex,
                )

    @pytest.fixture(scope="class", params=["complex", "not-complex"])
    def steerpyr(self, request):
        if request.param == "complex":
            is_complex = True
        elif request.param == "not-complex":
            is_complex = False
        return po.simul.SteerablePyramidFreq(
            (32, 32), height=2, order=1, is_complex=is_complex
        ).to(DEVICE)

    @pytest.mark.parametrize("channel_idx", [None, 0, [0, 1]])
    @pytest.mark.parametrize("batch_idx", [None, 0, [0, 1]])
    @pytest.mark.parametrize("show_residuals", [True, False])
    def test_pyrshow(self, steerpyr, channel_idx, batch_idx, show_residuals, curie_img):
        fails = False
        if not isinstance(channel_idx, int) or not isinstance(batch_idx, int):
            fails = True
        n_axes = 4
        if steerpyr.is_complex:
            n_axes *= 2
        if show_residuals:
            n_axes += 2
        img = curie_img.clone()
        img = img[..., : steerpyr.lo0mask.shape[-2], : steerpyr.lo0mask.shape[-1]]
        coeffs = steerpyr(img)
        if not fails:
            # unfortunately, can't figure out how to properly parametrize this
            # and use the steerpyr fixture
            for comp in ["rectangular", "polar", "logpolar"]:
                fig = po.pyrshow(
                    coeffs,
                    show_residuals=show_residuals,
                    plot_complex=comp,
                    batch_idx=batch_idx,
                    channel_idx=channel_idx,
                )
                # get all the axes that have an image (so, get all non-empty
                # axes)
                axes = [ax for ax in fig.axes if ax.images]
                if len(axes) != n_axes:
                    raise Exception(
                        f"Created {len(fig.axes)} axes, but expected {n_axes}!"
                    )
                plt.close("all")
        else:
            with pytest.raises(TypeError):
                po.pyrshow(coeffs, batch_idx=batch_idx, channel_idx=channel_idx)

    def test_display_test_signals(self, basic_stim):
        po.imshow(basic_stim)

    @pytest.mark.parametrize("as_rgb", [True, False])
    @pytest.mark.parametrize("channel_idx", [None, 0, [0, 1]])
    @pytest.mark.parametrize("batch_idx", [None, 0, [0, 1]])
    @pytest.mark.parametrize("is_complex", [False, "logpolar", "rectangular", "polar"])
    @pytest.mark.parametrize("mini_vid", [True, False])
    def test_animshow(self, as_rgb, channel_idx, batch_idx, is_complex, mini_vid):
        fails = False
        if is_complex:
            # this is 2 (the two complex components) * 4 (the four channels) *
            # 2 (the two batches)
            n_axes = 16
            dtype = torch.complex64
        else:
            # this is 4 (the four channels) * 2 (the two batches)
            n_axes = 8
            dtype = torch.float32
        vid = torch.rand((2, 4, 10, 10, 10), dtype=dtype)
        if mini_vid:
            # n_axes here follows the same logic as above
            if is_complex:
                n_axes += 16
            else:
                n_axes += 8
            # same number of batches, channels, and frames, then half the
            # height and width
            shape = [2, 4, 10, 5, 5]
            vid = [vid, torch.rand(shape, dtype=dtype)]
        if not is_complex:
            # need to change this to one of the acceptable strings
            is_complex = "rectangular"
        if batch_idx is None and channel_idx is None and not as_rgb:
            # then we'd have a 4d array we want to plot in grayscale -- don't
            # know how to do that
            fails = True
        else:
            if batch_idx is not None:
                # then we're only plotting one of the two batches
                n_axes /= 2
            if channel_idx is not None:
                # then we're only plotting one of the four channels
                n_axes /= 4
                # if channel_idx is not None, then we don't have all the
                # channels necessary for plotting RGB, so this will fail
                if as_rgb:
                    fails = True
            # when channel_idx=0, as_rgb does nothing, so don't want to
            # double-count
            elif as_rgb:
                # if we're plotting as_rgb, the four channels just specify
                # RGBA, so we only have one video for them
                n_axes /= 4
        if isinstance(batch_idx, list) or isinstance(channel_idx, list):
            # neither of these are supported
            fails = True
        if not fails:
            anim = po.animshow(
                vid,
                as_rgb=as_rgb,
                channel_idx=channel_idx,
                batch_idx=batch_idx,
                plot_complex=is_complex,
            )
            fig = anim._fig
            assert len(fig.axes) == n_axes, (
                f"Created {len(fig.axes)} axes, but expected {n_axes}!"
                " Probably plotting color as grayscale or vice versa"
            )
            plt.close("all")
        if fails:
            with pytest.raises(Exception):
                po.animshow(
                    vid,
                    as_rgb=as_rgb,
                    channel_idx=channel_idx,
                    batch_idx=batch_idx,
                    plot_complex=is_complex,
                )

    def test_update_plot_shape_fail(self, einstein_img):
        # update_plot expects 3 or 4d data -- this checks that update_plot
        # fails with 2d data and raises the proper exception
        fig = po.imshow(einstein_img)
        with pytest.raises(ValueError, match="3 or 4 dimensional"):
            po.tools.update_plot(fig.axes[0], einstein_img.squeeze())

    def test_synthesis_plot_shape_fail(self, einstein_img):
        # Synthesis plot_synthesis_status and animate expect 3 or 4d data --
        # this checks that plot_synthesis_status() and animate() both fail with
        # 2d data and raise the proper exception
        class TestModel(po.simul.LinearNonlinear):
            def __init__(self):
                super().__init__((7, 7))

            def forward(self, *args, **kwargs):
                return super().forward(*args, **kwargs)

        model = TestModel().to(DEVICE)
        model.eval()
        po.tools.remove_grad(model)
        met = po.synth.Metamer(einstein_img, model)
        met.synthesize(max_iter=3, store_progress=True)
        met._metamer = met.metamer.squeeze()
        with pytest.raises(ValueError, match="3 or 4d"):
            po.synth.metamer.plot_synthesis_status(met)
        with pytest.raises(ValueError, match="3 or 4d"):
            po.synth.metamer.animate(met)


def template_test_synthesis_all_plot(
    synthesis_object,
    iteration,
    display_synth,
    loss,
    representation_error,
    pixel_values,
    fig_creation,
):
    # template function to test whether we can plot all possible combinations
    # of plots. test_custom_fig tests whether these animate correctly. Any
    # synthesis object that has had synthesis() called should work with this
    if sum([display_synth, loss, representation_error, pixel_values]) == 0:
        # then there's nothing to plot here
        return
    included_plots = []
    plot_kwargs = {}
    # need to figure out which plotting function to call
    if isinstance(synthesis_object, po.synth.Metamer):
        containing_file = po.synth.metamer
        as_rgb = synthesis_object.image.shape[1] > 1
        plot_kwargs["plot_representation_error_as_rgb"] = as_rgb
        if display_synth:
            included_plots.append("display_metamer")
    elif isinstance(synthesis_object, po.synth.MADCompetition):
        containing_file = po.synth.mad_competition
        if display_synth:
            included_plots.append("display_mad_image")
    if loss:
        included_plots.append("plot_loss")
    if representation_error:
        included_plots.append("plot_representation_error")
    if pixel_values:
        included_plots.append("plot_pixel_values")
    width_ratios = {}
    if fig_creation.startswith("auto"):
        fig = None
        axes_idx = {}
        if fig_creation.endswith("ratios"):
            if loss:
                width_ratios["plot_loss"] = 2
            elif display_synth:
                if isinstance(synthesis_object, po.synth.Metamer):
                    width_ratios["display_metamer"] = 2
                elif isinstance(synthesis_object, po.synth.MADCompetition):
                    width_ratios["display_mad_image"] = 2
    elif fig_creation.startswith("pass"):
        fig, axes, axes_idx = containing_file._setup_synthesis_fig(
            None, {}, None, included_plots=included_plots
        )
        if fig_creation.endswith("without"):
            axes_idx = {}
    containing_file.plot_synthesis_status(
        synthesis_object,
        iteration=iteration,
        included_plots=included_plots,
        fig=fig,
        axes_idx=axes_idx,
        **plot_kwargs,
        width_ratios=width_ratios,
    )
    plt.close("all")


def template_test_synthesis_custom_fig(synthesis_object, func, fig_creation, tmp_path):
    # template function to test whether we can create our own figure and pass
    # it to the plotting and animating functions, specifying some or all of the
    # locations for the plots. Any synthesis object that has had synthesis()
    # called should work with this
    plot_kwargs = {}
    included_plots = ["plot_loss", "plot_pixel_values"]
    # need to figure out which plotting function to call
    if isinstance(synthesis_object, po.synth.Metamer):
        containing_file = po.synth.metamer
        as_rgb = synthesis_object.image.shape[1] > 1
        plot_kwargs["plot_representation_error_as_rgb"] = as_rgb
        included_plots.append("plot_representation_error")
        if fig_creation.endswith("extra"):
            included_plots.append("display_metamer")
        axes_idx = {"display_metamer": 0, "plot_representation_error": 8}
    elif isinstance(synthesis_object, po.synth.MADCompetition):
        containing_file = po.synth.mad_competition
        if fig_creation.endswith("extra"):
            included_plots.append("display_mad_image")
        axes_idx = {"display_mad_image": 0}
    fig, axes = plt.subplots(3, 3, figsize=(35, 17))
    if "-" in fig_creation:
        axes_idx["misc"] = [1, 4]
    if fig_creation.split("-")[-1] not in ["without"]:
        axes_idx.update({"plot_loss": 6, "plot_pixel_values": 7})
    if func == "plot" or fig_creation.endswith("preplot"):
        fig, axes_idx = containing_file.plot_synthesis_status(
            synthesis_object,
            included_plots=included_plots,
            fig=fig,
            axes_idx=axes_idx,
            **plot_kwargs,
        )
    if func == "animate":
        path = tmp_path / "test_anim.html"
        containing_file.animate(
            synthesis_object,
            fig=fig,
            axes_idx=axes_idx,
            included_plots=included_plots,
            **plot_kwargs,
        ).save(path)
        plt.close("all")


class TestMADDisplay:
    @pytest.fixture(scope="class", params=["rgb", "grayscale"])
    def synthesized_mad(self, request):
        # make the images really small so nothing takes as long
        if request.param == "rgb":
            img = po.load_images(IMG_DIR / "256" / "color_wheel.jpg", False).to(DEVICE)
            img = img[..., :16, :16]
        else:
            img = po.load_images(IMG_DIR / "256" / "nuts.pgm").to(DEVICE)
            img = img[..., :16, :16]

        # to serve as a metric, need to return a single value, but SSIM and MSE
        # will return a separate value for each RGB channel. Additionally, MAD
        # requires metrics are *dis*-similarity metrics, so that they return 0
        # if two images are identical (SSIM normally returns 1)
        def rgb_ssim(*args, **kwargs):
            return 1 - po.metric.ssim(*args, **kwargs).mean()

        def rgb_mse(*args, **kwargs):
            return po.metric.mse(*args, **kwargs).mean()

        mad = po.synth.MADCompetition(
            img, rgb_mse, rgb_ssim, "min", metric_tradeoff_lambda=0.1
        )
        mad.synthesize(max_iter=2, store_progress=True)
        return mad

    @pytest.mark.parametrize("iteration", [None, 1, -1])
    @pytest.mark.parametrize("display_mad", [True, False])
    @pytest.mark.parametrize("loss", [True, False])
    @pytest.mark.parametrize("pixel_values", [True, False])
    @pytest.mark.parametrize(
        "fig_creation", ["auto", "auto-ratios", "pass-with", "pass-without"]
    )
    def test_all_plot(
        self,
        synthesized_mad,
        iteration,
        display_mad,
        loss,
        pixel_values,
        fig_creation,
    ):
        # tests whether we can plot all possible combinations of plots.
        # test_custom_fig tests whether these animate correctly.
        template_test_synthesis_all_plot(
            synthesized_mad,
            iteration,
            display_mad,
            loss,
            False,
            pixel_values,
            fig_creation,
        )

    @pytest.mark.parametrize("func", ["plot", "animate"])
    @pytest.mark.parametrize(
        "fig_creation",
        [
            "custom",
            "custom-misc",
            "custom-without",
            "custom-extra",
            "custom-preplot",
        ],
    )
    def test_custom_fig(self, synthesized_mad, func, fig_creation, tmp_path):
        # tests whether we can create our own figure and pass it to
        # MADCompetition's plotting and animating functions, specifying some or
        # all of the locations for the plots
        template_test_synthesis_custom_fig(
            synthesized_mad, func, fig_creation, tmp_path
        )

    @pytest.fixture(scope="class")
    def all_mad(self):
        # run synthesis for all 4 MAD images.
        img = po.load_images(IMG_DIR / "256" / "nuts.pgm").to(DEVICE)
        img = img[..., :16, :16]
        model1 = po.metric.mse

        # MAD requires metrics are *dis*-similarity metrics, so that they
        # return 0 if two images are identical (SSIM normally returns 1)
        def model2(*args):
            return 1 - po.metric.ssim(*args).mean()

        mad = po.synth.MADCompetition(
            img, model1, model2, "min", metric_tradeoff_lambda=1
        )
        mad.synthesize(max_iter=2)
        mad2 = po.synth.MADCompetition(
            img, model1, model2, "max", metric_tradeoff_lambda=1
        )
        mad2.synthesize(max_iter=2)
        mad3 = po.synth.MADCompetition(
            img, model2, model1, "min", metric_tradeoff_lambda=1
        )
        mad3.synthesize(max_iter=2)
        mad4 = po.synth.MADCompetition(
            img, model2, model1, "max", metric_tradeoff_lambda=1
        )
        mad4.synthesize(max_iter=2)
        return mad, mad2, mad3, mad4

    @pytest.mark.parametrize("func", ["loss", "image"])
    def test_helper_funcs(self, all_mad, func):
        if func == "loss":
            func = po.synth.mad_competition.plot_loss_all
        elif func == "image":
            func = po.synth.mad_competition.display_mad_image_all
        func(*all_mad)

    @pytest.mark.parametrize("func", ["plot", "animate"])
    # plot_representation_error is an allowed value for metamer, but not MAD.
    # the second is just a typo
    @pytest.mark.parametrize("val", ["plot_representation_error", "plot_mad_image"])
    @pytest.mark.parametrize("variable", ["included_plots", "width_ratios", "axes_idx"])
    def test_allowed_plots_exception(self, synthesized_mad, func, val, variable):
        if func == "plot":
            func = po.synth.mad_competition.plot_synthesis_status
        elif func == "animate":
            func = po.synth.mad_competition.animate
        kwargs = {}
        if variable == "included_plots":
            kwargs["included_plots"] = [val, "plot_loss"]
        elif variable == "width_ratios":
            kwargs["width_ratios"] = {val: 1, "plot_loss": 1}
        elif variable == "axes_idx":
            kwargs["axes_idx"] = {val: 0, "plot_loss": 1}
        with pytest.raises(ValueError, match=f"{variable} contained value"):
            func(synthesized_mad, **kwargs)


class TestMetamerDisplay:
    @pytest.fixture(scope="class", params=["rgb", "grayscale"])
    def synthesized_met(self, request):
        img = request.param
        # make the images really small so nothing takes as long
        if img == "rgb":
            img = po.load_images(IMG_DIR / "256" / "color_wheel.jpg", False).to(DEVICE)
            img = img[..., :16, :16]
        else:
            img = po.load_images(IMG_DIR / "256" / "nuts.pgm").to(DEVICE)
            img = img[..., :16, :16]

        #  height=1 and order=0 to limit the time this takes, and then we
        #  only return one of the tensors so that everything is easy for
        #  plotting code to figure out (if we downsampled and were on an
        #  RGB image, we'd have a tensor of shape [1, 9, h, w], because
        #  we'd have the residuals and one filter output for each channel,
        #  and our code doesn't know how to handle that)
        class SPyr(po.simul.SteerablePyramidFreq):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def forward(self, *args, **kwargs):
                return super().forward(*args, **kwargs)[(0, 0)]

        model = SPyr(img.shape[-2:], height=1, order=1).to(DEVICE)
        met = po.synth.Metamer(img, model)
        met.synthesize(max_iter=2, store_progress=True)
        return met

    # mix together func and iteration, because iteration doesn't make sense to
    # pass to animate
    @pytest.mark.parametrize("iteration", [None, 1, -1])
    @pytest.mark.parametrize("display_metamer", [True, False])
    @pytest.mark.parametrize("loss", [True, False])
    @pytest.mark.parametrize("representation_error", [True, False])
    @pytest.mark.parametrize("pixel_values", [True, False])
    @pytest.mark.parametrize(
        "fig_creation", ["auto", "auto-ratios", "pass-with", "pass-without"]
    )
    def test_all_plot(
        self,
        synthesized_met,
        iteration,
        display_metamer,
        loss,
        representation_error,
        pixel_values,
        fig_creation,
    ):
        # tests whether we can plot all possible combinations of plots.
        # test_custom_fig tests whether these animate correctly.
        template_test_synthesis_all_plot(
            synthesized_met,
            iteration,
            display_metamer,
            loss,
            representation_error,
            pixel_values,
            fig_creation,
        )

    @pytest.mark.parametrize("func", ["plot", "animate"])
    @pytest.mark.parametrize(
        "fig_creation",
        [
            "custom",
            "custom-misc",
            "custom-without",
            "custom-extra",
            "custom-preplot",
        ],
    )
    def test_custom_fig(self, synthesized_met, func, fig_creation, tmp_path):
        # tests whether we can create our own figure and pass it to Metamer's
        # plotting and animating functions, specifying some or all of the
        # locations for the plots
        template_test_synthesis_custom_fig(
            synthesized_met, func, fig_creation, tmp_path
        )

    @pytest.mark.parametrize("func", ["plot", "animate"])
    # display_mad_image is an allowed value for MAD but not metamer.
    # the second is just a typo
    @pytest.mark.parametrize("val", ["display_mad_image", "plot_metamer"])
    @pytest.mark.parametrize("variable", ["included_plots", "width_ratios", "axes_idx"])
    def test_allowed_plots_exception(self, synthesized_met, func, val, variable):
        if func == "plot":
            func = po.synth.metamer.plot_synthesis_status
        elif func == "animate":
            func = po.synth.metamer.animate
        kwargs = {}
        if variable == "included_plots":
            kwargs["included_plots"] = [val, "plot_loss"]
        elif variable == "width_ratios":
            kwargs["width_ratios"] = {val: 1, "plot_loss": 1}
        elif variable == "axes_idx":
            kwargs["axes_idx"] = {val: 0, "plot_loss": 1}
        with pytest.raises(ValueError, match=f"{variable} contained value"):
            func(synthesized_met, **kwargs)
