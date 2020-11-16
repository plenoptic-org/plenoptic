#!/usr/bin/env python3

import pytest
import matplotlib.pyplot as plt
import plenoptic as po
import torch
import os.path as op
import numpy as np
import pyrtools as pt
from test_plenoptic import DEVICE, DATA_DIR


class TestDisplay(object):

    def test_update_plot_line(self):
        x = np.linspace(0, 100)
        y1 = np.random.rand(*x.shape)
        y2 = np.random.rand(*x.shape)
        fig, ax = plt.subplots(1, 1)
        ax.plot(x, y1, '-o', label='hi')
        po.update_plot(ax, torch.tensor(y2).reshape(1, 1, len(x)))
        assert len(ax.lines) == 1, "Too many lines were plotted!"
        _, ax_y = ax.lines[0].get_data()
        if not np.allclose(ax_y, y2):
            raise Exception("Didn't update line correctly!")
        plt.close('all')

    @pytest.mark.parametrize('how', ['dict', 'tensor'])
    def test_update_plot_line_multi_axes(self, how):
        x = np.linspace(0, 100)
        y1 = np.random.rand(*x.shape)
        y2 = np.random.rand(2, *y1.shape)
        if how == 'tensor':
            y2 = torch.tensor(y2).reshape(1, 2, *y1.shape)
        elif how == 'dict':
            y2 = {i: torch.tensor(y2[i]).reshape(1, 1, *y1.shape) for i in range(2)}
        fig, axes = plt.subplots(1, 2)
        for ax in axes:
            ax.plot(x, y1, '-o', label='hi')
        po.update_plot(axes, y2)
        for i, ax in enumerate(axes):
            assert len(ax.lines) == 1, "Too many lines were plotted!"
            _, ax_y = ax.lines[0].get_data()
            if how == 'tensor':
                y_check = y2[0, i]
            else:
                y_check = y2[i]
            if not np.allclose(ax_y, y_check):
                raise Exception("Didn't update line correctly!")
        plt.close('all')

    @pytest.mark.parametrize('how', ['dict-single', 'dict-multi', 'tensor'])
    def test_update_plot_line_multi_channel(self, how):
        if how == 'dict-single':
            n_data = 1
        else:
            n_data = 2
        x = np.linspace(0, 100)
        y1 = np.random.rand(2, *x.shape)
        y2 = np.random.rand(n_data, *x.shape)
        if how == 'tensor':
            y2 = torch.tensor(y2).reshape(1, 2, len(x))
        elif how == 'dict-multi':
            y2 = {i: torch.tensor(y2[i]).reshape(1, 1, len(x)) for i in range(2)}
        elif how == 'dict-single':
            y2 = {0: torch.tensor(y2[0]).reshape(1, 1, len(x))}
        fig, ax = plt.subplots(1, 1)
        for i in range(2):
            ax.plot(x, y1[i], label=i)
        po.update_plot(ax, y2)
        assert len(ax.lines) == 2, "Incorrect number of lines were plotted!"
        for i in range(2):
            _, ax_y = ax.lines[i].get_data()
            if how == 'tensor':
                y_check = y2[0, i]
            elif how == 'dict-multi':
                y_check = y2[i]
            elif how == 'dict-single':
                y_check = {0: y2[0], 1: y1[1]}[i]
            if not np.allclose(ax_y, y_check):
                raise Exception("Didn't update line correctly!")
        plt.close('all')

    def test_update_plot_stem(self):
        x = np.linspace(0, 100)
        y1 = np.random.rand(*x.shape)
        y2 = np.random.rand(*x.shape)
        fig, ax = plt.subplots(1, 1)
        ax.stem(x, y1, '-o', label='hi', use_line_collection=True)
        po.update_plot(ax, torch.tensor(y2).reshape(1, 1, len(x)))
        assert len(ax.containers) == 1, "Too many stems were plotted!"
        ax_y = ax.containers[0].markerline.get_ydata()
        if not np.allclose(ax_y, y2):
            raise Exception("Didn't update stems correctly!")
        plt.close('all')

    @pytest.mark.parametrize('how', ['dict', 'tensor'])
    def test_update_plot_stem_multi_axes(self, how):
        x = np.linspace(0, 100)
        y1 = np.random.rand(*x.shape)
        y2 = np.random.rand(2, *y1.shape)
        if how == 'tensor':
            y2 = torch.tensor(y2).reshape(1, 2, *y1.shape)
        elif how == 'dict':
            y2 = {i: torch.tensor(y2[i]).reshape(1, 1, *y1.shape) for i in range(2)}
        fig, axes = plt.subplots(1, 2)
        for ax in axes:
            ax.stem(x, y1, label='hi', use_line_collection=True)
        po.update_plot(axes, y2)
        for i, ax in enumerate(axes):
            assert len(ax.containers) == 1, "Too many stems were plotted!"
            ax_y = ax.containers[0].markerline.get_ydata()
            if how == 'tensor':
                y_check = y2[0, i]
            else:
                y_check = y2[i]
            if not np.allclose(ax_y, y_check):
                raise Exception("Didn't update stem correctly!")
        plt.close('all')

    @pytest.mark.parametrize('how', ['dict-single', 'dict-multi', 'tensor'])
    def test_update_plot_stem_multi_channel(self, how):
        if how == 'dict-single':
            n_data = 1
        else:
            n_data = 2
        x = np.linspace(0, 100)
        y1 = np.random.rand(2, *x.shape)
        y2 = np.random.rand(n_data, *x.shape)
        if how == 'tensor':
            y2 = torch.tensor(y2).reshape(1, 2, len(x))
        elif how == 'dict-multi':
            y2 = {i: torch.tensor(y2[i]).reshape(1, 1, len(x)) for i in range(2)}
        elif how == 'dict-single':
            y2 = {0: torch.tensor(y2[0]).reshape(1, 1, len(x))}
        fig, ax = plt.subplots(1, 1)
        for i in range(2):
            ax.stem(x, y1[i], label=i, use_line_collection=True)
        po.update_plot(ax, y2)
        assert len(ax.containers) == 2, "Incorrect number of stems were plotted!"
        for i in range(2):
            ax_y = ax.containers[i].markerline.get_ydata()
            if how == 'tensor':
                y_check = y2[0, i]
            elif how == 'dict-multi':
                y_check = y2[i]
            elif how == 'dict-single':
                y_check = {0: y2[0], 1: y1[1]}[i]
            if not np.allclose(ax_y, y_check):
                raise Exception("Didn't update stem correctly!")
        plt.close('all')

    def test_update_plot_image(self):
        y1 = np.random.rand(1, 1, 100, 100)
        y2 = np.random.rand(*y1.shape)
        fig = pt.imshow(y1.squeeze())
        ax = fig.axes[0]
        po.update_plot(ax, torch.tensor(y2))
        assert len(ax.images) == 1, "Too many images were plotted!"
        ax_y = ax.images[0].get_array().data
        if not np.allclose(ax_y, y2):
            raise Exception("Didn't update image correctly!")
        plt.close('all')

    @pytest.mark.parametrize('how', ['dict', 'tensor'])
    def test_update_plot_image_multi_axes(self, how):
        y1 = np.random.rand(1, 2, 100, 100)
        y2 = np.random.rand(1, 2, 100, 100)
        if how == 'tensor':
            y2 = torch.tensor(y2)
        elif how == 'dict':
            y2 = {i: torch.tensor(y2[0, i]).reshape(1, 1, 100, 100) for i in range(2)}
        fig = pt.imshow([y for y in y1.squeeze()])
        po.update_plot(fig.axes, y2)
        for i, ax in enumerate(fig.axes):
            assert len(ax.images) == 1, "Too many lines were plotted!"
            ax_y = ax.images[0].get_array().data
            if how == 'tensor':
                y_check = y2[0, i]
            else:
                y_check = y2[i]
            if not np.allclose(ax_y, y_check):
                raise Exception("Didn't update image correctly!")
        plt.close('all')

    def test_update_plot_scatter(self):
        x1 = np.random.rand(100)
        x2 = np.random.rand(100)
        y1 = np.random.rand(*x1.shape)
        y2 = np.random.rand(*x2.shape)
        fig, ax = plt.subplots(1, 1)
        ax.scatter(x1, y1)
        data = torch.stack((torch.tensor(x2), torch.tensor(y2)), -1).reshape(1, 1, len(x2), 2)
        po.update_plot(ax, data)
        assert len(ax.collections) == 1, "Too many scatter plots created"
        ax_data = ax.collections[0].get_offsets()
        if not np.allclose(ax_data, data):
            raise Exception("Didn't update points of the scatter plot correctly!")
        plt.close('all')

    @pytest.mark.parametrize('how', ['dict', 'tensor'])
    def test_update_plot_scatter_multi_axes(self, how):
        x1 = np.random.rand(100)
        x2 = np.random.rand(2, 100)
        y1 = np.random.rand(*x1.shape)
        y2 = np.random.rand(2, *y1.shape)
        if how == 'tensor':
            data = torch.stack((torch.tensor(x2), torch.tensor(y2)), -1).reshape(1, 2, len(x1), 2)
        elif how == 'dict':
            data = {i: torch.stack((torch.tensor(x2[i]), torch.tensor(y2[i])), -1).reshape(1, 1, len(x1), 2) for i in range(2)}
        fig, axes = plt.subplots(1, 2)
        for ax in axes:
            ax.scatter(x1, y1)
        po.update_plot(axes, data)
        for i, ax in enumerate(axes):
            assert len(ax.collections) == 1, "Too many scatter plots created"
            ax_data = ax.collections[0].get_offsets()
            if how == 'tensor':
                data_check = data[0, i]
            else:
                data_check = data[i]
            if not np.allclose(ax_data, data_check):
                raise Exception("Didn't update points of the scatter plot correctly!")
        plt.close('all')

    @pytest.mark.parametrize('how', ['dict-single', 'dict-multi', 'tensor'])
    def test_update_plot_scatter_multi_channel(self, how):
        if how == 'dict-single':
            n_data = 1
        else:
            n_data = 2
        x1 = np.random.rand(2, 100)
        x2 = np.random.rand(n_data, 100)
        y1 = np.random.rand(*x1.shape)
        y2 = np.random.rand(*x2.shape)
        if how == 'tensor':
            data = torch.stack((torch.tensor(x2), torch.tensor(y2)), -1).reshape(1, 2, x1.shape[-1], 2)
        elif how == 'dict-multi':
            data = {i: torch.stack((torch.tensor(x2[i]), torch.tensor(y2[i])), -1).reshape(1, 1, x1.shape[-1], 2) for i in range(2)}
        elif how == 'dict-single':
            data = {0: torch.stack((torch.tensor(x2[0]), torch.tensor(y2[0])), -1).reshape(1, 1, x1.shape[-1], 2)}
        fig, ax = plt.subplots(1, 1)
        for i in range(2):
            ax.scatter(x1[i], y1[i], label=i)
        po.update_plot(ax, data)
        assert len(ax.collections) == 2, "Incorrect number of scatter plots created"
        for i in range(2):
            ax_data = ax.collections[i].get_offsets()
            if how == 'tensor':
                data_check = data[0, i]
            elif how == 'dict-multi':
                data_check = data[i]
            elif how == 'dict-single':
                tmp = torch.stack((torch.tensor(x1), torch.tensor(y1)), -1)
                data_check = {0: data[0], 1: tmp[1]}[i]
            if not np.allclose(ax_data, data_check):
                raise Exception("Didn't update points of the scatter plot correctly!")

    def test_update_plot_mixed_multi_axes(self):
        x1 = np.linspace(0, 1, 100)
        x2 = np.random.rand(2, 100)
        y1 = np.random.rand(*x1.shape)
        y2 = np.random.rand(*x2.shape)
        data = {0: torch.stack((torch.tensor(x2[0]), torch.tensor(y2[0])),
                               -1).reshape(1, 1, x2.shape[-1], 2)}
        data[1] = torch.tensor(y2[1]).reshape(1, 1, x2.shape[-1])
        fig, axes = plt.subplots(1, 2)
        for i, ax in enumerate(axes):
            if i == 0:
                ax.scatter(x1, y1)
            else:
                ax.plot(x1, y1)
        po.update_plot(axes, data)
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
        plt.close('all')

    @pytest.mark.parametrize('as_rgb', [True, False])
    @pytest.mark.parametrize('channel_idx', [None, 0, [0, 1]])
    @pytest.mark.parametrize('batch_idx', [None, 0, [0, 1]])
    @pytest.mark.parametrize('is_complex', [False, 'logpolar', 'rectangular', 'polar'])
    @pytest.mark.parametrize('mini_im', [True, False])
    def test_imshow(self, as_rgb, channel_idx, batch_idx, is_complex, mini_im):
        fails = False
        if is_complex:
            im = torch.rand((2, 4, 10, 10, 2))
            # this is 2 (the two complex components) * 4 (the four channels) *
            # 2 (the two batches)
            n_axes = 16
        else:
            im = torch.rand((2, 4, 10, 10))
            # this is 4 (the four channels) * 2 (the two batches)
            n_axes = 8
        if mini_im:
            # n_axes here follows the same logic as above
            if is_complex:
                shape = [2, 4, 5, 5, 2]
                n_axes += 16
            else:
                shape = [2, 4, 5, 5]
                n_axes += 8
            im = [im, torch.rand(shape)]
        if not is_complex:
            # need to change this to one of the acceptable strings
            is_complex = 'rectangular'
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
            fig = po.imshow(im, as_rgb=as_rgb, channel_idx=channel_idx,
                            batch_idx=batch_idx, plot_complex=is_complex)
            assert len(fig.axes) == n_axes, f"Created {len(fig.axes)} axes, but expected {n_axes}! Probably plotting color as grayscale or vice versa"
            plt.close('all')
        if fails:
            with pytest.raises(Exception):
                po.imshow(im, as_rgb=as_rgb, channel_idx=channel_idx,
                          batch_idx=batch_idx, plot_complex=is_complex)

    @pytest.mark.parametrize('as_rgb', [True, False])
    @pytest.mark.parametrize('channel_idx', [None, 0, [0, 1]])
    @pytest.mark.parametrize('batch_idx', [None, 0, [0, 1]])
    @pytest.mark.parametrize('is_complex', [False, 'logpolar', 'rectangular', 'polar'])
    @pytest.mark.parametrize('mini_vid', [True, False])
    def test_animshow(self, as_rgb, channel_idx, batch_idx, is_complex, mini_vid):
        fails = False
        if is_complex:
            vid = torch.rand((2, 4, 10, 10, 10, 2))
            # this is 2 (the two complex components) * 4 (the four channels) *
            # 2 (the two batches)
            n_axes = 16
        else:
            vid = torch.rand((2, 4, 10, 10, 10))
            # this is 4 (the four channels) * 2 (the two batches)
            n_axes = 8
        if mini_vid:
            # n_axes here follows the same logic as above
            if is_complex:
                shape = [2, 4, 10, 5, 5, 2]
                n_axes += 16
            else:
                shape = [2, 4, 10, 5, 5]
                n_axes += 8
            vid = [vid, torch.rand(shape)]
        if not is_complex:
            # need to change this to one of the acceptable strings
            is_complex = 'rectangular'
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
            anim = po.animshow(vid, as_rgb=as_rgb, channel_idx=channel_idx,
                               batch_idx=batch_idx, plot_complex=is_complex)
            fig = anim._fig
            assert len(fig.axes) == n_axes, f"Created {len(fig.axes)} axes, but expected {n_axes}! Probably plotting color as grayscale or vice versa"
            plt.close('all')
        if fails:
            with pytest.raises(Exception):
                po.animshow(vid, as_rgb=as_rgb, channel_idx=channel_idx,
                            batch_idx=batch_idx, plot_complex=is_complex)


class TestMADDisplay(object):

    @pytest.mark.parametrize('func', ['plot', 'animate'])
    @pytest.mark.parametrize('iteration', [None, 1, -1])
    @pytest.mark.parametrize('plot_synthesized_image', [True, False])
    @pytest.mark.parametrize('plot_loss', [True, False])
    @pytest.mark.parametrize('plot_representation_error', [True, False])
    @pytest.mark.parametrize('plot_image_hist', [True, False])
    @pytest.mark.parametrize('plot_rep_comparison', [True, False])
    @pytest.mark.parametrize('plot_signal_comparison', [False, 'scatter', 'hist2d'])
    @pytest.mark.parametrize('fig_creation', ['auto', 'pass-with', 'pass-without'])
    def test_all_plot_animate(self, func, iteration, plot_synthesized_image,
                              plot_loss, plot_representation_error,
                              plot_image_hist, plot_rep_comparison,
                              plot_signal_comparison, fig_creation):
        img = po.load_images(op.join(DATA_DIR, 'nuts.pgm'))
        model1 = po.simul.models.naive.Identity().to(DEVICE)
        model2 = po.metric.nlpd
        func = 'scatter'
        if plot_signal_comparison:
            func = plot_signal_comparison
            plot_signal_comparison = True
        mad = po.synth.MADCompetition(img, model1, model2)
        if fig_creation == 'auto':
            fig = None
            axes_idx = {}
        elif fig_creation.startswith('pass'):
            fig, axes, axes_idx = mad._setup_synthesis_fig(None, {}, None)
            if fig_creation.endswith('without'):
                axes_idx = {}
        mad.synthesize('model_1_min', max_iter=3, store_progress=True)
        if func == 'plot':
            mad.plot_synthesis_status(iteration=iteration,
                                      plot_synthesized_image=plot_synthesized_image,
                                      plot_loss=plot_loss,
                                      plot_representation_error=plot_representation_error,
                                      plot_image_hist=plot_image_hist,
                                      plot_rep_comparison=plot_rep_comparison,
                                      plot_signal_comparison=plot_signal_comparison,
                                      signal_comp_func=func, fig=fig,
                                      axes_idx=axes_idx)
        elif func == 'animate':
            mad.animate(iteration=iteration,
                        plot_synthesized_image=plot_synthesized_image,
                        plot_loss=plot_loss,
                        plot_representation_error=plot_representation_error,
                        plot_image_hist=plot_image_hist,
                        plot_rep_comparison=plot_rep_comparison,
                        plot_signal_comparison=plot_signal_comparison,
                        signal_comp_func=func, fig=fig, axes_idx=axes_idx)

    @pytest.mark.parametrize('func', ['plot', 'animate'])
    @pytest.mark.parametrize('fig_creation', ['custom', 'custom-misc', 'custom-without',
                                              'custom-extra', 'custom-preplot'])
    def test_custom_fig(self, func, fig_creation):
        img = po.load_images(op.join(DATA_DIR, 'nuts.pgm'))
        model1 = po.simul.models.naive.Identity().to(DEVICE)
        model2 = po.metric.nlpd
        mad = po.synth.MADCompetition(img, model1, model2)
        init_fig = True
        fig, axes = plt.subplots(3, 3, figsize=(25, 17))
        axes_idx = {'image': 0, 'signal_comp': 2, 'rep_comp': 3,
                    'rep_error': 8}
        if '-' in fig_creation:
            axes_idx['misc'] = [1, 4]
        if not fig_creation.split('-')[-1] in ['without']:
            axes_idx.update({'loss': 6, 'hist': 7})
        if fig_creation.endswith('extra'):
            plot_synthesized_image = False
        else:
            plot_synthesized_image = True
        if fig_creation.endswith('preplot'):
            init_fig = False
        mad.synthesize('model_1_min', max_iter=3, store_progress=True)
        if func == 'plot' or fig_creation.endswith('preplot'):
            fig = mad.plot_synthesis_status(plot_synthesized_image=plot_synthesized_image,
                                            plot_loss=True,
                                            plot_representation_error=True,
                                            plot_image_hist=True,
                                            plot_rep_comparison=True,
                                            plot_signal_comparison=True, fig=fig,
                                            axes_idx=axes_idx)
        if func == 'animate':
            mad.animate(plot_synthesized_image=plot_synthesized_image,
                        plot_loss=True, plot_representation_error=True,
                        plot_image_hist=True, plot_rep_comparison=True,
                        plot_signal_comparison=True, fig=fig,
                        axes_idx=axes_idx, init_figure=init_fig)


class TestMetamerDisplay(object):

    @pytest.mark.parametrize('func', ['plot', 'animate'])
    @pytest.mark.parametrize('model', ['class', 'function'])
    @pytest.mark.parametrize('iteration', [None, 1, -1])
    @pytest.mark.parametrize('plot_synthesized_image', [True, False])
    @pytest.mark.parametrize('plot_loss', [True, False])
    @pytest.mark.parametrize('plot_representation_error', [True, False])
    @pytest.mark.parametrize('plot_image_hist', [True, False])
    @pytest.mark.parametrize('plot_rep_comparison', [True, False])
    @pytest.mark.parametrize('plot_signal_comparison', [False, 'scatter', 'hist2d'])
    @pytest.mark.parametrize('fig_creation', ['auto', 'pass-with', 'pass-without'])
    def test_all_plot_animate(self, func, model, iteration,
                              plot_synthesized_image, plot_loss,
                              plot_representation_error, plot_image_hist,
                              plot_rep_comparison, plot_signal_comparison,
                              fig_creation):
        im = po.load_images(op.join(DATA_DIR, 'nuts.pgm'))
        if model == 'class':
            model = po.simul.PooledV1(.5, im.shape[2:]).to(DEVICE)
        else:
            model = po.metric.nlpd
        func = 'scatter'
        if plot_signal_comparison:
            func = plot_signal_comparison
            plot_signal_comparison = True
        met = po.synth.Metamer(im, model)
        if fig_creation == 'auto':
            fig = None
            axes_idx = {}
        elif fig_creation.startswith('pass'):
            fig, axes, axes_idx = met._setup_synthesis_fig(None, {}, None)
            if fig_creation.endswith('without'):
                axes_idx = {}
        met.synthesize(max_iter=3, store_progress=True)
        if func == 'plot':
            met.plot_synthesis_status(iteration=iteration,
                                      plot_synthesized_image=plot_synthesized_image,
                                      plot_loss=plot_loss,
                                      plot_representation_error=plot_representation_error,
                                      plot_image_hist=plot_image_hist,
                                      plot_rep_comparison=plot_rep_comparison,
                                      plot_signal_comparison=plot_signal_comparison,
                                      signal_comp_func=func, fig=fig,
                                      axes_idx=axes_idx)
        elif func == 'animate':
            met.animate(iteration=iteration,
                        plot_synthesized_image=plot_synthesized_image,
                        plot_loss=plot_loss,
                        plot_representation_error=plot_representation_error,
                        plot_image_hist=plot_image_hist,
                        plot_rep_comparison=plot_rep_comparison,
                        plot_signal_comparison=plot_signal_comparison,
                        signal_comp_func=func, fig=fig, axes_idx=axes_idx)


    @pytest.mark.parametrize('func', ['plot', 'animate'])
    @pytest.mark.parametrize('fig_creation', ['custom', 'custom-misc', 'custom-without',
                                              'custom-extra', 'custom-preplot'])
    def test_custom_fig(self, func, fig_creation):
        im = po.load_images(op.join(DATA_DIR, 'nuts.pgm'))
        model = po.simul.PooledV1(.5, im.shape[2:]).to(DEVICE)
        met = po.synth.Metamer(im, model)
        init_fig = True
        fig, axes = plt.subplots(3, 3, figsize=(17, 17))
        axes_idx = {'image': 0, 'signal_comp': 2, 'rep_comp': 3,
                    'rep_error': 8}
        if '-' in fig_creation:
            axes_idx['misc'] = [1, 4]
        if not fig_creation.split('-')[-1] in ['without']:
            axes_idx.update({'loss': 6, 'hist': 7})
        if fig_creation.endswith('extra'):
            plot_synthesized_image = False
        else:
            plot_synthesized_image = True
        if fig_creation.endswith('preplot'):
            init_fig = False
        met.synthesize(max_iter=3, store_progress=True)
        if func == 'plot' or fig_creation.endswith('preplot'):
            fig = met.plot_synthesis_status(plot_synthesized_image=plot_synthesized_image,
                                            plot_loss=True,
                                            plot_representation_error=True,
                                            plot_image_hist=True,
                                            plot_rep_comparison=True,
                                            plot_signal_comparison=True, fig=fig,
                                            axes_idx=axes_idx)
        if func == 'animate':
            met.animate(plot_synthesized_image=plot_synthesized_image,
                        plot_loss=True, plot_representation_error=True,
                        plot_image_hist=True, plot_rep_comparison=True,
                        plot_signal_comparison=True, fig=fig,
                        axes_idx=axes_idx, init_figure=init_fig)
