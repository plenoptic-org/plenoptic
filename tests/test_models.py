#!/usr/bin/env python3
import os.path as op
import torch
import plenoptic as po
import matplotlib.pyplot as plt
import pytest
from test_plenoptic import DEVICE, DATA_DIR, DTYPE


class TestLinear(object):

    def test_linear(self):
        model = po.simul.Linear()
        x = po.make_basic_stimuli()
        assert model(x).requires_grad

    def test_linear_metamer(self):
        model = po.simul.Linear()
        image = plt.imread(op.join(DATA_DIR, 'nuts.pgm')).astype(float) / 255.
        im0 = torch.tensor(image, requires_grad=True, dtype=DTYPE).squeeze().unsqueeze(0).unsqueeze(0)
        M = po.synth.Metamer(im0, model)
        synthesized_signal, synthesized_representation = M.synthesize(max_iter=3, learning_rate=1, seed=1)


class TestLinearNonlinear(object):

    def test_linear_nonlinear(self):
        model = po.simul.Linear_Nonlinear()
        x = po.make_basic_stimuli()
        assert model(x).requires_grad

    def test_linear_nonlinear_metamer(self):
        model = po.simul.Linear_Nonlinear()
        image = plt.imread(op.join(DATA_DIR, 'metal.pgm')).astype(float) / 255.
        im0 = torch.tensor(image,requires_grad=True,dtype = torch.float32).squeeze().unsqueeze(0).unsqueeze(0)
        M = po.synth.Metamer(im0, model)
        synthesized_signal, synthesized_representation = M.synthesize(max_iter=3, learning_rate=1,seed=0)


# class TestConv(object):
# TODO expand, arbitrary shapes, dim


class TestLaplacianPyramid(object):

    def test_grad(self):
        L = po.simul.Laplacian_Pyramid()
        y = L.analysis(po.make_basic_stimuli())
        assert y[0].requires_grad


class TestPooledVentralStream(object):

    def test_rgc(self):
        im = po.load_images(op.join(DATA_DIR, 'nuts.pgm'))
        rgc = po.simul.PooledRGC(.5, im.shape[2:])
        rgc = rgc.to(DEVICE)
        rgc(im)
        _ = rgc.plot_window_widths('degrees')
        _ = rgc.plot_window_widths('degrees', jitter=0)
        _ = rgc.plot_window_widths('pixels')
        _ = rgc.plot_window_areas('degrees')
        _ = rgc.plot_window_areas('degrees')
        _ = rgc.plot_window_areas('pixels')
        fig = po.imshow(im)
        _ = rgc.plot_windows(fig.axes[0])
        rgc.plot_representation()
        rgc.plot_representation_image()
        fig, axes = plt.subplots(2, 1, figsize=(5, 12))
        rgc.plot_representation(ax=axes[1])
        rgc.plot_representation_image(ax=axes[0])
        plt.close('all')

    def test_rgc_2(self):
        im = po.load_images(op.join(DATA_DIR, 'nuts.pgm'))
        rgc = po.simul.PooledRGC(.5, im.shape[2:], transition_region_width=1)
        rgc = rgc.to(DEVICE)
        rgc(im)
        _ = rgc.plot_window_widths('degrees')
        _ = rgc.plot_window_widths('degrees', jitter=0)
        _ = rgc.plot_window_widths('pixels')
        _ = rgc.plot_window_areas('degrees')
        _ = rgc.plot_window_areas('degrees')
        _ = rgc.plot_window_areas('pixels')
        fig = po.imshow(im)
        _ = rgc.plot_windows(fig.axes[0])
        rgc.plot_representation()
        rgc.plot_representation_image()
        fig, axes = plt.subplots(2, 1, figsize=(5, 12))
        rgc.plot_representation(ax=axes[1])
        rgc.plot_representation_image(ax=axes[0])
        plt.close('all')

    def test_rgc_metamer(self):
        # literally just testing that it runs
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        rgc = po.simul.PooledRGC(.5, im.shape[2:])
        rgc = rgc.to(DEVICE)
        metamer = po.synth.Metamer(im, rgc)
        metamer.synthesize(max_iter=3)
        assert not torch.isnan(metamer.synthesized_signal).any(), "There's a NaN here!"

    def test_rgc_save_load(self, tmp_path):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
        # first time we cache the windows...
        rgc = po.simul.PooledRGC(.5, im.shape[2:], cache_dir=tmp_path)
        rgc = rgc.to(DEVICE)
        rgc(im)
        rgc.save_reduced(op.join(tmp_path, 'test_rgc_save_load.pt'))
        rgc_copy = po.simul.PooledRGC.load_reduced(op.join(tmp_path,
                                                                      'test_rgc_save_load.pt'))
        rgc_copy = rgc_copy.to(DEVICE)
        if not len(rgc.PoolingWindows.angle_windows) == len(rgc_copy.PoolingWindows.angle_windows):
            raise Exception("Something went wrong saving and loading, the lists of angle windows"
                            " are not the same length!")
        if not len(rgc.PoolingWindows.ecc_windows) == len(rgc_copy.PoolingWindows.ecc_windows):
            raise Exception("Something went wrong saving and loading, the lists of ecc windows"
                            " are not the same length!")
        # we don't recreate everything, e.g., the representation, but windows is the most important
        for i in range(len(rgc.PoolingWindows.angle_windows)):
            if not rgc.PoolingWindows.angle_windows[i].allclose(rgc_copy.PoolingWindows.angle_windows[i]):
                raise Exception("Something went wrong saving and loading, the angle_windows %d are"
                                " not identical!" % i)
        for i in range(len(rgc.PoolingWindows.ecc_windows)):
            if not rgc.PoolingWindows.ecc_windows[i].allclose(rgc_copy.PoolingWindows.ecc_windows[i]):
                raise Exception("Something went wrong saving and loading, the ecc_windows %d are"
                                " not identical!" % i)
        # ...second time we load them
        rgc = po.simul.PooledRGC(.5, im.shape[2:], cache_dir=tmp_path)

    @pytest.mark.parametrize('pretrained', [True, False])
    @pytest.mark.parametrize('requires_grad', [True, False])
    def test_frontend(self, pretrained, requires_grad):
        im = po.make_basic_stimuli()
        frontend = po.simul.FrontEnd(pretrained=pretrained, requires_grad=requires_grad)
        frontend(im)

    @pytest.mark.parametrize('pretrained', [True, False])
    @pytest.mark.parametrize('requires_grad', [True, False])
    def test_frontend_display_filters(self, pretrained, requires_grad):
        frontend = po.simul.FrontEnd(pretrained=pretrained, requires_grad=requires_grad)
        frontend.display_filters()
        plt.close('all')

    @pytest.mark.parametrize('requires_grad', [True, False])
    def test_frontend_plot(self, requires_grad):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        frontend = po.simul.FrontEnd(pretrained=True, requires_grad=requires_grad)
        po.tools.display.plot_representation(data=frontend(im), figsize=(11, 5))
        metamer = po.synth.Metamer(im, frontend)
        metamer.synthesize(max_iter=3, store_progress=1)
        metamer.plot_synthesis_status(figsize=(35, 5))
        metamer.animate(figsize=(35, 5))
        plt.close('all')

    @pytest.mark.parametrize('requires_grad', [True, False])
    def test_frontend_PoolingWindows(self, requires_grad):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        frontend = po.simul.FrontEnd(pretrained=True, requires_grad=requires_grad)
        pw = po.simul.PoolingWindows(.5, (256, 256))
        pw(frontend(im))
        po.tools.display.plot_representation(data=pw(frontend(im)))
        plt.close('all')

    def test_v1(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PooledV1(.5, im.shape[2:])
        v1 = v1.to(DEVICE)
        v1(im)
        _ = v1.plot_window_widths('pixels')
        _ = v1.plot_window_areas('pixels')
        for i in range(v1.num_scales):
            _ = v1.plot_window_widths('pixels', i)
            _ = v1.plot_window_areas('pixels', i)
        v1.plot_representation()
        v1.plot_representation_image()
        fig, axes = plt.subplots(2, 1, figsize=(27, 12))
        v1.plot_representation(ax=axes[1])
        v1.plot_representation_image(ax=axes[0])
        plt.close('all')

    def test_v1_norm(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PooledV1(.5, im.shape[2:])
        stats = po.optim.generate_norm_stats(v1, DATA_DIR, img_shape=(256, 256))
        v1 = po.simul.PooledV1(.5, im.shape[2:], normalize_dict=stats)
        v1 = v1.to(DEVICE)
        v1(im)
        _ = v1.plot_window_widths('pixels')
        _ = v1.plot_window_areas('pixels')
        for i in range(v1.num_scales):
            _ = v1.plot_window_widths('pixels', i)
            _ = v1.plot_window_areas('pixels', i)
        v1.plot_representation()
        v1.plot_representation_image()
        fig, axes = plt.subplots(2, 1, figsize=(27, 12))
        v1.plot_representation(ax=axes[1])
        v1.plot_representation_image(ax=axes[0])
        plt.close('all')

    def test_v1_2(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PooledV1(.5, im.shape[2:], transition_region_width=1)
        v1 = v1.to(DEVICE)
        v1(im)
        _ = v1.plot_window_widths('pixels')
        _ = v1.plot_window_areas('pixels')
        for i in range(v1.num_scales):
            _ = v1.plot_window_widths('pixels', i)
            _ = v1.plot_window_areas('pixels', i)
        v1.plot_representation()
        v1.plot_representation_image()
        fig, axes = plt.subplots(2, 1, figsize=(27, 12))
        v1.plot_representation(ax=axes[1])
        v1.plot_representation_image(ax=axes[0])
        plt.close('all')

    def test_v1_mean_luminance(self):
        for fname in ['nuts', 'einstein']:
            im = plt.imread(op.join(DATA_DIR, fname+'.pgm'))
            im = torch.tensor(im, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
            v1 = po.simul.PooledV1(.5, im.shape[2:])
            v1 = v1.to(DEVICE)
            v1_rep = v1(im)
            rgc = po.simul.PooledRGC(.5, im.shape[2:])
            rgc = rgc.to(DEVICE)
            rgc_rep = rgc(im)
            if not torch.allclose(rgc.representation['mean_luminance'], v1.mean_luminance):
                raise Exception("Somehow RGC and V1 mean luminance representations are not the "
                                "same for image %s!" % fname)
            if not torch.allclose(rgc_rep, v1_rep[..., -rgc_rep.shape[-1]:]):
                raise Exception("Somehow V1's representation does not have the mean luminance "
                                "in the location expected! for image %s!" % fname)

    def test_v1_save_load(self, tmp_path):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
        # first time we cache the windows...
        v1 = po.simul.PooledV1(.5, im.shape[2:], cache_dir=tmp_path)
        v1 = v1.to(DEVICE)
        v1(im)
        v1.save_reduced(op.join(tmp_path, 'test_v1_save_load.pt'))
        v1_copy = po.simul.PooledV1.load_reduced(op.join(tmp_path,
                                                                    'test_v1_save_load.pt'))
        v1_copy = v1_copy.to(DEVICE)
        if not len(v1.PoolingWindows.angle_windows) == len(v1_copy.PoolingWindows.angle_windows):
            raise Exception("Something went wrong saving and loading, the lists of angle windows"
                            " are not the same length!")
        if not len(v1.PoolingWindows.ecc_windows) == len(v1_copy.PoolingWindows.ecc_windows):
            raise Exception("Something went wrong saving and loading, the lists of ecc windows"
                            " are not the same length!")
        # we don't recreate everything, e.g., the representation, but windows is the most important
        for i in range(len(v1.PoolingWindows.angle_windows)):
            if not v1.PoolingWindows.angle_windows[i].allclose(v1_copy.PoolingWindows.angle_windows[i]):
                raise Exception("Something went wrong saving and loading, the angle_windows %d are"
                                " not identical!" % i)
        for i in range(len(v1.PoolingWindows.ecc_windows)):
            if not v1.PoolingWindows.ecc_windows[i].allclose(v1_copy.PoolingWindows.ecc_windows[i]):
                raise Exception("Something went wrong saving and loading, the ecc_windows %d are"
                                " not identical!" % i)
        # ...second time we load them
        v1 = po.simul.PooledV1(.5, im.shape[2:], cache_dir=tmp_path)

    @pytest.mark.parametrize('window', ['cosine', 'gaussian'])
    def test_v1_scales(self, window):
        im = po.load_images(op.join(DATA_DIR, 'einstein.pgm'))
        v1 = po.simul.PooledV1(1, im.shape[-2:], std_dev=1, window_type=window)
        lum_rep = v1(im, ['mean_luminance'])
        more_rep = v1(im, ['mean_luminance', 0])
        if lum_rep.numel() >= more_rep.numel():
            raise Exception("V1 not properly restricting output!")
        if any([(i, 0) in v1.representation.keys() for i in [1, 2, 3]]):
            raise Exception("Extra keys are showing up in v1.representation!")
        if lum_rep.numel() != v1(im, ['mean_luminance']).numel():
            raise Exception("V1 is not dropping unnecessary output!")

    def test_v1_metamer(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PooledV1(.5, im.shape[2:])
        v1 = v1.to(DEVICE)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=3)
