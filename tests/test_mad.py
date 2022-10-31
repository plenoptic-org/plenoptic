# necessary to avoid issues with animate:
# https://github.com/matplotlib/matplotlib/issues/10287/
import matplotlib as mpl
# use the html backend, so we don't need to have ffmpeg
mpl.rcParams['animation.writer'] = 'html'
mpl.use('agg')
import pytest
import matplotlib.pyplot as plt
import plenoptic as po
import torch
import os.path as op
from conftest import DEVICE, DATA_DIR, get_model
import numpy as np

class TestMAD(object):

    @pytest.mark.parametrize('target', ['min', 'max'])
    @pytest.mark.parametrize('model_order', ['mse-ssim', 'ssim-mse'])
    @pytest.mark.parametrize('store_progress', [False, True, 2])
    def test_basic(self, curie_img, target, model_order, store_progress):
        if model_order == 'mse-ssim':
            model = po.metric.mse
            model2 = lambda *args: 1 - po.metric.ssim(*args)
        elif model_order == 'ssim-mse':
            model = lambda *args: 1 - po.metric.ssim(*args)
            model2 = po.metric.mse
        mad = po.synth.MADCompetition(curie_img, model, model2, target)
        mad.synthesize(max_iter=5, store_progress=store_progress)
        if store_progress:
            mad.synthesize(max_iter=5, store_progress=store_progress)

    @pytest.mark.parametrize('fail', [False, 'img', 'metric1', 'metric2', 'target'])
    @pytest.mark.parametrize('rgb', [False, True])
    @pytest.mark.parametrize('model', ['ColorModel'], indirect=True)
    def test_save_load(self, curie_img, fail, rgb, model, tmp_path):
        # this works with either rgb or grayscale images
        def metric(*args):
            return po.metric.mse(*args).mean()
        if rgb:
            curie_img = curie_img.repeat(1, 3, 1, 1)
            def metric2(x1, x2):
                return po.metric.mse(model(x1), model(x2)).mean()
        # MAD requires metrics are *dis*-similarity metrics, so that they
        # return 0 if two images are identical (SSIM normally returns 1)
        else:
            def metric2(*args):
                return 1-po.metric.ssim(*args)
        target = 'min'
        mad = po.synth.MADCompetition(curie_img, metric, metric2, target)
        mad.synthesize(max_iter=4, store_progress=True)
        mad.save(op.join(tmp_path, 'test_mad_save_load.pt'))
        if fail:
            if fail == 'img':
                curie_img = torch.rand_like(curie_img)
            elif fail == 'metric1':
                # this works with either rgb or grayscale images (though note
                # that SSIM just operates on each RGB channel independently,
                # which is probably not the right thing to do)
                def metric(x1, x2):
                    return 2*(1 - po.metric.ssim(x1, x2)).mean()
            elif fail == 'metric2':
                # this works with either rgb or grayscale images
                def metric2(*args):
                    return po.metric.mse(*args).mean()
            elif fail == 'target':
                target = 'max'
            mad_copy = po.synth.MADCompetition(curie_img, metric, metric2, target)
            with pytest.raises(Exception):
                mad_copy.load(op.join(tmp_path, "test_mad_save_load.pt"),
                              map_location=DEVICE)
        else:
            mad_copy = po.synth.MADCompetition(curie_img, metric, metric2, target)
            mad_copy.load(op.join(tmp_path, "test_mad_save_load.pt"), map_location=DEVICE)
            # check that can resume
            mad_copy.synthesize(max_iter=5, store_progress=True)
        if rgb:
            # since this is a fixture, get this back to a grayscale image
            curie_img = curie_img.mean(1, True)

    @pytest.mark.parametrize('optimizer', ['Adam', None, 'Scheduler'])
    def test_optimizer_opts(self, curie_img, optimizer):
        mad = po.synth.MADCompetition(curie_img, po.metric.mse,
                                      lambda *args: 1-po.metric.ssim(*args),
                                      'min')
        scheduler = None
        if optimizer == 'Adam' or optimizer == 'Scheduler':
            optimizer = torch.optim.Adam([mad.synthesized_signal])
            if optimizer == 'Scheduler':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        mad.synthesize(max_iter=5, optimizer=optimizer, scheduler=scheduler)

    @pytest.mark.parametrize('model', ['ssim', 'class'])
    def test_require_metric(self, curie_img, model):
        # test that we fail if we get a model or a function that's not a metric
        # (i.e., doesn't return 0 on identical images)
        if model == 'ssim':
            model = po.metric.ssim
        elif model == 'class':
            model = po.simul.OnOff((8, 8))
        with pytest.raises(Exception):
            po.synth.MADCompetition(curie_img, po.metric.mse)

    @pytest.mark.parametrize('to_type', ['dtype', 'device'])
    def test_to(self, curie_img, to_type):
        mad = po.synth.MADCompetition(curie_img, po.metric.mse,
                                      po.tools.optim.l2_norm, 'min')
        mad.synthesize(max_iter=5)
        if to_type == 'dtype':
            mad.to(torch.float16)
            assert mad.initial_signal.dtype == torch.float16
            assert mad.reference_signal.dtype == torch.float16
            assert mad.synthesized_signal.dtype == torch.float16
        # can only run this one if we're on a device with CPU and GPU.
        elif to_type == 'device' and DEVICE.type != 'cpu':
            mad.to('cpu')
        # initial_signal doesn't get used anywhere after init, so check it like
        # this
        mad.initial_signal - mad.reference_signal
        mad.synthesized_signal - mad.reference_signal


    def test_map_location(self, curie_img, tmp_path):
        # only run this test if we have a gpu available
        if DEVICE.type != 'cpu':
            curie_img = curie_img.to(DEVICE)
            mad = po.synth.MADCompetition(curie_img, po.metric.mse,
                                          po.tools.optim.l2_norm, 'min')
            mad.synthesize(max_iter=4, store_progress=True)
            mad.save(op.join(tmp_path, 'test_mad_map_location.pt'))
            curie_img = curie_img.to('cpu')
            mad_copy = po.synth.MADCompetition(curie_img, po.metric.mse,
                                               po.tools.optim.l2_norm, 'min')
            assert mad_copy.reference_signal.device.type == 'cpu'
            mad_copy.load(op.join(tmp_path, 'test_mad_map_location.pt'),
                          map_location='cpu')
            assert mad_copy.synthesized_signal.device.type == 'cpu'
