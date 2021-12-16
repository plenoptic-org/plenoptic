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


class TestMAD(object):

    @pytest.mark.parametrize('target', ['min', 'max'])
    @pytest.mark.parametrize('model_order', ['mse-nlpd', 'nlpd-mse'])
    @pytest.mark.parametrize('store_progress', [False, True, 2])
    def test_basic(self, curie_img, target, model_order, store_progress):
        if model_order == 'mse-nlpd':
            model = po.metric.mse
            model2 = lambda *args: 1 - po.metric.ssim(*args)
        elif model_order == 'nlpd-mse':
            model = lambda *args: 1 - po.metric.ssim(*args)
            model2 = po.metric.mse
        mad = po.synth.MADCompetition(curie_img, model, model2, target)
        mad.synthesize(max_iter=5, store_progress=store_progress)
        if store_progress:
            mad.synthesize(max_iter=5, store_progress=store_progress)
           
    @pytest.mark.parametrize('fail', [False, 'img', 'model1', 'model2', 'target'])
    def test_save_load(self, curie_img, fail, tmp_path):
        # for ease of use
        model = po.metric.mse
        # MAD requires metrics are *dis*-similarity metrics, so that they
        # return 0 if two images are identical (SSIM normally returns 1)
        model2 = lambda *args: 1-po.metric.ssim(*args)
        target = 'min'
        mad = po.synth.MADCompetition(curie_img, model, model2, target)
        mad.synthesize(max_iter=4, store_progress=True)
        mad.save(op.join(tmp_path, 'test_mad_save_load.pt'))
        # when the model is a function, the loss_function is ignored and thus
        # we won't actually fail to load here (we check against the specific
        # callable because we've overwritten the model input arg)
        if fail:
            if fail == 'img':
                curie_img = torch.rand_like(curie_img)
            elif fail == 'model1':
                model = lambda *args: 2*(1 - po.metric.ssim(*args))
            elif fail == 'model2':
                model2 = po.metric.mse
            elif fail == 'target':
                target = 'max'
            mad_copy = po.synth.MADCompetition(curie_img, model, model2, target)
            with pytest.raises(Exception):
                mad_copy.load(op.join(tmp_path, "test_mad_save_load.pt"),
                              map_location=DEVICE)
        else:
            mad_copy = po.synth.MADCompetition(curie_img, model, model2, target)
            mad_copy.load(op.join(tmp_path, "test_mad_save_load.pt"), map_location=DEVICE)
            # check that can resume
            mad_copy.synthesize(max_iter=5, store_progress=True)

    @pytest.mark.parametrize('optimizer', ['Adam', None, 'Scheduler'])
    def test_optimizer_opts(self, curie_img, optimizer):
        mad = po.synth.MADCompetition(curie_img, po.metric.mse, lambda *args:
                                      1-po.metric.ssim(*args), 'min')
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
