#!/usr/bin/env python3

# necessary to avoid issues with animate:
# https://github.com/matplotlib/matplotlib/issues/10287/
import matplotlib
matplotlib.use('agg')
import os.path as op
import torch
import plenoptic as po
import pytest
from conftest import DEVICE


# in order for pickling to work with functions, they must be defined at top of
# module: https://stackoverflow.com/a/36995008
def custom_loss(x1, x2):
    return (x1-x2).sum()


class TestMetamers(object):

    @pytest.mark.parametrize('model', ['frontend.LinearNonlinear'], indirect=True)
    @pytest.mark.parametrize('loss_func', ['mse', 'l2', 'custom'])
    @pytest.mark.parametrize('fail', [False, 'img', 'model', 'loss', 'range_penalty'])
    @pytest.mark.parametrize('range_penalty', [.1, 0])
    def test_metamer_save_load(self, einstein_img, model, loss_func, fail, range_penalty, tmp_path):
        if loss_func == 'mse':
            loss = po.tools.optim.mse
        elif loss_func == 'l2':
            loss = po.tools.optim.l2_norm
        elif loss_func == 'custom':
            loss = custom_loss
        met = po.synth.Metamer(einstein_img, model, loss_function=loss,
                               range_penalty_lambda=range_penalty)
        met.synthesize(max_iter=4, store_progress=True)
        met.save(op.join(tmp_path, 'test_metamer_save_load.pt'))
        if fail:
            if fail == 'img':
                einstein_img = torch.rand_like(einstein_img)
            elif fail == 'model':
                model = po.simul.Gaussian(30).to(DEVICE)
            elif fail == 'loss':
                loss = po.metric.ssim
            elif fail == 'range_penalty':
                range_penalty = .5
            met_copy = po.synth.Metamer(einstein_img, model,
                                        loss_function=loss,
                                        range_penalty_lambda=range_penalty)
            with pytest.raises(Exception):
                met_copy.load(op.join(tmp_path, "test_metamer_save_load.pt"),
                              map_location=DEVICE)
        else:
            met_copy = po.synth.Metamer(einstein_img, model,
                                        loss_function=loss,
                                        range_penalty_lambda=range_penalty)
            met_copy.load(op.join(tmp_path, "test_metamer_save_load.pt"),
                          map_location=DEVICE)
            for k in ['image', 'saved_metamer', 'metamer', 'target_representation']:
                if not getattr(met, k).allclose(getattr(met_copy, k), rtol=1e-2):
                    raise Exception("Something went wrong with saving and loading! %s not the same"
                                    % k)
            # check loss functions correctly saved
            met_loss = met.loss_function(met.model(met.metamer),
                                         met.target_representation)
            met_copy_loss = met_copy.loss_function(met.model(met.metamer),
                                                   met_copy.target_representation)
            if not torch.allclose(met_loss, met_copy_loss, rtol=1E-2):
                raise Exception(f"Loss function not properly saved! Before saving was {met_loss}, "
                                f"after loading was {met_copy_loss}")
            # check that can resume
            met_copy.synthesize(max_iter=4, store_progress=True,)

    @pytest.mark.parametrize('model', ['frontend.LinearNonlinear'], indirect=True)
    @pytest.mark.parametrize('store_progress', [True, 2, 3])
    def test_metamer_store_rep(self, einstein_img, model, store_progress):
        metamer = po.synth.Metamer(einstein_img, model)
        max_iter = 3
        if store_progress == 3:
            max_iter = 6
        metamer.synthesize(max_iter=max_iter, store_progress=store_progress)
        # we initialize saved_metamer the first time it's called, so it will
        # have 1 extra saved
        assert len(metamer.saved_metamer) == (max_iter//store_progress)+1, "Didn't end up with enough saved signal!"

    @pytest.mark.parametrize('model', ['frontend.LinearNonlinear'], indirect=True)
    def test_metamer_continue(self, einstein_img, model):
        metamer = po.synth.Metamer(einstein_img, model)
        metamer.synthesize(max_iter=3, store_progress=True)
        metamer.synthesize(max_iter=3, store_progress=True)

    @pytest.mark.parametrize('model', ['SPyr'], indirect=True)
    @pytest.mark.parametrize('coarse_to_fine', ['separate', 'together'])
    def test_coarse_to_fine(self, einstein_img, model, coarse_to_fine, tmp_path):
        metamer = po.synth.Metamer(einstein_img, model)
        metamer.synthesize(max_iter=5, stop_iters_to_check=1, coarse_to_fine=coarse_to_fine,
                           coarse_to_fine_kwargs={'change_scale_criterion': 10,
                                                  'ctf_iters_to_check': 1})
        assert len(metamer.scales_finished) > 0, "Didn't actually switch scales!"

        metamer.save(op.join(tmp_path, 'test_metamer_ctf.pt'))
        metamer_copy = po.synth.Metamer(einstein_img, model)
        metamer_copy.load(op.join(tmp_path, "test_metamer_ctf.pt"),
                          map_location=DEVICE)
        # check the ctf-related attributes all saved correctly
        for k in ['coarse_to_fine', 'scales', 'scales_loss', 'scales_timing',
                  'scales_finished']:
            if not getattr(metamer, k) == (getattr(metamer_copy, k)):
                raise Exception("Something went wrong with saving and loading! %s not the same"
                                % k)
        # check we can resume
        metamer.synthesize(max_iter=5, stop_iters_to_check=1, coarse_to_fine=coarse_to_fine,
                           coarse_to_fine_kwargs={'change_scale_criterion': 10,
                                                  'ctf_iters_to_check': 1})

    @pytest.mark.parametrize('model', ['NLP'], indirect=True)
    @pytest.mark.parametrize('optimizer', ['Adam', None, 'Scheduler'])
    def test_optimizer(self, curie_img, model, optimizer):
        met = po.synth.Metamer(curie_img, model)
        scheduler = None
        if optimizer == 'Adam' or optimizer == 'Scheduler':
            optimizer = torch.optim.Adam([met.metamer])
            if optimizer == 'Scheduler':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        met.synthesize(max_iter=5, optimizer=optimizer,
                       scheduler=scheduler)

    @pytest.mark.parametrize('model', ['Identity'], indirect=True)
    def test_map_location(self, curie_img, model, tmp_path):
        # only run this test if we have a gpu available
        if DEVICE.type != 'cpu':
            curie_img = curie_img.to(DEVICE)
            model.to(DEVICE)
            met = po.synth.Metamer(curie_img, model)
            met.synthesize(max_iter=4, store_progress=True)
            met.save(op.join(tmp_path, 'test_metamer_map_location.pt'))
            # calling load with map_location effectively switches everything
            # over to that device
            met_copy = po.synth.Metamer(curie_img, model)
            met_copy.load(op.join(tmp_path, 'test_metamer_map_location.pt'),
                          map_location='cpu')
            assert met_copy.metamer.device.type == 'cpu'
            assert met_copy.image.device.type == 'cpu'
            met.synthesize(max_iter=4, store_progress=True)

    @pytest.mark.parametrize('model', ['Identity'], indirect=True)
    @pytest.mark.parametrize('to_type', ['dtype', 'device'])
    def test_to(self, curie_img, model, to_type):
        met = po.synth.Metamer(curie_img, model)
        met.synthesize(max_iter=5)
        if to_type == 'dtype':
            met.to(torch.float16)
            assert met.image.dtype == torch.float16
            assert met.metamer.dtype == torch.float16
        # can only run this one if we're on a device with CPU and GPU.
        elif to_type == 'device' and DEVICE.type != 'cpu':
            met.to('cpu')
        met.metamer - met.image
