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


class TestMetamers(object):

    @pytest.mark.parametrize('model', ['frontend.LinearNonlinear', 'nlpd'], indirect=True)
    @pytest.mark.parametrize('loss_func', [None, 'l2', 'range_penalty_w_beta'])
    @pytest.mark.parametrize('fail', [False, 'img', 'model', 'loss'])
    def test_metamer_save_load(self, einstein_img, model, loss_func, fail, tmp_path):
        loss_kwargs = {}
        if loss_func is None:
            loss = None
        elif loss_func == 'l2':
            loss = po.optim.l2_norm
        elif loss_func == 'range_penalty_w_beta':
            loss = po.optim.l2_and_penalize_range
            loss_kwargs['beta'] = .9
        met = po.synth.Metamer(einstein_img, model, loss_function=loss, loss_function_kwargs=loss_kwargs)
        met.synthesize(max_iter=10, store_progress=True)
        met.save(op.join(tmp_path, 'test_metamer_save_load.pt'))
        # when the model is a function, the loss_function is ignored and thus
        # we won't actually fail to load here (we check against the specific
        # callable because we've overwritten the model input arg)
        if fail and not (fail == 'loss' and model == po.metric.nlpd):
            if fail == 'img':
                einstein_img = torch.rand_like(einstein_img)
            elif fail == 'model':
                model = po.metric.mse
            elif fail == 'loss':
                loss = lambda *args, **kwargs: 1
                loss_kwargs = {}
            met_copy = po.synth.Metamer(einstein_img, model, loss_function=loss, loss_function_kwargs=loss_kwargs)
            with pytest.raises(Exception):
                met_copy.load(op.join(tmp_path, "test_metamer_save_load.pt"),
                              map_location=DEVICE)
        else:
            met_copy = po.synth.Metamer(einstein_img, model, loss_function=loss, loss_function_kwargs=loss_kwargs)
            met_copy.load(op.join(tmp_path, "test_metamer_save_load.pt"),
                          map_location=DEVICE)
            for k in ['base_signal', 'saved_representation', 'saved_signal', 'synthesized_representation',
                      'synthesized_signal', 'base_representation']:
                if not getattr(met, k).allclose(getattr(met_copy, k)):
                    raise Exception("Something went wrong with saving and loading! %s not the same"
                                    % k)
            assert not isinstance(met_copy.synthesized_representation, torch.nn.Parameter), "matched_rep shouldn't be a parameter!"
            # check loss functions correctly saved
            met_loss = met.loss_function(met.synthesized_representation, met.base_representation,
                                         met.synthesized_signal, met.base_signal)
            met_copy_loss = met_copy.loss_function(met_copy.synthesized_representation,
                                                   met_copy.base_representation,
                                                   met_copy.synthesized_signal, met_copy.base_signal)
            if met_loss != met_copy_loss:
                raise Exception(f"Loss function not properly saved! Before saving was {met_loss}, "
                                f"after loading was {met_copy_loss}")
            # check that can resume
            met_copy.synthesize(max_iter=10, loss_change_iter=5, store_progress=True,
                                learning_rate=None)

    @pytest.mark.parametrize('model', ['frontend.LinearNonlinear'], indirect=True)
    @pytest.mark.parametrize('store_progress', [True, 2, 3])
    def test_metamer_store_rep(self, einstein_img, model, store_progress):
        metamer = po.synth.Metamer(einstein_img, model)
        max_iter = 3
        if store_progress == 3:
            max_iter = 6
        metamer.synthesize(max_iter=max_iter, store_progress=store_progress)

    @pytest.mark.parametrize('model', ['frontend.LinearNonlinear'], indirect=True)
    def test_metamer_store_rep_fail(self, einstein_img, model):
        metamer = po.synth.Metamer(einstein_img, model)
        with pytest.raises(Exception):
            # save_progress cannot be True if store_progress is False
            metamer.synthesize(max_iter=3, store_progress=False, save_progress=True)


    @pytest.mark.parametrize('model', ['frontend.LinearNonlinear'], indirect=True)
    def test_metamer_continue(self, einstein_img, model):
        metamer = po.synth.Metamer(einstein_img, model)
        metamer.synthesize(max_iter=3, store_progress=True)
        metamer.synthesize(max_iter=3, store_progress=True, learning_rate=None,
                           seed=None)


    @pytest.mark.parametrize('model', ['SPyr'], indirect=True)
    @pytest.mark.parametrize('coarse_to_fine', ['separate', 'together'])
    def test_coarse_to_fine(self, einstein_img, model, coarse_to_fine, tmp_path):
        metamer = po.synth.Metamer(einstein_img, model)
        metamer.synthesize(max_iter=10, loss_change_iter=1, loss_change_thresh=10,
                           coarse_to_fine=coarse_to_fine)
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
        metamer_copy.synthesize(max_iter=10, loss_change_iter=1, loss_change_thresh=10,
                                coarse_to_fine=coarse_to_fine)

    @pytest.mark.parametrize('model', ['frontend.LinearNonlinear'], indirect=True)
    @pytest.mark.parametrize("clamp_each_iter", [True, False])
    def test_metamer_clamper(self, einstein_img, model, clamp_each_iter):
        clamper = po.RangeClamper((0, 1))

        metamer = po.synth.Metamer(einstein_img, model)
        metamer.synthesize(max_iter=3, clamper=clamper, clamp_each_iter=clamp_each_iter)

    @pytest.mark.parametrize('model', ['frontend.LinearNonlinear'], indirect=True)
    def test_metamer_no_clamper(self, einstein_img, model):
        metamer = po.synth.Metamer(einstein_img, model)
        metamer.synthesize(max_iter=3, clamper=None)

    @pytest.mark.parametrize('model', ['NLP'], indirect=True)
    @pytest.mark.parametrize('loss_func', [None, 'l2', 'mse', 'range_penalty',
                                           'range_penalty_w_beta'])
    @pytest.mark.parametrize('store_progress', [False, True, 2])
    def test_loss_func(self, curie_img, model, loss_func, store_progress, tmp_path):
        loss_kwargs = {}
        if loss_func is None:
            loss = None
        elif loss_func == 'l2':
            loss = po.optim.l2_norm
        elif loss_func == 'mse':
            loss = po.optim.mse
        elif loss_func == 'range_penalty':
            loss = po.optim.l2_and_penalize_range
        elif loss_func == 'range_penalty_w_beta':
            loss = po.optim.l2_and_penalize_range
            loss_kwargs['beta'] = .9
        met = po.synth.Metamer(curie_img, model, loss_function=loss, loss_function_kwargs=loss_kwargs)
        met.synthesize(max_iter=10, loss_change_iter=5, store_progress=store_progress,
                       save_progress=store_progress, save_path=op.join(tmp_path, 'test_mad.pt'))
        if store_progress:
            met.synthesize(max_iter=10, loss_change_iter=5, store_progress=store_progress,
                           save_progress=store_progress,
                           save_path=op.join(tmp_path, 'test_mad.pt'), learning_rate=None)

    @pytest.mark.parametrize('model', ['NLP'], indirect=True)
    @pytest.mark.parametrize('optimizer', ['Adam', 'SGD', 'Adam-args'])
    def test_optimizer_opts(self, curie_img, model, optimizer):
        if '-' in optimizer:
            optimizer = optimizer.split('-')[0]
            optimizer_kwargs = {'weight_decay': .1}
        else:
            optimizer_kwargs = {}
        met = po.synth.Metamer(curie_img, model)
        met.synthesize(max_iter=10, loss_change_iter=5, optimizer=optimizer,
                       optimizer_kwargs=optimizer_kwargs)
