#!/usr/bin/env python3

# necessary to avoid issues with animate:
# https://github.com/matplotlib/matplotlib/issues/10287/
import matplotlib
matplotlib.use('agg')
import os.path as op
import torch
import plenoptic as po
import matplotlib.pyplot as plt
import pytest
from test_plenoptic import DEVICE, DATA_DIR, DTYPE


class TestMetamers(object):

    @pytest.mark.parametrize('model', ['class', 'function'])
    @pytest.mark.parametrize('loss_func', [None, 'l2', 'range_penalty_w_beta'])
    def test_metamer_save_load(self, model, loss_func, tmp_path):

        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        save_reduced = False
        model_constructor = None
        if model == 'class':
            model = po.simul.Linear_Nonlinear().to(DEVICE)
        elif model == 'function':
            model = po.metric.nlpd
        loss_kwargs = {}
        if loss_func is None:
            loss = None
        elif loss_func == 'l2':
            loss = po.optim.l2_norm
        elif loss_func == 'range_penalty_w_beta':
            loss = po.optim.l2_and_penalize_range
            loss_kwargs['beta'] = .9
        met = po.synth.Metamer(im, model, loss_function=loss, loss_function_kwargs=loss_kwargs)
        met.synthesize(max_iter=10, store_progress=True)
        met.save(op.join(tmp_path, 'test_metamer_save_load.pt'), save_reduced)
        met_copy = po.synth.Metamer.load(op.join(tmp_path, "test_metamer_save_load.pt"),
                                         map_location=DEVICE,
                                         model_constructor=model_constructor)
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

    def test_metamer_store_rep(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        lnl = po.simul.Linear_Nonlinear().to(DEVICE)
        metamer = po.synth.Metamer(im, lnl)
        metamer.synthesize(max_iter=3, store_progress=2)

    def test_metamer_store_rep_2(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        lnl = po.simul.Linear_Nonlinear().to(DEVICE)
        metamer = po.synth.Metamer(im, lnl)
        metamer.synthesize(max_iter=3, store_progress=True)

    def test_metamer_store_rep_3(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        lnl = po.simul.Linear_Nonlinear().to(DEVICE)
        metamer = po.synth.Metamer(im, lnl)
        metamer.synthesize(max_iter=6, store_progress=3)

    def test_metamer_store_rep_4(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        lnl = po.simul.Linear_Nonlinear().to(DEVICE)
        metamer = po.synth.Metamer(im, lnl)
        with pytest.raises(Exception):
            metamer.synthesize(max_iter=3, store_progress=False, save_progress=True)

    def test_metamer_continue(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        lnl = po.simul.Linear_Nonlinear().to(DEVICE)
        metamer = po.synth.Metamer(im, lnl)
        metamer.synthesize(max_iter=3, store_progress=True)
        metamer.synthesize(max_iter=3, store_progress=True, learning_rate=None,
                           seed=None)

    def test_metamer_save_progress(self, tmp_path):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        lnl = po.simul.Linear_Nonlinear().to(DEVICE)
        im = torch.tensor(im, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
        metamer = po.synth.Metamer(im, lnl)
        save_path = op.join(tmp_path, 'test_metamer_save_progress.pt')
        metamer.synthesize(max_iter=3, store_progress=True, save_progress=True,
                           save_path=save_path)
        po.synth.Metamer.load(save_path)

    @pytest.mark.parametrize('coarse_to_fine', ['separate', 'together'])
    def test_coarse_to_fine(self, coarse_to_fine, tmp_path):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        # with downsample=False, we get a tensor back. setting height=1 and
        # order=1 limits the size
        spyr = po.simul.Steerable_Pyramid_Freq(im.shape[-2:], downsample=False,
                                               height=1, order=1).to(DEVICE)
        metamer = po.synth.Metamer(im, spyr)
        metamer.synthesize(max_iter=10, loss_change_iter=1, loss_change_thresh=10,
                           coarse_to_fine=coarse_to_fine)
        assert len(metamer.scales_finished) > 0, "Didn't actually switch scales!"

        metamer.save(op.join(tmp_path, 'test_metamer_ctf.pt'))
        metamer_copy = po.synth.Metamer.load(op.join(tmp_path, "test_metamer_ctf.pt"),
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

    @pytest.mark.parametrize("clamp_each_iter", [True, False])
    def test_metamer_clamper(self, clamp_each_iter):
        clamper = po.RangeClamper((0, 1))
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        lnl = po.simul.Linear_Nonlinear().to(DEVICE)
        metamer = po.synth.Metamer(im, lnl)
        metamer.synthesize(max_iter=3, clamper=clamper, clamp_each_iter=clamp_each_iter)

    def test_metamer_no_clamper(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        lnl = po.simul.Linear_Nonlinear().to(DEVICE)
        metamer = po.synth.Metamer(im, lnl)
        metamer.synthesize(max_iter=3, clamper=None)

    @pytest.mark.parametrize('loss_func', [None, 'l2', 'mse', 'range_penalty',
                                           'range_penalty_w_beta'])
    @pytest.mark.parametrize('store_progress', [False, True, 2])
    @pytest.mark.parametrize('resume', [False, True])
    def test_loss_func(self, loss_func, store_progress, resume, tmp_path):
        img = po.tools.data.load_images(op.join(DATA_DIR, 'curie.pgm')).to(DEVICE)
        model = po.metric.NLP().to(DEVICE)
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
        met = po.synth.Metamer(img, model, loss_function=loss, loss_function_kwargs=loss_kwargs)
        met.synthesize(max_iter=10, loss_change_iter=5, store_progress=store_progress,
                       save_progress=store_progress, save_path=op.join(tmp_path, 'test_mad.pt'))
        if resume and store_progress:
            met.synthesize(max_iter=10, loss_change_iter=5, store_progress=store_progress,
                           save_progress=store_progress,
                           save_path=op.join(tmp_path, 'test_mad.pt'), learning_rate=None)
        met.plot_synthesis_status()
        plt.close('all')

    @pytest.mark.parametrize('optimizer', ['Adam', 'SGD', 'Adam-args'])
    @pytest.mark.parametrize('swa', [True, False])
    def test_optimizer_opts(self, optimizer, swa):
        img = po.tools.data.load_images(op.join(DATA_DIR, 'curie.pgm')).to(DEVICE)
        model = po.metric.NLP().to(DEVICE)
        if '-' in optimizer:
            optimizer = optimizer.split('-')[0]
            optimizer_kwargs = {'weight_decay': .1}
        else:
            optimizer_kwargs = {}
        if swa:
            swa_kwargs = {'swa_start': 1, 'swa_freq': 1, 'swa_lr': .05}
        else:
            swa_kwargs = {}
        met = po.synth.Metamer(img, model)
        met.synthesize(max_iter=10, loss_change_iter=5, swa=swa, swa_kwargs=swa_kwargs,
                       optimizer=optimizer, optimizer_kwargs=optimizer_kwargs)

