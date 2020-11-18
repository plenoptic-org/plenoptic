#!/usr/bin/env python3
import os.path as op
import torch
import plenoptic as po
import matplotlib.pyplot as plt
import pytest
from test_plenoptic import DEVICE, DATA_DIR, DTYPE


class TestMetamers(object):

    @pytest.mark.parametrize('model', ['class', 'class_reduced', 'function'])
    @pytest.mark.parametrize('loss_func', [None, 'l2', 'range_penalty_w_beta'])
    def test_metamer_save_load(self, model, loss_func, tmp_path):

        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        save_reduced = False
        model_constructor = None
        if model == 'class':
            model = po.simul.PooledV1(.5, im.shape[2:]).to(DEVICE)
        elif model == 'class_reduced':
            model = po.simul.PooledV1(.5, im.shape[2:]).to(DEVICE)
            save_reduced = True
            model_constructor = po.simul.PooledV1.from_state_dict_reduced
        else:
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
        v1 = po.simul.PooledV1(.5, im.shape[2:])
        v1 = v1.to(DEVICE)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=3, store_progress=2)

    def test_metamer_store_rep_2(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PooledV1(.5, im.shape[2:])
        v1 = v1.to(DEVICE)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=3, store_progress=True)

    def test_metamer_store_rep_3(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PooledV1(.5, im.shape[2:])
        v1 = v1.to(DEVICE)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=6, store_progress=3)

    def test_metamer_store_rep_4(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PooledV1(.5, im.shape[2:])
        v1 = v1.to(DEVICE)
        metamer = po.synth.Metamer(im, v1)
        with pytest.raises(Exception):
            metamer.synthesize(max_iter=3, store_progress=False, save_progress=True)

    def test_metamer_plotting_v1(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PooledV1(.5, im.shape[2:])
        v1 = v1.to(DEVICE)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=6, store_progress=True)
        metamer.plot_representation_error()
        metamer.model.plot_representation_image(data=metamer.representation_error())
        metamer.plot_synthesis_status()
        metamer.plot_synthesis_status(iteration=1)
        plt.close('all')

    def test_metamer_plotting_rgc(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        rgc = po.simul.PooledRGC(.5, im.shape[2:])
        rgc = rgc.to(DEVICE)
        metamer = po.synth.Metamer(im, rgc)
        metamer.synthesize(max_iter=6, store_progress=True)
        metamer.plot_representation_error()
        metamer.model.plot_representation_image(data=metamer.representation_error())
        metamer.plot_synthesis_status()
        metamer.plot_synthesis_status(iteration=1)
        plt.close('all')

    def test_metamer_continue(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        rgc = po.simul.PooledRGC(.5, im.shape[2:])
        rgc = rgc.to(DEVICE)
        metamer = po.synth.Metamer(im, rgc)
        metamer.synthesize(max_iter=3, store_progress=True)
        metamer.synthesize(max_iter=3, store_progress=True, learning_rate=None,
                           seed=None)

    def test_metamer_animate(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        rgc = po.simul.PooledRGC(.5, im.shape[2:])
        rgc = rgc.to(DEVICE)
        metamer = po.synth.Metamer(im, rgc)
        metamer.synthesize(max_iter=3, store_progress=True)
        # this will test several related functions for us:
        # plot_synthesis_status, plot_representation_error,
        # representation_error
        metamer.animate(figsize=(17, 5), plot_representation_error=True, ylim='rescale100',
                        framerate=40).to_html5_video()
        plt.close('all')

    def test_metamer_save_progress(self, tmp_path):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PooledV1(.5, im.shape[2:])
        v1 = v1.to(DEVICE)
        metamer = po.synth.Metamer(im, v1)
        save_path = op.join(tmp_path, 'test_metamer_save_progress.pt')
        metamer.synthesize(max_iter=3, store_progress=True, save_progress=True,
                           save_path=save_path)
        po.synth.Metamer.load(save_path, po.simul.PooledV1.from_state_dict_reduced)

    def test_metamer_loss_change(self):
        # literally just testing that it runs
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        rgc = po.simul.PooledRGC(.5, im.shape[2:])
        rgc = rgc.to(DEVICE)
        metamer = po.synth.Metamer(im, rgc)
        metamer.synthesize(max_iter=10, loss_change_iter=1, loss_change_thresh=1,
                           loss_change_fraction=.5)
        metamer.synthesize(max_iter=10, loss_change_iter=1, loss_change_thresh=1,
                           loss_change_fraction=.5, fraction_removed=.1)

    @pytest.mark.parametrize('fraction_removed', [0, .1])
    @pytest.mark.parametrize('loss_change_fraction', [.5, 1])
    @pytest.mark.parametrize('coarse_to_fine', ['separate', 'together'])
    def test_metamer_coarse_to_fine(self, fraction_removed, loss_change_fraction, coarse_to_fine,
                                    tmp_path):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PooledV1(.5, im.shape[2:])
        v1 = v1.to(DEVICE)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=10, loss_change_iter=1, loss_change_thresh=10,
                           coarse_to_fine=coarse_to_fine, fraction_removed=fraction_removed,
                           loss_change_fraction=loss_change_fraction)
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
                                coarse_to_fine=coarse_to_fine, fraction_removed=fraction_removed,
                                loss_change_fraction=loss_change_fraction)

    @pytest.mark.parametrize("clamp_each_iter", [True, False])
    def test_metamer_clamper(self, clamp_each_iter):
        clamper = po.RangeClamper((0, 1))
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        rgc = po.simul.PooledRGC(.5, im.shape[2:])
        rgc = rgc.to(DEVICE)
        metamer = po.synth.Metamer(im, rgc)
        metamer.synthesize(max_iter=3, clamper=clamper, clamp_each_iter=clamp_each_iter)

    def test_metamer_no_clamper(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        rgc = po.simul.PooledRGC(.5, im.shape[2:])
        rgc = rgc.to(DEVICE)
        metamer = po.synth.Metamer(im, rgc)
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
        if store_progress:
            met.animate().to_html5_video()
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

