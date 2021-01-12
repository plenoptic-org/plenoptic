#!/usr/bin/env python3
import os.path as op
import torch
import plenoptic as po
import matplotlib.pyplot as plt
import pytest
from test_plenoptic import DEVICE, DATA_DIR, DTYPE


class TestMetamers(object):

    def test_metamer_save_load(self, tmp_path):

        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PooledV1(.5, im.shape[2:])
        v1 = v1.to(DEVICE)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=3, store_progress=True)
        metamer.save(op.join(tmp_path, 'test_metamer_save_load.pt'))
        met_copy = po.synth.Metamer.load(op.join(tmp_path, "test_metamer_save_load.pt"),
                                         map_location=DEVICE)
        for k in ['base_signal', 'saved_representation', 'saved_signal', 'synthesized_representation',
                  'synthesized_signal', 'base_representation']:
            if not getattr(metamer, k).allclose(getattr(met_copy, k)):
                raise Exception("Something went wrong with saving and loading! %s not the same"
                                % k)

    def test_metamer_save_load_reduced(self, tmp_path):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PooledV1(.5, im.shape[2:])
        v1 = v1.to(DEVICE)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=3, store_progress=True)
        metamer.save(op.join(tmp_path, 'test_metamer_save_load_reduced.pt'), True)
        with pytest.raises(Exception):
            met_copy = po.synth.Metamer.load(op.join(tmp_path,
                                                     "test_metamer_save_load_reduced.pt"))
        met_copy = po.synth.Metamer.load(op.join(tmp_path, 'test_metamer_save_load_reduced.pt'),
                                         po.simul.PooledV1.from_state_dict_reduced,
                                         map_location=DEVICE)
        for k in ['base_signal', 'saved_representation', 'saved_signal', 'synthesized_representation',
                  'synthesized_signal', 'base_representation']:
            if not getattr(metamer, k).allclose(getattr(met_copy, k)):
                raise Exception("Something went wrong with saving and loading! %s not the same" % k)

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

    def test_metamer_continue(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        rgc = po.simul.PooledRGC(.5, im.shape[2:])
        rgc = rgc.to(DEVICE)
        metamer = po.synth.Metamer(im, rgc)
        metamer.synthesize(max_iter=3, store_progress=True)
        metamer.synthesize(max_iter=3, store_progress=True,
                           initial_image=metamer.synthesized_signal.detach().clone())

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
                        framerate=40)

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

    def test_metamer_coarse_to_fine(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        v1 = po.simul.PooledV1(.5, im.shape[2:])
        v1 = v1.to(DEVICE)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=10, loss_change_iter=1, loss_change_thresh=10,
                           coarse_to_fine=True)
        metamer.synthesize(max_iter=10, loss_change_iter=1, loss_change_thresh=10,
                           coarse_to_fine=True, fraction_removed=.1)
        metamer.synthesize(max_iter=10, loss_change_iter=1, loss_change_thresh=10,
                           coarse_to_fine=True, loss_change_fraction=.5)
        metamer.synthesize(max_iter=10, loss_change_iter=1, loss_change_thresh=10,
                           coarse_to_fine=True, loss_change_fraction=.5, fraction_removed=.1)

    # @pytest.mark.parametrize("clamp_each_iter", [True, False])
    # @pytest.mark.parametrize("cone_power", [1, 1/3])
    # def test_metamer_clamper(self, clamp_each_iter, cone_power):
    #     clamper = po.RangeClamper((0, 1))
    #     im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
    #     im = torch.tensor(im/255, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
    #     rgc = po.simul.PooledRGC(.5, im.shape[2:], cone_power=cone_power)
    #     rgc = rgc.to(DEVICE)
    #     metamer = po.synth.Metamer(im, rgc)
    #     if cone_power == 1/3 and not clamp_each_iter:
    #         # these will fail because we'll end up outside the 0, 1 range
    #         with pytest.raises(IndexError):
    #             metamer.synthesize(max_iter=3, clamper=clamper, clamp_each_iter=clamp_each_iter)
    #     else:
    #         metamer.synthesize(max_iter=3, clamper=clamper, clamp_each_iter=clamp_each_iter)

    def test_metamer_no_clamper(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im/255, dtype=DTYPE, device=DEVICE).unsqueeze(0).unsqueeze(0)
        rgc = po.simul.PooledRGC(.5, im.shape[2:], cone_power=1)
        rgc = rgc.to(DEVICE)
        metamer = po.synth.Metamer(im, rgc)
        metamer.synthesize(max_iter=3, clamper=None)
