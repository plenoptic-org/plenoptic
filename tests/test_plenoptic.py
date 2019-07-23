import pytest
import torch
import requests
import math
import tqdm
import tarfile
import os
import numpy as np
import pyrtools as pt
import plenoptic as po
import os.path as op
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
DATA_DIR = op.join(op.dirname(op.realpath(__file__)), '..', 'data')


@pytest.fixture
def test_files_dir():
    path = op.join(op.dirname(op.realpath(__file__)), '..', 'data', 'plenoptic-test-files')
    if not op.exists(path):
        print("matfiles required for testing not found, downloading now...")
        # Streaming, so we can iterate over the response.
        r = requests.get("https://osf.io/q9kn8/download", stream=True)

        # Total size in bytes.
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024*1024
        wrote = 0
        with open(path + ".tar.gz", 'wb') as f:
            for data in tqdm.tqdm(r.iter_content(block_size), unit='MB', unit_scale=True,
                                  total=math.ceil(total_size//block_size)):
                wrote += len(data)
                f.write(data)
        if total_size != 0 and wrote != total_size:
            raise Exception("Error downloading test files!")
        with tarfile.open(path + ".tar.gz") as f:
            f.extractall(op.dirname(path))
        os.remove(path + ".tar.gz")
    return path


def to_numpy(x):
    """helper function to detach tensor, get it on cpu, and convert it to numpy array

    """
    return x.detach().cpu().numpy().squeeze()


class TestBasics(object):
    def test_one(self):
        model = po.simul.Linear()
        x = po.make_basic_stimuli()
        assert model(x).requires_grad


def test_find_files(test_files_dir):
    assert op.exists(op.join(test_files_dir, 'buildSCFpyr0.mat'))


class TestPooling(object):
    def test_creation(self):
        windows, theta, ecc = po.simul.pooling.create_pooling_windows(.87)

    def test_creation_args(self):
        with pytest.raises(Exception):
            # we can't create these with transition_region_width != .5
            windows, theta, ecc = po.simul.pooling.create_pooling_windows(.87, .2, 30, 1.2, .7, 100, 100)
        windows, theta, ecc = po.simul.pooling.create_pooling_windows(.87, .2, 30, 1.2, .5, 100, 100)

    def test_ecc_windows(self):
        ecc, windows = po.simul.pooling.log_eccentricity_windows(n_windows=4)
        ecc, windows = po.simul.pooling.log_eccentricity_windows(n_windows=4.5)
        ecc, windows = po.simul.pooling.log_eccentricity_windows(window_width=.5)
        ecc, windows = po.simul.pooling.log_eccentricity_windows(window_width=1)

    def test_angle_windows(self):
        theta, windows = po.simul.pooling.polar_angle_windows(4)
        theta, windows = po.simul.pooling.polar_angle_windows(4, 1000)
        with pytest.raises(Exception):
            theta, windows = po.simul.pooling.polar_angle_windows(1.5)
        with pytest.raises(Exception):
            theta, windows = po.simul.pooling.polar_angle_windows(1)

    def test_calculations(self):
        # these really shouldn't change, but just in case...
        assert po.simul.pooling.calc_angular_window_width(2) == np.pi
        assert po.simul.pooling.calc_angular_n_windows(2) == np.pi
        with pytest.raises(Exception):
            po.simul.pooling.calc_eccentricity_window_width()
        assert po.simul.pooling.calc_eccentricity_window_width(n_windows=4) == 0.8502993454155389
        assert po.simul.pooling.calc_eccentricity_window_width(scaling=.87) == 0.8446653390527211
        assert po.simul.pooling.calc_eccentricity_window_width(5, 10, scaling=.87) == 0.8446653390527211
        assert po.simul.pooling.calc_eccentricity_window_width(5, 10, n_windows=4) == 0.1732867951399864
        assert po.simul.pooling.calc_eccentricity_n_windows(0.8502993454155389) == 4
        assert po.simul.pooling.calc_eccentricity_n_windows(0.1732867951399864, 5, 10) == 4
        assert po.simul.pooling.calc_scaling(4) == 0.8761474337786708
        assert po.simul.pooling.calc_scaling(4, 5, 10) == 0.17350368946058647
        assert np.isinf(po.simul.pooling.calc_scaling(4, 0))


class TestVentralStream(object):
    def test_rgc(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=device)
        rgc = po.simul.RetinalGanglionCells(.5, im.shape)
        rgc(im)
        _ = rgc.plot_window_sizes('degrees')
        _ = rgc.plot_window_sizes('degrees', jitter=0)
        _ = rgc.plot_window_sizes('pixels')
        fig = pt.imshow(im.detach())
        _ = rgc.plot_windows(fig.axes[0])
        rgc.plot_representation()
        fig, axes = plt.subplots(2, 1)
        rgc.plot_representation(ax=axes[1])

    def test_rgc_metamer(self):
        # literally just testing that it runs
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=device)
        rgc = po.simul.RetinalGanglionCells(.5, im.shape)
        metamer = po.synth.Metamer(im, rgc)
        metamer.synthesize(max_iter=10)

    def test_rgc_save_load(self, tmp_path):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=device)
        rgc = po.simul.PrimaryVisualCortex(.5, im.shape)
        rgc(im)
        rgc.save_sparse(op.join(tmp_path, 'test_rgc_save_load.pt'))
        rgc_copy = po.simul.RetinalGanglionCells.load_sparse(op.join(tmp_path,
                                                                     'test_rgc_save_load.pt'))
        if not len(rgc.PoolingWindows.windows) == len(rgc_copy.PoolingWindows.windows):
            raise Exception("Something went wrong saving and loading, the lists of windows are"
                            " not the same length!")
        # we don't recreate everything, e.g., the representation, but windows is the most important
        for i in range(len(rgc.PoolingWindows.windows)):
            if not rgc.PoolingWindows.windows[i].allclose(rgc_copy.PoolingWindows.windows[i]):
                raise Exception("Something went wrong saving and loading, the windows %d are"
                                " not identical!" % i)

    def test_v1(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        v1(im)
        _ = v1.plot_window_sizes('pixels')
        for i in range(v1.num_scales):
            _ = v1.plot_window_sizes('pixels', i)
        v1.plot_representation()
        fig, axes = plt.subplots(2, 1)
        v1.plot_representation(ax=axes[1])

    def test_v1_mean_luminance(self):
        for fname in ['nuts', 'einstein']:
            im = plt.imread(op.join(DATA_DIR, fname+'.pgm'))
            im = torch.tensor(im, dtype=torch.float32, device=device)
            v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
            v1_rep = v1(im)
            rgc = po.simul.RetinalGanglionCells(.5, im.shape)
            rgc_rep = rgc(im)
            if not torch.allclose(rgc_rep, v1.mean_luminance):
                raise Exception("Somehow RGC and V1 mean luminance representations are not the "
                                "same for image %s!" % fname)
            if not torch.allclose(rgc_rep, v1_rep[-rgc_rep.shape[0]:]):
                raise Exception("Somehow V1's representation does not have the mean luminance "
                                "in the location expected! for image %s!" % fname)

    def test_v1_save_load(self, tmp_path):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        v1(im)
        v1.save_sparse(op.join(tmp_path, 'test_v1_save_load.pt'))
        v1_copy = po.simul.PrimaryVisualCortex.load_sparse(op.join(tmp_path,
                                                                   'test_v1_save_load.pt'))
        if not len(v1.PoolingWindows.windows) == len(v1_copy.PoolingWindows.windows):
            raise Exception("Something went wrong saving and loading, the lists of windows are"
                            " not the same length!")
        # we don't recreate everything, e.g., the representation, but windows is the most important
        for i in range(len(v1.PoolingWindows.windows)):
            if not v1.PoolingWindows.windows[i].allclose(v1_copy.PoolingWindows.windows[i]):
                raise Exception("Something went wrong saving and loading, the windows %d are"
                                " not identical!" % i)

    def test_v1_metamer(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=10)


class TestMetamers(object):
    def test_metamer_save_load(self, tmp_path):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=10, save_representation=True, save_image=True)
        metamer.save(op.join(tmp_path, 'test_metamer_save_load.pt'))
        met_copy = po.synth.Metamer.load(op.join(tmp_path, "test_metamer_save_load.pt"))
        for k in ['target_image', 'saved_representation', 'saved_image', 'matched_representation',
                  'matched_image', 'target_representation']:
            if not getattr(metamer, k).allclose(getattr(met_copy, k)):
                raise Exception("Something went wrong with saving and loading! %s not the same"
                                % k)

    def test_metamer_save_load_sparse(self, tmp_path):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=10, save_representation=True, save_image=True)
        metamer.save(op.join(tmp_path, 'test_metamer_save_load_sparse.pt'), True)
        with pytest.raises(Exception):
            met_copy = po.synth.Metamer.load(op.join(tmp_path, "test_metamer_save_load_sparse.pt"))
        met_copy = po.synth.Metamer.load(op.join(tmp_path, 'test_metamer_save_load_sparse.pt'),
                                         po.simul.PrimaryVisualCortex.from_state_dict_sparse)
        for k in ['target_image', 'saved_representation', 'saved_image', 'matched_representation',
                  'matched_image', 'target_representation']:
            if not getattr(metamer, k).allclose(getattr(met_copy, k)):
                raise Exception("Something went wrong with saving and loading! %s not the same"
                                % k)

    def test_metamer_save_rep(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=10, save_representation=2, save_image=2)

    def test_metamer_save_rep_2(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=10, save_representation=2, save_image=True)

    def test_metamer_save_rep_3(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=10, save_representation=3, save_image=True)

    def test_metamer_save_rep_4(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        metamer = po.synth.Metamer(im, v1)
        metamer.synthesize(max_iter=10, save_representation=3, save_image=3)

    def test_metamer_animate(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=device)
        rgc = po.simul.RetinalGanglionCells(.5, im.shape)
        metamer = po.synth.Metamer(im, rgc)
        metamer.synthesize(max_iter=10, save_image=True, save_representation=True)
        # this will test several related functions for us:
        # plot_metamer_status, plot_representation_ratio,
        # representation_ratio
        metamer.animate(figsize=(17, 5), plot_representation_ratio=True, ylim='rescale100',
                        framerate=40)

    def test_metamer_nans(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = im / 255
        im = torch.tensor(im, dtype=torch.float32, device=device)
        initial_image = .5*torch.ones_like(im, requires_grad=True, device=device,
                                           dtype=torch.float32)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        metamer = po.synth.Metamer(im, v1)
        clamper = po.RangeClamper((0, 1))
        # this gets raised because we try to use saved_image_ticker,
        # which was never initialized, since we're not saving images
        with pytest.raises(UnboundLocalError):
            metamer.synthesize(clamper=clamper, learning_rate=10, max_iter=10, loss_thresh=1e-8,
                               initial_image=initial_image)
        # need to re-initialize this for the following run
        initial_image = .5*torch.ones_like(im, requires_grad=True, device=device,
                                           dtype=torch.float32)
        matched_im, _ = metamer.synthesize(clamper=clamper, learning_rate=10, save_image=True,
                                           save_representation=True, max_iter=10, loss_thresh=1e-8,
                                           initial_image=initial_image)
        # this should hit a nan as it runs, leading the second saved
        # image to be all nans, but, because of our way of handling
        # this, matched_image should have no nans
        assert torch.isnan(metamer.saved_image[-1]).all(), "This should be all NaNs!"
        assert not torch.isnan(metamer.matched_image).any(), "There should be no NaNs!"

    def test_metamer_save_progress(self, tmp_path):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        metamer = po.synth.Metamer(im, v1)
        save_path = op.join(tmp_path, 'test_metamer_save_progress.pt')
        metamer.synthesize(max_iter=10, save_representation=True, save_image=True,
                           save_progress=True, save_path=save_path)
        po.synth.Metamer.load(save_path, po.simul.PrimaryVisualCortex.from_state_dict_sparse)

# class SteerablePyramid(unittest.TestCase):
#     def test1(self):
