import pytest
import torch
import requests
import math
import tqdm
import tarfile
import os
import numpy as np
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
        windows, theta, ecc = po.simul.create_pooling_windows(.87)

    def test_creation_args(self):
        windows, theta, ecc = po.simul.create_pooling_windows(.87, .2, 30, 1.2, .7, 100, 100)

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
        assert po.simul.pooling.calc_polar_window_width(2) == np.pi
        assert po.simul.pooling.calc_polar_n_windows(2) == np.pi
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
        # because of this issue
        # https://github.com/pytest-dev/pytest/issues/5017, need to cast
        # the path object to a string on python 3.5 (just wrapping in
        # pathlib.Path, as I took the final comment in that issue to
        # mean, does not work)
        rgc.save_sparse(op.join(tmp_path.as_posix(), 'test_rgc_save_load.pt'))
        rgc_copy = po.simul.RetinalGanglionCells.load_sparse(op.join(tmp_path.as_posix(),
                                                                     'test_rgc_save_load.pt'))
        if not len(rgc.windows) == len(rgc_copy.windows):
            raise Exception("Something went wrong saving and loading, the lists of windows are"
                            " not the same length!")
        # we don't recreate everything, e.g., the representation, but windows is the most important
        for i in range(len(rgc.windows)):
            if not rgc.windows[i].allclose(rgc_copy.windows[i]):
                raise Exception("Something went wrong saving and loading, the windows %d are"
                                " not identical!" % i)

    def test_v1(self):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        v1(im)

    def test_v1_save_load(self, tmp_path):
        im = plt.imread(op.join(DATA_DIR, 'nuts.pgm'))
        im = torch.tensor(im, dtype=torch.float32, device=device)
        v1 = po.simul.PrimaryVisualCortex(.5, im.shape)
        v1(im)
        # because of this issue
        # https://github.com/pytest-dev/pytest/issues/5017, need to cast
        # the path object to a string on python 3.5 (just wrapping in
        # pathlib.Path, as I took the final comment in that issue to
        # mean, does not work)
        v1.save_sparse(op.join(tmp_path.as_posix(), 'test_v1_save_load.pt'))
        v1_copy = po.simul.PrimaryVisualCortex.load_sparse(op.join(tmp_path.as_posix(),
                                                                   'test_v1_save_load.pt'))
        if not len(v1.windows) == len(v1_copy.windows):
            raise Exception("Something went wrong saving and loading, the lists of windows are"
                            " not the same length!")
        # we don't recreate everything, e.g., the representation, but windows is the most important
        for i in range(len(v1.windows)):
            if not v1.windows[i].allclose(v1_copy.windows[i]):
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
        # because of this issue
        # https://github.com/pytest-dev/pytest/issues/5017, need to cast
        # the path object to a string on python 3.5 (just wrapping in
        # pathlib.Path, as I took the final comment in that issue to
        # mean, does not work)
        metamer.save(op.join(tmp_path.as_posix(), 'test_metamer_save_load.pt'))
        met_copy = po.synth.Metamer.load(op.join(tmp_path.as_posix(), "test_metamer_save_load.pt"))
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
        # because of this issue
        # https://github.com/pytest-dev/pytest/issues/5017, need to cast
        # the path object to a string on python 3.5 (just wrapping in
        # pathlib.Path, as I took the final comment in that issue to
        # mean, does not work)
        metamer.save(op.join(tmp_path.as_posix(), 'test_metamer_save_load_sparse.pt'), True)
        with pytest.raises(Exception):
            met_copy = po.synth.Metamer.load(op.join(tmp_path.as_posix(),
                                                     "test_metamer_save_load_sparse.pt"))
        met_copy = po.synth.Metamer.load(op.join(tmp_path.as_posix(),
                                                 'test_metamer_save_load_sparse.pt'),
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


# class SteerablePyramid(unittest.TestCase):
#     def test1(self):
