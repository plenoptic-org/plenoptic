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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32


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

# class SteerablePyramid(unittest.TestCase):
#     def test1(self):
