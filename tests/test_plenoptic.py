import pytest
import torch
import requests
import math
import tqdm
import tarfile
import os
import plenoptic as po
import os.path as op


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# self.assertEqual('foo'.upper(), 'FOO')
# self.assertTrue
# self.assertFalse


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
        print('hi')
        assert model(x).requires_grad


def test_find_files(test_files_dir):
    assert op.exists(op.join(test_files_dir, 'buildSCFpyr0.mat'))

# class SteerablePyramid(unittest.TestCase):
#     def test1(self):
