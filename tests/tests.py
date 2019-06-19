import unittest
import torch
import plenoptic as po
import os.path as op


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
DATA_PATH = op.join(op.dirname(op.realpath(__file__)), '..', 'data')


# self.assertEqual('foo'.upper(), 'FOO')
# self.assertTrue
# self.assertFalse

def to_numpy(x):
    """helper function to detach tensor, get it on cpu, and convert it to numpy array

    """
    return x.detach().cpu().numpy().squeeze()


class Basics(unittest.TestCase):
    def test1(self):
        model = po.simul.Linear()
        x = po.make_basic_stimuli()
        self.assertTrue(model(x).requires_grad)


# class SteerablePyramid(unittest.TestCase):
#     def test1(self):


if __name__ == '__main__':
    print('torch version ', torch.__version__)
    unittest.main()
