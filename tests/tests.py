import unittest
import numpy as np
import torch
import plenoptic as po

numpy = lambda x : x.detach().cpu().numpy().squeeze()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
data_path = '../data/'


# self.assertEqual('foo'.upper(), 'FOO')
# self.assertTrue
# self.assertFalse


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
