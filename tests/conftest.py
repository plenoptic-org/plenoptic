#!/usr/bin/env python3
import pytest
import plenoptic as po
import os.path as op
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
DATA_DIR = op.join(op.dirname(op.realpath(__file__)), '..', 'data')


class ColorModel(torch.nn.Module):
    """Simple model that takes color image as input and outputs 2d conv."""
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, 1)

    def forward(self, x):
        return self.conv(x)


@pytest.fixture(scope='package')
def curie_img():
    return po.load_images(op.join(DATA_DIR, 'curie.pgm')).to(DEVICE)


@pytest.fixture(scope='package')
def einstein_img():
    return po.load_images(op.join(DATA_DIR, 'einstein.pgm')).to(DEVICE)


@pytest.fixture(scope='package')
def color_img():
    img = po.load_images(op.join(DATA_DIR, 'color_wheel.jpg'),
                         as_gray=False).to(DEVICE)
    return img[..., :256, :256]


def get_model(name):
    if name == 'LNL':
        return po.simul.Linear_Nonlinear().to(DEVICE)
    elif name == 'SPyr':
        # with downsample=False, we get a tensor back. setting height=1 and
        # order=1 limits the size
        return po.simul.Steerable_Pyramid_Freq((256, 256), downsample=False,
                                               height=1, order=1).to(DEVICE)
    elif name == 'Identity':
        return po.simul.models.naive.Identity().to(DEVICE)
    elif name == 'NLP':
        return po.metric.NLP().to(DEVICE)
    elif name == 'nlpd':
        return po.metric.nlpd
    elif name == 'mse':
        return po.metric.naive.mse
    elif name == 'ColorModel':
        return ColorModel().to(DEVICE)
    elif name == 'FrontEnd':
        return po.simul.FrontEnd(pretrained=True, requires_grad=False).to(DEVICE)


@pytest.fixture(scope='package')
def model(request):
    return get_model(request.param)

# this is the same as model() fixture above, in order to get two independent
# fixtures.
@pytest.fixture(scope='package')
def model2(request):
    return get_model(request.param)
