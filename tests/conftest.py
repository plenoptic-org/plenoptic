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


@pytest.fixture(scope='package')
def basic_stim():
    return po.make_basic_stimuli().to(DEVICE)


def get_model(name):
    if name == 'SPyr':
        # with downsample=False, we get a tensor back. setting height=1 and
        # order=1 limits the size
        return po.simul.Steerable_Pyramid_Freq((256, 256), downsample=False,
                                               height=1, order=1).to(DEVICE)
    elif name == 'Identity':
        return po.simul.models.Identity().to(DEVICE)
    elif name == 'NLP':
        return po.metric.NLP().to(DEVICE)
    elif name == 'nlpd':
        return po.metric.nlpd
    elif name == 'mse':
        return po.metric.naive.mse
    elif name == 'ColorModel':
        return ColorModel().to(DEVICE)

    # FrontEnd models:
    elif name == 'frontend.CenterSurround':
        return po.simul.CenterSurround((31, 31)).to(DEVICE)
    elif name == 'frontend.Gaussian':
        return po.simul.Gaussian((31, 31)).to(DEVICE)
    elif name == 'frontend.LinearNonlinear':
        return po.simul.LinearNonlinear((31, 31)).to(DEVICE)
    elif name == 'frontend.LuminanceGainControl':
        return po.simul.LuminanceGainControl((31, 31)).to(DEVICE)
    elif name == 'frontend.LuminanceContrastGainControl':
        return po.simul.LuminanceContrastGainControl((31, 31)).to(DEVICE)
    elif name == 'frontend.OnOff':
        return po.simul.OnOff((31, 31), pretrained=True).to(DEVICE)


@pytest.fixture(scope='package')
def model(request):
    return get_model(request.param)

# this is the same as model() fixture above, in order to get two independent
# fixtures.
@pytest.fixture(scope='package')
def model2(request):
    return get_model(request.param)
