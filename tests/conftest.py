#!/usr/bin/env python3
import pytest
import plenoptic as po
import os.path as op
import torch
from torchvision.transforms.functional import center_crop

import plenoptic.simulate.canonical_computations.filters as filters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
DATA_DIR = op.join(op.dirname(op.realpath(__file__)), '..', 'data')

torch.set_num_threads(1)  # torch uses all avail threads which will slow tests
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
class ColorModel(torch.nn.Module):
    """Simple model that takes color image as input and outputs 2d conv."""
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, 1)

    def forward(self, x):
        return self.conv(x)


@pytest.fixture(scope='package')
def curie_img():
    return po.load_images(op.join(DATA_DIR, '256/curie.pgm')).to(DEVICE)


@pytest.fixture(scope='package')
def einstein_img():
    return po.load_images(op.join(DATA_DIR, '256/einstein.pgm')).to(DEVICE)

@pytest.fixture(scope='package')
def einstein_img_small(einstein_img):
    return center_crop(einstein_img, [64]).to(DEVICE)

@pytest.fixture(scope='package')
def color_img():
    img = po.load_images(op.join(DATA_DIR, 'color_wheel.jpg'),
                         as_gray=False).to(DEVICE)
    return img[..., :256, :256]


@pytest.fixture(scope='package')
def basic_stim():
    return po.tools.make_synthetic_stimuli().to(DEVICE)


def get_model(name):
    if name == 'LNL':
        return po.simul.Linear_Nonlinear().to(DEVICE)
    elif name == 'SPyr':
        # in order to get a tensor back, need to wrap steerable pyramid so that
        # we can call convert_pyr_to_tensor in the forward call. in order for
        # that to work, downsample must be False
        class spyr(po.simul.Steerable_Pyramid_Freq):
            def __init__(self, *args, **kwargs):
                kwargs.pop('downsample', None)
                super().__init__(*args, downsample=False, **kwargs)
            def forward(self, *args, **kwargs):
                coeffs = super().forward(*args, **kwargs)
                pyr_tensor, _ = self.convert_pyr_to_tensor(coeffs)
                return pyr_tensor
        # setting height=1 and # order=1 limits the size
        return spyr((256, 256), height=1, order=1).to(DEVICE)
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

    # naive models
    elif name in ['Identity', "naive.Identity"]:
        return po.simul.Identity().to(DEVICE)
    elif name == 'naive.CenterSurround':
        return po.simul.CenterSurround((31, 31)).to(DEVICE)
    elif name == 'naive.Gaussian':
        return po.simul.Gaussian((31, 31)).to(DEVICE)
    elif name == 'naive.Linear':
        return po.simul.Linear((31, 31)).to(DEVICE)

    # FrontEnd models:
    elif name == 'frontend.LinearNonlinear':
        return po.simul.LinearNonlinear((31, 31)).to(DEVICE)
    elif name == 'frontend.LuminanceGainControl':
        return po.simul.LuminanceGainControl((31, 31)).to(DEVICE)
    elif name == 'frontend.LuminanceContrastGainControl':
        return po.simul.LuminanceContrastGainControl((31, 31)).to(DEVICE)
    elif name == 'frontend.OnOff':
        return po.simul.OnOff((31, 31), pretrained=True, cache_filt=True).to(DEVICE)
    elif name == 'frontend.OnOff.nograd':
        mdl = po.simul.OnOff((31, 31), pretrained=True, cache_filt=True).to(DEVICE)
        for p in mdl.parameters():
            p.detach_()
        return mdl


@pytest.fixture(scope='package')
def model(request):
    return get_model(request.param)

# this is the same as model() fixture above, in order to get two independent
# fixtures.
@pytest.fixture(scope='package')
def model2(request):
    return get_model(request.param)
