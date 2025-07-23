import os

import pytest
import torch

import plenoptic as po
from plenoptic.data.fetch import fetch_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if we have a second gpu, can use it for some tests
DEVICE2 = torch.device(1) if torch.cuda.device_count() > 1 else DEVICE
IMG_DIR = fetch_data("test_images.tar.gz")

# we do this to enable deterministic behavior on the gpu, for
# PortillaSimoncelli, see
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility for
# details
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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


@pytest.fixture(scope="package")
def curie_img():
    return po.load_images(IMG_DIR / "256" / "curie.pgm").to(DEVICE)


@pytest.fixture(scope="package")
def einstein_img():
    return po.load_images(IMG_DIR / "256" / "einstein.pgm").to(DEVICE)


@pytest.fixture(scope="package")
def einstein_img_small(einstein_img):
    return po.tools.center_crop(einstein_img, 64).to(DEVICE)


@pytest.fixture(scope="package")
def color_img():
    img = po.load_images(IMG_DIR / "256" / "color_wheel.jpg", as_gray=False).to(DEVICE)
    return img[..., :256, :256]


@pytest.fixture(scope="package")
def parrot_square():
    img = po.load_images(IMG_DIR / "mixed" / "Parrot.png").to(DEVICE)
    return po.tools.center_crop(img, 254)


@pytest.fixture(scope="package")
def parrot_square_double(parrot_square):
    return parrot_square.to(torch.float64)


@pytest.fixture(scope="package")
def einstein_img_double(einstein_img):
    return einstein_img.to(torch.float64)


@pytest.fixture(scope="package")
def basic_stim():
    return po.load_images(IMG_DIR / "256").to(DEVICE)


def get_model(name):
    if name == "SPyr":
        # in order to get a tensor back, need to wrap steerable pyramid so that
        # we can call convert_pyr_to_tensor in the forward call. in order for
        # that to work, downsample must be False
        class spyr(po.simul.SteerablePyramidFreq):
            def __init__(self, *args, **kwargs):
                kwargs.pop("downsample", None)
                super().__init__(*args, downsample=False, **kwargs)

            def forward(self, *args, **kwargs):
                coeffs = super().forward(*args, **kwargs)
                pyr_tensor, _ = po.simul.SteerablePyramidFreq.convert_pyr_to_tensor(
                    coeffs
                )
                return pyr_tensor

        # setting height=1 and # order=1 limits the size
        return spyr((256, 256), height=1, order=1).to(DEVICE)
    elif name == "LPyr":
        # in order to get a tensor back, need to wrap laplacian pyramid so that
        # we can flatten the output. in practice, not the best way to use this
        class lpyr(po.simul.LaplacianPyramid):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def forward(self, *args, **kwargs):
                coeffs = super().forward(*args, **kwargs)
                return torch.cat([c.flatten(-2) for c in coeffs], -1)

        return lpyr().to(DEVICE)
    elif name == "nlpd":
        return po.metric.nlpd
    elif name == "mse":
        return po.metric.naive.mse
    elif name == "ColorModel":
        model = ColorModel().to(DEVICE)
        po.tools.remove_grad(model)
        model.eval()
        return model

    # naive models
    elif name == "naive.Identity":
        model = po.simul.Identity().to(DEVICE)
        model.eval()
        return model
    elif name == "naive.CenterSurround":
        model = po.simul.CenterSurround((31, 31)).to(DEVICE)
        model.eval()
        return model
    elif name == "naive.Gaussian":
        model = po.simul.Gaussian((31, 31)).to(DEVICE)
        model.eval()
        return model
    elif name == "naive.Linear":
        model = po.simul.Linear((31, 31)).to(DEVICE)
        model.eval()
        return model

    # FrontEnd models:
    elif name == "frontend.LinearNonlinear":
        model = po.simul.LinearNonlinear((31, 31)).to(DEVICE)
        model.eval()
        return model
    elif name == "frontend.LinearNonlinear.nograd":
        model = po.simul.LinearNonlinear((31, 31)).to(DEVICE)
        po.tools.remove_grad(model)
        model.eval()
        return model
    elif name == "frontend.LuminanceGainControl":
        model = po.simul.LuminanceGainControl((31, 31)).to(DEVICE)
        model.eval()
        return model
    elif name == "frontend.LuminanceContrastGainControl":
        model = po.simul.LuminanceContrastGainControl((31, 31)).to(DEVICE)
        model.eval()
        return model
    elif name == "frontend.OnOff":
        model = po.simul.OnOff((31, 31), pretrained=True, cache_filt=True).to(DEVICE)
        model.eval()
        return model
    elif name == "frontend.OnOff.nograd":
        model = po.simul.OnOff((31, 31), pretrained=True, cache_filt=True).to(DEVICE)
        po.tools.remove_grad(model)
        model.eval()
        return model
    elif name == "VideoModel":
        # super simple model that combines across the batch dimension, as a
        # model with a temporal component would do
        class VideoModel(po.simul.OnOff):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def forward(self, *args, **kwargs):
                # this will do on/off on each batch separately
                rep = super().forward(*args, **kwargs)
                return rep.mean(0)

        model = VideoModel((31, 31), pretrained=True, cache_filt=True).to(DEVICE)
        po.tools.remove_grad(model)
        model.eval()
        return model
    elif name == "PortillaSimoncelli":
        return po.simul.PortillaSimoncelli((256, 256)).to(DEVICE)
    elif name == "NonModule":

        class NonModule:
            def __init__(self):
                self.name = "nonmodule"

            def __call__(self, x):
                return 1 * x

        return NonModule()
    elif "diff_dims" in name:

        class DimModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.target_dims = int(name.replace("diff_dims-", ""))

            def forward(self, img):
                img = img.flatten()
                while img.ndimension() != self.target_dims:
                    img = img.unsqueeze(0)
                return img

        model = DimModel()
        model.eval()
        return model


@pytest.fixture(scope="package")
def model(request):
    return get_model(request.param)


# this is the same as model() fixture above, in order to get two independent
# fixtures.
@pytest.fixture(scope="package")
def model2(request):
    return get_model(request.param)
