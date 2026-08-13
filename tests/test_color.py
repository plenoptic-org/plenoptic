import pytest
import torch

import plenoptic as po
from conftest import DEVICE


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_opc(batched, dtype):
    image = torch.rand(1, 3, 8, 8, device=DEVICE, dtype=dtype)
    if not batched:
        image = image.squeeze(0)
    transform = po.process.OPC().to(device=DEVICE, dtype=dtype)
    rgb_to_lms = torch.tensor(
        [
            [0.3811, 0.5783, 0.0402],
            [0.1967, 0.7244, 0.0782],
            [0.0241, 0.1288, 0.8444],
        ]
    ).to(device=DEVICE, dtype=dtype)
    lms_to_opponent = torch.tensor(
        [
          [0.5, 0.5, 0.0],
          [-4.0, 4.0, 0.0],
          [0.5, 0.5, -1.0],
        ]
    ).to(device=DEVICE, dtype=dtype)
    expected = torch.einsum("ij,...jhw->...ihw", lms_to_opponent @ rgb_to_lms, image)
    torch.testing.assert_close(transform(image), expected)
    assert transform(image).shape == image.shape


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pca(batched, dtype):
    target = torch.rand(1, 3, 16, 16, device=DEVICE, dtype=dtype)
    if not batched:
        target = target.squeeze(0)
    transform = po.process.PCA(target)
    transformed = transform(target)
    assert transformed.shape == target.shape

    # Transformed data should have zero mean and identity covariance
    transformed = transformed.reshape(3, -1)
    torch.testing.assert_close(
        transformed.mean(-1),
        torch.zeros(3, device=DEVICE, dtype=dtype),
        atol=1e-6,
        rtol=0,
    )
    covariance = transformed @ transformed.mT / transformed.shape[-1]
    torch.testing.assert_close(
        covariance,
        torch.eye(3, device=DEVICE, dtype=dtype),
        atol=1e-5,
        rtol=1e-5,
    )


def test_pca_batch_application_and_gradients():
    target = torch.rand(1, 3, 8, 8, device=DEVICE)
    image = torch.rand(2, 3, 8, 8, device=DEVICE, requires_grad=True)
    transform = po.process.PCA(target)
    transformed = transform(image)
    assert transformed.shape == image.shape
    transformed.square().mean().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()


@pytest.mark.parametrize(
    "image",
    [
      torch.rand(2, 3, 8, 8), # Invalid: batch size > 1
      torch.rand(1, 2, 8, 8), # Invalid: channel size != 3
      torch.rand(3, 8), # Invalid: not 3D or 4D
    ],
)
def test_color_invalid_fit_image(image):
    # Test PCA initialization with invalid images
    with pytest.raises(ValueError):
        po.process.PCA(image)

    # Multi-batch images are not allowed for PCA initialization,
    # but are allowed for OPC and PCA application
    is_multibatch = (image.ndim == 4) & (image.shape[0] > 1)
    if not is_multibatch:
        # Test PCA application with invalid images
        transform = po.process.PCA(torch.rand(1, 3, 8, 8))
        with pytest.raises(ValueError):
            transform(image)
        # Test OPC application with invalid images
        with pytest.raises(ValueError):
            po.process.OPC()(image)
