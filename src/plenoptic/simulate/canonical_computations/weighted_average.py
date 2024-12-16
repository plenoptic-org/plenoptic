#!/usr/bin/env python3

import torch
from torch import Tensor
import einops
import warnings
from ...tools.conv import blur_downsample
from typing import Literal


class SimpleAverage(torch.nn.Module):
    """Module to average over the last two dimensions of input
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, keepdim: bool = False):
        """Average over the last two dimensions of input.
        """
        return torch.mean(x, dim=(-2, -1), keepdim=keepdim)


class WeightedAverage(torch.nn.Module):
    """Module to take a weighted average over last two dimensions of tensors.

    - Weights are set at initialization, must be non-negative, and 3d Tensors (different
      masks indexed on first dimension, height and width on last two).

    - If two weights are set, they are multiplied together when taking the average
      (e.g., separable polar angle and eccentricity weights).

    - Weights are normalized at initialization so that they sum to 1 (as best as
      possible). If any mask sums to near-zero, an exception is raised. If there's
      variation across mask sums, a warning is raised.

    Parameters
    ----------
    weights_1, weights_2 :
        3d Tensors defining the weights for the average.

    Attributes
    ----------
    weights :
        List of one or two 3d Tensors (depending on whether weights_2 set at
        initialization) containing the normalized weights.

    """
    def __init__(self, weights_1: Tensor, weights_2: Tensor | None = None):
        super().__init__()
        self._validate_weights(weights_1, "_1")
        self.register_buffer("_weights_1", weights_1)
        self._n_weights = 1
        input_einsum = "m1 h w"
        output_einsum = "m1"
        if weights_2 is not None:
            self._validate_weights(weights_2, "_2")
            if weights_1.shape[-2:] != weights_2.shape[-2:]:
                raise ValueError("weights_1 and weights_2 must have same height and width!")
            self._n_weights += 1
            input_einsum += ", m2 h w"
            output_einsum += " m2"
            self.register_buffer("_weights_2", weights_2)
        self._image_shape = weights_1.shape[-2:]
        self._output_einsum = output_einsum
        self._weight_einsum = f"{input_einsum} -> {output_einsum}"
        self._forward_einsum = f"{input_einsum}, b c {{extra_dims}} h w -> b c {{extra_dims}} {output_einsum} {{extra_out}}"
        self._extra_dims = ["", "i1", "i1 i2"]
        self._normalize_weights()

    def _normalize_weights(self):
        """Normalize
        """
        weight_sums = self.sum_weights()
        if torch.isclose(weight_sums, torch.zeros_like(weight_sums)).any():
            raise ValueError("Some of the weights sum to zero! This will not work out well.")
        var = weight_sums.var()
        if not torch.isclose(var, torch.zeros_like(var)):
            warnings.warn("Looks like there's some variation in the sums across your weights."
                          " That might be fine, but just wanted to make sure you knew...")
        mode = torch.mode(weight_sums.flatten()).values
        if not torch.isclose(mode, torch.ones_like(mode)):
            warnings.warn("Weights don't sum to 1, normalizing...")
            if self._n_weights == 1:
                self._weights_1 = self._weights_1 / mode
            else:
                self._weights_1 = self._weights_1 / mode.sqrt()
                self._weights_2 = self._weights_2 / mode.sqrt()

    @property
    def weights(self):
        # inspired by
        # https://discuss.pytorch.org/t/why-no-nn-bufferlist-like-function-for-registered-buffer-tensor/18884/10
        return [getattr(self, f"_weights_{j}") for j in range(1, self._n_weights+1)]

    @staticmethod
    def _validate_weights(weights: Tensor, idx: 'str' = '_1'):
        if weights.ndim != 3:
            raise ValueError(f"weights{idx} must be 3d!")
        if weights.min() < 0:
            raise ValueError(f"weights{idx} must be non-negative!")

    def forward(self, image: Tensor, keepdim: bool = False) -> Tensor:
        """Take the weighted average over last two dimensions of input.

        All other dimensions are preserved.

        Parameters
        ----------
        image :
            4d to 6d Tensor.
        keepdim :
            Whether to take the weighted average (False) or just apply the weights
            (True).

        Returns
        -------
        weighted_avg :
            Weighted average. Dimensionality depends on both the input's dimensionality
            and ``len(weights)``

        """
        extra_out = ""
        if keepdim:
            extra_out = " h w"
        extra_dims = self._extra_dims[image.ndim - 4]
        einsum_str = self._forward_einsum.format(extra_dims=extra_dims, extra_out=extra_out)
        return einops.einsum(*self.weights, image, einsum_str)

    def sum_weights(self) -> Tensor:
        """Sum weights, largely used for diagnostic purposes.

        Returns
        -------
        sum :
            1d or 2d tensor (depending on ``len(weights)``) containing the sum of all
            weights.

        """
        return einops.einsum(*self.weights, self._weight_einsum)


class WeightedAveragePyramid(torch.nn.Module):
    """Module to take weighted average across scales.

    This initializes a ``WeightedAverage`` per scale, down-sampling by a factor of 2
    using the ``blur_downsample`` method (and normalizing independently).

    As with ``WeightedAverage``:

    - Weights are set at initialization, must be non-negative, and 3d Tensors (different
      masks indexed on first dimension, height and width on last two).

    - If two weights are set, they are multiplied together when taking the average
      (e.g., separable polar angle and eccentricity weights).

    - Weights are normalized at initialization so that they sum to 1 (as best as
      possible). If any mask sums to near-zero, an exception is raised. If there's
      variation across mask sums, a warning is raised.

    Parameters
    ----------
    weights_1, weights_2 :
        3d Tensors defining the weights for the average.
    n_scales :
        Number of scales.

    Attributes
    ----------
    weights :
        ModuleList of ``WeightedAverage`` at each scale

    """
    def __init__(self, weights_1: Tensor,
                 weights_2: Tensor | None = None,
                 n_scales: int = 4):
        super().__init__()
        self._n_weights = 1 if weights_2 is None else 2
        self.n_scales = n_scales
        for i in range(n_scales):
            if i != 0:
                weights_1 = blur_downsample(weights_1.unsqueeze(0), 1, scale_filter=True)
                weights_1 = weights_1.squeeze(0).clip(min=0)
                if weights_2 is not None:
                    weights_2 = blur_downsample(weights_2.unsqueeze(0), 1, scale_filter=True)
                    weights_2 = weights_2.squeeze(0).clip(min=0)
            # it's possible negative values will get introduced by the downsampling
            # above, in which case we remove them
            weights = WeightedAverage(weights_1, weights_2)
            setattr(self, f"_weights_scale_{i}", weights)

    @property
    def weights(self):
        return torch.nn.ModuleList([getattr(self, f"_weights_scale_{i}") for i in range(self.n_scales)])

    def forward(self, image: list[Tensor]) -> list[Tensor]:
        """Take the weighted average over last two dimensions of each input in list.

        All other dimensions are preserved.

        Parameters
        ----------
        image :
            List of 4d to 6d Tensor, each of which has been downsampled by a factor of 2.
        keepdim :
            Whether to take the weighted average (False) or just apply the weights
            (True).

        Returns
        -------
        weighted_avg :
            Weighted average. Dimensionality depends on both the input's dimensionality
            and whether ``weights_2`` was set at initialization.

        """
        return torch.stack([w(x) for x, w in zip(image, self.weights)], dim=2)

    def sum_weights(self) -> Tensor:
        """Sum weights, largely used for diagnostic purposes.

        Returns
        -------
        sum :
            2d or 3d tensor (depending on whether ``weights_2`` was set at
            initialization) containing the sum of all weights on each scale.

        """
        sums = []
        for i in range(self.n_scales):
            sums.append(getattr(self, f"_weights_scale_{i}").sum_weights())
        return einops.pack(sums, f"* {self._weights_scale_0.output_einsum}")[0]
