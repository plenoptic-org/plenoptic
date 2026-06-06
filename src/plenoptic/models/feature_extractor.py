"""Adapter for compatibility with torchvision and timm models."""
# numpydoc ignore=EX01

import warnings
from collections import OrderedDict
from typing import Literal

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.models import feature_extraction

from ..plot.display import _clean_up_axes, imshow


class FeatureExtractorModel(torch.nn.Module):
    """
    Return features from model.

    This adapter combines a torch model with a feature extractor and optional transform,
    allowing us to target the output of a particular layer in a network for use with
    synthesis objects.

    This adapter is intended to work with :ref:`TorchVision` and :ref:`timm`, two model
    zoos from the deep learning community that contain a large number of models.

    For more details on the node naming conventions used here, please see the
    :ref:`relevant subheading <about-node-names>` in the `torchvision documentation
    <https://pytorch.org/vision/stable/feature_extraction.html>`_.

    Parameters
    ----------
    model
        The pytorch module to use.
    return_node
        The names of the nodes to return. See Examples and
        :func:`torchvision.feature_extraction`.
    transform
        Pre-processing transform to apply to image before passing to model. If
        ``None``, will not apply any transform.

    Examples
    --------
    Use with a torchvision model:

    >>> import plenoptic as po
    >>> import torchvision
    >>> weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    >>> tv_model = torchvision.models.resnet50(weights=weights)
    >>> tv_model.eval()
    >>> # This model's transform consists of resizing, cropping, and normalizing.
    >>> # We recommend only including the normalizing in the transform.
    >>> tv_transform = weights.transforms()
    >>> norm = torchvision.transforms.Normalize(tv_transform.mean, tv_transform.std)
    >>> model = FeatureExtractorModel(tv_model, "layer2", norm)
    >>> # this model requires a 3d input, and expects it to have a certain input size.
    >>> img = po.process.center_crop(po.data.einstein(False), tv_transform.crop_size[0])
    >>> img.shape)
    (1, 3, 224, 224)
    >>> model(img).shape
    (1, 401408)

    Use with timm a model. The primary difference is in the syntax for retrieving
    the model and the transform:

    >>> import timm
    >>> from timm.data import resolve_data_config
    >>> from timm.data.transforms_factory import create_transform
    >>> timm_model = timm.create_model(
    ...     "hf-hub:nateraw/resnet50-oxford-iiit-pet", pretrained=True
    ... )
    >>> timm_model.eval()
    >>> # Create Transform
    >>> timm_transform = create_transform(
    ...     **resolve_data_config(model.pretrained_cfg, model=model)
    ... )
    >>> # This model has the same resizing, cropping, normalizing transform as above,
    >>> # but timm allows us to explicitly select only the final step.
    >>> transform
    >>> timm_norm = timm_transform.transforms[-1]
    >>> timm_crop = timm_transform.transforms[1]
    >>> model = FeatureExtractorModel(tv_model, "layer2", timm_norm)
    >>> # this model requires a 3d input, and expects it to have a certain input size.
    >>> img = timm_crop(po.data.einstein(False))
    >>> img.shape
    (1, 3, 224, 224)
    >>> model(img).shape
    (1, 401408)

    The torchvision function
    :func:`torchvision.models.feature_extraction.get_graph_node_names` allows us to view
    possible node names:

    >>> from torchvision.models import feature_extraction
    >>> # This function returns two lists, one for nodes in train mode, one for those in
    >>> # eval mode. We want the eval mode list:
    >>> node_names = feature_extraction.get_node_names(tv_model)[1]
    >>> len(node_names)
    176
    >>> node_names[77:81]
    ['layer2.3.add', 'layer2.3.relu_2', 'layer3.0.conv1', 'layer3.0.bn1']
    >>> model = FeatureExtractorModel(tv_model, node_names[78], norm)
    >>> model(img).shape
    (1, 401408)

    We can even pass multiple node names, in which case all corresponding outputs are
    concatenated together.

    >>> model = FeatureExtractorModel(tv_model, ["layer2", "layer4"], norm)
    >>> model(img).shape
    (1, 501760)

    The order of elements in ``return_nodes`` does not matter: the outputs are always
    returned based on their order in ``model``.

    >>> rep = model(img)
    >>> model = FeatureExtractorModel(tv_model, ["layer4", "layer2"], norm)
    >>> rep[0, 0] == model(img)[0, 0]
    True

    The function :meth:`convert_to_dict` will convert the output of :meth:`forward` to a
    dictionary and return its elements to their original shape. This may be useful for
    plotting or investigation.

    >>> [(k, v.shape) for k, v in model.convert_to_dict(model(img))]
    [("layer2", torch.Size([1, 512, 28, 28])), ("layer4", torch.Size([1, 2048, 7, 7]))]

    Visualize model representation with :meth:`plot_representation`:

    .. plot::
       :context: reset

       >>> fig, axes = model.plot_representation(model(img))
    """

    def __init__(
        self,
        model: torch.nn.Module,
        return_node: str | list[str] | dict[str, str],
        transform: torch.nn.Module | None = None,
    ):
        super().__init__()
        self.transform = transform
        if isinstance(return_node, str):
            return_node = [return_node]
        self.extractor = feature_extraction.create_feature_extractor(model, return_node)
        self.model = model
        if hasattr(model, "training") and model.training:
            warnings.warn(
                "model is in training mode, you probably want to call eval()"
                " to switch to evaluation mode"
            )
        self._out_keys = None
        self._packed_shapes = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute feature activity of an input.

        We flatten across all dimensions except the batch / first dimension. This allows
        us to support returning features of different dimensionality (as is common
        across layers in deep nets), while still returning only a single tensor, as is
        necessary for our synthesis methods.

        Parameters
        ----------
        x
            The tensor to analyze.

        Returns
        -------
        representation_tensor
            The feature activity as a 2d tensor, of shape ``(batch, representation)``.

        See Also
        --------
        convert_to_dict
            Convert tensor representation to a dictionary, whose keys are the feature
            names, with their original shapes.
        """
        if self.transform is not None:
            x = self.transform(x)
        original_out = self.extractor(x)
        return self.convert_to_tensor(original_out)

    def convert_to_tensor(
        self, representation_dict: OrderedDict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Convert dictionary of statistics to a tensor.

        The output has shape ``(batch, representation)``, flattening and concatenating
        across all representation features, channels, and additional dimensions. The
        dictionary representation may be easier to make sense of.

        Parameters
        ----------
        representation_dict
             Dictionary of representation, whose keys are the feature names and whose
             values are tensors in the original shape.

        Returns
        -------
        representation_tensor
            Feature activity as a 2d tensor, of shape ``(batch, representation)``.

        See Also
        --------
        convert_to_dict
            Convert tensor representation to a dictionary, whose keys are the feature
            names, with their original shapes.
        """
        self._out_keys = representation_dict.keys()
        packed_out, self._packed_shapes = einops.pack(
            list(representation_dict.values()), "b *"
        )
        return packed_out

    def convert_to_dict(
        self, representation_tensor: torch.Tensor
    ) -> OrderedDict[str, torch.Tensor]:
        """
        Convert tensor of statistics to a dictionary.

        The output of :meth:`forward` is flattened so as to allow us to return a single
        tensor, regardless of the specified features. This function undoes that
        flattening, returning a dictionary whose keys are the feature names and whose
        values have the original shape. This may be useful for investigation or
        plotting.

        This function requires calling either :func:`forward` or
        :func:`convert_to_tensor` first, so that it knows how to properly reshape the
        input.

        Parameters
        ----------
        representation_tensor
            2d tensor of model representation, e.g., the output of :meth:`forward`.

        Returns
        -------
        representation_dict
            Dictionary of representation, with informative keys.

        Raises
        ------
        ValueError
            If :func:`forward` or :func:`convert_to_tensor` was not called before this
            one, because then we don't know how to properly reshape.

        See Also
        --------
        convert_to_tensor
            Convert dictionary representation to a 2d tensor.
        """
        if self._packed_shapes is None or self._out_keys is None:
            raise ValueError(
                "Call forward or convert_to_tensor before this function,"
                " otherwise we don't know how to properly reshape!"
            )
        unpacked = einops.unpack(representation_tensor, self._packed_shapes, "b *")
        return OrderedDict({k: v for k, v in zip(self._out_keys, unpacked)})

    def plot_representation(
        self,
        data: torch.Tensor | dict[str, torch.Tensor],
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] | None = None,
        ylim: tuple[float, float] | Literal[False] | None = None,
        batch_idx: int = 0,
        title: str | None = None,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """
        Plot model representation.

        A description goes here

        Parameters
        ----------
        data
            The data to show on the plot. Should look like the output of
            :meth:`forward` or :meth:`convert_to_dict`, with the exact same
            structure (e.g., as returned by another instance of this class).
        ax
            Axes where we will plot the data. If a ``plt.Axes`` instance, will
            subdivide into 6 or 8 new axes (depending on self.n_scales). If
            ``None``, we create a new figure.
        figsize
            The size of the figure to create. Must be ``None`` if ax is not ``None``. If
            both figsize and ax are ``None``, then we set ``figsize=(12, 15)``.
        ylim
            If not None, the y-limits to use for this plot. If None, we use the
            default, slightly adjusted so that the minimum is 0. If False, do not
            change y-limits.
        batch_idx
            Which index to take from the batch dimension (the first one).
        title
            Title for the plot.

        Returns
        -------
        fig
            Figure containing the plot.
        axes
            List of 6 or 8 axes containing the plot (depending on ``self.n_scales``).

        Raises
        ------
        ValueError
            If both ``figsize`` and ``ax`` are not ``None``.
        """
        if isinstance(data, torch.Tensor):
            data = self.convert_to_dict(data)["layer2"]

        # Select the batch index
        data = data[batch_idx]

        # Compute across channels spatal error
        spatial_error = torch.abs(data).mean(dim=0).detach().cpu().numpy()

        # Compute per-channel error
        error = torch.abs(data).mean(dim=(1, 2))  # Shape: (C,)
        sorted_idx = torch.argsort(error, descending=True)
        sorted_error = error[sorted_idx].detach().cpu().numpy()

        # Determine figure layout
        if ax is None:
            fig, axes = plt.subplots(
                2, 1, figsize=figsize, gridspec_kw={"height_ratios": [1, 1]}
            )
        else:
            ax = _clean_up_axes(
                ax, False, ["top", "right", "bottom", "left"], ["x", "y"]
            )
            gs = ax.get_subplotspec().subgridspec(2, 1, height_ratios=[3, 1])
            fig = ax.figure
            axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]

        # Plot average error across channels
        imshow(
            ax=axes[0],
            image=spatial_error[None, None, ...],
            title="Average Error Across Channels",
            vrange="auto0",
        )
        # axes[0].set_title()

        # Plot channel error distribution
        x_pos = np.arange(20)
        axes[1].bar(x_pos, sorted_error[:20], color="C1", alpha=0.7)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(sorted_idx[:20].tolist(), rotation=45)
        axes[1].set_xlabel("Channel")
        axes[1].set_ylabel("Absolute error")
        axes[1].set_title("Top 20 Channels Contributions to Error")

        if title is not None:
            fig.suptitle(title)

        return fig, axes
