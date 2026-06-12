"""Adapter for compatibility with torchvision and timm models."""
# numpydoc ignore=EX01

import warnings
from collections import OrderedDict

import einops
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.models import feature_extraction

from ..plot import display


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
    >>> # This model's transform consists of resizing, cropping, and normalizing.
    >>> # We recommend only including the normalizing in the transform.
    >>> tv_transform = weights.transforms()
    >>> norm = torchvision.transforms.Normalize(tv_transform.mean, tv_transform.std)
    >>> model = po.models.FeatureExtractorModel(tv_model, "layer2", norm).eval()
    >>> # this model requires a 3d input, and expects it to have a certain input size.
    >>> img = po.process.center_crop(po.data.einstein(False), tv_transform.crop_size[0])
    >>> img.shape
    torch.Size([1, 3, 224, 224])
    >>> model(img).shape
    torch.Size([1, 401408])
    >>> po.remove_grad(model)
    >>> po.validate.validate_model(model, image_shape=img.shape)

    Use with timm a model. The primary difference is in the syntax for retrieving
    the model and the transform:

    >>> import timm
    >>> from timm.data import resolve_data_config
    >>> from timm.data.transforms_factory import create_transform
    >>> timm_model = timm.create_model(
    ...     "hf-hub:nateraw/resnet50-oxford-iiit-pet", pretrained=True
    ... )
    >>> # Create Transform
    >>> timm_transform = create_transform(
    ...     **resolve_data_config(model.pretrained_cfg, model=model)
    ... )
    >>> # This model has the same resizing, cropping, normalizing transform as above,
    >>> # but timm allows us to explicitly select the different steps
    >>> transform
    >>> timm_norm = timm_transform.transforms[-1]
    >>> timm_crop = timm_transform.transforms[1]
    >>> model = po.models.FeatureExtractorModel(tv_model, "layer2", timm_norm).eval()
    >>> # this model requires a 3d input, and expects it to have a certain input size.
    >>> img = timm_crop(po.data.einstein(False))
    >>> img.shape
    torch.Size([1, 3, 224, 224])
    >>> model(img).shape
    torch.Size([1, 401408])
    >>> po.remove_grad(model)
    >>> po.validate.validate_model(model, image_shape=img.shape)

    The torchvision function
    :func:`torchvision.models.feature_extraction.get_graph_graph_node_names` allows us
    to view possible node names:

    >>> from torchvision.models import feature_extraction
    >>> # This function returns two lists, one for nodes in train mode, one for those in
    >>> # eval mode. We want the eval mode list:
    >>> node_names = feature_extraction.get_graph_node_names(tv_model)[1]
    >>> len(node_names)
    176
    >>> node_names[77:81]
    ['layer2.3.add', 'layer2.3.relu_2', 'layer3.0.conv1', 'layer3.0.bn1']
    >>> model = po.models.FeatureExtractorModel(tv_model, node_names[78], norm)
    >>> model(img).shape
    torch.Size([1, 401408])

    We can even pass multiple node names, in which case all corresponding outputs are
    concatenated together.

    >>> model = po.models.FeatureExtractorModel(tv_model, ["layer2", "layer4"], norm)
    >>> model(img).shape
    torch.Size([1, 501760])

    The order of elements in ``return_nodes`` does not matter: the outputs are always
    returned based on their order in ``model``.

    >>> rep = model(img)
    >>> model = po.models.FeatureExtractorModel(tv_model, ["layer4", "layer2"], norm)
    >>> rep[0, 0] == model(img)[0, 0]
    tensor(True)

    The function :meth:`convert_to_dict` will convert the output of :meth:`forward` to a
    dictionary and return its elements to their original shape. This may be useful for
    plotting or investigation.

    >>> [(k, v.shape) for k, v in model.convert_to_dict(model(img)).items()]
    [('layer2', torch.Size([1, 512, 28, 28])), ('layer4', torch.Size([1, 2048, 7, 7]))]

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
        elif hasattr(model, "training") and not model.training:
            # by default, all torch modules are in training mode. make sure
            # FeatureExtractor mode matches that of the underlying model
            self.eval()
        self._out_keys = None
        self._packed_shapes = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute feature activity of an input.

        We flatten across all dimensions except the batch / first dimension. This allows
        us to support returning features of different shapes and dimensionality (as is
        common across layers in deep nets), while still returning only a single tensor,
        as is necessary for our synthesis methods.

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

        Examples
        --------
        >>> import plenoptic as po
        >>> import torchvision
        >>> weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        >>> tv_model = torchvision.models.resnet50(weights=weights)
        >>> # This model's transform consists of resizing, cropping, and normalizing.
        >>> # We recommend only including the normalizing in the transform.
        >>> tv_transform = weights.transforms()
        >>> norm = torchvision.transforms.Normalize(tv_transform.mean, tv_transform.std)
        >>> model = po.models.FeatureExtractorModel(tv_model, "layer2", norm).eval()
        >>> # this model requires a 3d input, and expects it to have a certain input
        >>> # size.
        >>> img = po.process.center_crop(
        ...     po.data.einstein(False), tv_transform.crop_size[0]
        ... )
        >>> model(img).shape
        torch.Size([1, 401408])
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

        Examples
        --------
        >>> import plenoptic as po
        >>> import torchvision
        >>> weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        >>> tv_model = torchvision.models.resnet50(weights=weights)
        >>> # This model's transform consists of resizing, cropping, and normalizing.
        >>> # We recommend only including the normalizing in the transform.
        >>> tv_transform = weights.transforms()
        >>> norm = torchvision.transforms.Normalize(tv_transform.mean, tv_transform.std)
        >>> model = po.models.FeatureExtractorModel(
        ...     tv_model, ["layer2", "layer4"], norm
        ... )
        >>> # this model requires a 3d input, and expects it to have a certain input
        >>> # size.
        >>> img = po.process.center_crop(
        ...     po.data.einstein(False), tv_transform.crop_size[0]
        ... )
        >>> representation_tensor = model(img)
        >>> representation_dict = model.convert_to_dict(representation_tensor)
        >>> representation_tensor_new = model.convert_to_tensor(representation_dict)
        >>> torch.equal(representation_tensor, representation_tensor_new)
        True
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

        Examples
        --------
        >>> import plenoptic as po
        >>> import torchvision
        >>> weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        >>> tv_model = torchvision.models.resnet50(weights=weights)
        >>> # This model's transform consists of resizing, cropping, and normalizing.
        >>> # We recommend only including the normalizing in the transform.
        >>> tv_transform = weights.transforms()
        >>> norm = torchvision.transforms.Normalize(tv_transform.mean, tv_transform.std)
        >>> model = po.models.FeatureExtractorModel(
        ...     tv_model, ["layer2", "layer4"], norm
        ... )
        >>> # this model requires a 3d input, and expects it to have a certain input
        >>> # size.
        >>> img = po.process.center_crop(
        ...     po.data.einstein(False), tv_transform.crop_size[0]
        ... )
        >>> representation_dict = model.convert_to_dict(model(img))
        >>> [(k, v.shape) for k, v in representation_dict.items()]
        [('layer2', torch.Size([1, 512, 28, 28])),
         ('layer4', torch.Size([1, 2048, 7, 7]))]
        """
        if self._packed_shapes is None or self._out_keys is None:
            raise ValueError(
                "Call forward or convert_to_tensor before this function,"
                " otherwise we don't know how to properly reshape!"
            )
        unpacked = einops.unpack(representation_tensor, self._packed_shapes, "b *")
        return OrderedDict({k: v for k, v in zip(self._out_keys, unpacked)})

    def update_plot(
        self,
        axes: mpl.axes.Axes | list[mpl.axes.Axes],
        data: torch.Tensor | dict,
        batch_idx: int = 0,
        rescale_ylim: bool = False,
    ) -> list:
        """
        Update representation plot (for an animation).

        This is a helper function for creating an animation over time.

        Parameters
        ----------
        axes
            The list of axes to update. We assume that these are the axes created by
            :func:`plot_representation` and so contain artists in the correct order.
        data
            The new data to plot.
        batch_idx
            Which index to take from the batch dimension.
        rescale_ylim
            Whether to rescale the ylimits of the per-channel plot or not.

        Returns
        -------
        artists
            A list of the artists used to update the information on the
            plots.

        See Also
        --------
        plot_representation
            Create plots to summarize model representation, which we assume created
            the axes passed to this function for updating.
        :func:`plenoptic.plot.update_plot`
            Generic ``update_plot`` function.
        :func:`plenoptic.plot.synthesis_animate`
            Function which creates a video of synthesis process over time, makes use
            of this function.

        Examples
        --------
        .. plot::
           :context: reset

           >>> import plenoptic as po
           >>> import torchvision
           >>> weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
           >>> tv_model = torchvision.models.resnet50(weights=weights)
           >>> # This model's transform consists of resizing, cropping, and normalizing.
           >>> # We recommend only including the normalizing in the transform.
           >>> tv_transform = weights.transforms()
           >>> norm = torchvision.transforms.Normalize(
           ...     tv_transform.mean, tv_transform.std
           ... )
           >>> model = po.models.FeatureExtractorModel(tv_model, "layer2", norm)
           >>> # this model requires a 3d input, and expects it to have a certain input
           >>> # size.
           >>> img = po.process.center_crop(
           ...     po.data.einstein(False), tv_transform.crop_size[0]
           ... )
           >>> fig, axes = model.plot_representation(model(img))
           >>> img = po.process.center_crop(
           ...     po.data.curie(False), tv_transform.crop_size[0]
           ... )
           >>> model.update_plot(axes, model(img))
           [<matplotlib...>]
        """
        if isinstance(data, torch.Tensor):
            data = self.convert_to_dict(data)

        artists = []
        per_channel_reps = []
        for i, (k, v) in enumerate(data.items()):
            # Average representation across channels
            avg_channel_rep = v.mean(dim=1, keepdim=True)
            # Average representation across additional dimensions (probably space)
            per_channel_rep = v.mean(dim=tuple(np.arange(2, v.ndim)))
            while per_channel_rep.ndim < 3:
                per_channel_rep = per_channel_rep.unsqueeze(0)
            per_channel_reps.append(per_channel_rep)
            art = display.update_plot(
                axes[2 * i : 2 * (i + 1)],
                {"00": avg_channel_rep, "01": per_channel_rep},
                batch_idx=batch_idx,
            )
            artists.extend(art)
        if rescale_ylim:
            display._rescale_ylim(axes[1::2], torch.cat(per_channel_reps))
        return artists

    def plot_representation(
        self,
        data: torch.Tensor | dict[str, torch.Tensor],
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] | None = None,
        batch_idx: int = 0,
        title: str | None = None,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """
        Plot model representation.

        This creates two plots: one containing the representation averaged across all
        channels, and one containing the per-channel representation, i.e., the
        representation averaged across all dimensions *except* channels.

        Intended for neural networks, e.g., models whose output at each node has
        many channels and one or two additional dimensions.

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
            both figsize and ax are ``None``, then we set ``figsize=(7, 5)``.
        batch_idx
            Which index to take from the batch dimension (the first one).
        title
            Title for the plot.

        Returns
        -------
        fig
            Figure containing the plot.
        axes
            List of axes containing the plot. Number of axes will be two per node.

        Raises
        ------
        ValueError
            If both ``figsize`` and ``ax`` are not ``None``.

        Examples
        --------
        .. plot::
           :context: reset

           >>> import plenoptic as po
           >>> import torchvision
           >>> weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
           >>> tv_model = torchvision.models.resnet50(weights=weights)
           >>> # This model's transform consists of resizing, cropping, and normalizing.
           >>> # We recommend only including the normalizing in the transform.
           >>> tv_transform = weights.transforms()
           >>> norm = torchvision.transforms.Normalize(
           ...     tv_transform.mean, tv_transform.std
           ... )
           >>> model = po.models.FeatureExtractorModel(tv_model, "layer2", norm)
           >>> # this model requires a 3d input, and expects it to have a certain input
           >>> # size.
           >>> img = po.process.center_crop(
           ...     po.data.einstein(False), tv_transform.crop_size[0]
           ... )
           >>> model.plot_representation(model(img))
           (<Figure ...>, [<Axes...>, <Axes...>])

        This function creates two axes per node, one showing the representation averaged
        across channels, one showing it per channel (averaging across any additional
        dimensions):

        .. plot::
           :context: close-figs

           >>> model = po.models.FeatureExtractorModel(
           ...     tv_model, ["layer2", "layer4"], norm
           ... )
           >>> model.plot_representation(model(img))
           (<Figure ...>, [<Axes...>, <Axes...>, <Axes...>, <Axes...>])

        Plot the dictionary representation:

        .. plot::
           :context: close-figs

           >>> model.plot_representation(model.convert_to_dict(model(img)))
           (<Figure ...>, [<Axes...>, <Axes...>, <Axes...>, <Axes...>])

        Plot on an existing axes object:

        .. plot::
           :context: close-figs

           >>> fig, axes = plt.subplots(1, 2)
           >>> model.plot_representation(model.convert_to_dict(model(img)), ax=axes[1])
           (<Figure ...>, [<Axes...>, <Axes...>, <Axes...>, <Axes...>])
        """
        if ax is None and figsize is None:
            figsize = (7, 5)
        elif ax is not None and figsize is not None:
            raise ValueError("figsize can't be set if ax is not None")

        if isinstance(data, torch.Tensor):
            data = self.convert_to_dict(data)

        # Determine figure layout
        n_cols = len(data)
        axes = []
        if ax is None:
            fig = plt.figure(figsize=figsize)
            gs = mpl.gridspec.GridSpec(1, n_cols, fig)
        else:
            ax = display._clean_up_axes(
                ax, False, ["top", "right", "bottom", "left"], ["x", "y"]
            )
            gs = ax.get_subplotspec().subgridspec(1, n_cols)
            fig = ax.figure

        for i, (k, v) in enumerate(data.items()):
            ax = fig.add_subplot(gs[i])

            # Average representation across channels
            avg_channel_rep = v.mean(dim=1, keepdim=True)
            # Average representation across additional dimensions (probably space)
            per_channel_rep = v.mean(dim=tuple(np.arange(2, v.ndim)))
            while per_channel_rep.ndim < 3:
                per_channel_rep = per_channel_rep.unsqueeze(0)

            if avg_channel_rep.ndim == 3:
                height_ratios = [1, 1]
            elif avg_channel_rep.ndim == 4:
                height_ratios = [2, 1]

            # this warning is not relevant here
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="data has keys, so we're ignoring title"
                )
                ax = display.plot_representation(
                    data={
                        f"{k} avg across channels": avg_channel_rep,
                        f"{k} per channel (n={v.shape[1]})": per_channel_rep,
                    },
                    ax=ax,
                    batch_idx=batch_idx,
                    axes_direction="vertical",
                    gridspec_kwargs={"height_ratios": height_ratios},
                )
            axes.extend(ax)

        if title is not None:
            fig.suptitle(title)

        return fig, axes
