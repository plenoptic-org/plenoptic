"""Simple Metamer Class"""

import torch
from tqdm.auto import tqdm

from ..tools import optim
from ..tools.validate import validate_input, validate_model
from .synthesis import Synthesis


class SimpleMetamer(Synthesis):
    r"""Simple version of metamer synthesis.

    This doesn't have any of the bells and whistles of the full Metamer class,
    but does perform basic metamer synthesis: given a target image and a model,
    synthesize a new image (initialized with uniform noise) that has the same
    model output.

    This is meant as a demonstration of the basic logic of synthesis.

    Parameters
    ----------
    image
        A 4d tensor, this is the image whose model representation we wish to
        match.
    model
        The visual model whose representation we wish to match.

    """

    def __init__(self, image: torch.Tensor, model: torch.nn.Module):
        validate_model(
            model,
            image_shape=image.shape,
            image_dtype=image.dtype,
            device=image.device,
        )
        self.model = model
        validate_input(image)
        self.image = image
        self.metamer = torch.rand_like(self.image, requires_grad=True)
        self.target_representation = self.model(self.image).detach()
        self.optimizer = None
        self.losses = []

    def synthesize(
        self,
        max_iter: int = 100,
        optimizer: None | torch.optim.Optimizer = None,
    ) -> torch.Tensor:
        """Synthesize a simple metamer.

        If called multiple times, will continue where we left off.

        Parameters
        ----------
        max_iter
            Number of iterations to run synthesis for.
        optimizer
            The optimizer to use. If None and this is the first time calling
            synthesize, we use Adam(lr=.01, amsgrad=True); if synthesize has
            been called before, we reuse the previous optimizer.

        Returns
        -------
        metamer
            The synthesized metamer

        """
        if optimizer is None:
            if self.optimizer is None:
                self.optimizer = torch.optim.Adam([self.metamer], lr=0.01, amsgrad=True)
        else:
            self.optimizer = optimizer

        pbar = tqdm(range(max_iter))
        for _ in pbar:

            def closure():
                self.optimizer.zero_grad()
                metamer_representation = self.model(self.metamer)
                # We want to make sure our metamer ends up in the range [0, 1],
                # so we penalize all values outside that range in the loss
                # function. You could theoretically also just clamp metamer on
                # each step of the iteration, but the penalty in the loss seems
                # to work better in practice
                loss = optim.mse(metamer_representation, self.target_representation)
                loss = loss + 0.1 * optim.penalize_range(self.metamer, (0, 1))
                self.losses.append(loss.item())
                loss.backward(retain_graph=False)
                pbar.set_postfix(loss=loss.item())
                return loss

            self.optimizer.step(closure)

    def save(self, file_path: str):
        r"""Save all relevant (non-model) variables in .pt file.

        Parameters
        ----------
        file_path :
            The path to save the SimpleMetamer object to.

        """
        super().save(file_path, attrs=None)

    def load(self, file_path: str, map_location: str | None = None):
        r"""Load all relevant attributes from a .pt file.

        Note this operates in place and so doesn't return anything.

        Parameters
        ----------
        file_path
            The path to load the synthesis object from
        """
        check_attributes = ["target_representation", "image"]
        super().load(
            file_path,
            check_attributes=check_attributes,
            map_location=map_location,
        )

    def to(self, *args, **kwargs):
        r"""Move and/or cast the parameters and buffers.

        This can be called as
        .. function:: to(device=None, dtype=None, non_blocking=False)
        .. function:: to(dtype, non_blocking=False)
        .. function:: to(tensor, non_blocking=False)
        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point desired :attr:`dtype` s. In addition, this method will
        only cast the floating point parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices. When calling this method to move tensors
        to a CUDA device, items in ``attrs`` that start with "saved_" will not
        be moved.
        .. note::
            This method modifies the module in-place.
        Args:
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module
            dtype (:class:`torch.dtype`): the desired floating point type of
                the floating point parameters and buffers in this module
            tensor (torch.Tensor): Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module
            attrs (:class:`list`): list of strs containing the attributes of
                this object to move to the specified device/dtype
        Returns:
            Module: self
        """
        attrs = ["model", "image", "target_representation", "metamer"]
        super().to(*args, attrs=attrs, **kwargs)
        return self
