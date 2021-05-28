"""Simple Metamer Class
"""

import torch
from tqdm.auto import tqdm
from .synthesis import Synthesis


class SimpleMetamer(Synthesis):
    r"""Abstract super-class for synthesis methods

    All synthesis methods share a variety of similarities and thus need
    to have similar methods. Some of these can be implemented here and
    simply inherited, some of them will need to be different for each
    sub-class and thus are marked as abstract methods here

    """

    def __init__(self, model: torch.nn.Module, target_signal: torch.Tensor, max_iter:int=100, lr:float=.01):
        self.model = model
        self.max_iter = max_iter
        self.lr = lr
        self.target_signal = target_signal
        self.synthesized_signal = torch.rand_like(
            self.target_signal,
            requires_grad=True
        )
        self.target_model_response = self.model(self.target_signal)
        self.loss = torch.nn.MSELoss()

    def synthesize(self) -> torch.Tensor:

        self.optimizer = torch.optim.SGD([self.synthesized_signal], lr=self.lr)

        step = 0
        self.losses = []
        
        pbar = tqdm(range(self.max_iter))
        for step in pbar:

            def closure():
                self.optimizer.zero_grad()
                synthesized_model_response = self.model(self.synthesized_signal)
                loss = self.loss(self.target_model_response,synthesized_model_response)
                self.losses.append(loss.item())
                loss.backward(retain_graph=True)
                pbar.set_postfix(loss=loss.item())
                return loss

            self.optimizer.step(closure)
        


        return self.synthesized_signal


    def save(self, file_path):
        r"""Save all relevant (non-model) variables in .pt file.
        
        Parameters
        ----------
        file_path : str
            The path to save the synthesis object to
        attrs : list
            List of strs containing the names of the attributes of this
            object to save.
        """


        attributes=['target_model_response','target_signal','synthesized_signal','losses','optimizer']
        super().save(file_path,attributes=None)


    def load(self, file_path, map_location=None):
        r"""Load all relevant attributes from a .pt file.
        
        Note this operates in place and so doesn't return anything.
        Parameters
        ----------
        file_path : str
            The path to load the synthesis object from
        Examples
        --------
        >>> 
        """
        check_attributes = ['target_model_response','target_signal']
        super().load(file_path,check_attributes=check_attributes,map_location=map_location)
        
    

    def to(self, *args, **kwargs):
        r"""Moves and/or casts the parameters and buffers.
        Similar to ``save``, this is an abstract method only because you
        need to define the attributes to call to on.
        
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
        
        super().to(*args,attrs=['model','target_signal','target_model_response','synthesized_signal'],**kwargs)


        return self