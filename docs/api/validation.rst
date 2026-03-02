.. _validation-api:

Validation
----------

As plenoptic requires models do not have gradients attached to their parameters, we provide a convenience function for removing them.

.. currentmodule:: plenoptic
.. autosummary::
   :signatures: none
   :toctree: generated

   ~remove_grad

.. rubric:: Validation functions
   :heading-level: 3

The following functions are used to validate that the user-supplied inputs are compatible with our synthesis objects.

.. autosummary::
   :signatures: none
   :toctree: generated

   ~validate.validate_model
   ~validate.validate_input
   ~validate.validate_metric
   ~validate.validate_coarse_to_fine
   ~validate.validate_convert_tensor_dict
