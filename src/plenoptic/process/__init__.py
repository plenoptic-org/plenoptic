"""Image-processing methods.

These classes and functions all process images in some way, and thus may be helpful for
constructing custom models or metrics. As is, they are not compatible with any of the
synthesis methods.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
