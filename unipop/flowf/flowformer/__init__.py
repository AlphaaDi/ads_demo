from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = [
    "configs",
    "core",
    "OpticalFlowFormer",
]

from flowformer import configs
from flowformer import core
from flowformer.optical_flow_former import OpticalFlowFormer

# Needs to be last line
__version__ = "0.1"