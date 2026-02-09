"""
TCI-Polaron Core Library

This package contains the core implementations for:
- Tensor Cross Interpolation (TCI)
- Quantized Tensor Train (QTT) encoding
- Adaptive Cross Interpolation (ACI)
"""

from .tci_core import TCIFitter
from .qtt_utils import QTTEncoder
from .tci_utils import compute_tci_integral, compute_tci_integral_reference
from .physics_models import vectorized_gaussian
from .aci_core import AdaptiveTCI

__all__ = [
    'TCIFitter',
    'QTTEncoder', 
    'compute_tci_integral',
    'compute_tci_integral_reference',
    'vectorized_gaussian',
    'AdaptiveTCI',
]
