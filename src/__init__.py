"""
TCI-Polaron Core Library

This package contains the core implementations for:
- Tensor Cross Interpolation (TCI)
- Quantized Tensor Train (QTT) encoding
- Adaptive Cross Interpolation (ACI)
- Holstein Polaron self-energy
"""

from .tci_core import TCIFitter
from .qtt_utils import QTTEncoder
from .tci_utils import compute_tci_integral, compute_tci_integral_reference
from .physics_models import (vectorized_gaussian, bare_electron_gf,
                              bare_phonon_gf, epsilon_k,
                              matsubara_freq_fermion, matsubara_freq_boson)
from .aci_core import AdaptiveTCI
from .holstein import HolsteinParams, compute_sigma2_brute_force, compute_sigma2_tci

__all__ = [
    'TCIFitter',
    'QTTEncoder',
    'compute_tci_integral',
    'compute_tci_integral_reference',
    'vectorized_gaussian',
    'AdaptiveTCI',
    'HolsteinParams',
    'compute_sigma2_brute_force',
    'compute_sigma2_tci',
]

