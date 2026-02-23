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
                              matsubara_freq_fermion, matsubara_freq_boson,
                              bare_electron_gf_tau, bare_phonon_gf_tau)
from .aci_core import AdaptiveTCI
from .holstein import (HolsteinParams, compute_sigma2_brute_force, compute_sigma2_tci,
                        compute_sigma4_brute_force, compute_sigma4_vectorized,
                        compute_sigma4_tci,
                        compute_sigma4_direct_tci,
                        compute_sigma2_tau, sigma_tau_to_matsubara,
                        compute_sigma4_tau_brute_force, compute_sigma4_tau_tci)

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
    'compute_sigma4_brute_force',
    'compute_sigma4_vectorized',
    'compute_sigma4_tci',
    'compute_sigma4_direct_tci',
    'bare_electron_gf_tau',
    'bare_phonon_gf_tau',
    'compute_sigma2_tau',
    'sigma_tau_to_matsubara',
    'compute_sigma4_tau_brute_force',
    'compute_sigma4_tau_tci',
]
