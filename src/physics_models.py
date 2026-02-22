import numpy as np


def vectorized_gaussian(coords):
    """
    coords: (M, N) 物理坐标
    """
    coords = np.atleast_2d(coords)
    return np.exp(-np.sum(coords**2, axis=1))


# ============================================================
# Holstein Model Propagators
# ============================================================

def epsilon_k(k, t=1.0):
    """
    1D tight-binding dispersion: ε(k) = -2t cos(k)
    
    Args:
        k: momentum (scalar or array), in [-π, π]
        t: hopping parameter
    """
    return -2.0 * t * np.cos(k)


def bare_electron_gf(k, iwn, t=1.0):
    """
    Bare electron Matsubara Green's function:
        G_0(k, iω_n) = 1 / (iω_n - ε_k)
    
    Args:
        k: momentum
        iwn: fermionic Matsubara frequency (imaginary, real-valued ω_n)
        t: hopping
    
    Returns:
        complex G_0
    """
    ek = epsilon_k(k, t)
    return 1.0 / (1j * iwn - ek)


def bare_phonon_gf(inu_m, omega0):
    """
    Bare phonon Matsubara Green's function (dispersionless Holstein):
        D_0(iν_m) = -2ω_0 / (ν_m² + ω_0²)
    
    No q-dependence (Einstein phonons).
    
    Args:
        inu_m: bosonic Matsubara frequency (real-valued ν_m)
        omega0: phonon frequency
    
    Returns:
        real D_0
    """
    return -2.0 * omega0 / (inu_m**2 + omega0**2)


def matsubara_freq_fermion(n, beta):
    """Fermionic Matsubara frequency: ω_n = (2n+1)π/β"""
    return (2 * n + 1) * np.pi / beta


def matsubara_freq_boson(m, beta):
    """Bosonic Matsubara frequency: ν_m = 2mπ/β"""
    return 2 * m * np.pi / beta