import numpy as np


def vectorized_gaussian(coords):
    """
    coords: (M, N) physical coordinates
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


# ============================================================
# Imaginary-Time Propagators
# ============================================================

def bare_electron_gf_tau(k, tau, beta, t=1.0):
    """
    Bare electron Green's function in imaginary time:
        G₀(k, τ) = -exp(-εₖ τ) / (1 + exp(-β εₖ))

    For τ ∈ [0, β), this is a smooth exponential decay.
    Much lower TT rank than the Matsubara frequency version.

    Args:
        k: momentum (scalar or array)
        tau: imaginary time, τ ∈ [0, β)
        beta: inverse temperature
        t: hopping parameter
    """
    ek = epsilon_k(k, t)
    # Use numerically stable form: exp(-ek*tau) / (1 + exp(-beta*ek))
    # For large |ek|, use: -exp(-ek*tau) * fermi(ek)
    return -np.exp(-ek * tau) / (1.0 + np.exp(-beta * ek))


def bare_phonon_gf_tau(tau, beta, omega0):
    """
    Bare phonon Green's function in imaginary time (dispersionless):
        D₀(τ) = -cosh[ω₀(β/2 - τ)] / sinh(ω₀ β/2)

    For τ ∈ [0, β), smooth cosh envelope.
    Fourier pair: D₀(iνm) = ∫₀^β dτ e^{iνmτ} D₀(τ) = -2ω₀/(νm² + ω₀²)

    Args:
        tau: imaginary time, τ ∈ [0, β)
        beta: inverse temperature
        omega0: phonon frequency
    """
    return -np.cosh(omega0 * (beta / 2.0 - tau)) / np.sinh(omega0 * beta / 2.0)