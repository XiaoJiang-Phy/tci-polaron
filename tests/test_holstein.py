"""
Tests for Holstein polaron self-energy calculations
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.physics_models import (bare_electron_gf, bare_phonon_gf, epsilon_k,
                                 matsubara_freq_fermion, matsubara_freq_boson)
from src.holstein import HolsteinParams, compute_sigma2_brute_force, compute_sigma2_tci


def test_dispersion():
    """ε(k=0) = -2t, ε(k=π) = +2t"""
    assert np.isclose(epsilon_k(0, t=1.0), -2.0)
    assert np.isclose(epsilon_k(np.pi, t=1.0), 2.0, atol=1e-10)
    assert np.isclose(epsilon_k(np.pi / 2, t=1.0), 0.0, atol=1e-10)
    print("✅ test_dispersion passed")


def test_electron_gf():
    """G0(k=π/2, iω0) at β=10: ε=0, ω0=π/10"""
    beta = 10.0
    wn = matsubara_freq_fermion(0, beta)  # ω_0 = π/β
    g0 = bare_electron_gf(np.pi / 2, wn, t=1.0)
    # ε(π/2) = 0, so G0 = 1/(iω_0) = -i/ω_0
    expected = 1.0 / (1j * wn)
    assert np.isclose(g0, expected), f"G0={g0}, expected={expected}"
    print("✅ test_electron_gf passed")


def test_phonon_gf():
    """D0(ν=0) = -2ω0/ω0² = -2/ω0"""
    omega0 = 0.5
    d0 = bare_phonon_gf(0.0, omega0)
    expected = -2.0 / omega0
    assert np.isclose(d0, expected), f"D0={d0}, expected={expected}"
    print("✅ test_phonon_gf passed")


def test_matsubara_frequencies():
    """Check frequency values"""
    beta = 10.0
    # Fermionic: ω_0 = π/β
    assert np.isclose(matsubara_freq_fermion(0, beta), np.pi / beta)
    # Bosonic: ν_0 = 0
    assert np.isclose(matsubara_freq_boson(0, beta), 0.0)
    # Bosonic: ν_1 = 2π/β
    assert np.isclose(matsubara_freq_boson(1, beta), 2 * np.pi / beta)
    print("✅ test_matsubara_frequencies passed")


def test_sigma2_brute_force():
    """Compute Σ(2) and check basic physical properties"""
    params = HolsteinParams(t=1.0, omega0=0.5, g=0.3, beta=10.0, N_k=64, N_nu=128)
    
    sigma = compute_sigma2_brute_force(params, k_ext=0.0, n_ext=0)
    
    print(f"  Σ(2)(k=0, iω_0) = {sigma:.6f}")
    print(f"  Re[Σ] = {sigma.real:.6f}, Im[Σ] = {sigma.imag:.6f}")
    
    # Physical checks:
    # 1. Re[Σ] should be negative (attractive phonon-mediated correction)
    assert sigma.real < 0, f"Re[Σ] = {sigma.real} should be < 0"
    
    # 2. |Σ| should scale as g² = 0.09 (perturbative regime)
    assert abs(sigma) < 1.0, f"|Σ| = {abs(sigma)} too large for weak coupling"
    
    print("✅ test_sigma2_brute_force passed")


def test_sigma2_g_squared_scaling():
    """Σ(2) ∝ g², verify by doubling g"""
    params1 = HolsteinParams(t=1.0, omega0=0.5, g=0.1, beta=10.0, N_k=32, N_nu=64)
    params2 = HolsteinParams(t=1.0, omega0=0.5, g=0.2, beta=10.0, N_k=32, N_nu=64)
    
    s1 = compute_sigma2_brute_force(params1, k_ext=0.0, n_ext=0)
    s2 = compute_sigma2_brute_force(params2, k_ext=0.0, n_ext=0)
    
    ratio = s2 / s1
    expected_ratio = (0.2 / 0.1) ** 2  # = 4.0
    
    print(f"  Σ(g=0.1) = {s1:.6f}")
    print(f"  Σ(g=0.2) = {s2:.6f}")
    print(f"  Ratio = {ratio:.4f} (expected ≈ {expected_ratio})")
    
    assert np.isclose(abs(ratio), expected_ratio, rtol=0.01), f"Ratio {ratio} != {expected_ratio}"
    print("✅ test_sigma2_g_squared_scaling passed")


def test_sigma2_tci_vs_brute_force():
    """Compare TCI result with brute-force"""
    params = HolsteinParams(t=1.0, omega0=0.5, g=0.3, beta=10.0, N_k=32, N_nu=64)
    
    sigma_bf = compute_sigma2_brute_force(params, k_ext=0.0, n_ext=0)
    sigma_tci = compute_sigma2_tci(params, k_ext=0.0, n_ext=0, rank=5)
    
    rel_error = abs(sigma_tci - sigma_bf) / abs(sigma_bf)
    
    print(f"  Brute force: {sigma_bf:.6f}")
    print(f"  TCI (rank=5): {sigma_tci:.6f}")
    print(f"  Relative error: {rel_error:.2%}")
    
    assert rel_error < 0.05, f"TCI error {rel_error:.2%} too large"
    print("✅ test_sigma2_tci_vs_brute_force passed")


if __name__ == "__main__":
    print("="*60)
    print("Holstein Polaron Self-Energy Tests")
    print("="*60)
    
    test_dispersion()
    test_electron_gf()
    test_phonon_gf()
    test_matsubara_frequencies()
    test_sigma2_brute_force()
    test_sigma2_g_squared_scaling()
    test_sigma2_tci_vs_brute_force()
    
    print("\n" + "="*60)
    print("All tests passed! ✅")
    print("="*60)
