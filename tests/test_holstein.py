"""
Tests for Holstein polaron self-energy calculations
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.physics_models import (bare_electron_gf, bare_phonon_gf, epsilon_k,
                                 matsubara_freq_fermion, matsubara_freq_boson)
from src.holstein import (HolsteinParams, compute_sigma2_brute_force, compute_sigma2_tci,
                          compute_sigma4_brute_force, compute_sigma4_vectorized,
                          compute_sigma4_tci, compute_sigma4_direct_tci)


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
    assert np.isclose(matsubara_freq_fermion(0, beta), np.pi / beta)
    assert np.isclose(matsubara_freq_boson(0, beta), 0.0)
    assert np.isclose(matsubara_freq_boson(1, beta), 2 * np.pi / beta)
    print("✅ test_matsubara_frequencies passed")


def test_sigma2_brute_force():
    """Compute Σ(2) and check basic physical properties"""
    params = HolsteinParams(t=1.0, omega0=0.5, g=0.3, beta=10.0, N_k=64, N_nu=128)
    sigma = compute_sigma2_brute_force(params, k_ext=0.0, n_ext=0)
    print(f"  Σ(2)(k=0, iω_0) = {sigma:.6f}")
    assert sigma.real < 0 or abs(sigma.real) < 1e-10, f"Re[Σ] = {sigma.real}"
    assert abs(sigma) < 1.0, f"|Σ| = {abs(sigma)} too large"
    print("✅ test_sigma2_brute_force passed")


def test_sigma2_g_squared_scaling():
    """Σ(2) ∝ g²"""
    params1 = HolsteinParams(t=1.0, omega0=0.5, g=0.1, beta=10.0, N_k=32, N_nu=64)
    params2 = HolsteinParams(t=1.0, omega0=0.5, g=0.2, beta=10.0, N_k=32, N_nu=64)
    s1 = compute_sigma2_brute_force(params1, k_ext=0.0, n_ext=0)
    s2 = compute_sigma2_brute_force(params2, k_ext=0.0, n_ext=0)
    ratio = s2 / s1
    assert np.isclose(abs(ratio), 4.0, rtol=0.01), f"Ratio {ratio} != 4"
    print(f"  |Σ(g=0.2)/Σ(g=0.1)| = {abs(ratio):.4f}")
    print("✅ test_sigma2_g_squared_scaling passed")


def test_sigma2_tci_vs_brute_force():
    """TCI matches brute-force for Σ(2)"""
    params = HolsteinParams(t=1.0, omega0=0.5, g=0.3, beta=10.0, N_k=32, N_nu=64)
    sigma_bf = compute_sigma2_brute_force(params, k_ext=0.0, n_ext=0)
    sigma_tci = compute_sigma2_tci(params, k_ext=0.0, n_ext=0, rank=5)
    rel_error = abs(sigma_tci - sigma_bf) / abs(sigma_bf)
    print(f"  Σ(2) BF: {sigma_bf:.6f}, TCI: {sigma_tci:.6f}, err: {rel_error:.2%}")
    assert rel_error < 0.05
    print("✅ test_sigma2_tci_vs_brute_force passed")


def test_sigma4_g_fourth_scaling():
    """Σ(4) ∝ g⁴"""
    params1 = HolsteinParams(t=1.0, omega0=0.5, g=0.1, beta=10.0, N_k=8, N_nu=16)
    params2 = HolsteinParams(t=1.0, omega0=0.5, g=0.2, beta=10.0, N_k=8, N_nu=16)
    s1 = compute_sigma4_vectorized(params1, k_ext=0.0, n_ext=0)
    s2 = compute_sigma4_vectorized(params2, k_ext=0.0, n_ext=0)
    ratio = abs(s2) / abs(s1)
    expected = (0.2 / 0.1) ** 4  # = 16.0
    print(f"  |Σ(4)(g=0.2)/Σ(4)(g=0.1)| = {ratio:.4f} (expected {expected})")
    assert np.isclose(ratio, expected, rtol=0.01)
    print("✅ test_sigma4_g_fourth_scaling passed")


def test_sigma4_tci_vs_brute_force():
    """TCI matches brute-force for Σ(4)"""
    params = HolsteinParams(t=1.0, omega0=0.5, g=0.3, beta=10.0, N_k=8, N_nu=16)
    s_bf = compute_sigma4_brute_force(params, k_ext=0.0, n_ext=0)
    s_tci = compute_sigma4_tci(params, k_ext=0.0, n_ext=0, rank=5)
    rel_error = abs(s_tci - s_bf) / abs(s_bf)
    print(f"  Σ(4) BF: {s_bf:.8f}, TCI: {s_tci:.8f}, err: {rel_error:.2%}")
    assert rel_error < 0.01
    print("✅ test_sigma4_tci_vs_brute_force passed")


def test_sigma4_smaller_than_sigma2():
    """In weak coupling: |Σ(4)| << |Σ(2)|"""
    params = HolsteinParams(t=1.0, omega0=0.5, g=0.3, beta=10.0, N_k=16, N_nu=32)
    s2 = compute_sigma2_tci(params, k_ext=0.0, n_ext=0)
    s4 = compute_sigma4_vectorized(params, k_ext=0.0, n_ext=0)
    ratio = abs(s4) / abs(s2)
    print(f"  |Σ(4)|/|Σ(2)| = {ratio:.4f} (should be << 1 for g=0.3)")
    assert ratio < 1.0, f"Σ(4) too large relative to Σ(2)"
    print("✅ test_sigma4_smaller_than_sigma2 passed")


def test_sigma4_direct_tci_vs_brute_force():
    """Direct 4D TCI matches brute-force for Σ(4)"""
    params = HolsteinParams(t=1.0, omega0=0.5, g=0.3, beta=10.0, N_k=8, N_nu=16)
    s_bf = compute_sigma4_brute_force(params, k_ext=0.0, n_ext=0)
    s_tci = compute_sigma4_direct_tci(params, k_ext=0.0, n_ext=0, rank=10, n_sweeps=4)
    rel_error = abs(s_tci - s_bf) / abs(s_bf)
    print(f"  Σ(4) BF: {s_bf:.8f}, Direct-4D-TCI: {s_tci:.8f}, err: {rel_error:.2%}")
    assert rel_error < 0.10, f"Direct 4D TCI error {rel_error:.2%} too large"
    print("✅ test_sigma4_direct_tci_vs_brute_force passed")


def test_sigma4_direct_tci_rank_convergence():
    """Higher rank -> lower error for direct 4D TCI"""
    params = HolsteinParams(t=1.0, omega0=0.5, g=0.3, beta=10.0, N_k=8, N_nu=16)
    s_ref = compute_sigma4_vectorized(params, k_ext=0.0, n_ext=0)
    err_low = abs(compute_sigma4_direct_tci(params, 0.0, 0, rank=3) - s_ref) / abs(s_ref)
    err_high = abs(compute_sigma4_direct_tci(params, 0.0, 0, rank=10) - s_ref) / abs(s_ref)
    print(f"  rank=3 err: {err_low:.2%}, rank=10 err: {err_high:.2%}")
    assert err_high <= err_low + 0.01, f"Higher rank did not improve: {err_high:.2%} vs {err_low:.2%}"
    print("✅ test_sigma4_direct_tci_rank_convergence passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Holstein Polaron Self-Energy Tests")
    print("=" * 60)

    test_dispersion()
    test_electron_gf()
    test_phonon_gf()
    test_matsubara_frequencies()
    test_sigma2_brute_force()
    test_sigma2_g_squared_scaling()
    test_sigma2_tci_vs_brute_force()
    test_sigma4_g_fourth_scaling()
    test_sigma4_tci_vs_brute_force()
    test_sigma4_smaller_than_sigma2()
    test_sigma4_direct_tci_vs_brute_force()
    test_sigma4_direct_tci_rank_convergence()

    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)

