"""
Phase 3 Benchmark: Holstein Polaron Self-Energy

Tests speed, convergence, and momentum dispersion for Σ(2).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from src.holstein import (HolsteinParams, compute_sigma2_brute_force,
                          compute_sigma2_brute_force_vectorized,
                          compute_sigma2_tci)


def benchmark_speed():
    """Compare wall time: brute-force loop vs vectorized vs TCI"""
    print("=" * 65)
    print("1. SPEED BENCHMARK")
    print("=" * 65)
    print(f"{'N_k':>5} {'N_ν':>5} | {'Loop (s)':>10} {'Vec (s)':>10} {'TCI (s)':>10} | {'Speedup':>8}")
    print("-" * 65)

    configs = [
        (16, 32),
        (32, 64),
        (64, 128),
    ]

    for N_k, N_nu in configs:
        params = HolsteinParams(t=1.0, omega0=0.5, g=0.3, beta=10.0, N_k=N_k, N_nu=N_nu)

        # Brute-force loop
        t0 = time.time()
        s_loop = compute_sigma2_brute_force(params, 0.0, 0)
        t_loop = time.time() - t0

        # Vectorized
        t0 = time.time()
        s_vec = compute_sigma2_brute_force_vectorized(params, 0.0, 0)
        t_vec = time.time() - t0

        # TCI path (vectorized Matsubara + direct q-sum)
        t0 = time.time()
        s_tci = compute_sigma2_tci(params, 0.0, 0, rank=5)
        t_tci = time.time() - t0

        speedup = t_loop / t_vec if t_vec > 0 else float('inf')
        print(f"{N_k:5d} {N_nu:5d} | {t_loop:10.4f} {t_vec:10.4f} {t_tci:10.4f} | {speedup:7.1f}×")

    print()


def benchmark_convergence():
    """Check Σ(2) convergence with Matsubara cutoff N_ν"""
    print("=" * 65)
    print("2. MATSUBARA CONVERGENCE (N_k=64 fixed)")
    print("=" * 65)
    print(f"{'N_ν':>6} | {'Re[Σ]':>14} {'Im[Σ]':>14} | {'ΔIm[Σ]':>12}")
    print("-" * 65)

    prev_im = None
    for N_nu in [16, 32, 64, 128, 256, 512]:
        params = HolsteinParams(t=1.0, omega0=0.5, g=0.3, beta=10.0, N_k=64, N_nu=N_nu)
        sigma = compute_sigma2_brute_force_vectorized(params, 0.0, 0)

        delta = "" if prev_im is None else f"{abs(sigma.imag - prev_im):.2e}"
        print(f"{N_nu:6d} | {sigma.real:14.8f} {sigma.imag:14.8f} | {delta:>12}")
        prev_im = sigma.imag

    print()


def benchmark_dispersion():
    """Compute Σ(2)(k) over the Brillouin zone"""
    print("=" * 65)
    print("3. MOMENTUM DISPERSION Σ(2)(k, iω_0)")
    print("=" * 65)

    params = HolsteinParams(t=1.0, omega0=0.5, g=0.3, beta=10.0, N_k=64, N_nu=128)
    k_points = np.linspace(0, np.pi, 9)  # half BZ (symmetry)

    print(f"{'k/π':>6} | {'Re[Σ]':>14} {'Im[Σ]':>14} | {'|Σ|':>10}")
    print("-" * 65)

    for k in k_points:
        sigma = compute_sigma2_brute_force_vectorized(params, k, 0)
        print(f"{k/np.pi:6.3f} | {sigma.real:14.8f} {sigma.imag:14.8f} | {abs(sigma):10.6f}")

    print()


def benchmark_coupling():
    """Σ(2) vs coupling strength g"""
    print("=" * 65)
    print("4. COUPLING DEPENDENCE Σ(2)(k=0) vs g")
    print("=" * 65)
    print(f"{'g':>6} | {'Re[Σ]':>14} {'Im[Σ]':>14} | {'|Σ|/g²':>10}")
    print("-" * 65)

    for g_val in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
        params = HolsteinParams(t=1.0, omega0=0.5, g=g_val, beta=10.0, N_k=64, N_nu=128)
        sigma = compute_sigma2_brute_force_vectorized(params, 0.0, 0)
        ratio = abs(sigma) / g_val**2
        print(f"{g_val:6.2f} | {sigma.real:14.8f} {sigma.imag:14.8f} | {ratio:10.6f}")

    print()


if __name__ == "__main__":
    print("\n" + "★" * 65)
    print("  Holstein Polaron Σ(2) Benchmark Suite")
    print("★" * 65 + "\n")

    benchmark_speed()
    benchmark_convergence()
    benchmark_dispersion()
    benchmark_coupling()

    print("=" * 65)
    print("Benchmark complete.")
    print("=" * 65)
