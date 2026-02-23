"""
Phase 4 Benchmark: Direct 4D TCI for Σ(4)

Compares vectorized brute-force vs direct 4D TCI:
  - Speed at various grid sizes
  - Rank convergence
  - Scaling study
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from src.holstein import (HolsteinParams, compute_sigma4_vectorized,
                          compute_sigma4_direct_tci)


def benchmark_speed():
    """Speed: vectorized vs direct-4D-TCI at various grid sizes"""
    print("=" * 70)
    print("1. SPEED BENCHMARK: Vectorized vs Direct-4D-TCI")
    print("=" * 70)
    print(f"{'N_k':>5} {'N_ν':>5} {'4D pts':>12} | {'Vec (s)':>10} {'TCI (s)':>10} | {'Err':>8}")
    print("-" * 70)

    configs = [
        (8,  16, 10),
        (16, 32, 10),
        (16, 64, 15),
    ]

    for N_k, N_nu, rank in configs:
        params = HolsteinParams(t=1.0, omega0=0.5, g=0.3, beta=10.0, N_k=N_k, N_nu=N_nu)
        n_pts = N_k**2 * (2 * N_nu)**2

        t0 = time.time()
        s_vec = compute_sigma4_vectorized(params, 0.0, 0)
        t_vec = time.time() - t0

        t0 = time.time()
        s_tci = compute_sigma4_direct_tci(params, 0.0, 0, rank=rank)
        t_tci = time.time() - t0

        err = abs(s_tci - s_vec) / abs(s_vec) * 100
        print(f"{N_k:5d} {N_nu:5d} {n_pts:12.2e} | {t_vec:10.3f} {t_tci:10.3f} | {err:7.2f}%")

    print()


def benchmark_rank_convergence():
    """Relative error vs TCI rank"""
    print("=" * 70)
    print("2. RANK CONVERGENCE (N_k=16, N_ν=32)")
    print("=" * 70)

    params = HolsteinParams(t=1.0, omega0=0.5, g=0.3, beta=10.0, N_k=16, N_nu=32)
    s_ref = compute_sigma4_vectorized(params, 0.0, 0)
    print(f"Reference Σ(4) = {s_ref:.8f}\n")

    print(f"{'Rank':>6} | {'Re[Σ]':>14} {'Im[Σ]':>14} | {'Rel Err':>10} {'Time (s)':>10}")
    print("-" * 70)

    for rank in [3, 5, 8, 10, 15, 20]:
        t0 = time.time()
        s_tci = compute_sigma4_direct_tci(params, 0.0, 0, rank=rank)
        dt = time.time() - t0
        err = abs(s_tci - s_ref) / abs(s_ref) * 100
        print(f"{rank:6d} | {s_tci.real:14.8f} {s_tci.imag:14.8f} | {err:9.2f}% {dt:10.2f}")

    print()


def benchmark_scaling():
    """Wall time vs N_k (fixed N_ν=16) to show complexity"""
    print("=" * 70)
    print("3. SCALING STUDY: Wall time vs N_k (N_ν=16, rank=10)")
    print("=" * 70)
    print(f"{'N_k':>5} {'4D pts':>12} | {'Vec (s)':>10} {'TCI (s)':>10} | {'Ratio':>8}")
    print("-" * 70)

    for N_k in [4, 8, 12, 16, 24]:
        params = HolsteinParams(t=1.0, omega0=0.5, g=0.3, beta=10.0, N_k=N_k, N_nu=16)
        n_pts = N_k**2 * (2 * 16)**2

        t0 = time.time()
        _ = compute_sigma4_vectorized(params, 0.0, 0)
        t_vec = time.time() - t0

        t0 = time.time()
        _ = compute_sigma4_direct_tci(params, 0.0, 0, rank=10)
        t_tci = time.time() - t0

        ratio = t_vec / t_tci if t_tci > 0 else float('inf')
        print(f"{N_k:5d} {n_pts:12.2e} | {t_vec:10.3f} {t_tci:10.3f} | {ratio:7.2f}×")

    print()


if __name__ == "__main__":
    print("\n" + "★" * 70)
    print("  Phase 4: Direct 4D TCI Benchmark for Σ(4)")
    print("★" * 70 + "\n")

    benchmark_speed()
    benchmark_rank_convergence()
    benchmark_scaling()

    print("=" * 70)
    print("Benchmark complete.")
    print("=" * 70)
