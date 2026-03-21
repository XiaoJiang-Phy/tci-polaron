import numpy as np
import time
from src.tci_core import TCIFitter
from src.qtt_utils import QTTEncoder
from src.tci_utils import compute_tci_integral
from src.physics_models import vectorized_gaussian
from src.holstein import (HolsteinParams, compute_sigma2_brute_force, compute_sigma2_tci,
                          compute_sigma4_vectorized, compute_sigma4_direct_tci,
                          compute_sigma4_tau_brute_force, compute_sigma4_tau_tci)

THEORETICAL = 5.56832

def run_normal_demo():
    print("\n--- Mode 1: Standard Grid TCI ---")
    grid = [np.linspace(-3, 3, 64) for _ in range(3)]
    dx = grid[0][1] - grid[0][0]
    solver = TCIFitter(vectorized_gaussian, grid, rank=1)
    solver.build_cores()
    res = compute_tci_integral(solver, dx_vol=dx**3)
    error = abs(res - THEORETICAL) / THEORETICAL * 100
    print(f"Standard grid integral: {res:.6f} (theory: {THEORETICAL}, error: {error:.2f}%)")

def run_qtt_demo():
    print("\n--- Mode 2: QTT Fused-Bit TCI (rank=10) ---")
    encoder = QTTEncoder(num_vars=3, num_bits=20, bounds=[(-3, 3)]*3)
    def wrapped_f(idx): return vectorized_gaussian(encoder.decode(idx))
    
    domain = [np.arange(encoder.d, dtype=int)] * encoder.R
    solver = TCIFitter(wrapped_f, domain, rank=10)
    solver.build_cores(anchors=encoder.get_anchors())
    
    dx_vol = (6.0**3) / (encoder.d ** encoder.R)
    res = compute_tci_integral(solver, dx_vol=dx_vol)
    error = abs(res - THEORETICAL) / THEORETICAL * 100
    print(f"QTT integral: {res:.6f} (theory: {THEORETICAL}, error: {error:.2f}%)")

def run_high_rank_qtt_demo():
    print("\n--- Mode 3: High-Rank QTT TCI (rank=50) ---")
    encoder = QTTEncoder(num_vars=3, num_bits=20, bounds=[(-3, 3)]*3)
    def wrapped_f(idx): return vectorized_gaussian(encoder.decode(idx))
    
    domain = [np.arange(encoder.d, dtype=int)] * encoder.R
    solver = TCIFitter(wrapped_f, domain, rank=50)
    solver.build_cores(anchors=encoder.get_anchors(), n_sweeps=5)
    
    dx_vol = (6.0**3) / (encoder.d ** encoder.R)
    res = compute_tci_integral(solver, dx_vol=dx_vol)
    error = abs(res - THEORETICAL) / THEORETICAL * 100
    print(f"High-rank QTT integral: {res:.6f} (theory: {THEORETICAL}, error: {error:.2f}%)")

def run_holstein_demo():
    print("\n--- Mode 4: Holstein Polaron 2nd-Order Self-Energy ---")
    params = HolsteinParams(t=1.0, omega0=0.5, g=0.3, beta=10.0, N_k=64, N_nu=128)
    print(f"Params: t={params.t}, omega0={params.omega0}, g={params.g}, beta={params.beta}")
    print(f"Grid: N_k={params.N_k}, N_nu={params.N_nu}")
    
    k_ext, n_ext = 0.0, 0
    
    # Brute force
    print("\nComputing brute-force sum...")
    sigma_bf = compute_sigma2_brute_force(params, k_ext, n_ext)
    print(f"Sigma(2) brute force: {sigma_bf:.8f}")
    print(f"  Re[Sigma] = {sigma_bf.real:.8f}, Im[Sigma] = {sigma_bf.imag:.8f}")
    
    # TCI
    print("\nComputing TCI-accelerated...")
    sigma_tci = compute_sigma2_tci(params, k_ext, n_ext, rank=5)
    print(f"Sigma(2) TCI (rank=5): {sigma_tci:.8f}")
    
    rel_error = abs(sigma_tci - sigma_bf) / abs(sigma_bf) * 100
    print(f"\nRelative error: {rel_error:.2f}%")

def run_sigma4_direct_tci_demo():
    print("\n--- Mode 5: Sigma(4) Direct 4D TCI (no dim reduction) ---")
    params = HolsteinParams(t=1.0, omega0=0.5, g=0.3, beta=10.0, N_k=16, N_nu=32)
    print(f"Params: t={params.t}, omega0={params.omega0}, g={params.g}, beta={params.beta}")
    print(f"Grid: N_k={params.N_k}, N_nu={params.N_nu}")
    print(f"4D total points: {params.N_k**2 * (2*params.N_nu)**2:.2e}")

    k_ext, n_ext = 0.0, 0

    # Vectorized brute-force (reference)
    print("\nComputing vectorized brute-force (reference)...")
    t0 = time.time()
    sigma_vec = compute_sigma4_vectorized(params, k_ext, n_ext)
    t_vec = time.time() - t0
    print(f"  Sigma(4) vectorized: {sigma_vec:.8f}  ({t_vec:.2f}s)")

    # Direct 4D TCI
    for rank in [5, 10, 20]:
        print(f"\nComputing direct 4D TCI (rank={rank})...")
        t0 = time.time()
        sigma_tci = compute_sigma4_direct_tci(params, k_ext, n_ext, rank=rank, verbose=True)
        t_tci = time.time() - t0
        rel_err = abs(sigma_tci - sigma_vec) / abs(sigma_vec) * 100
        print(f"  Sigma(4) TCI: {sigma_tci:.8f}  ({t_tci:.2f}s)")
        print(f"  Relative error: {rel_err:.2f}%")

def run_tau_demo():
    print("\n--- Mode 6: Sigma(4) Imaginary-Time tau Representation ---")
    params = HolsteinParams(t=1.0, omega0=0.5, g=0.3, beta=10.0, N_k=8, N_nu=16)
    print(f"Params: t={params.t}, omega0={params.omega0}, g={params.g}, beta={params.beta}")
    print(f"Grid: N_k={params.N_k}, N_nu={params.N_nu}")

    k_ext, n_ext = 0.0, 0

    # Matsubara reference
    t0 = time.time()
    sigma_mat = compute_sigma4_vectorized(params, k_ext, n_ext)
    t_mat = time.time() - t0
    print(f"\n  Sigma(4) Matsubara ref:     {sigma_mat:.8f}  ({t_mat:.2f}s)")

    # tau-BF (now exact, uses Matsubara h)
    t0 = time.time()
    s_bf = compute_sigma4_tau_brute_force(params, k_ext, n_ext)
    t_bf = time.time() - t0
    err_bf = abs(s_bf - sigma_mat) / abs(sigma_mat) * 100
    print(f"  Sigma(4) tau-BF (exact):   {s_bf:.8f}  ({t_bf:.2f}s, error {err_bf:.4f}%)")

    # tau-TCI convergence with N_tau (with tail subtraction)
    print("\n  tau-TCI (tail subtraction, O(1/N_tau^2)) convergence:")
    for N_tau in [32, 64, 128, 256]:
        t0 = time.time()
        s_tci = compute_sigma4_tau_tci(params, k_ext, n_ext, N_tau=N_tau)
        dt = time.time() - t0
        err = abs(s_tci - sigma_mat) / abs(sigma_mat) * 100
        print(f"    N_tau={N_tau:4d}: {s_tci:.8f}  ({dt:.2f}s, error {err:.4f}%)")


if __name__ == "__main__":
    run_normal_demo()
    run_holstein_demo()
    run_sigma4_direct_tci_demo()
    run_tau_demo()