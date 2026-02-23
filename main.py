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
    print("\n--- 模式 1: 普通网格 TCI ---")
    grid = [np.linspace(-3, 3, 64) for _ in range(3)]
    dx = grid[0][1] - grid[0][0]
    solver = TCIFitter(vectorized_gaussian, grid, rank=1)
    solver.build_cores()
    res = compute_tci_integral(solver, dx_vol=dx**3)
    error = abs(res - THEORETICAL) / THEORETICAL * 100
    print(f"普通网格积分: {res:.6f} (理论: {THEORETICAL}, 误差: {error:.2f}%)")

def run_qtt_demo():
    print("\n--- 模式 2: QTT 融合比特 TCI (rank=10) ---")
    encoder = QTTEncoder(num_vars=3, num_bits=20, bounds=[(-3, 3)]*3)
    def wrapped_f(idx): return vectorized_gaussian(encoder.decode(idx))
    
    domain = [np.arange(encoder.d, dtype=int)] * encoder.R
    solver = TCIFitter(wrapped_f, domain, rank=10)
    solver.build_cores(anchors=encoder.get_anchors())
    
    dx_vol = (6.0**3) / (encoder.d ** encoder.R)
    res = compute_tci_integral(solver, dx_vol=dx_vol)
    error = abs(res - THEORETICAL) / THEORETICAL * 100
    print(f"QTT 积分: {res:.6f} (理论: {THEORETICAL}, 误差: {error:.2f}%)")

def run_high_rank_qtt_demo():
    print("\n--- 模式 3: 高秩 QTT TCI (rank=50) ---")
    encoder = QTTEncoder(num_vars=3, num_bits=20, bounds=[(-3, 3)]*3)
    def wrapped_f(idx): return vectorized_gaussian(encoder.decode(idx))
    
    domain = [np.arange(encoder.d, dtype=int)] * encoder.R
    solver = TCIFitter(wrapped_f, domain, rank=50)
    solver.build_cores(anchors=encoder.get_anchors(), n_sweeps=5)
    
    dx_vol = (6.0**3) / (encoder.d ** encoder.R)
    res = compute_tci_integral(solver, dx_vol=dx_vol)
    error = abs(res - THEORETICAL) / THEORETICAL * 100
    print(f"高秩 QTT 积分: {res:.6f} (理论: {THEORETICAL}, 误差: {error:.2f}%)")

def run_holstein_demo():
    print("\n--- 模式 4: Holstein Polaron 2阶自能 ---")
    params = HolsteinParams(t=1.0, omega0=0.5, g=0.3, beta=10.0, N_k=64, N_nu=128)
    print(f"参数: t={params.t}, ω₀={params.omega0}, g={params.g}, β={params.beta}")
    print(f"网格: N_k={params.N_k}, N_ν={params.N_nu}")
    
    k_ext, n_ext = 0.0, 0
    
    # Brute force
    print("\n计算暴力求和...")
    sigma_bf = compute_sigma2_brute_force(params, k_ext, n_ext)
    print(f"Σ(2) 暴力求和: {sigma_bf:.8f}")
    print(f"  Re[Σ] = {sigma_bf.real:.8f}, Im[Σ] = {sigma_bf.imag:.8f}")
    
    # TCI
    print("\n计算 TCI 加速...")
    sigma_tci = compute_sigma2_tci(params, k_ext, n_ext, rank=5)
    print(f"Σ(2) TCI (rank=5): {sigma_tci:.8f}")
    
    rel_error = abs(sigma_tci - sigma_bf) / abs(sigma_bf) * 100
    print(f"\n相对误差: {rel_error:.2f}%")

def run_sigma4_direct_tci_demo():
    print("\n--- 模式 5: Σ(4) 直接 4D TCI (无降维) ---")
    params = HolsteinParams(t=1.0, omega0=0.5, g=0.3, beta=10.0, N_k=16, N_nu=32)
    print(f"参数: t={params.t}, ω₀={params.omega0}, g={params.g}, β={params.beta}")
    print(f"网格: N_k={params.N_k}, N_ν={params.N_nu}")
    print(f"4D 总点数: {params.N_k**2 * (2*params.N_nu)**2:.2e}")

    k_ext, n_ext = 0.0, 0

    # Vectorized brute-force (reference)
    print("\n计算向量化暴力求和 (参考)...")
    t0 = time.time()
    sigma_vec = compute_sigma4_vectorized(params, k_ext, n_ext)
    t_vec = time.time() - t0
    print(f"  Σ(4) 向量化: {sigma_vec:.8f}  ({t_vec:.2f}s)")

    # Direct 4D TCI
    for rank in [5, 10, 20]:
        print(f"\n计算直接 4D TCI (rank={rank})...")
        t0 = time.time()
        sigma_tci = compute_sigma4_direct_tci(params, k_ext, n_ext, rank=rank, verbose=True)
        t_tci = time.time() - t0
        rel_err = abs(sigma_tci - sigma_vec) / abs(sigma_vec) * 100
        print(f"  Σ(4) TCI: {sigma_tci:.8f}  ({t_tci:.2f}s)")
        print(f"  相对误差: {rel_err:.2f}%")

def run_tau_demo():
    print("\n--- 模式 6: Σ(4) 虚时间 τ 表示 ---")
    params = HolsteinParams(t=1.0, omega0=0.5, g=0.3, beta=10.0, N_k=8, N_nu=16)
    print(f"参数: t={params.t}, ω₀={params.omega0}, g={params.g}, β={params.beta}")
    print(f"网格: N_k={params.N_k}, N_ν={params.N_nu}")

    k_ext, n_ext = 0.0, 0

    # Matsubara reference
    t0 = time.time()
    sigma_mat = compute_sigma4_vectorized(params, k_ext, n_ext)
    t_mat = time.time() - t0
    print(f"\n  Σ(4) Matsubara 参考:     {sigma_mat:.8f}  ({t_mat:.2f}s)")

    # τ-BF (now exact, uses Matsubara h)
    t0 = time.time()
    s_bf = compute_sigma4_tau_brute_force(params, k_ext, n_ext)
    t_bf = time.time() - t0
    err_bf = abs(s_bf - sigma_mat) / abs(sigma_mat) * 100
    print(f"  Σ(4) τ-BF (精确):       {s_bf:.8f}  ({t_bf:.2f}s, 误差 {err_bf:.4f}%)")

    # τ-TCI convergence with N_tau
    print("\n  τ-TCI (direct 2D sum, τ-space h) 收敛:")
    for N_tau in [128, 256, 512]:
        t0 = time.time()
        s_tci = compute_sigma4_tau_tci(params, k_ext, n_ext, N_tau=N_tau)
        dt = time.time() - t0
        err = abs(s_tci - sigma_mat) / abs(sigma_mat) * 100
        print(f"    N_τ={N_tau:4d}: {s_tci:.8f}  ({dt:.2f}s, 误差 {err:.2f}%)")


if __name__ == "__main__":
    run_normal_demo()
    run_holstein_demo()
    run_sigma4_direct_tci_demo()
    run_tau_demo()