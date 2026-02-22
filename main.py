import numpy as np
from src.tci_core import TCIFitter
from src.qtt_utils import QTTEncoder
from src.tci_utils import compute_tci_integral
from src.physics_models import vectorized_gaussian
from src.holstein import HolsteinParams, compute_sigma2_brute_force, compute_sigma2_tci

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

if __name__ == "__main__":
    run_normal_demo()
    run_holstein_demo()