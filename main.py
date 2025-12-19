import numpy as np
from tci_core import TCIFitter
from qtt_utils import QTTEncoder
from tci_utils import compute_tci_integral
from physics_models import vectorized_gaussian

def run_normal_demo():
    print("\n--- 模式 1: 普通网格 TCI ---")
    grid = [np.linspace(-3, 3, 64) for _ in range(3)]
    dx = grid[0][1] - grid[0][0]
    solver = TCIFitter(vectorized_gaussian, grid, rank=1)
    solver.build_cores()
    res = compute_tci_integral(solver, dx_vol=dx**3)
    print(f"普通网格积分: {res:.6f} (理论: 5.56832)")

def run_qtt_demo():
    print("\n--- 模式 2: QTT 融合比特 TCI ---")
    encoder = QTTEncoder(num_vars=3, num_bits=20, bounds=[(-3, 3)]*3)
    def wrapped_f(idx): return vectorized_gaussian(encoder.decode(idx))
    
    # 构建 domain
    domain = [np.arange(encoder.d, dtype=int)] * encoder.R
    solver = TCIFitter(wrapped_f, domain, rank=10)
    
    # 显式使用修复后的 get_anchors
    solver.build_cores(anchors=encoder.get_anchors())
    
    # 计算物理体积元
    dx_vol = (6.0**3) / (encoder.d ** encoder.R)
    res = compute_tci_integral(solver, dx_vol=dx_vol)
    print(f"QTT 高分辨率积分: {res:.6f} (分辨率: 2^60)")

if __name__ == "__main__":
    run_normal_demo()
    run_qtt_demo()