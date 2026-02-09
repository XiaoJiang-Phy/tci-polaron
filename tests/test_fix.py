
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.tci_core import TCIFitter
from src.qtt_utils import QTTEncoder
from src.tci_utils import compute_tci_integral
from src.physics_models import vectorized_gaussian

def test_fixed_qtt():
    # 模拟 main.py 的设置
    encoder = QTTEncoder(num_vars=3, num_bits=20, bounds=[(-3, 3)]*3)
    def wrapped_f(idx): return vectorized_gaussian(encoder.decode(idx))
    domain = [np.arange(encoder.d, dtype=int)] * encoder.R
    
    # 手动设置更好的锚点：[7, 0, 0, ...] 对应物理坐标 (0,0,0)
    best_anchor = np.zeros(encoder.R, dtype=int)
    best_anchor[0] = 7 # 在第 0 层设置所有变量的最高位为 1 (即 0.5)
    
    # 锚点 1: 全 0, 锚点 2: 中心
    anchors = np.array([np.zeros(encoder.R, dtype=int), best_anchor])
    
    print("--- 运行测试 (修正锚点后) ---")
    solver = TCIFitter(wrapped_f, domain, rank=10)
    solver.build_cores(anchors=anchors)
    
    dx_vol = (6.0**3) / (encoder.d ** encoder.R)
    res = compute_tci_integral(solver, dx_vol=dx_vol)
    print(f"修正后的 QTT 积分: {res:.6f} (期望 ~5.568)")

if __name__ == "__main__":
    test_fixed_qtt()
