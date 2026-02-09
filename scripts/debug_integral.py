"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QTT 积分调试脚本
"""
import numpy as np
from src.tci_core import TCIFitter
from src.qtt_utils import QTTEncoder
from src.physics_models import vectorized_gaussian

def debug_qtt_integral():
    print("="*60)
    print("QTT TCI 积分调试")
    print("="*60)
    
    encoder = QTTEncoder(num_vars=3, num_bits=20, bounds=[(-3, 3)]*3)
    def wrapped_f(idx): return vectorized_gaussian(encoder.decode(idx))
    domain = [np.arange(encoder.d, dtype=int)] * encoder.R
    
    solver = TCIFitter(wrapped_f, domain, rank=10)
    anchors = encoder.get_anchors()
    solver.build_cores(anchors=anchors, n_sweeps=3, verbose=True)
    
    rank = solver.rank
    n_dims = solver.n_dims
    
    print(f"\n配置: rank={rank}, n_dims={n_dims}")
    
    # 追踪积分过程
    left_vec = np.ones((1, rank))
    scale_log = 0.0  # 记录归一化的对数
    
    for d in range(min(5, n_dims)):  # 只看前 5 层
        l_indices = solver.pivot_paths[:, :d] if d > 0 else np.zeros((rank, 0), dtype=int)
        r_indices = solver.pivot_paths[:, d+1:] if d < n_dims - 1 else np.zeros((rank, 0), dtype=int)
        
        n_curr = len(solver.domain[d])
        
        # 构建 fiber
        total_samples = rank * n_curr * rank
        paths = np.zeros((total_samples, n_dims), dtype=int)
        
        idx = 0
        for l in range(rank):
            for i in range(n_curr):
                for r in range(rank):
                    if d > 0:
                        paths[idx, :d] = l_indices[l]
                    paths[idx, d] = i
                    if d < n_dims - 1:
                        paths[idx, d+1:] = r_indices[r]
                    idx += 1
        
        coords = np.array([solver.domain[dim][paths[:, dim]] for dim in range(n_dims)]).T
        vals = solver.func(coords)
        fiber = vals.reshape(rank, n_curr, rank)
        
        # 边缘化
        M = np.sum(fiber, axis=1)
        
        print(f"\nLayer {d}:")
        print(f"  fiber max: {np.max(fiber):.6e}, sum: {np.sum(fiber):.6e}")
        print(f"  M diagonal: {np.diag(M)[:3]}...")
        print(f"  M max: {np.max(M):.6e}")
        
        # Pivot 校正
        if d > 0:
            pivot_coords = np.array([solver.domain[dim][solver.pivot_paths[:, dim]] for dim in range(n_dims)]).T
            pivot_vals = solver.func(pivot_coords)
            print(f"  Pivot vals: {pivot_vals[:3]}...")
            
            # 稳定校正
            max_pivot = np.max(np.abs(pivot_vals))
            threshold = max_pivot * 1e-10 if max_pivot > 0 else 1e-15
            correction = np.ones(rank)
            valid_mask = np.abs(pivot_vals) > threshold
            correction[valid_mask] = 1.0 / pivot_vals[valid_mask]
            
            print(f"  Correction: {correction[:3]}...")
            M = M * correction[np.newaxis, :]
        
        # 收缩
        left_vec = left_vec @ M
        print(f"  Result: {left_vec[0, :3]}...")
        print(f"  Result max: {np.max(np.abs(left_vec)):.6e}")

if __name__ == "__main__":
    debug_qtt_integral()
