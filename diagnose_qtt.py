"""
QTT 数值爆炸诊断脚本
目标：逐层追踪 Pivot 值和积分收缩过程，定位爆炸源头
"""
import numpy as np
from tci_core import TCIFitter
from qtt_utils import QTTEncoder
from physics_models import vectorized_gaussian

def diagnose_qtt():
    print("="*60)
    print("QTT 数值爆炸诊断")
    print("="*60)
    
    # 1. 设置编码器
    encoder = QTTEncoder(num_vars=3, num_bits=20, bounds=[(-3, 3)]*3)
    print(f"QTT 配置: {encoder.n_vars}变量 × {encoder.R}层, 融合维度 d={encoder.d}")
    
    # 2. 验证锚点解码
    print("\n--- 锚点验证 ---")
    anchors = encoder.get_anchors()
    for i, anchor in enumerate(anchors):
        coords = encoder.decode(anchor.reshape(1, -1))
        val = vectorized_gaussian(coords)
        print(f"锚点 {i}: 索引[0]={anchor[0]}, 物理坐标={coords[0]}, f(x)={val[0]:.6e}")
    
    # 3. 构建 TCI 并检查 Pivot 值
    print("\n--- TCI 构建 (verbose) ---")
    def wrapped_f(idx): return vectorized_gaussian(encoder.decode(idx))
    domain = [np.arange(encoder.d, dtype=int)] * encoder.R
    
    solver = TCIFitter(wrapped_f, domain, rank=10)
    solver.build_cores(anchors=anchors, n_sweeps=3, verbose=True)
    
    # 4. 检查最终 Pivot 路径的函数值
    print("\n--- 最终 Pivot 诊断 ---")
    for r in range(min(3, solver.rank)):  # 只看前 3 个秩
        path = solver.pivot_paths[r]
        coords = encoder.decode(path.reshape(1, -1))
        val = vectorized_gaussian(coords)
        print(f"Rank {r}: 路径[0:3]={path[:3]}, 物理坐标={coords[0][:3]}, f(x)={val[0]:.6e}")
    
    # 5. 逐层追踪积分收缩
    print("\n--- 积分收缩追踪 ---")
    trace_integral(solver, encoder)

def trace_integral(solver, encoder):
    """逐层追踪积分计算过程"""
    rank = solver.rank
    SAFE_EPSILON = 1e-200
    
    # Layer 0
    l_idx = np.zeros((1, 0), dtype=int)
    r_idx = solver.pivot_paths[:, 1:]
    fiber_0 = solver._build_sweep_matrix_vectorized(0, l_idx, r_idx)
    curr_vec = np.sum(fiber_0, axis=0, keepdims=True)
    print(f"Layer 0: curr_vec max={np.max(np.abs(curr_vec)):.6e}")
    
    for d in range(1, min(5, solver.n_dims)):  # 只看前 5 层
        # Pivot Matrix
        p_left = solver.pivot_paths[:, :d]
        p_right = solver.pivot_paths[:, d:]
        combined_paths = np.hstack([p_left, p_right])
        coords_transposed = solver._path_to_coords(combined_paths.T).T
        P_vals = solver.func(coords_transposed)
        
        print(f"Layer {d}: P_vals = {P_vals[:5]} (min={np.min(np.abs(P_vals)):.2e})")
        
        # 检查是否有极小值导致爆炸
        if np.min(np.abs(P_vals)) < 1e-10:
            print(f"  ⚠️ 警告: Pivot 值过小，求逆将放大 {1/np.min(np.abs(P_vals[P_vals != 0])):.2e} 倍!")
        
        # 使用 pinv
        P_diag = np.diag(P_vals)
        inv_P_diag = np.linalg.pinv(P_diag, rcond=1e-10)
        inv_P = np.diag(inv_P_diag)
        curr_vec = curr_vec * inv_P
        
        # 下一层积分矩阵
        l_idx_next = solver.pivot_paths[:, :d]
        r_idx_next = solver.pivot_paths[:, d+1:]
        fiber_mat = solver._build_sweep_matrix_vectorized(d, l_idx_next, r_idx_next)
        n_curr = fiber_mat.shape[0] // rank
        fiber_tensor = fiber_mat.reshape(rank, n_curr, rank)
        M = np.sum(fiber_tensor, axis=1)
        curr_vec = curr_vec @ M
        
        print(f"  After contraction: curr_vec max={np.max(np.abs(curr_vec)):.6e}")
        
        if np.max(np.abs(curr_vec)) > 1e20:
            print(f"  ❌ 爆炸检测! 在 Layer {d} 发生数值溢出")
            break

if __name__ == "__main__":
    diagnose_qtt()
