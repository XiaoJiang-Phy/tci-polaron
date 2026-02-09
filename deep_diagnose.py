"""
深度诊断：为什么 Pivot 从正确的锚点 (0,0,0) 跳到 (-3,-3,0)？
"""
import numpy as np
from tci_core import TCIFitter
from qtt_utils import QTTEncoder
from physics_models import vectorized_gaussian

def deep_diagnose():
    encoder = QTTEncoder(num_vars=3, num_bits=20, bounds=[(-3, 3)]*3)
    def wrapped_f(idx): return vectorized_gaussian(encoder.decode(idx))
    domain = [np.arange(encoder.d, dtype=int)] * encoder.R
    
    anchors = encoder.get_anchors()
    
    # 手动创建 solver 并检查初始化
    solver = TCIFitter(wrapped_f, domain, rank=10)
    
    print("=== 初始化前 (随机) ===")
    print(f"pivot_paths[0] = {solver.pivot_paths[0][:5]}...")
    
    # 手动执行锚点初始化
    anchor_coords = np.array([domain[d][anchors[:,d]] for d in range(solver.n_dims)]).T
    vals = solver.func(anchor_coords)
    print(f"\n锚点函数值: {vals}")
    best = np.argmax(np.abs(vals))
    print(f"最佳锚点索引: {best}, 值: {vals[best]:.6e}")
    
    # 初始化 pivot_paths
    for r in range(solver.rank): 
        solver.pivot_paths[r, :] = anchors[best]
    
    print(f"\n=== 初始化后 (锚点 {best}) ===")
    print(f"pivot_paths[0] = {solver.pivot_paths[0][:5]}...")
    
    # 检查第一个站点优化
    print("\n=== 第一次站点优化 (d=0) ===")
    d = 0
    l = np.zeros((1, 0), dtype=int)
    r = solver.pivot_paths[:, 1:]
    
    print(f"右侧环境 r[0] = {r[0][:5]}...")
    
    sweep_matrix = solver._build_sweep_matrix_vectorized(d, l, r)
    print(f"sweep_matrix shape: {sweep_matrix.shape}")
    print(f"sweep_matrix 最大值: {np.max(sweep_matrix):.6e}")
    print(f"sweep_matrix 非零元素: {np.sum(sweep_matrix != 0)}")
    
    # 找出 MaxVol 选择了什么
    new_I = solver._get_maxvol(sweep_matrix)
    print(f"MaxVol 选择的索引: {new_I}")
    
    # 解码这些索引
    for idx in new_I[:3]:
        curr_val_idx = idx % len(domain[d])
        print(f"  idx={idx} -> domain[0][{curr_val_idx}] = {domain[d][curr_val_idx]}")
        # 解码完整路径
        full_path = np.zeros(solver.n_dims, dtype=int)
        full_path[0] = curr_val_idx
        full_path[1:] = solver.pivot_paths[0, 1:]  # 使用初始化的右侧
        coords = encoder.decode(full_path.reshape(1, -1))
        val = wrapped_f(full_path.reshape(1, -1))
        print(f"    物理坐标: {coords[0][:3]}, f(x)={val[0]:.6e}")

if __name__ == "__main__":
    deep_diagnose()
