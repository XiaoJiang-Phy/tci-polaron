import numpy as np

def compute_tci_integral(solver, dx_vol=1.0):
    """
    基于全张量收缩 (Full Tensor Contraction) 的积分计算。
    修正：大幅放宽数值截断阈值，防止物理信号被误杀。
    """
    rank = solver.rank
    
    # 极小值保护：只要大于 1e-200 就不视为 0，保留物理尾部
    # float64 的精度极限约为 1e-308
    SAFE_EPSILON = 1e-200 
    
    # --- 1. 初始化 (Layer 0) ---
    l_idx = np.zeros((1, 0), dtype=int)
    r_idx = solver.pivot_paths[:, 1:]
    
    # fiber_0: (n_curr, Rank)
    fiber_0 = solver._build_sweep_matrix_vectorized(0, l_idx, r_idx)
    
    # curr_vec: (1, Rank) -> 对物理维度求和
    curr_vec = np.sum(fiber_0, axis=0, keepdims=True)
    
    # 初始检查：如果连第一层都全是 0，那确实没东西
    if np.max(np.abs(curr_vec)) < SAFE_EPSILON:
        return 0.0

    # --- 2. 迭代收缩 (Layer 1 to d) ---
    for d in range(1, solver.n_dims):
        # A. 计算 Pivot Matrix P
        p_left = solver.pivot_paths[:, :d]   # (Rank, d)
        p_right = solver.pivot_paths[:, d:]  # (Rank, n-d)
        combined_paths = np.hstack([p_left, p_right]) 
        
        # 形状修正：确保传给 func 的是 (N, n_dims)
        coords_transposed = solver._path_to_coords(combined_paths.T).T
        P_vals = solver.func(coords_transposed) # (Rank, )
        
        # B. 计算安全的逆矩阵 (Safe Inverse)
        inv_P = np.zeros_like(P_vals)
        
        # 仅过滤掉极其微小的数值 (接近下溢出的值)
        # 对于 QTT 高斯函数，尾部数值可能在 1e-50 级别，必须保留！
        mask = np.abs(P_vals) > SAFE_EPSILON
        
        # 正常求逆
        inv_P[mask] = 1.0 / P_vals[mask]
        # mask 为 False 处保持 0 (视为该 Rank 已死)
        
        # 应用逆矩阵
        curr_vec = curr_vec * inv_P
        
        # C. 计算下一层积分矩阵 M
        l_idx_next = solver.pivot_paths[:, :d]
        r_idx_next = solver.pivot_paths[:, d+1:]
        
        fiber_mat = solver._build_sweep_matrix_vectorized(d, l_idx_next, r_idx_next)
        
        # Reshape & Sum
        n_curr = fiber_mat.shape[0] // rank
        fiber_tensor = fiber_mat.reshape(rank, n_curr, rank)
        M = np.sum(fiber_tensor, axis=1) # (Rank, Rank)
        
        # D. 收缩
        curr_vec = curr_vec @ M
        
        # 过程监控：如果中途全归零，提前退出（优化性能）
        if np.max(np.abs(curr_vec)) < SAFE_EPSILON:
            return 0.0

    # 最终结果求和
    final_val = np.sum(curr_vec)
    
    if np.isnan(final_val) or np.isinf(final_val):
        return 0.0
        
    return final_val * dx_vol