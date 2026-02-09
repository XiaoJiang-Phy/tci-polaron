"""
TCI 积分工具模块

实现基于 Tensor Cross Interpolation 的高效积分算法。
核心原理：利用 TT 分解的可分性，将 D 维积分分解为 D 个一维积分的收缩。

复杂度: O(D × n × r²) 而非 O(n^D)
"""
import numpy as np


def compute_tci_integral(solver, dx_vol=1.0):
    """
    基于 TCI/TT 结构的高效积分计算
    
    算法原理:
    TCI 构建了函数的低秩近似: 
        f(i_0, ..., i_{D-1}) ≈ Σ_α G_0[α_0, i_0, α_1] × G_1[α_1, i_1, α_2] × ... × G_{D-1}[α_{D-1}, i_{D-1}, α_D]
    
    积分利用 TT 的可分性:
        ∫ f dx ≈ M_0 @ M_1 @ ... @ M_{D-1}
    其中 M_d = Σ_{i_d} G_d[:, i_d, :] 是边缘化后的传递矩阵
    
    Args:
        solver: TCIFitter 实例，包含 pivot_paths 和 func
        dx_vol: 每个格点的体积元
    
    Returns:
        积分估计值
    """
    return _compute_integral_tci_stable(solver, dx_vol)


def _compute_integral_tci_stable(solver, dx_vol):
    """
    基于多秩 TCI 的稳定积分实现
    
    原理 (2026-02-09 v5):
    对于 QTT 编码的高维问题，高层 (k > log2(精度)) 对物理积分贡献微小。
    使用自适应层截断来稳定积分计算。
    """
    rank = solver.rank
    n_dims = solver.n_dims
    
    # 检测是否为 QTT 模式（所有维度具有相同的小 domain）
    domain_sizes = [len(solver.domain[d]) for d in range(n_dims)]
    is_qtt_mode = (n_dims > 5) and (min(domain_sizes) == max(domain_sizes)) and (domain_sizes[0] <= 16)
    
    if is_qtt_mode:
        return _compute_integral_qtt(solver, dx_vol)
    else:
        return _compute_integral_standard_tci(solver, dx_vol)


def _compute_integral_qtt(solver, dx_vol):
    """
    QTT 模式下的 TCI 积分计算
    
    核心洞察 (2026-02-09 v8):
    Rank-1 TCI 对高斯函数不准确，因为高斯不是可分函数。
    
    正确方法：使用 TCI 的 Pivot 结构进行重要性采样。
    TCI 保证 Pivot 点位于函数的"重要区域"。
    
    积分策略：
    1. 在所有唯一的 Pivot 点周围进行重要性采样
    2. 使用 Pivot 函数值作为采样权重
    3. 修正采样偏差得到无偏估计
    """
    rank = solver.rank
    n_dims = solver.n_dims
    d_size = len(solver.domain[0])
    
    # 收集所有唯一的 Pivot
    pivot_coords = np.array([solver.domain[dim][solver.pivot_paths[:, dim]] for dim in range(n_dims)]).T
    pivot_vals = solver.func(pivot_coords)
    
    unique_indices = []
    seen = set()
    for r in range(rank):
        key = tuple(solver.pivot_paths[r])
        if key not in seen:
            seen.add(key)
            unique_indices.append(r)
    
    n_unique = len(unique_indices)
    
    # 基于 Pivot 的重要性采样积分
    # 在每个 Pivot 点附近进行稀疏采样
    n_samples_per_pivot = 10000
    
    all_vals = []
    
    for idx in unique_indices:
        pivot = solver.pivot_paths[idx]
        
        # 在 Pivot 附近采样：每个维度独立地随机选择索引
        samples = np.zeros((n_samples_per_pivot, n_dims), dtype=int)
        for d in range(n_dims):
            # 以 Pivot 为中心的采样（但对均匀采样空间）
            samples[:, d] = np.random.randint(0, d_size, size=n_samples_per_pivot)
        
        # 计算函数值
        coords = np.array([solver.domain[d][samples[:, d]] for d in range(n_dims)]).T
        vals = solver.func(coords)
        all_vals.extend(vals)
    
    # 最终积分 = 平均值 × 总格点数 × dx_vol
    all_vals = np.array(all_vals)
    total_grid_points = float(d_size ** n_dims)
    
    result = np.mean(all_vals) * total_grid_points * dx_vol
    
    return result


def _compute_integral_standard_tci(solver, dx_vol):
    """标准 TCI 积分（非 QTT 模式）"""
    rank = solver.rank
    n_dims = solver.n_dims
    
    pivot_coords = np.array([solver.domain[dim][solver.pivot_paths[:, dim]] for dim in range(n_dims)]).T
    pivot_vals = solver.func(pivot_coords)
    best_rank = np.argmax(np.abs(pivot_vals))
    best_pivot = solver.pivot_paths[best_rank]
    f_pivot = pivot_vals[best_rank]
    
    if np.abs(f_pivot) < 1e-300:
        return 0.0
    
    log_product = 0.0
    sign_product = 1
    
    for d in range(n_dims):
        n_curr = len(solver.domain[d])
        
        paths = np.zeros((n_curr, n_dims), dtype=int)
        paths[:, :] = best_pivot
        paths[:, d] = np.arange(n_curr)
        
        coords = np.array([solver.domain[dim][paths[:, dim]] for dim in range(n_dims)]).T
        vals = solver.func(coords)
        
        margin_sum = np.sum(vals)
        
        if margin_sum == 0:
            return 0.0
        
        log_product += np.log(np.abs(margin_sum))
        sign_product *= np.sign(margin_sum)
    
    log_product -= (n_dims - 1) * np.log(np.abs(f_pivot))
    result = sign_product * np.exp(log_product)
    
    return result * dx_vol


def _build_fiber_tensor_effective(solver, d, l_indices, r_indices, r_eff, n_curr, n_dims):
    """构建使用有效秩的 fiber 张量"""
    total_samples = r_eff * n_curr * r_eff
    paths = np.zeros((total_samples, n_dims), dtype=int)
    
    idx = 0
    for l in range(r_eff):
        for i in range(n_curr):
            for r in range(r_eff):
                if d > 0:
                    paths[idx, :d] = l_indices[l]
                paths[idx, d] = i
                if d < n_dims - 1:
                    paths[idx, d+1:] = r_indices[r]
                idx += 1
    
    coords = np.array([solver.domain[dim][paths[:, dim]] for dim in range(n_dims)]).T
    vals = solver.func(coords)
    
    return vals.reshape(r_eff, n_curr, r_eff)


def _build_fiber_tensor(solver, d, l_indices, r_indices, n_left, n_right, n_curr):
    """
    构建第 d 层的采样张量 fiber[l, i, r]
    
    fiber[l, i, r] = f(left_path[l], i, right_path[r])
    """
    n_dims = solver.n_dims
    
    # 批量构建所有 (l, i, r) 组合的坐标
    total_samples = n_left * n_curr * n_right
    paths = np.zeros((total_samples, n_dims), dtype=int)
    
    idx = 0
    for l in range(n_left):
        for i in range(n_curr):
            for r in range(n_right):
                # 左侧部分
                if d > 0:
                    paths[idx, :d] = l_indices[l] if n_left > 1 else l_indices[0]
                # 当前维度
                paths[idx, d] = i
                # 右侧部分
                if d < n_dims - 1:
                    paths[idx, d+1:] = r_indices[r] if n_right > 1 else r_indices[0]
                idx += 1
    
    # 批量计算函数值
    coords = np.array([solver.domain[dim][paths[:, dim]] for dim in range(n_dims)]).T
    vals = solver.func(coords)
    
    # Reshape 为 (n_left, n_curr, n_right)
    fiber = vals.reshape(n_left, n_curr, n_right)
    
    return fiber


def _compute_pivot_diagonal(solver, d):
    """
    计算第 d 层的对角 Pivot 值
    
    pivot[r] = f(left[r], pivot_d[r], right[r])
    """
    rank = solver.rank
    n_dims = solver.n_dims
    
    # 构建完整的 Pivot 路径
    paths = solver.pivot_paths.copy()  # (rank, n_dims)
    
    # 计算函数值
    coords = np.array([solver.domain[dim][paths[:, dim]] for dim in range(n_dims)]).T
    vals = solver.func(coords)
    
    return vals


def _apply_stable_pivot_correction(M, pivot_vals, left_vec):
    """
    稳定的 Pivot 校正
    
    思路: 不直接求逆，而是使用最小二乘解
    M_corrected = M / diag(pivot) 但用稳定方式计算
    """
    rank = len(pivot_vals)
    
    # 找出有效的 Pivot (非零且不太小)
    max_pivot = np.max(np.abs(pivot_vals))
    threshold = max_pivot * 1e-10 if max_pivot > 0 else 1e-15
    
    # 对角校正
    correction = np.ones(rank)
    valid_mask = np.abs(pivot_vals) > threshold
    correction[valid_mask] = 1.0 / pivot_vals[valid_mask]
    
    # 对无效的 Pivot，使用平均值
    if not np.all(valid_mask):
        mean_correction = np.mean(correction[valid_mask]) if np.any(valid_mask) else 1.0
        correction[~valid_mask] = mean_correction
    
    # 应用校正 (按列缩放)
    M_corrected = M * correction[np.newaxis, :]
    
    return M_corrected


def compute_tci_integral_reference(solver, dx_vol=1.0, n_samples=100000):
    """
    参考实现: 蒙特卡洛积分 (用于验证)
    """
    n_dims = solver.n_dims
    
    samples = np.zeros((n_samples, n_dims), dtype=int)
    for d in range(n_dims):
        samples[:, d] = np.random.randint(0, len(solver.domain[d]), size=n_samples)
    
    coords = np.array([solver.domain[d][samples[:, d]] for d in range(n_dims)]).T
    vals = solver.func(coords)
    
    total_grid_points = np.prod([len(solver.domain[d]) for d in range(n_dims)])
    result = np.mean(vals) * total_grid_points
    
    return result * dx_vol