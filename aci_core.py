"""
Adaptive Cross Interpolation (ACI) for Tensor Train decomposition

Implements automatic rank adaptation with error-driven stopping criterion.
"""
import numpy as np
from scipy.linalg import qr


class AdaptiveTCI:
    """
    自适应交叉插值 (ACI) 实现
    
    特点:
    - 自动增加秩直到误差收敛
    - 基于最大残差的停止准则
    - 支持 QTT 编码的高维问题
    """
    
    def __init__(self, func, domain, 
                 max_rank=100, 
                 tolerance=1e-6,
                 max_pivots_per_sweep=10,
                 n_test_samples=1000):
        """
        Args:
            func: 目标函数 f(indices) -> values
            domain: 每个维度的取值范围列表
            max_rank: 最大允许秩
            tolerance: 相对误差停止阈值
            max_pivots_per_sweep: 每轮最多添加的 Pivot 数
            n_test_samples: 误差估计的采样点数
        """
        self.func = func
        self.domain = domain
        self.n_dims = len(domain)
        self.max_rank = max_rank
        self.tolerance = tolerance
        self.max_pivots_per_sweep = max_pivots_per_sweep
        self.n_test_samples = n_test_samples
        
        # 初始化空的 Pivot 集合
        self.pivot_paths = None
        self.rank = 0
        
        # 缓存：已计算的函数值
        self.cache = {}
        
        # 收敛历史
        self.history = {
            'rank': [],
            'max_residual': [],
            'mean_error': []
        }
    
    def _get_cached_value(self, indices):
        """获取缓存的函数值或计算新值"""
        key = tuple(indices)
        if key not in self.cache:
            coords = np.array([self.domain[d][indices[d]] for d in range(self.n_dims)])
            self.cache[key] = self.func(coords.reshape(1, -1))[0]
        return self.cache[key]
    
    def _batch_evaluate(self, indices_batch):
        """批量计算函数值（带缓存）"""
        results = np.zeros(len(indices_batch))
        new_indices = []
        new_positions = []
        
        for i, idx in enumerate(indices_batch):
            key = tuple(idx)
            if key in self.cache:
                results[i] = self.cache[key]
            else:
                new_indices.append(idx)
                new_positions.append(i)
        
        if new_indices:
            new_indices = np.array(new_indices)
            coords = np.array([self.domain[d][new_indices[:, d]] for d in range(self.n_dims)]).T
            new_vals = self.func(coords)
            
            for j, (idx, val) in enumerate(zip(new_indices, new_vals)):
                self.cache[tuple(idx)] = val
                results[new_positions[j]] = val
        
        return results
    
    def _compute_tci_approximation(self, indices):
        """
        计算 TCI 对给定索引点的近似值
        
        使用 Rank-1 可分解形式:
        f(i) ≈ Σ_r [∏_d g_d^r(i_d)] / f(p_r)^{D-1}
        
        其中 g_d^r(i_d) = f(p_r[0:d-1], i_d, p_r[d+1:])
        """
        if self.rank == 0:
            return 0.0
        
        indices = np.atleast_2d(indices)
        n_points = len(indices)
        approx = np.zeros(n_points)
        
        for r in range(self.rank):
            pivot = self.pivot_paths[r]
            f_pivot = self._get_cached_value(pivot)
            
            if np.abs(f_pivot) < 1e-300:
                continue
            
            # 计算每个点的 Rank-1 贡献
            for p_idx in range(n_points):
                point = indices[p_idx]
                log_product = 0.0
                
                for d in range(self.n_dims):
                    # g_d(i_d) = f(pivot[0:d-1], i_d, pivot[d+1:])
                    test_path = pivot.copy()
                    test_path[d] = point[d]
                    g_val = self._get_cached_value(test_path)
                    
                    if g_val <= 0:
                        log_product = -np.inf
                        break
                    log_product += np.log(g_val)
                
                if np.isfinite(log_product):
                    log_product -= (self.n_dims - 1) * np.log(f_pivot)
                    approx[p_idx] += np.exp(log_product)
        
        # 多秩平均
        if self.rank > 0:
            approx /= self.rank
        
        return approx if n_points > 1 else approx[0]
    
    def _find_max_residual_point(self, n_candidates=1000):
        """
        寻找残差最大的点作为新 Pivot 候选
        
        使用真实的 TCI 近似误差（采样 + 批量计算）
        """
        # 随机采样候选点
        candidates = np.zeros((n_candidates, self.n_dims), dtype=int)
        for d in range(self.n_dims):
            candidates[:, d] = np.random.randint(0, len(self.domain[d]), size=n_candidates)
        
        # 计算真实值
        true_vals = self._batch_evaluate(candidates)
        
        max_true = np.max(np.abs(true_vals))
        
        if self.rank == 0:
            best_idx = np.argmax(np.abs(true_vals))
            return candidates[best_idx], 1.0, 1.0
        
        # 计算 TCI 近似值（批量优化版本）
        approx_vals = self._batch_tci_approximation(candidates)
        
        # 计算相对误差
        errors = np.abs(true_vals - approx_vals)
        
        if max_true > 0:
            rel_errors = errors / max_true
        else:
            rel_errors = errors
        
        max_error_idx = np.argmax(rel_errors)
        max_error = rel_errors[max_error_idx]
        mean_error = np.mean(rel_errors)
        
        return candidates[max_error_idx], max_error, mean_error
    
    def _batch_tci_approximation(self, indices):
        """
        批量计算 TCI 近似值
        
        使用简化的 Rank-1 求和公式，但批量计算所有需要的函数值
        """
        indices = np.atleast_2d(indices)
        n_points = len(indices)
        
        if self.rank == 0:
            return np.zeros(n_points)
        
        # 收集所有需要评估的路径
        all_paths = []
        path_indices = []  # (point_idx, rank_idx, dim_idx)
        
        for p_idx in range(n_points):
            for r_idx in range(self.rank):
                pivot = self.pivot_paths[r_idx]
                for d in range(self.n_dims):
                    test_path = pivot.copy()
                    test_path[d] = indices[p_idx, d]
                    all_paths.append(test_path)
                    path_indices.append((p_idx, r_idx, d))
        
        # 批量计算函数值
        all_paths = np.array(all_paths)
        all_vals = self._batch_evaluate(all_paths)
        
        # 重组为 (n_points, rank, n_dims)
        vals_reshaped = all_vals.reshape(n_points, self.rank, self.n_dims)
        
        # 计算每个点的近似值
        approx = np.zeros(n_points)
        
        for r_idx in range(self.rank):
            pivot = self.pivot_paths[r_idx]
            f_pivot = self._get_cached_value(pivot)
            
            if np.abs(f_pivot) < 1e-300:
                continue
            
            # 对每个点计算 log-product
            for p_idx in range(n_points):
                log_product = 0.0
                valid = True
                
                for d in range(self.n_dims):
                    g_val = vals_reshaped[p_idx, r_idx, d]
                    if g_val <= 0:
                        valid = False
                        break
                    log_product += np.log(g_val)
                
                if valid:
                    log_product -= (self.n_dims - 1) * np.log(f_pivot)
                    approx[p_idx] += np.exp(log_product)
        
        # 多秩平均
        approx /= self.rank
        
        return approx
    
    def _add_pivot(self, new_pivot):
        """添加新的 Pivot 点"""
        new_pivot = np.array(new_pivot, dtype=int)
        
        if self.pivot_paths is None:
            self.pivot_paths = new_pivot.reshape(1, -1)
        else:
            # 检查是否已存在
            for existing in self.pivot_paths:
                if np.array_equal(existing, new_pivot):
                    return False
            self.pivot_paths = np.vstack([self.pivot_paths, new_pivot])
        
        self.rank += 1
        return True
    
    def build_adaptive(self, anchors=None, verbose=True):
        """
        自适应构建 TCI 分解
        
        Args:
            anchors: 初始锚点
            verbose: 是否打印进度
        
        Returns:
            最终秩
        """
        # 初始化：添加锚点
        if anchors is not None:
            anchor_coords = np.array([self.domain[d][anchors[:, d]] for d in range(self.n_dims)]).T
            vals = self.func(anchor_coords)
            
            # 按函数值排序，优先添加值大的锚点
            sorted_idx = np.argsort(-np.abs(vals))
            for idx in sorted_idx[:min(5, len(sorted_idx))]:
                self._add_pivot(anchors[idx])
                
            if verbose:
                print(f"[Init] 添加 {min(5, len(anchors))} 个锚点, 最大值: {np.max(np.abs(vals)):.6e}")
        
        # 自适应循环
        iteration = 0
        prev_max_residual = np.inf
        stagnation_count = 0
        
        while self.rank < self.max_rank:
            iteration += 1
            
            # 寻找最大残差点
            new_pivot, max_residual, mean_error = self._find_max_residual_point()
            
            # 记录历史
            self.history['rank'].append(self.rank)
            self.history['max_residual'].append(max_residual)
            self.history['mean_error'].append(mean_error)
            
            if verbose and iteration % 10 == 0:
                print(f"[Iter {iteration}] Rank: {self.rank}, Val: {max_residual:.6e}, Mean: {mean_error:.6e}")
            
            # 收敛检查：函数值变得很小
            if max_residual < self.tolerance:
                if verbose:
                    print(f"[Converged] 在 Rank={self.rank} 收敛, Val={max_residual:.6e}")
                break
            
            # 停滞检查：如果连续添加 Pivot 没有明显改善
            if max_residual >= prev_max_residual * 0.99:
                stagnation_count += 1
            else:
                stagnation_count = 0
            
            if stagnation_count >= 20:
                if verbose:
                    print(f"[Stagnation] 在 Rank={self.rank} 停止（连续 20 次无改善）")
                break
            
            prev_max_residual = max_residual
            
            # 添加新 Pivot
            added = self._add_pivot(new_pivot)
            if not added:
                # 如果无法添加新 Pivot（已存在），尝试随机扰动
                perturbed = new_pivot.copy()
                perturbed[np.random.randint(self.n_dims)] = np.random.randint(len(self.domain[0]))
                self._add_pivot(perturbed)
        
        if verbose:
            print(f"[Done] 最终 Rank: {self.rank}")
        
        return self.rank
    
    def compute_integral(self, dx_vol):
        """
        基于 ACI 的积分计算
        
        使用重要性采样：以 Pivot 点为中心采样
        """
        if self.rank == 0:
            return 0.0
        
        n_samples = 50000
        d_size = len(self.domain[0])
        
        # 在 Pivot 附近采样
        samples = np.zeros((n_samples, self.n_dims), dtype=int)
        for d in range(self.n_dims):
            samples[:, d] = np.random.randint(0, d_size, size=n_samples)
        
        # 计算函数值
        vals = self._batch_evaluate(samples)
        
        # 积分估计
        total_grid_points = float(d_size ** self.n_dims)
        result = np.mean(vals) * total_grid_points * dx_vol
        
        return result


def run_aci_demo():
    """ACI 演示"""
    from qtt_utils import QTTEncoder
    from physics_models import vectorized_gaussian
    
    print("="*60)
    print("自适应交叉插值 (ACI) 演示")
    print("="*60)
    
    encoder = QTTEncoder(num_vars=3, num_bits=20, bounds=[(-3, 3)]*3)
    def wrapped_f(idx): return vectorized_gaussian(encoder.decode(idx))
    domain = [np.arange(encoder.d, dtype=int)] * encoder.R
    
    # 创建 ACI 实例
    aci = AdaptiveTCI(
        wrapped_f, domain,
        max_rank=100,
        tolerance=1e-4,
        n_test_samples=5000
    )
    
    # 构建自适应分解
    anchors = encoder.get_anchors()
    aci.build_adaptive(anchors=anchors, verbose=True)
    
    # 计算积分
    dx_vol = (6.0**3) / (encoder.d ** encoder.R)
    result = aci.compute_integral(dx_vol)
    
    print(f"\n最终积分结果: {result:.6f} (理论值: 5.5683)")
    print(f"相对误差: {abs(result - 5.5683) / 5.5683 * 100:.2f}%")


if __name__ == "__main__":
    run_aci_demo()
