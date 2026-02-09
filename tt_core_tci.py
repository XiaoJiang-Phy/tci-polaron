"""
TT-Cores based TCI implementation with proper integral computation

Key insight: Instead of using Rank-1 sum formula, we explicitly build
TT-cores and compute integrals through core contraction.
"""
import numpy as np
from scipy.linalg import svd, qr


class TTCoreTCI:
    """
    基于 TT-Core 的 TCI 实现
    
    核心思想：显式构建和存储 TT-cores，支持稳定的积分计算
    """
    
    def __init__(self, func, domain, max_rank=50, tolerance=1e-8):
        self.func = func
        self.domain = domain
        self.n_dims = len(domain)
        self.max_rank = max_rank
        self.tolerance = tolerance
        
        # TT-cores: cores[d] has shape (r_{d-1}, n_d, r_d)
        # r_0 = r_{D} = 1 (boundary conditions)
        self.cores = [None] * self.n_dims
        self.ranks = [1] + [1] * self.n_dims  # Bond dimensions
        
    def build_cores_from_sampling(self, n_samples_per_dim=100):
        """
        通过采样构建 TT-cores
        
        使用 TT-cross 算法的简化版本，正确处理 SVD 奇异值
        """
        n_dims = self.n_dims
        
        # 右侧采样集合（固定用于所有层）
        n_right_samples = min(n_samples_per_dim, self.max_rank)
        right_samples_full = np.zeros((n_right_samples, n_dims), dtype=int)
        for dd in range(n_dims):
            right_samples_full[:, dd] = np.random.randint(0, len(self.domain[dd]), size=n_right_samples)
        
        # 左侧累积（用于生成路径）
        left_paths = np.zeros((1, 0), dtype=int)
        carry_matrix = None  # 传递矩阵（包含奇异值信息）
        
        for d in range(n_dims):
            n_curr = len(self.domain[d])
            r_prev = self.ranks[d]
            
            # 右侧采样（从当前层之后开始）
            right_samples = right_samples_full[:, d+1:] if d < n_dims - 1 else np.zeros((n_right_samples, 0), dtype=int)
            n_right = n_right_samples if d < n_dims - 1 else 1
            
            # 构建采样张量
            sample_tensor = np.zeros((r_prev, n_curr, n_right))
            
            for l in range(r_prev):
                for i in range(n_curr):
                    for r in range(n_right):
                        # 构建完整路径
                        path = np.zeros(n_dims, dtype=int)
                        if d > 0:
                            path[:d] = left_paths[l % len(left_paths)]
                        path[d] = i
                        if d < n_dims - 1:
                            path[d+1:] = right_samples[r]
                        
                        # 计算函数值
                        coords = np.array([self.domain[dd][path[dd]] for dd in range(n_dims)])
                        sample_tensor[l, i, r] = self.func(coords.reshape(1, -1))[0]
            
            # 应用之前的传递矩阵
            if carry_matrix is not None:
                # sample_tensor: (r_prev, n_curr, n_right)
                # carry_matrix: (r_prev_old, r_prev)
                # 需要收缩：result[l, i, r] = sum_k carry[k, l] * sample[l, i, r]
                # 但这里 carry 的作用是调整左侧索引的权重
                for i in range(n_curr):
                    for r in range(n_right):
                        sample_tensor[:, i, r] = carry_matrix @ sample_tensor[:, i, r]
            
            # 重塑并进行 SVD
            matrix = sample_tensor.reshape(r_prev * n_curr, n_right)
            
            if matrix.size == 0:
                self.cores[d] = np.zeros((r_prev, n_curr, 1))
                self.ranks[d + 1] = 1
                continue
            
            try:
                U, s, Vh = svd(matrix, full_matrices=False)
            except:
                U = matrix
                s = np.ones(min(matrix.shape))
                Vh = np.eye(min(matrix.shape[1], len(s)))
            
            # 自适应截断
            if len(s) > 0:
                total_norm = np.sum(s**2)
                if total_norm > 0:
                    cumsum = np.cumsum(s**2)
                    r_next = np.searchsorted(cumsum, total_norm * (1 - self.tolerance**2)) + 1
                else:
                    r_next = 1
            else:
                r_next = 1
            
            r_next = min(r_next, self.max_rank, len(s)) if len(s) > 0 else 1
            r_next = max(r_next, 1)
            
            # 存储 core：U 包含左正交基
            # 将奇异值吸收到右侧（传递给下一层）
            self.cores[d] = U[:, :r_next].reshape(r_prev, n_curr, r_next)
            self.ranks[d + 1] = r_next
            
            # 传递矩阵：s * Vh 传给下一层
            carry_matrix = np.diag(s[:r_next]) @ Vh[:r_next, :]
            
            # 更新左侧路径（用于下一层采样）
            # 简化：使用随机路径
            left_paths = np.zeros((r_next, d + 1), dtype=int)
            for dd in range(d + 1):
                left_paths[:, dd] = np.random.randint(0, len(self.domain[dd]), size=r_next)
            
            print(f"  Layer {d}: shape={self.cores[d].shape}, rank={r_next}, max_s={s[0] if len(s) > 0 else 0:.2e}")
        
        # 最后一个 core 需要特殊处理
        # 将剩余的 carry_matrix 吸收进最后一个 core
        if carry_matrix is not None and self.cores[-1] is not None:
            # cores[-1]: (r_prev, n_d, r_next)
            # carry_matrix 的列数应该与右侧采样数匹配，我们只取第一列（对应边界）
            if carry_matrix.shape[1] >= 1:
                scale = carry_matrix[:, 0]
                for i in range(self.cores[-1].shape[1]):
                    self.cores[-1][:, i, 0] *= scale
        
    def _generate_left_paths(self, d, n_paths):
        """从已构建的 cores 生成左侧路径"""
        paths = np.zeros((n_paths, d), dtype=int)
        
        # 简化：随机生成路径
        for dd in range(d):
            paths[:, dd] = np.random.randint(0, len(self.domain[dd]), size=n_paths)
        
        return paths
    
    def compute_integral(self, dx_vol):
        """
        通过 TT-core 收缩计算积分
        
        积分 = Σ_i prod_d G_d(i_d) = contract(Σ_{i_0} G_0, Σ_{i_1} G_1, ..., Σ_{i_{D-1}} G_{D-1})
        """
        if self.cores[0] is None:
            return 0.0
        
        # 初始化左边界向量
        left_vec = np.ones((1,))
        
        for d in range(self.n_dims):
            core = self.cores[d]  # (r_prev, n_d, r_next)
            
            if core is None:
                continue
            
            # 边缘化当前维度：sum_i core[:, i, :]
            marginalized = np.sum(core, axis=1)  # (r_prev, r_next)
            
            # 收缩
            left_vec = left_vec @ marginalized  # (r_next,)
        
        result = left_vec[0] if len(left_vec) > 0 else 0.0
        
        return result * dx_vol


def run_ttcore_demo():
    """TT-Core 演示"""
    from qtt_utils import QTTEncoder
    from physics_models import vectorized_gaussian
    
    print("="*60)
    print("TT-Core TCI 积分演示")
    print("="*60)
    
    encoder = QTTEncoder(num_vars=3, num_bits=20, bounds=[(-3, 3)]*3)
    def wrapped_f(idx): return vectorized_gaussian(encoder.decode(idx))
    domain = [np.arange(encoder.d, dtype=int)] * encoder.R
    
    # 创建 TT-Core TCI
    tt = TTCoreTCI(wrapped_f, domain, max_rank=20, tolerance=1e-6)
    
    print("\n构建 TT-cores...")
    tt.build_cores_from_sampling(n_samples_per_dim=50)
    
    # 计算积分
    dx_vol = (6.0**3) / (encoder.d ** encoder.R)
    result = tt.compute_integral(dx_vol)
    
    print(f"\nTT-core 积分结果: {result:.6f} (理论值: 5.5683)")
    print(f"相对误差: {abs(result - 5.5683) / 5.5683 * 100:.2f}%")


if __name__ == "__main__":
    run_ttcore_demo()
