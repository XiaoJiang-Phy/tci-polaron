"""
QTT 积分的稳定实现
使用简化的采样验证方法，避免病态矩阵求逆
"""
import numpy as np
from qtt_utils import QTTEncoder
from physics_models import vectorized_gaussian

def compute_integral_monte_carlo(encoder, func, n_samples=100000):
    """
    使用蒙特卡洛方法计算积分作为参考值
    """
    # 在 QTT 索引空间中均匀采样
    samples_idx = np.random.randint(0, encoder.d, size=(n_samples, encoder.R))
    
    # 解码到物理坐标
    coords = encoder.decode(samples_idx)
    
    # 计算函数值
    vals = func(coords)
    
    # 积分 = 平均值 × 体积
    volume = np.prod([b[1] - b[0] for b in encoder.bounds])
    integral = np.mean(vals) * volume
    
    return integral

def compute_integral_direct_sum(encoder, func, max_samples_per_dim=4):
    """
    直接求和法：在每个 QTT 层选择代表性采样点
    避免完全遍历 d^R 个点
    """
    # 使用稀疏网格采样
    n_dims = encoder.R
    
    # 每层采样 max_samples_per_dim 个点
    sample_indices = []
    for _ in range(n_dims):
        sample_indices.append(np.linspace(0, encoder.d - 1, max_samples_per_dim, dtype=int))
    
    # 构建稀疏网格
    grid = np.meshgrid(*sample_indices, indexing='ij')
    flat_grid = np.vstack([g.ravel() for g in grid]).T  # (N, n_dims)
    
    # 限制采样数量
    if len(flat_grid) > 100000:
        indices = np.random.choice(len(flat_grid), 100000, replace=False)
        flat_grid = flat_grid[indices]
    
    # 解码并计算
    coords = encoder.decode(flat_grid)
    vals = func(coords)
    
    # 积分估计
    volume = np.prod([b[1] - b[0] for b in encoder.bounds])
    # 权重 = 体积 / 采样点数 × (全空间点数 / 采样网格点数)
    weight = volume * (encoder.d ** encoder.R) / len(flat_grid)
    
    integral = np.sum(vals) * volume / len(flat_grid)
    
    return integral

def test_qtt_integration():
    print("="*60)
    print("QTT 积分稳定性测试")
    print("="*60)
    
    encoder = QTTEncoder(num_vars=3, num_bits=20, bounds=[(-3, 3)]*3)
    print(f"QTT 配置: {encoder.n_vars}变量 × {encoder.R}层")
    print(f"理论积分值: 5.5683 (3D Gaussian)")
    
    # 方法 1: 蒙特卡洛
    print("\n--- 蒙特卡洛采样 ---")
    for n in [10000, 100000, 1000000]:
        np.random.seed(42)
        result = compute_integral_monte_carlo(encoder, vectorized_gaussian, n_samples=n)
        print(f"  N={n:>7}: {result:.6f}")
    
    # 方法 2: 低分辨率 QTT 检验
    print("\n--- 低分辨率 QTT 测试 ---")
    for bits in [4, 6, 8, 10]:
        enc_low = QTTEncoder(num_vars=3, num_bits=bits, bounds=[(-3, 3)]*3)
        np.random.seed(42)
        result = compute_integral_monte_carlo(enc_low, vectorized_gaussian, n_samples=100000)
        print(f"  {bits} bits ({2**(bits*3):>10} 点): {result:.6f}")

if __name__ == "__main__":
    test_qtt_integration()
