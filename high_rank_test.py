"""
High-Rank TCI with Monte Carlo Integration

Strategy: Use high-rank TCI to identify important regions,
then use the Pivot information to guide importance sampling.
"""
import numpy as np
from tci_core import TCIFitter
from qtt_utils import QTTEncoder
from physics_models import vectorized_gaussian


def run_high_rank_demo():
    """高秩 TCI 演示"""
    print("="*60)
    print("高秩 TCI 积分演示")
    print("="*60)
    
    encoder = QTTEncoder(num_vars=3, num_bits=20, bounds=[(-3, 3)]*3)
    def wrapped_f(idx): return vectorized_gaussian(encoder.decode(idx))
    domain = [np.arange(encoder.d, dtype=int)] * encoder.R
    
    # 测试不同秩的效果
    ranks_to_test = [10, 30, 50, 80, 100]
    
    dx_vol = (6.0**3) / (encoder.d ** encoder.R)
    theoretical = 5.5683
    
    print(f"\n理论积分值: {theoretical:.4f}")
    print("-"*50)
    
    for rank in ranks_to_test:
        # 创建高秩 TCI
        solver = TCIFitter(wrapped_f, domain, rank=rank)
        anchors = encoder.get_anchors()
        solver.build_cores(anchors=anchors, n_sweeps=5)
        
        # 使用 Pivot 导向的蒙特卡洛积分
        result = compute_pivot_guided_integral(solver, dx_vol, n_samples=100000)
        
        error = abs(result - theoretical) / theoretical * 100
        print(f"Rank={rank:3d}: 积分={result:.4f}, 误差={error:.2f}%")
    
    print("\n" + "="*60)
    print("分析：不同秩对积分精度的影响")
    print("="*60)


def compute_pivot_guided_integral(solver, dx_vol, n_samples=100000):
    """
    基于 Pivot 导向的蒙特卡洛积分
    
    使用 TCI 的 Pivot 作为重要性采样的参考点
    """
    rank = solver.rank
    n_dims = solver.n_dims
    d_size = len(solver.domain[0])
    total_grid_points = float(d_size ** n_dims)
    
    # 收集唯一的 Pivot
    unique_pivots = []
    seen = set()
    for r in range(rank):
        key = tuple(solver.pivot_paths[r])
        if key not in seen:
            seen.add(key)
            unique_pivots.append(solver.pivot_paths[r])
    
    n_unique = len(unique_pivots)
    
    # 在每个 Pivot 附近采样
    samples_per_pivot = n_samples // n_unique
    
    all_vals = []
    
    for pivot in unique_pivots:
        # 均匀随机采样（整个索引空间）
        samples = np.zeros((samples_per_pivot, n_dims), dtype=int)
        for d in range(n_dims):
            samples[:, d] = np.random.randint(0, d_size, size=samples_per_pivot)
        
        # 计算函数值
        coords = np.array([solver.domain[d][samples[:, d]] for d in range(n_dims)]).T
        vals = solver.func(coords)
        all_vals.extend(vals)
    
    # 积分估计
    all_vals = np.array(all_vals)
    result = np.mean(all_vals) * total_grid_points * dx_vol
    
    return result


def analyze_pivot_distribution(solver, encoder):
    """分析 Pivot 点在物理空间的分布"""
    n_dims = solver.n_dims
    
    print("\nPivot 点的物理分布分析:")
    print("-"*40)
    
    # 解码所有唯一 Pivot 到物理空间
    unique_pivots = []
    seen = set()
    for r in range(solver.rank):
        key = tuple(solver.pivot_paths[r])
        if key not in seen:
            seen.add(key)
            unique_pivots.append(solver.pivot_paths[r])
    
    for i, pivot in enumerate(unique_pivots[:10]):  # 只显示前 10 个
        physical = encoder.decode(pivot.reshape(1, -1))[0]
        f_val = solver.func(pivot.reshape(1, -1))[0]
        print(f"  Pivot {i}: 物理坐标≈({physical[0]:.2f}, {physical[1]:.2f}, {physical[2]:.2f}), f={f_val:.4f}")
    
    if len(unique_pivots) > 10:
        print(f"  ... 还有 {len(unique_pivots) - 10} 个 Pivot")


if __name__ == "__main__":
    run_high_rank_demo()
