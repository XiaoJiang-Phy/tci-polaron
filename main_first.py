import numpy as np
from scipy.linalg import lu
from scipy.linalg import solve

class TCIFitter:
    def __init__(self, func, domain, rank_limit=10):
        self.func = func
        self.domain = domain
        self.n_dims = len(domain)
        self.rank = rank_limit  # 统一使用 self.rank
        
        # --- 核心修复：在这里初始化路径矩阵 ---
        self.pivot_paths = np.zeros((self.rank, self.n_dims), dtype=int)
        for d in range(self.n_dims):
            for r in range(self.rank):
                self.pivot_paths[r, d] = np.random.choice(len(self.domain[d]))

    def _get_maxvol(self, matrix, tolerance=1.05):
        """
        任务 A: 提升工程能力。
        在这里实现一个简易的 MaxVol 算法。
        提示：可以使用 scipy.linalg.lu 的选主元结果，或者查阅 MaxVol 的迭代算法。
        :param matrix: 输入矩阵，形状为 (m, r)，其中 m 是样本数，r 是秩。
        :param tolerance: 容忍度，控制选择的行的质量。
        """
        # 1. 使用 LU 分解获得初始索引 I
        p, _, _ = lu(matrix, p_indices=True)
        I = list(p[:matrix.shape[1]])

        # 2. 迭代改进索引 I
        for _ in range(100):  # 最大迭代次数
            # 计算 Z = A * inv(A[I, :])
            # 思考点：直接求逆效率低，怎么用 solve 替代？
            #sub_matrix = matrix[I, :]
            #Z = matrix @ np.linalg.inv(sub_matrix)

            # 提示：sub_matrix.T @ Z.T = matrix.T
            Z = solve(matrix[I,:].T, matrix.T).T

            # 找到 Z 中绝对值最大的元素及其位置 (i, j)
            idx = np.unravel_index(np.argmax(np.abs(Z)), Z.shape)

            # 如果最大绝对值大于 tolerance，则更新索引
            if np.abs(Z[idx]) > tolerance:
                I[idx[1]] = idx[0]
            else:
                break  # 满足容忍度，停止迭代

        return np.array(I)


    def _build_sweep_matrix(self, dim_idx, left_pivots, right_pivots):
        """
        构造第 dim_idx 维度的采样矩阵
        :param dim_idx: 当前扫描的维度索引 (0 到 N-1)
        :param left_pivots: 左侧维度的路径集合 (r_prev, dim_idx)
        :param right_pivots: 右侧维度的路径集合 (r_next, N - dim_idx - 1)
        """
        n_curr = len(self.domain[dim_idx])
        r_prev = left_pivots.shape[0]
        r_next = right_pivots.shape[0]

        # 构造采样矩阵
        matrix = np.zeros((n_curr*r_prev, r_next))

        # 为了方便索引，我们预先取出当前维度的网格点
        current_grid = self.domain[dim_idx]

        # 填充采样矩阵
        for i in range(r_prev):
            for j in range(n_curr):
                row_idx = i*n_curr + j
                for k in range(r_next):
                    # 任务：拼凑出完整的 N 维坐标向量
                    # 1. 组合左侧、当前、右侧的坐标
                    full_coords = self._assemble_coords(left_pivots[i], current_grid[j], right_pivots[k])

                    # 2. 采样目标函数
                    matrix[row_idx, k] = self.func(full_coords)

        return matrix
    
    def _assemble_coords(self, left_indices, current_coord, right_indices):
        """
        拼凑出完整的 N 维坐标向量
        :param left_indices: 左侧维度的索引 (dim_idx 之前)
        :param current_coord: 当前维度的坐标 (dim_idx)
        :param right_indices: 右侧维度的索引 (dim_idx 之后)
        :return: 完整的 N 维坐标向量
        """
        left_vals = [self.domain[d][idx] for d, idx in enumerate(left_indices)]

        # 获取右侧各维度对应的具体数值
        right_vals = [self.domain[len(left_indices) + 1 + d][idx] for d, idx in enumerate(right_indices)]
        return np.array(left_vals + [current_coord] + right_vals)

    def build_cores(self):
        """
        任务 B: 提升算法能力。
        这是 TCI 的核心迭代。
        逻辑：
        1. 固定其他维度，对当前维度进行采样，构造采样矩阵。
        2. 调用 _get_maxvol 更新当前维度的最佳采样点（Pivots）。
        3. 计算交叉点上的核心矩阵（Inverse Core）。
        """
        # --- 正向扫一遍 ---
        for d in range(self.n_dims):
            # 1. 准备左侧右侧的“锚点”
            #左侧是之前维度已经选好的最优路径：取前d列
            left_pivots = self.pivot_paths[:, :d] if d > 0 else np.zeros((self.rank, 0), dtype=int)
            # 右侧是后续维度现有的路径：取 d+1 之后的所有列
            right_pivots = self.pivot_paths[:, d+1:] if d < self.n_dims - 1 else np.zeros((self.rank, 0), dtype=int)

            # 2. 构造采样矩阵
            matrix = self._build_sweep_matrix(d, left_pivots, right_pivots)

            # 3. 使用 MaxVol 更新当前维度的 Pivots
            new_indices = self._get_maxvol(matrix)

            # 4. 更新全局路径矩阵
            self._update_paths(d, new_indices, left_pivots)

    def _update_paths(self, d, new_I, left_pivots):
        n_curr = len(self.domain[d])
        old_paths = self.pivot_paths.copy()
        for r_idx, flat_idx in enumerate(new_I):
            # 计算对应的当前维度索引
            prev_path_idx = flat_idx // n_curr
            curr_dim_idx = flat_idx % n_curr
            self.pivot_paths[r_idx, :d] = old_paths[prev_path_idx, :d]
            self.pivot_paths[r_idx, d] = curr_dim_idx


    def evaluate(self, point_indices):
        """
        利用学到的最优路径（Pivot Paths）对任意网格点进行插值
        这里演示一个最基础的 2D/3D 重构逻辑：
        F(x, y) = f(x, P_y) * [f(P_x, P_y)]^-1 * f(P_x, y)
        """
        # 1. 找到对应的核心矩阵（Core Matrix）
        # 这是所有最优路径交叉点的函数值矩阵
        r = self.rank
        core_matrix = np.zeros((r, r))
        for i in range(r):
            for j in range(r):
                # 交叉点采样：左侧最优路径 + 右侧最优路径
                # 这里为了简化，直接用 build_cores 完结后的 pivot_paths
                coords = self._assemble_coords(self.pivot_paths[i, :0], # 假设 2D
                                              self.domain[0][self.pivot_paths[i, 0]], 
                                              self.pivot_paths[j, 1:])
                # 注意：实际 N 维重构更复杂，这里先以 2D 逻辑验证 MVP
                # 简易逻辑：直接返回插值点的近似
                pass 
        
        # 考虑到 N 维张量重构的复杂性，MVP 阶段最简单的验证方法是：
        # 看看 pivot_paths 是否收敛到了高斯函数的中心附近。
        return self.pivot_paths
    
    def _path_to_coords(self, path_indices):
        """将一整条路径索引 [i1, i2, ..., iN] 转换为物理坐标"""
        return np.array([self.domain[d][idx] for d, idx in enumerate(path_indices)])

    def get_tci_integral(self):
        # 1. 获取最优路径上的中心点值 (Core value)
        p = self.pivot_paths[0] # 假设 rank=1
        f_max = self.func(self._path_to_coords(p))
        
        # 防止分母为 0
        if np.abs(f_max) < 1e-15: return 0
        
        # 2. 计算每个维度的 1D 积分贡献
        integral = f_max
        for d in range(self.n_dims):
            dim_sum = 0
            for i_val, val in enumerate(self.domain[d]):
                # 构造采样路径：除了维度 d 变动，其他都固定在 pivot_paths 上
                full_idx = self.pivot_paths[0].copy()
                full_idx[d] = i_val
                dim_sum += self.func(self._path_to_coords(full_idx))
            
            # TCI 的积分恒等式：按维度归一化累乘
            integral *= (dim_sum / f_max)
        
        # 3. 乘以网格步长 dx^N
        dx = self.domain[0][1] - self.domain[0][0]
        return integral * (dx ** self.n_dims)



# 使用示例
if __name__ == "__main__":
    # 1. 定义目标：3D 高斯函数
    def target(x): 
        return np.exp(-np.sum(x**2))
    
    # 2. 定义网格：3个维度，每个 50 点
    grid = [np.linspace(-3, 3, 50) for _ in range(3)]
    
    # 3. 初始化并训练
    # 这里的 rank 设置为 1 即可，因为高斯函数是秩为 1 的张量
    solver = TCIFitter(target, grid, rank_limit=2)
    print("初始路径示例:\n", solver.pivot_paths)
    
    solver.build_cores()
    
    print("\n优化后的路径 (Pivot Paths):")
    print(solver.pivot_paths)
    
    # 4. 物理直觉验证：
    # 对于 exp(-x^2)，最大值在 x=0，即索引 25 附近
    # 如果优化后的索引接近 [24, 24, 24]，说明你的 TCI 逻辑完全正确！
    mid_idx = 50 // 2
    print(f"\n期望索引接近: [{mid_idx}, {mid_idx}, {mid_idx}]")

    # 5. 计算积分
    integral = solver.get_tci_integral()
    print(f"\nTCI 计算的积分结果: {integral}")