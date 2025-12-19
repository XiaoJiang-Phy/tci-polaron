import numpy as np
from scipy.linalg import lu, solve

class TCIFitter:
    """TCI 拟合器，用于基于张量交叉插值方法的快速原型验证。"""

    def __init__(self, func, domain, rank=5):
        self.func = func
        self.domain = domain
        self.n_dims = len(domain)
        self.rank = rank
        # 初始化路径矩阵 (r, N)。默认随机采样作为初始猜测。
        self.pivot_paths = np.zeros((rank, self.n_dims), dtype=int)
        for d in range(self.n_dims):
            self.pivot_paths[:, d] = np.random.choice(len(domain[d]), rank)

    # ... _get_maxvol, _update_paths, get_tci_integral 保持不变 ...
    def _get_maxvol(self, matrix, tolerance=1.05):
        """简易的 MaxVol 行选择算法。

        该实现用于从矩阵中挑选出使得子矩阵体积较大的 r 行，作为近似的 pivots。

        参数：
            matrix: 输入矩阵，形状为 (m, r)，其中 m 是样本数，r 是秩近似值。
            tolerance: 容忍度，若更新项的绝对值大于该阈值则替换对应的行索引。
        """
        # 1. 使用 LU 分解获得初始索引 I。此处仅作为初始猜测，具体实现可替换为更稳健的方法。
        p, _, _ = lu(matrix, p_indices=True)
        I = list(p[:matrix.shape[1]])

        # 2. 迭代改进索引 I
        for _ in range(100):  # 最大迭代次数
            # 计算 Z = A * inv(A[I, :])，用线性方程求解替代显式求逆以提高稳定性与效率。
            Z = solve(matrix[I, :].T, matrix.T).T

            # 找到 Z 中绝对值最大的元素及其位置 (i, j)。若超过容忍度则替换对应索引。
            idx = np.unravel_index(np.argmax(np.abs(Z)), Z.shape)
            if np.abs(Z[idx]) > tolerance:
                I[idx[1]] = idx[0]
            else:
                break

        return np.array(I)


    def _update_paths(self, d, new_I, left_pivots):
        """根据新的行索引（new_I）更新 pivot paths。

        new_I 中的每一项表示展平后矩阵的行索引，通过整除和取余可以恢复上一路径的索引与当前维度的索引。
        """

        n_curr = len(self.domain[d])
        old_paths = self.pivot_paths.copy()
        for r_idx, flat_idx in enumerate(new_I):
            # 计算对应的上一条路径索引与当前维度在网格中的索引。
            prev_path_idx = flat_idx // n_curr
            curr_dim_idx = flat_idx % n_curr
            self.pivot_paths[r_idx, :d] = old_paths[prev_path_idx, :d]
            self.pivot_paths[r_idx, d] = curr_dim_idx

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

        # 构造采样矩阵并逐项采样函数值。
        matrix = np.zeros((n_curr * r_prev, r_next))
        # 为了方便索引，预先取出当前维度的网格点。
        current_grid = self.domain[dim_idx]

        # 填充采样矩阵，逐行计算目标函数值并赋值。
        for i in range(r_prev):
            for j in range(n_curr):
                row_idx = i * n_curr + j
                for k in range(r_next):
                    # 拼接左侧、当前、右侧的坐标为完整的 N 维坐标，然后采样函数值。
                    full_coords = self._assemble_coords(left_pivots[i], current_grid[j], right_pivots[k])
                    matrix[row_idx, k] = self.func(full_coords)

        return matrix

    def _build_sweep_matrix_vectorized(self, d, left_indices, right_indices):
        """向量化构造采样矩阵，用批量处理替代三层嵌套循环以提高性能。"""
        n_curr = len(self.domain[d])
        r_prev = left_indices.shape[0]
        r_next = right_indices.shape[0]

        # 总样本数为 r_prev * n_curr * r_next。
        total_samples = r_prev * n_curr * r_next

        # 构造坐标大矩阵 (TotalSamples, N)，一次性拼凑出所有点的 N 维坐标。
        all_coords = self._assemble_all_coords(d, left_indices, right_indices)

        # 批量调用函数（这是提速的关键）。假设 func 支持 (M, N) 输入并返回长度为 M 的数组。
        all_values = self.func(all_coords)

        # 将结果重塑回 (r_prev * n_curr, r_next) 并返回。
        return all_values.reshape(r_prev * n_curr, r_next)
    
    def _assemble_coords(self, left_indices, current_coord, right_indices):
        """将左侧索引、当前坐标和右侧索引拼接为完整的 N 维坐标向量并返回。"""

        left_vals = [self.domain[d][idx] for d, idx in enumerate(left_indices)]
        # 获取右侧各维度对应的数值并拼接。
        right_vals = [self.domain[len(left_indices) + 1 + d][idx] for d, idx in enumerate(right_indices)]
        return np.array(left_vals + [current_coord] + right_vals)

    # def _assemble_all_coords(self, d, left_idx, right_idx):
    #     """利用广播机制一次性拼凑所有坐标（旧的示例实现，保留以供参考）。"""
    #     r_prev = left_idx.shape[0]
    #     n_curr = len(self.domain[d])
    #     r_next = right_idx.shape[0]

    #     # 构造最终的坐标数组容器，形状为 (r_prev, n_curr, r_next, N)。
    #     coords_tensor = np.zeros((r_prev, n_curr, r_next, self.n_dims))

    #     # 填充左侧维度（从之前的路径继承）。
    #     for i in range(d):
    #         vals = self.domain[i][left_idx[:, i]]
    #         coords_tensor[:, :, :, i] = vals[:, np.newaxis, np.newaxis]

    #     # 填充当前维度 d。
    #     curr_vals = self.domain[d]
    #     coords_tensor[:, :, :, d] = curr_vals[np.newaxis, :, np.newaxis]

    #     # 填充右侧维度。
    #     for i in range(self.n_dims - d - 1):
    #         dim_idx = d + 1 + i
    #         vals = self.domain[dim_idx][right_idx[:, i]]
    #         coords_tensor[:, :, :, dim_idx] = vals[np.newaxis, np.newaxis, :]

    #     # 展平为 (TotalSamples, N) 以供批量函数调用并返回。
    #     return coords_tensor.reshape(-1, self.n_dims)
    
    
    def _assemble_all_coords(self, d, left_idx, right_idx):
        """利用广播一次性拼凑所有采样点的 N 维坐标并返回。

        返回值形状为 (r_prev * n_curr * r_next, N)。"""
        r_prev = left_idx.shape[0]
        n_curr = len(self.domain[d])
        r_next = right_idx.shape[0]
        
        # 1. 预分配大矩阵 (r_prev, n_curr, r_next, N)，用于存储所有坐标组合。
        coords_tensor = np.zeros((r_prev, n_curr, r_next, self.n_dims))

        # 2. 填充左侧维度（索引范围为 0 到 d-1）。循环次数仅为 d，开销可接受。
        for i in range(d):
            coords_tensor[:, :, :, i] = self.domain[i][left_idx[:, i]][:, np.newaxis, np.newaxis]

        # 3. 填充当前维度 d（对 n_curr 个点广播）。
        coords_tensor[:, :, :, d] = self.domain[d][:, np.newaxis]

        # 4. 填充右侧维度（d+1 到 N-1）。同样使用广播进行填充。
        for i in range(self.n_dims - d - 1):
            dim_idx = d + 1 + i
            coords_tensor[:, :, :, dim_idx] = self.domain[dim_idx][right_idx[:, i]][np.newaxis, np.newaxis, :]

        # 展平为 (TotalSamples, N) 并返回。
        return coords_tensor.reshape(-1, self.n_dims)
    
    def _path_to_coords(self, path_indices):
        """将一整条路径索引 [i1, i2, ..., iN] 转换为对应的物理坐标向量并返回。"""

        return np.array([self.domain[d][idx] for d, idx in enumerate(path_indices)])

    def get_tci_integral(self):
        """基于当前 pivot paths 估计 TCI 积分的近似值。

        该实现为示例版，假设 rank=1 时更为合理；对于更高秩需要更完善的归一化策略。
        """

        # 1. 获取最优路径上的中心点值（Core value）。取第一条路径作为代表。
        p = self.pivot_paths[0]
        f_max = self.func(self._path_to_coords(p))

        # 防止除零或数值不稳定性。
        if np.abs(f_max) < 1e-15:
            return 0

        # 2. 计算每个维度的 1D 积分贡献，按维度逐个替换并累乘归一化因子。
        integral = f_max
        for d in range(self.n_dims):
            dim_sum = 0
            for i_val, val in enumerate(self.domain[d]):
                # 构造路径索引：固定其他维度为 pivot paths，改变第 d 维的索引。
                full_idx = self.pivot_paths[0].copy()
                full_idx[d] = i_val
                dim_sum += self.func(self._path_to_coords(full_idx))

            # TCI 的积分恒等式：按维度归一化累乘。
            integral *= (dim_sum / f_max)

        # 3. 乘以网格步长 dx^N 并返回。
        dx = self.domain[0][1] - self.domain[0][0]
        return integral * (dx ** self.n_dims)
    
    def evaluate(self, point_indices):
        """利用 pivot paths 对给定网格点进行简单的近似重构。

        说明：该方法为示例性实现，仅用于验证 pivot paths 的合理性，完整 N 维重构需更复杂的实现。
        """
        # 1. 找到对应的核心矩阵（Core Matrix），即 pivot paths 交叉点处的函数值矩阵。
        r = self.rank
        core_matrix = np.zeros((r, r))
        for i in range(r):
            for j in range(r):
                # 交叉点采样：组合左侧路径与右侧路径的索引。这里为简化处理，直接使用现有的 pivot_paths。
                coords = self._assemble_coords(self.pivot_paths[i, :0], # 假设 2D
                                              self.domain[0][self.pivot_paths[i, 0]], 
                                              self.pivot_paths[j, 1:])
                # 注意：实际 N 维重构更复杂，这里先以 2D 逻辑验证 MVP
                # 简易逻辑：直接返回插值点的近似
                pass 
        
        # 考虑到 N 维张量重构的复杂性，MVP 阶段最简单的验证方法是：
        # 看看 pivot_paths 是否收敛到了高斯函数的中心附近。
        return self.pivot_paths
    
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