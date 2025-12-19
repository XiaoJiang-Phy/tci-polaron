import numpy as np
from scipy.linalg import lu

class TCIFitter:
    def __init__(self, func, domain, rank=10):
        self.func = func
        self.domain = domain
        self.n_dims = len(domain)
        self.rank = rank
        # 保持 pivot_paths 为整数索引
        self.pivot_paths = np.zeros((rank, self.n_dims), dtype=int)
        for d in range(self.n_dims):
            self.pivot_paths[:, d] = np.random.randint(0, len(domain[d]), size=rank)

    def _path_to_coords(self, path_indices):
        """将离散索引路径转换为物理/逻辑坐标数组"""
        # 注意：这里返回的类型取决于 self.domain 的内容类型
        return np.array([self.domain[d][path_indices[d]] for d in range(self.n_dims)])

    def _get_maxvol(self, matrix, tolerance=1.05):
        """实现 MaxVol (MCI) 逻辑"""
        m, r = matrix.shape
        if r == 0: return np.array([], dtype=int)
        try:
            # 增加 check_finite=False 略微提升速度，防止极小值报错
            _, _, p = lu(matrix, p_indices=True, check_finite=False)
            k = min(self.rank, m, r)
            I = np.array(p[:k], dtype=int)
        except Exception:
            # 回退机制：随机选择
            I = np.random.choice(m, min(self.rank, m), replace=False)

        # 迭代优化 MaxVol
        for _ in range(20):
            try:
                sub_matrix = matrix[I, :]
                # 使用 pinv 处理可能的病态矩阵
                Z = matrix @ np.linalg.pinv(sub_matrix)
                idx = np.unravel_index(np.argmax(np.abs(Z)), Z.shape)
                if idx[1] >= len(I): break
                if np.abs(Z[idx]) > tolerance: I[idx[1]] = idx[0]
                else: break
            except: break
        return I.flatten().astype(int)

    def _build_sweep_matrix_vectorized(self, d, left, right):
        """
        向量化构造采样矩阵。
        【关键修正】移除了 dtype=float 的强制转换，支持 Integer 传递给 QTTEncoder
        """
        n_curr = len(self.domain[d])
        r_prev, r_next = left.shape[0], right.shape[0]
        
        # 1. 构建全索引张量 (Batch, n_dims)
        coords_idx = np.zeros((r_prev, n_curr, r_next, self.n_dims), dtype=int)
        
        # 填充左侧路径
        for i in range(d): 
            coords_idx[:,:,:,i] = left[:, i][:, None, None]
        # 填充当前维度 (遍历所有可能值)
        coords_idx[:,:,:,d] = np.arange(n_curr, dtype=int)[None, :, None]
        # 填充右侧路径
        for i in range(self.n_dims - d - 1):
            coords_idx[:,:,:,d+1+i] = right[:, i][None, None, :]
        
        flat_idx = coords_idx.reshape(-1, self.n_dims)
        
        # 2. 将索引映射回 domain 值 (核心修正点)
        # 我们先创建一个空容器，类型跟随 domain 的第一个元素，不强制 float
        sample_val = self.domain[0][0]
        flat_coords = np.zeros(flat_idx.shape, dtype=type(sample_val))
        
        for i in range(self.n_dims):
            flat_coords[:, i] = self.domain[i][flat_idx[:, i]]
            
        # 3. 调用物理函数
        # reshape 回 (Left * Current, Right) 以符合 TCI 矩阵结构
        return self.func(flat_coords).reshape(r_prev * n_curr, r_next)

    def build_cores(self, anchors=None):
        """TCI 扫频算法实现"""
        if anchors is not None:
            # Smart Init
            anchor_coords = np.array([self.domain[d][anchors[:,d]] for d in range(self.n_dims)]).T
            vals = self.func(anchor_coords)
            best = np.argmax(np.abs(vals))
            # 只有当锚点有显著值时才覆盖，防止全零初始化
            if np.abs(vals[best]) > 1e-15:
                # 广播最佳路径到所有秩，作为良好的起点
                for r in range(self.rank): 
                    self.pivot_paths[r, :] = anchors[best]

        for d in range(self.n_dims):
            l = np.zeros((1, 0), dtype=int) if d == 0 else self.pivot_paths[:, :d]
            r = np.zeros((1, 0), dtype=int) if d == self.n_dims-1 else self.pivot_paths[:, d+1:]
            
            sweep_matrix = self._build_sweep_matrix_vectorized(d, l, r)
            new_I = self._get_maxvol(sweep_matrix)
            
            old = self.pivot_paths.copy()
            for r_idx, f_idx in enumerate(new_I):
                if r_idx >= self.rank: break
                # 解码 f_idx:它是 (r_prev * n_curr) 的展平索引
                # 需要还原出 prev_rank_idx 和 curr_dim_idx
                prev_r_idx = f_idx // len(self.domain[d])
                curr_val_idx = f_idx % len(self.domain[d])
                
                if d > 0: 
                    self.pivot_paths[r_idx, :d] = old[prev_r_idx, :d]
                self.pivot_paths[r_idx, d] = curr_val_idx
            
            # 广播填补空缺的秩
            if 0 < len(new_I) < self.rank:
                for r_idx in range(len(new_I), self.rank):
                    self.pivot_paths[r_idx, :] = self.pivot_paths[r_idx % len(new_I), :]