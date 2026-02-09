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
        """
        实现 MaxVol (Maximum Volume) 行选择
        
        修复说明 (2026-02-09):
        使用 QR 分解的列主元方法替代 LU，更稳定且易于控制输出长度。
        """
        m, n = matrix.shape
        if n == 0: 
            return np.array([], dtype=int)
        
        k = min(self.rank, m, n)  # 最多选择 k 行
        
        try:
            # 对转置矩阵做带主元的 QR 分解
            # matrix.T: (n, m), Q: (n, k), R: (k, m), P: (m,)
            from scipy.linalg import qr
            Q, R, P = qr(matrix.T, mode='economic', pivoting=True)
            I = np.array(P[:k], dtype=int)
        except Exception:
            # 回退：基于行范数选择
            row_norms = np.linalg.norm(matrix, axis=1)
            I = np.argsort(row_norms)[-k:][::-1]
        
        # 迭代优化 MaxVol (贪婪改进)
        for _ in range(20):
            try:
                sub_matrix = matrix[I, :]
                if sub_matrix.shape[0] == 0:
                    break
                # Z[i, j] = (替换后的体积增益)
                Z = matrix @ np.linalg.pinv(sub_matrix)
                max_idx = np.unravel_index(np.argmax(np.abs(Z)), Z.shape)
                if max_idx[1] >= len(I): 
                    break
                if np.abs(Z[max_idx]) > tolerance: 
                    I[max_idx[1]] = max_idx[0]
                else: 
                    break
            except:
                break
        
        return I.astype(int)

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

    def build_cores(self, anchors=None, n_sweeps=3, verbose=False):
        """
        TCI 双向扫频算法实现 (DMRG-like)
        
        修复说明 (2026-02-09):
        单向扫频在深层 TT 结构中无法保证全局一致性。
        双向扫频让每个站点能"看到"两端的优化结果。
        
        Args:
            anchors: 战略锚点数组
            n_sweeps: 完整扫频轮数 (前向+反向 = 1轮)
            verbose: 是否打印收敛信息
        """
        # 1. 智能初始化
        if anchors is not None:
            anchor_coords = np.array([self.domain[d][anchors[:,d]] for d in range(self.n_dims)]).T
            vals = self.func(anchor_coords)
            best = np.argmax(np.abs(vals))
            if np.abs(vals[best]) > 1e-15:
                for r in range(self.rank): 
                    self.pivot_paths[r, :] = anchors[best]
                if verbose:
                    print(f"[Init] 最佳锚点值: {vals[best]:.6e}")

        # 2. 双向扫频
        for sweep in range(n_sweeps):
            pivot_change = 0
            
            # 2a. 前向扫频 (d: 0 -> n_dims-1)
            for d in range(self.n_dims):
                old_pivots = self.pivot_paths[:, d].copy()
                self._optimize_site(d)
                pivot_change += np.sum(self.pivot_paths[:, d] != old_pivots)
            
            # 2b. 反向扫频 (d: n_dims-2 -> 0)
            for d in range(self.n_dims - 2, -1, -1):
                old_pivots = self.pivot_paths[:, d].copy()
                self._optimize_site(d)
                pivot_change += np.sum(self.pivot_paths[:, d] != old_pivots)
            
            if verbose:
                print(f"[Sweep {sweep+1}/{n_sweeps}] Pivot 变化数: {pivot_change}")
            
            # 早停条件: Pivot 不再变化
            if pivot_change == 0:
                if verbose:
                    print(f"[Early Stop] 在第 {sweep+1} 轮收敛")
                break
    
    def _optimize_site(self, d):
        """优化单个站点的 Pivot 选择"""
        l = np.zeros((1, 0), dtype=int) if d == 0 else self.pivot_paths[:, :d]
        r = np.zeros((1, 0), dtype=int) if d == self.n_dims-1 else self.pivot_paths[:, d+1:]
        
        sweep_matrix = self._build_sweep_matrix_vectorized(d, l, r)
        new_I = self._get_maxvol(sweep_matrix)
        
        old = self.pivot_paths.copy()
        for r_idx, f_idx in enumerate(new_I):
            if r_idx >= self.rank: break
            prev_r_idx = f_idx // len(self.domain[d])
            curr_val_idx = f_idx % len(self.domain[d])
            
            if d > 0: 
                self.pivot_paths[r_idx, :d] = old[prev_r_idx, :d]
            self.pivot_paths[r_idx, d] = curr_val_idx
        
        # 广播填补空缺的秩
        if 0 < len(new_I) < self.rank:
            for r_idx in range(len(new_I), self.rank):
                self.pivot_paths[r_idx, :] = self.pivot_paths[r_idx % len(new_I), :]