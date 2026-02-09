import numpy as np

class QTTEncoder:
    """遵循 Ritter et al. 2024 的 Fused Representation"""
    def __init__(self, num_vars, num_bits, bounds):
        self.n_vars = num_vars    
        self.R = num_bits         
        self.bounds = bounds      
        self.d = 2 ** num_vars    # 融合站点的物理维度

    def decode(self, qtt_indices):
        """
        将 Fused QTT 索引解码回物理坐标 
        qtt_indices: (M, R) 整数索引矩阵
        """
        # 【关键】确保是整数，否则位运算会报错
        qtt_indices = np.atleast_2d(qtt_indices).astype(int)
        M, R = qtt_indices.shape
        u = np.zeros((M, self.n_vars))
        
        for k in range(R):
            # 取出第 k 列 (即第 k 个融合比特层)
            val = qtt_indices[:, k] 
            weight = 2.0 ** -(k + 1)
            for v in range(self.n_vars):
                # 解包 fused bits: 从高位到低位
                bit = (val >> (self.n_vars - 1 - v)) & 1
                u[:, v] += bit * weight
        
        # 映射到物理范围 [min, max]
        res = []
        for i, b in enumerate(self.bounds):
            val = b[0] + (b[1]-b[0]) * u[:, i]
            res.append(val)
        return np.array(res).T

    def encode(self, physical_coords):
        """
        将物理坐标编码为 QTT 索引 (decode 的逆过程)
        
        physical_coords: (M, n_vars) 物理坐标
        返回: (M, R) QTT 索引
        """
        physical_coords = np.atleast_2d(physical_coords)
        M = physical_coords.shape[0]
        
        # 1. 归一化到 [0, 1]
        u = np.zeros((M, self.n_vars))
        for i, b in enumerate(self.bounds):
            u[:, i] = (physical_coords[:, i] - b[0]) / (b[1] - b[0])
        
        # 2. 逐层提取比特并融合
        qtt_indices = np.zeros((M, self.R), dtype=int)
        for k in range(self.R):
            fused_val = 0
            for v in range(self.n_vars):
                # 提取第 k 个比特 (最高位优先)
                bit = int(u[0, v] * (2 ** (k + 1))) % 2
                fused_val |= (bit << (self.n_vars - 1 - v))
            qtt_indices[:, k] = fused_val
        
        return qtt_indices

    def get_anchors(self):
        """
        生成 QTT 战略锚点
        
        修复说明 (2026-02-09 v2):
        直接通过 encode() 从物理坐标计算正确的 QTT 索引。
        """
        # 物理坐标点 -> QTT 索引
        physical_anchors = np.array([
            [-3.0, -3.0, -3.0],  # 左边界
            [0.0, 0.0, 0.0],     # 高斯峰值中心 (最重要!)
            [3.0, 3.0, 3.0],     # 右边界
            [1.0, 1.0, 1.0],     # 偏移探测点
            [-1.0, -1.0, -1.0],  # 另一个探测点
        ])
        
        qtt_anchors = []
        for coord in physical_anchors:
            qtt_idx = self.encode(coord.reshape(1, -1))
            qtt_anchors.append(qtt_idx[0])
        
        return np.array(qtt_anchors)