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

    def get_anchors(self):
        """生成 QTT 战略锚点"""
        # 中点锚点：第一层比特全为 1 (二进制 0.100...)，其余为 0
        # 注意：这里返回的形状必须与 solver.domain 一致
        # domain 长度为 R (层数)，每层取值范围 [0, d-1]
        
        # 锚点 1: 全 0 (对应物理左边界)
        start = np.zeros(self.R, dtype=int)
        
        # 锚点 2: 中点 (最高位比特设为 1，其余为 0)
        # 对于 fused dimension d=2^n，最高位实际上是 d-1 (如果 n=1)
        # 更精确的做法是：每一层的 "0.5" 实际上只发生在第一层 (k=0)
        # 0.5 = 2^-1. 第一层权重大。
        mid = np.zeros(self.R, dtype=int)
        
        # 简单的中点启发式：设置所有层为中间值 (往往能捕捉到高斯峰)
        # 或者仅设置第一层为 max_val / 2
        mid[:] = (self.d - 1) // 2 
        
        return np.array([start, mid])