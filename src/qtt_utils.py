import numpy as np

class QTTEncoder:
    """Fused Representation following Ritter et al. 2024"""
    def __init__(self, num_vars, num_bits, bounds):
        self.n_vars = num_vars    
        self.R = num_bits         
        self.bounds = bounds      
        self.d = 2 ** num_vars    # physical dimension of the fused site

    def decode(self, qtt_indices):
        """
        Decode fused QTT indices back to physical coordinates.
        qtt_indices: (M, R) integer index matrix
        """
        # Ensure integer type; bitwise operations fail otherwise
        qtt_indices = np.atleast_2d(qtt_indices).astype(int)
        M, R = qtt_indices.shape
        u = np.zeros((M, self.n_vars))
        
        for k in range(R):
            # Extract column k (the k-th fused bit layer)
            val = qtt_indices[:, k] 
            weight = 2.0 ** -(k + 1)
            for v in range(self.n_vars):
                # Unpack fused bits: from most significant to least significant
                bit = (val >> (self.n_vars - 1 - v)) & 1
                u[:, v] += bit * weight
        
        # Map to physical range [min, max]
        res = []
        for i, b in enumerate(self.bounds):
            val = b[0] + (b[1]-b[0]) * u[:, i]
            res.append(val)
        return np.array(res).T

    def encode(self, physical_coords):
        """
        Encode physical coordinates to QTT indices (inverse of decode).
        
        physical_coords: (M, n_vars) physical coordinates
        Returns: (M, R) QTT indices
        """
        physical_coords = np.atleast_2d(physical_coords)
        M = physical_coords.shape[0]
        
        # 1. Normalize to [0, 1]
        u = np.zeros((M, self.n_vars))
        for i, b in enumerate(self.bounds):
            u[:, i] = (physical_coords[:, i] - b[0]) / (b[1] - b[0])
        
        # 2. Extract bits layer by layer and fuse
        qtt_indices = np.zeros((M, self.R), dtype=int)
        for k in range(self.R):
            fused_val = 0
            for v in range(self.n_vars):
                # Extract the k-th bit (most significant bit first)
                bit = int(u[0, v] * (2 ** (k + 1))) % 2
                fused_val |= (bit << (self.n_vars - 1 - v))
            qtt_indices[:, k] = fused_val
        
        return qtt_indices

    def get_anchors(self):
        """
        Generate strategic QTT anchor points.
        
        Fix note (2026-02-09 v2):
        Compute correct QTT indices directly from physical coordinates via encode().
        """
        # Physical coordinate points -> QTT indices
        physical_anchors = np.array([
            [-3.0, -3.0, -3.0],  # left boundary
            [0.0, 0.0, 0.0],     # Gaussian peak center (most important!)
            [3.0, 3.0, 3.0],     # right boundary
            [1.0, 1.0, 1.0],     # offset probe point
            [-1.0, -1.0, -1.0],  # another probe point
        ])
        
        qtt_anchors = []
        for coord in physical_anchors:
            qtt_idx = self.encode(coord.reshape(1, -1))
            qtt_anchors.append(qtt_idx[0])
        
        return np.array(qtt_anchors)