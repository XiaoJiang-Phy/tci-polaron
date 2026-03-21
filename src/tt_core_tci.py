"""
TT-Cores based TCI implementation with proper integral computation

Key insight: Instead of using Rank-1 sum formula, we explicitly build
TT-cores and compute integrals through core contraction.
"""
import numpy as np
from scipy.linalg import svd, qr


class TTCoreTCI:
    """
    TCI implementation based on explicit TT-Cores.
    
    Core idea: explicitly build and store TT-cores to support stable
    integral computation.
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
        Build TT-cores via sampling.
        
        Uses a simplified TT-cross algorithm with proper SVD singular value handling.
        """
        n_dims = self.n_dims
        
        # Right-side sample set (fixed for all layers)
        n_right_samples = min(n_samples_per_dim, self.max_rank)
        right_samples_full = np.zeros((n_right_samples, n_dims), dtype=int)
        for dd in range(n_dims):
            right_samples_full[:, dd] = np.random.randint(0, len(self.domain[dd]), size=n_right_samples)
        
        # Left-side accumulation (for path generation)
        left_paths = np.zeros((1, 0), dtype=int)
        carry_matrix = None  # Transfer matrix (carries singular value information)
        
        for d in range(n_dims):
            n_curr = len(self.domain[d])
            r_prev = self.ranks[d]
            
            # Right-side samples (starting from the layer after current)
            right_samples = right_samples_full[:, d+1:] if d < n_dims - 1 else np.zeros((n_right_samples, 0), dtype=int)
            n_right = n_right_samples if d < n_dims - 1 else 1
            
            # Build sampling tensor
            sample_tensor = np.zeros((r_prev, n_curr, n_right))
            
            for l in range(r_prev):
                for i in range(n_curr):
                    for r in range(n_right):
                        # Build complete path
                        path = np.zeros(n_dims, dtype=int)
                        if d > 0:
                            path[:d] = left_paths[l % len(left_paths)]
                        path[d] = i
                        if d < n_dims - 1:
                            path[d+1:] = right_samples[r]
                        
                        # Evaluate function
                        coords = np.array([self.domain[dd][path[dd]] for dd in range(n_dims)])
                        sample_tensor[l, i, r] = self.func(coords.reshape(1, -1))[0]
            
            # Apply previous transfer matrix
            if carry_matrix is not None:
                # sample_tensor: (r_prev, n_curr, n_right)
                # carry_matrix: (r_prev_old, r_prev)
                # Contract: result[l, i, r] = sum_k carry[k, l] * sample[l, i, r]
                # carry adjusts the weights of left-side indices
                for i in range(n_curr):
                    for r in range(n_right):
                        sample_tensor[:, i, r] = carry_matrix @ sample_tensor[:, i, r]
            
            # Reshape and perform SVD
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
            
            # Adaptive truncation
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
            
            # Store core: U contains the left-orthogonal basis
            # Absorb singular values into the right side (pass to next layer)
            self.cores[d] = U[:, :r_next].reshape(r_prev, n_curr, r_next)
            self.ranks[d + 1] = r_next
            
            # Transfer matrix: s * Vh passed to the next layer
            carry_matrix = np.diag(s[:r_next]) @ Vh[:r_next, :]
            
            # Update left paths (for next-layer sampling)
            # Simplified: use random paths
            left_paths = np.zeros((r_next, d + 1), dtype=int)
            for dd in range(d + 1):
                left_paths[:, dd] = np.random.randint(0, len(self.domain[dd]), size=r_next)
            
            print(f"  Layer {d}: shape={self.cores[d].shape}, rank={r_next}, max_s={s[0] if len(s) > 0 else 0:.2e}")
        
        # Last core needs special treatment
        # Absorb residual carry_matrix into the last core
        if carry_matrix is not None and self.cores[-1] is not None:
            # cores[-1]: (r_prev, n_d, r_next)
            # Take the first column of carry_matrix (corresponding to boundary)
            if carry_matrix.shape[1] >= 1:
                scale = carry_matrix[:, 0]
                for i in range(self.cores[-1].shape[1]):
                    self.cores[-1][:, i, 0] *= scale
        
    def _generate_left_paths(self, d, n_paths):
        """Generate left-side paths from previously built cores."""
        paths = np.zeros((n_paths, d), dtype=int)
        
        # Simplified: randomly generate paths
        for dd in range(d):
            paths[:, dd] = np.random.randint(0, len(self.domain[dd]), size=n_paths)
        
        return paths
    
    def compute_integral(self, dx_vol):
        """
        Compute integral via TT-core contraction.
        
        Integral = Σ_i prod_d G_d(i_d) = contract(Σ_{i_0} G_0, Σ_{i_1} G_1, ..., Σ_{i_{D-1}} G_{D-1})
        """
        if self.cores[0] is None:
            return 0.0
        
        # Initialize left boundary vector
        left_vec = np.ones((1,))
        
        for d in range(self.n_dims):
            core = self.cores[d]  # (r_prev, n_d, r_next)
            
            if core is None:
                continue
            
            # Marginalize current dimension: sum_i core[:, i, :]
            marginalized = np.sum(core, axis=1)  # (r_prev, r_next)
            
            # Contract
            left_vec = left_vec @ marginalized  # (r_next,)
        
        result = left_vec[0] if len(left_vec) > 0 else 0.0
        
        return result * dx_vol


def run_ttcore_demo():
    """TT-Core demo."""
    from .qtt_utils import QTTEncoder
    from .physics_models import vectorized_gaussian
    
    print("="*60)
    print("TT-Core TCI Integration Demo")
    print("="*60)
    
    encoder = QTTEncoder(num_vars=3, num_bits=20, bounds=[(-3, 3)]*3)
    def wrapped_f(idx): return vectorized_gaussian(encoder.decode(idx))
    domain = [np.arange(encoder.d, dtype=int)] * encoder.R
    
    # Create TT-Core TCI
    tt = TTCoreTCI(wrapped_f, domain, max_rank=20, tolerance=1e-6)
    
    print("\nBuilding TT-cores...")
    tt.build_cores_from_sampling(n_samples_per_dim=50)
    
    # Compute integral
    dx_vol = (6.0**3) / (encoder.d ** encoder.R)
    result = tt.compute_integral(dx_vol)
    
    print(f"\nTT-core integral result: {result:.6f} (theory: 5.5683)")
    print(f"Relative error: {abs(result - 5.5683) / 5.5683 * 100:.2f}%")


if __name__ == "__main__":
    run_ttcore_demo()
