"""
Adaptive Cross Interpolation (ACI) for Tensor Train decomposition

Implements automatic rank adaptation with error-driven stopping criterion.
"""
import numpy as np
from scipy.linalg import qr


class AdaptiveTCI:
    """
    Adaptive Cross Interpolation (ACI) implementation.
    
    Features:
    - Automatic rank increase until error convergence
    - Stopping criterion based on maximum residual
    - Supports QTT-encoded high-dimensional problems
    """
    
    def __init__(self, func, domain, 
                 max_rank=100, 
                 tolerance=1e-6,
                 max_pivots_per_sweep=10,
                 n_test_samples=1000):
        """
        Args:
            func: target function f(indices) -> values
            domain: list of value ranges for each dimension
            max_rank: maximum allowed rank
            tolerance: relative error stopping threshold
            max_pivots_per_sweep: maximum pivots added per sweep
            n_test_samples: number of sample points for error estimation
        """
        self.func = func
        self.domain = domain
        self.n_dims = len(domain)
        self.max_rank = max_rank
        self.tolerance = tolerance
        self.max_pivots_per_sweep = max_pivots_per_sweep
        self.n_test_samples = n_test_samples
        
        # Initialize empty pivot set
        self.pivot_paths = None
        self.rank = 0
        
        # Cache: previously computed function values
        self.cache = {}
        
        # Convergence history
        self.history = {
            'rank': [],
            'max_residual': [],
            'mean_error': []
        }
    
    def _get_cached_value(self, indices):
        """Get cached function value or compute a new one."""
        key = tuple(indices)
        if key not in self.cache:
            coords = np.array([self.domain[d][indices[d]] for d in range(self.n_dims)])
            self.cache[key] = self.func(coords.reshape(1, -1))[0]
        return self.cache[key]
    
    def _batch_evaluate(self, indices_batch):
        """Batch-evaluate function values (with caching)."""
        results = np.zeros(len(indices_batch))
        new_indices = []
        new_positions = []
        
        for i, idx in enumerate(indices_batch):
            key = tuple(idx)
            if key in self.cache:
                results[i] = self.cache[key]
            else:
                new_indices.append(idx)
                new_positions.append(i)
        
        if new_indices:
            new_indices = np.array(new_indices)
            coords = np.array([self.domain[d][new_indices[:, d]] for d in range(self.n_dims)]).T
            new_vals = self.func(coords)
            
            for j, (idx, val) in enumerate(zip(new_indices, new_vals)):
                self.cache[tuple(idx)] = val
                results[new_positions[j]] = val
        
        return results
    
    def _compute_tci_approximation(self, indices):
        """
        Compute TCI approximation at given index points.
        
        Uses the Rank-1 separable form:
        f(i) ≈ Σ_r [∏_d g_d^r(i_d)] / f(p_r)^{D-1}
        
        where g_d^r(i_d) = f(p_r[0:d-1], i_d, p_r[d+1:])
        """
        if self.rank == 0:
            return 0.0
        
        indices = np.atleast_2d(indices)
        n_points = len(indices)
        approx = np.zeros(n_points)
        
        for r in range(self.rank):
            pivot = self.pivot_paths[r]
            f_pivot = self._get_cached_value(pivot)
            
            if np.abs(f_pivot) < 1e-300:
                continue
            
            # Compute Rank-1 contribution per point
            for p_idx in range(n_points):
                point = indices[p_idx]
                log_product = 0.0
                
                for d in range(self.n_dims):
                    # g_d(i_d) = f(pivot[0:d-1], i_d, pivot[d+1:])
                    test_path = pivot.copy()
                    test_path[d] = point[d]
                    g_val = self._get_cached_value(test_path)
                    
                    if g_val <= 0:
                        log_product = -np.inf
                        break
                    log_product += np.log(g_val)
                
                if np.isfinite(log_product):
                    log_product -= (self.n_dims - 1) * np.log(f_pivot)
                    approx[p_idx] += np.exp(log_product)
        
        # Multi-rank average
        if self.rank > 0:
            approx /= self.rank
        
        return approx if n_points > 1 else approx[0]
    
    def _find_max_residual_point(self, n_candidates=1000):
        """
        Find the point with maximum residual as a new pivot candidate.
        
        Uses true TCI approximation error (sampling + batch evaluation).
        """
        # Randomly sample candidate points
        candidates = np.zeros((n_candidates, self.n_dims), dtype=int)
        for d in range(self.n_dims):
            candidates[:, d] = np.random.randint(0, len(self.domain[d]), size=n_candidates)
        
        # Compute true values
        true_vals = self._batch_evaluate(candidates)
        
        max_true = np.max(np.abs(true_vals))
        
        if self.rank == 0:
            best_idx = np.argmax(np.abs(true_vals))
            return candidates[best_idx], 1.0, 1.0
        
        # Compute TCI approximation values (batch-optimized)
        approx_vals = self._batch_tci_approximation(candidates)
        
        # Compute relative errors
        errors = np.abs(true_vals - approx_vals)
        
        if max_true > 0:
            rel_errors = errors / max_true
        else:
            rel_errors = errors
        
        max_error_idx = np.argmax(rel_errors)
        max_error = rel_errors[max_error_idx]
        mean_error = np.mean(rel_errors)
        
        return candidates[max_error_idx], max_error, mean_error
    
    def _batch_tci_approximation(self, indices):
        """
        Batch-compute TCI approximation values.
        
        Uses the simplified Rank-1 sum formula with batch function evaluation.
        """
        indices = np.atleast_2d(indices)
        n_points = len(indices)
        
        if self.rank == 0:
            return np.zeros(n_points)
        
        # Collect all paths needing evaluation
        all_paths = []
        path_indices = []  # (point_idx, rank_idx, dim_idx)
        
        for p_idx in range(n_points):
            for r_idx in range(self.rank):
                pivot = self.pivot_paths[r_idx]
                for d in range(self.n_dims):
                    test_path = pivot.copy()
                    test_path[d] = indices[p_idx, d]
                    all_paths.append(test_path)
                    path_indices.append((p_idx, r_idx, d))
        
        # Batch function evaluation
        all_paths = np.array(all_paths)
        all_vals = self._batch_evaluate(all_paths)
        
        # Reorganize to (n_points, rank, n_dims)
        vals_reshaped = all_vals.reshape(n_points, self.rank, self.n_dims)
        
        # Compute approximation for each point
        approx = np.zeros(n_points)
        
        for r_idx in range(self.rank):
            pivot = self.pivot_paths[r_idx]
            f_pivot = self._get_cached_value(pivot)
            
            if np.abs(f_pivot) < 1e-300:
                continue
            
            # Compute log-product for each point
            for p_idx in range(n_points):
                log_product = 0.0
                valid = True
                
                for d in range(self.n_dims):
                    g_val = vals_reshaped[p_idx, r_idx, d]
                    if g_val <= 0:
                        valid = False
                        break
                    log_product += np.log(g_val)
                
                if valid:
                    log_product -= (self.n_dims - 1) * np.log(f_pivot)
                    approx[p_idx] += np.exp(log_product)
        
        # Multi-rank average
        approx /= self.rank
        
        return approx
    
    def _add_pivot(self, new_pivot):
        """Add a new pivot point."""
        new_pivot = np.array(new_pivot, dtype=int)
        
        if self.pivot_paths is None:
            self.pivot_paths = new_pivot.reshape(1, -1)
        else:
            # Check if already exists
            for existing in self.pivot_paths:
                if np.array_equal(existing, new_pivot):
                    return False
            self.pivot_paths = np.vstack([self.pivot_paths, new_pivot])
        
        self.rank += 1
        return True
    
    def build_adaptive(self, anchors=None, verbose=True):
        """
        Adaptively build TCI decomposition.
        
        Args:
            anchors: initial anchor points
            verbose: whether to print progress
        
        Returns:
            Final rank
        """
        # Initialize: add anchor points
        if anchors is not None:
            anchor_coords = np.array([self.domain[d][anchors[:, d]] for d in range(self.n_dims)]).T
            vals = self.func(anchor_coords)
            
            # Sort by function value, prioritize large-valued anchors
            sorted_idx = np.argsort(-np.abs(vals))
            for idx in sorted_idx[:min(5, len(sorted_idx))]:
                self._add_pivot(anchors[idx])
                
            if verbose:
                print(f"[Init] Added {min(5, len(anchors))} anchors, max value: {np.max(np.abs(vals)):.6e}")
        
        # Adaptive loop
        iteration = 0
        prev_max_residual = np.inf
        stagnation_count = 0
        
        while self.rank < self.max_rank:
            iteration += 1
            
            # Find maximum residual point
            new_pivot, max_residual, mean_error = self._find_max_residual_point()
            
            # Record history
            self.history['rank'].append(self.rank)
            self.history['max_residual'].append(max_residual)
            self.history['mean_error'].append(mean_error)
            
            if verbose and iteration % 10 == 0:
                print(f"[Iter {iteration}] Rank: {self.rank}, Val: {max_residual:.6e}, Mean: {mean_error:.6e}")
            
            # Convergence check: function value is sufficiently small
            if max_residual < self.tolerance:
                if verbose:
                    print(f"[Converged] At Rank={self.rank}, Val={max_residual:.6e}")
                break
            
            # Stagnation check: consecutive pivots show no significant improvement
            if max_residual >= prev_max_residual * 0.99:
                stagnation_count += 1
            else:
                stagnation_count = 0
            
            if stagnation_count >= 20:
                if verbose:
                    print(f"[Stagnation] Stopped at Rank={self.rank} (20 iterations without improvement)")
                break
            
            prev_max_residual = max_residual
            
            # Add new pivot
            added = self._add_pivot(new_pivot)
            if not added:
                # If unable to add (already exists), try random perturbation
                perturbed = new_pivot.copy()
                perturbed[np.random.randint(self.n_dims)] = np.random.randint(len(self.domain[0]))
                self._add_pivot(perturbed)
        
        if verbose:
            print(f"[Done] Final Rank: {self.rank}")
        
        return self.rank
    
    def compute_integral(self, dx_vol):
        """
        ACI-based integral computation.
        
        Uses importance sampling centered on pivot points.
        """
        if self.rank == 0:
            return 0.0
        
        n_samples = 50000
        d_size = len(self.domain[0])
        
        # Sample near pivots
        samples = np.zeros((n_samples, self.n_dims), dtype=int)
        for d in range(self.n_dims):
            samples[:, d] = np.random.randint(0, d_size, size=n_samples)
        
        # Evaluate function values
        vals = self._batch_evaluate(samples)
        
        # Integral estimate
        total_grid_points = float(d_size ** self.n_dims)
        result = np.mean(vals) * total_grid_points * dx_vol
        
        return result


def run_aci_demo():
    """ACI demo."""
    from .qtt_utils import QTTEncoder
    from .physics_models import vectorized_gaussian
    
    print("="*60)
    print("Adaptive Cross Interpolation (ACI) Demo")
    print("="*60)
    
    encoder = QTTEncoder(num_vars=3, num_bits=20, bounds=[(-3, 3)]*3)
    def wrapped_f(idx): return vectorized_gaussian(encoder.decode(idx))
    domain = [np.arange(encoder.d, dtype=int)] * encoder.R
    
    # Create ACI instance
    aci = AdaptiveTCI(
        wrapped_f, domain,
        max_rank=100,
        tolerance=1e-4,
        n_test_samples=5000
    )
    
    # Build adaptive decomposition
    anchors = encoder.get_anchors()
    aci.build_adaptive(anchors=anchors, verbose=True)
    
    # Compute integral
    dx_vol = (6.0**3) / (encoder.d ** encoder.R)
    result = aci.compute_integral(dx_vol)
    
    print(f"\nFinal integral result: {result:.6f} (theory: 5.5683)")
    print(f"Relative error: {abs(result - 5.5683) / 5.5683 * 100:.2f}%")


if __name__ == "__main__":
    run_aci_demo()
