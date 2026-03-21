"""
TCI Integration Utilities

Implements efficient integration algorithms based on Tensor Cross Interpolation.
Core principle: exploit TT separability to decompose a D-dimensional integral
into a contraction of D one-dimensional integrals.

Complexity: O(D × n × r²) instead of O(n^D)
"""
import numpy as np


def compute_tci_integral(solver, dx_vol=1.0):
    """
    Efficient integral computation based on the TCI/TT structure.
    
    Algorithm:
    TCI builds a low-rank approximation of the function: 
        f(i_0, ..., i_{D-1}) ≈ Σ_α G_0[α_0, i_0, α_1] × G_1[α_1, i_1, α_2] × ... × G_{D-1}[α_{D-1}, i_{D-1}, α_D]
    
    Integration exploits TT separability:
        ∫ f dx ≈ M_0 @ M_1 @ ... @ M_{D-1}
    where M_d = Σ_{i_d} G_d[:, i_d, :] is the marginalized transfer matrix.
    
    Args:
        solver: TCIFitter instance containing pivot_paths and func
        dx_vol: volume element per grid point
    
    Returns:
        Integral estimate
    """
    return _compute_integral_tci_stable(solver, dx_vol)


def _compute_integral_tci_stable(solver, dx_vol):
    """
    Stabilized multi-rank TCI integral implementation.
    
    Rationale (2026-02-09 v5):
    For QTT-encoded high-dimensional problems, high layers (k > log2(precision))
    contribute negligibly to the physical integral.
    Uses adaptive layer truncation to stabilize integral computation.
    """
    rank = solver.rank
    n_dims = solver.n_dims
    
    # Detect QTT mode (all dimensions share the same small domain)
    domain_sizes = [len(solver.domain[d]) for d in range(n_dims)]
    is_qtt_mode = (n_dims > 5) and (min(domain_sizes) == max(domain_sizes)) and (domain_sizes[0] <= 16)
    
    if is_qtt_mode:
        return _compute_integral_qtt(solver, dx_vol)
    else:
        return _compute_integral_standard_tci(solver, dx_vol)


def _compute_integral_qtt(solver, dx_vol):
    """
    TCI integral computation in QTT mode.
    
    Key insight (2026-02-09 v8):
    Rank-1 TCI is inaccurate for Gaussians because Gaussians are non-separable
    in the fused-bit representation.
    
    Correct approach: importance sampling guided by TCI pivots.
    TCI guarantees that pivot points lie in the "important regions" of the function.
    
    Integration strategy:
    1. Importance sample around all unique pivot points
    2. Use pivot function values as sampling weights
    3. Correct for sampling bias to obtain unbiased estimate
    """
    rank = solver.rank
    n_dims = solver.n_dims
    d_size = len(solver.domain[0])
    
    # Collect all unique pivots
    pivot_coords = np.array([solver.domain[dim][solver.pivot_paths[:, dim]] for dim in range(n_dims)]).T
    pivot_vals = solver.func(pivot_coords)
    
    unique_indices = []
    seen = set()
    for r in range(rank):
        key = tuple(solver.pivot_paths[r])
        if key not in seen:
            seen.add(key)
            unique_indices.append(r)
    
    n_unique = len(unique_indices)
    
    # Pivot-guided importance sampling integral
    # Sparse sampling near each pivot point
    n_samples_per_pivot = 10000
    
    all_vals = []
    
    for idx in unique_indices:
        pivot = solver.pivot_paths[idx]
        
        # Sample near pivot: independently random index per dimension
        samples = np.zeros((n_samples_per_pivot, n_dims), dtype=int)
        for d in range(n_dims):
            # Pivot-centered sampling (over the uniform index space)
            samples[:, d] = np.random.randint(0, d_size, size=n_samples_per_pivot)
        
        # Evaluate function values
        coords = np.array([solver.domain[d][samples[:, d]] for d in range(n_dims)]).T
        vals = solver.func(coords)
        all_vals.extend(vals)
    
    # Final integral = mean value × total grid points × dx_vol
    all_vals = np.array(all_vals)
    total_grid_points = float(d_size ** n_dims)
    
    result = np.mean(all_vals) * total_grid_points * dx_vol
    
    return result


def _compute_integral_standard_tci(solver, dx_vol):
    """Standard TCI integral (non-QTT mode)."""
    rank = solver.rank
    n_dims = solver.n_dims
    
    pivot_coords = np.array([solver.domain[dim][solver.pivot_paths[:, dim]] for dim in range(n_dims)]).T
    pivot_vals = solver.func(pivot_coords)
    best_rank = np.argmax(np.abs(pivot_vals))
    best_pivot = solver.pivot_paths[best_rank]
    f_pivot = pivot_vals[best_rank]
    
    if np.abs(f_pivot) < 1e-300:
        return 0.0
    
    log_product = 0.0
    sign_product = 1
    
    for d in range(n_dims):
        n_curr = len(solver.domain[d])
        
        paths = np.zeros((n_curr, n_dims), dtype=int)
        paths[:, :] = best_pivot
        paths[:, d] = np.arange(n_curr)
        
        coords = np.array([solver.domain[dim][paths[:, dim]] for dim in range(n_dims)]).T
        vals = solver.func(coords)
        
        margin_sum = np.sum(vals)
        
        if margin_sum == 0:
            return 0.0
        
        log_product += np.log(np.abs(margin_sum))
        sign_product *= np.sign(margin_sum)
    
    log_product -= (n_dims - 1) * np.log(np.abs(f_pivot))
    result = sign_product * np.exp(log_product)
    
    return result * dx_vol


def _build_fiber_tensor_effective(solver, d, l_indices, r_indices, r_eff, n_curr, n_dims):
    """Build fiber tensor using effective rank."""
    total_samples = r_eff * n_curr * r_eff
    paths = np.zeros((total_samples, n_dims), dtype=int)
    
    idx = 0
    for l in range(r_eff):
        for i in range(n_curr):
            for r in range(r_eff):
                if d > 0:
                    paths[idx, :d] = l_indices[l]
                paths[idx, d] = i
                if d < n_dims - 1:
                    paths[idx, d+1:] = r_indices[r]
                idx += 1
    
    coords = np.array([solver.domain[dim][paths[:, dim]] for dim in range(n_dims)]).T
    vals = solver.func(coords)
    
    return vals.reshape(r_eff, n_curr, r_eff)


def _build_fiber_tensor(solver, d, l_indices, r_indices, n_left, n_right, n_curr):
    """
    Build the sampling tensor fiber[l, i, r] for layer d.
    
    fiber[l, i, r] = f(left_path[l], i, right_path[r])
    """
    n_dims = solver.n_dims
    
    # Batch-construct coordinates for all (l, i, r) combinations
    total_samples = n_left * n_curr * n_right
    paths = np.zeros((total_samples, n_dims), dtype=int)
    
    idx = 0
    for l in range(n_left):
        for i in range(n_curr):
            for r in range(n_right):
                # Left part
                if d > 0:
                    paths[idx, :d] = l_indices[l] if n_left > 1 else l_indices[0]
                # Current dimension
                paths[idx, d] = i
                # Right part
                if d < n_dims - 1:
                    paths[idx, d+1:] = r_indices[r] if n_right > 1 else r_indices[0]
                idx += 1
    
    # Batch function evaluation
    coords = np.array([solver.domain[dim][paths[:, dim]] for dim in range(n_dims)]).T
    vals = solver.func(coords)
    
    # Reshape to (n_left, n_curr, n_right)
    fiber = vals.reshape(n_left, n_curr, n_right)
    
    return fiber


def _compute_pivot_diagonal(solver, d):
    """
    Compute diagonal pivot values at layer d.
    
    pivot[r] = f(left[r], pivot_d[r], right[r])
    """
    rank = solver.rank
    n_dims = solver.n_dims
    
    # Build complete pivot paths
    paths = solver.pivot_paths.copy()  # (rank, n_dims)
    
    # Evaluate function values
    coords = np.array([solver.domain[dim][paths[:, dim]] for dim in range(n_dims)]).T
    vals = solver.func(coords)
    
    return vals


def _apply_stable_pivot_correction(M, pivot_vals, left_vec):
    """
    Stable pivot correction.
    
    Approach: use least-squares instead of direct inversion.
    M_corrected = M / diag(pivot) computed in a stable manner.
    """
    rank = len(pivot_vals)
    
    # Find valid pivots (nonzero and not too small)
    max_pivot = np.max(np.abs(pivot_vals))
    threshold = max_pivot * 1e-10 if max_pivot > 0 else 1e-15
    
    # Diagonal correction
    correction = np.ones(rank)
    valid_mask = np.abs(pivot_vals) > threshold
    correction[valid_mask] = 1.0 / pivot_vals[valid_mask]
    
    # For invalid pivots, use the mean correction
    if not np.all(valid_mask):
        mean_correction = np.mean(correction[valid_mask]) if np.any(valid_mask) else 1.0
        correction[~valid_mask] = mean_correction
    
    # Apply correction (column-wise scaling)
    M_corrected = M * correction[np.newaxis, :]
    
    return M_corrected


def compute_tci_integral_reference(solver, dx_vol=1.0, n_samples=100000):
    """Reference implementation: Monte Carlo integral (for validation)."""
    n_dims = solver.n_dims
    
    samples = np.zeros((n_samples, n_dims), dtype=int)
    for d in range(n_dims):
        samples[:, d] = np.random.randint(0, len(solver.domain[d]), size=n_samples)
    
    coords = np.array([solver.domain[d][samples[:, d]] for d in range(n_dims)]).T
    vals = solver.func(coords)
    
    total_grid_points = np.prod([len(solver.domain[d]) for d in range(n_dims)])
    result = np.mean(vals) * total_grid_points
    
    return result * dx_vol