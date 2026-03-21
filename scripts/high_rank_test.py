"""
High-Rank TCI with Monte Carlo Integration

Strategy: Use high-rank TCI to identify important regions,
then use the Pivot information to guide importance sampling.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.tci_core import TCIFitter
from src.qtt_utils import QTTEncoder
from src.physics_models import vectorized_gaussian


def run_high_rank_demo():
    """High-rank TCI demo."""
    print("="*60)
    print("High-Rank TCI Integration Demo")
    print("="*60)
    
    encoder = QTTEncoder(num_vars=3, num_bits=20, bounds=[(-3, 3)]*3)
    def wrapped_f(idx): return vectorized_gaussian(encoder.decode(idx))
    domain = [np.arange(encoder.d, dtype=int)] * encoder.R
    
    # Test different ranks
    ranks_to_test = [10, 30, 50, 80, 100]
    
    dx_vol = (6.0**3) / (encoder.d ** encoder.R)
    theoretical = 5.5683
    
    print(f"\nTheoretical integral value: {theoretical:.4f}")
    print("-"*50)
    
    for rank in ranks_to_test:
        # Create high-rank TCI
        solver = TCIFitter(wrapped_f, domain, rank=rank)
        anchors = encoder.get_anchors()
        solver.build_cores(anchors=anchors, n_sweeps=5)
        
        # Pivot-guided Monte Carlo integration
        result = compute_pivot_guided_integral(solver, dx_vol, n_samples=100000)
        
        error = abs(result - theoretical) / theoretical * 100
        print(f"Rank={rank:3d}: integral={result:.4f}, error={error:.2f}%")
    
    print("\n" + "="*60)
    print("Analysis: effect of rank on integral accuracy")
    print("="*60)


def compute_pivot_guided_integral(solver, dx_vol, n_samples=100000):
    """
    Pivot-guided Monte Carlo integration.
    
    Uses TCI pivots as reference points for importance sampling.
    """
    rank = solver.rank
    n_dims = solver.n_dims
    d_size = len(solver.domain[0])
    total_grid_points = float(d_size ** n_dims)
    
    # Collect unique pivots
    unique_pivots = []
    seen = set()
    for r in range(rank):
        key = tuple(solver.pivot_paths[r])
        if key not in seen:
            seen.add(key)
            unique_pivots.append(solver.pivot_paths[r])
    
    n_unique = len(unique_pivots)
    
    # Sample near each pivot
    samples_per_pivot = n_samples // n_unique
    
    all_vals = []
    
    for pivot in unique_pivots:
        # Uniform random sampling (entire index space)
        samples = np.zeros((samples_per_pivot, n_dims), dtype=int)
        for d in range(n_dims):
            samples[:, d] = np.random.randint(0, d_size, size=samples_per_pivot)
        
        # Evaluate function values
        coords = np.array([solver.domain[d][samples[:, d]] for d in range(n_dims)]).T
        vals = solver.func(coords)
        all_vals.extend(vals)
    
    # Integral estimate
    all_vals = np.array(all_vals)
    result = np.mean(all_vals) * total_grid_points * dx_vol
    
    return result


def analyze_pivot_distribution(solver, encoder):
    """Analyze pivot point distribution in physical space."""
    n_dims = solver.n_dims
    
    print("\nPivot distribution analysis in physical space:")
    print("-"*40)
    
    # Decode all unique pivots to physical space
    unique_pivots = []
    seen = set()
    for r in range(solver.rank):
        key = tuple(solver.pivot_paths[r])
        if key not in seen:
            seen.add(key)
            unique_pivots.append(solver.pivot_paths[r])
    
    for i, pivot in enumerate(unique_pivots[:10]):  # show first 10 only
        physical = encoder.decode(pivot.reshape(1, -1))[0]
        f_val = solver.func(pivot.reshape(1, -1))[0]
        print(f"  Pivot {i}: phys_coord~({physical[0]:.2f}, {physical[1]:.2f}, {physical[2]:.2f}), f={f_val:.4f}")
    
    if len(unique_pivots) > 10:
        print(f"  ... {len(unique_pivots) - 10} more pivots")


if __name__ == "__main__":
    run_high_rank_demo()
