"""
Stable QTT integration implementation.
Uses simplified sampling-based validation to avoid ill-conditioned matrix inversion.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.qtt_utils import QTTEncoder
from src.physics_models import vectorized_gaussian

def compute_integral_monte_carlo(encoder, func, n_samples=100000):
    """
    Compute integral via Monte Carlo as a reference value.
    """
    # Uniformly sample in QTT index space
    samples_idx = np.random.randint(0, encoder.d, size=(n_samples, encoder.R))
    
    # Decode to physical coordinates
    coords = encoder.decode(samples_idx)
    
    # Evaluate function values
    vals = func(coords)
    
    # Integral = mean value × volume
    volume = np.prod([b[1] - b[0] for b in encoder.bounds])
    integral = np.mean(vals) * volume
    
    return integral

def compute_integral_direct_sum(encoder, func, max_samples_per_dim=4):
    """
    Direct summation: select representative sample points at each QTT layer.
    Avoids full traversal of d^R points.
    """
    # Sparse grid sampling
    n_dims = encoder.R
    
    # Sample max_samples_per_dim points per layer
    sample_indices = []
    for _ in range(n_dims):
        sample_indices.append(np.linspace(0, encoder.d - 1, max_samples_per_dim, dtype=int))
    
    # Build sparse grid
    grid = np.meshgrid(*sample_indices, indexing='ij')
    flat_grid = np.vstack([g.ravel() for g in grid]).T  # (N, n_dims)
    
    # Limit number of samples
    if len(flat_grid) > 100000:
        indices = np.random.choice(len(flat_grid), 100000, replace=False)
        flat_grid = flat_grid[indices]
    
    # Decode and evaluate
    coords = encoder.decode(flat_grid)
    vals = func(coords)
    
    # Integral estimate
    volume = np.prod([b[1] - b[0] for b in encoder.bounds])
    # Weight = volume / n_samples × (total grid points / sampled grid points)
    weight = volume * (encoder.d ** encoder.R) / len(flat_grid)
    
    integral = np.sum(vals) * volume / len(flat_grid)
    
    return integral

def test_qtt_integration():
    print("="*60)
    print("QTT Integration Stability Test")
    print("="*60)
    
    encoder = QTTEncoder(num_vars=3, num_bits=20, bounds=[(-3, 3)]*3)
    print(f"QTT config: {encoder.n_vars} vars x {encoder.R} layers")
    print(f"Theoretical integral: 5.5683 (3D Gaussian)")
    
    # Method 1: Monte Carlo
    print("\n--- Monte Carlo Sampling ---")
    for n in [10000, 100000, 1000000]:
        np.random.seed(42)
        result = compute_integral_monte_carlo(encoder, vectorized_gaussian, n_samples=n)
        print(f"  N={n:>7}: {result:.6f}")
    
    # Method 2: Low-resolution QTT check
    print("\n--- Low-Resolution QTT Test ---")
    for bits in [4, 6, 8, 10]:
        enc_low = QTTEncoder(num_vars=3, num_bits=bits, bounds=[(-3, 3)]*3)
        np.random.seed(42)
        result = compute_integral_monte_carlo(enc_low, vectorized_gaussian, n_samples=100000)
        print(f"  {bits} bits ({2**(bits*3):>10} pts): {result:.6f}")

if __name__ == "__main__":
    test_qtt_integration()
