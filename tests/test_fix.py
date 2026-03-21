
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.tci_core import TCIFitter
from src.qtt_utils import QTTEncoder
from src.tci_utils import compute_tci_integral
from src.physics_models import vectorized_gaussian

def test_fixed_qtt():
    # Replicate main.py setup
    encoder = QTTEncoder(num_vars=3, num_bits=20, bounds=[(-3, 3)]*3)
    def wrapped_f(idx): return vectorized_gaussian(encoder.decode(idx))
    domain = [np.arange(encoder.d, dtype=int)] * encoder.R
    
    # Manually set better anchors: [7, 0, 0, ...] maps to physical coords (0,0,0)
    best_anchor = np.zeros(encoder.R, dtype=int)
    best_anchor[0] = 7 # Set MSB of all variables at layer 0 to 1 (i.e., 0.5)
    
    # Anchor 1: all zeros, Anchor 2: center
    anchors = np.array([np.zeros(encoder.R, dtype=int), best_anchor])
    
    print("--- Running test (with corrected anchors) ---")
    solver = TCIFitter(wrapped_f, domain, rank=10)
    solver.build_cores(anchors=anchors)
    
    dx_vol = (6.0**3) / (encoder.d ** encoder.R)
    res = compute_tci_integral(solver, dx_vol=dx_vol)
    print(f"Corrected QTT integral: {res:.6f} (expected ~5.568)")

if __name__ == "__main__":
    test_fixed_qtt()
