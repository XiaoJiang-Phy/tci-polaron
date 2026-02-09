# tci-polaron

**Fast & Scalable Polaron Solver using Tensor Cross Interpolation (TCI)**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Status](https://img.shields.io/badge/status-work--in--progress-orange)

## 🚀 Overview

**tci-polaron** is a computational framework designed to solve the **Electron-Phonon interaction (Polaron)** problem with exponential efficiency. 

Unlike traditional **Diagrammatic Monte Carlo (DiagMC)** methods which suffer from the sign problem and slow convergence in high-order expansion, this project leverages **Tensor Cross Interpolation (TCI)** and **Tensor Train (TT/MPS)** decomposition to compress the high-dimensional integrals found in Feynman diagrams.

## ✨ Features

### Core Algorithms

| Module | Description |
|--------|-------------|
| `TCIFitter` | Tensor Cross Interpolation with bidirectional sweeping and MaxVol pivot selection |
| `QTTEncoder` | Quantized Tensor Train encoder for ultra-high resolution grids ($2^{60}$ virtual points) |
| `AdaptiveTCI` | Adaptive Cross Interpolation with automatic rank determination |
| `TTCoreTCI` | Explicit TT-core construction experiments |

### Integration Methods

- **Standard TCI Integral**: Rank-1 separable approximation for low-dimensional problems
- **QTT Guided Sampling**: Monte Carlo integration guided by TCI pivots for high-dimensional QTT
- **Reference MC**: Pure Monte Carlo for validation

### Key Capabilities

- ✅ **Tensor Cross Interpolation (TCI)**: Deterministic tensor interpolation replacing stochastic sampling
- ✅ **Quantized Tensor Train (QTT)**: Represents $2^{60}$ grid points with $O(D \log n)$ storage
- ✅ **Bidirectional Sweeping**: DMRG-like optimization for stable pivot selection
- ✅ **Adaptive Rank Control**: Automatic rank determination based on error convergence
- ✅ **Numerical Stability**: QR-based MaxVol, stable integral computation

## 📁 Project Structure

```
tci-polaron/
├── main.py                    # Main entry point
├── src/                       # Core library
│   ├── __init__.py           # Package exports
│   ├── tci_core.py           # TCIFitter - core TCI algorithm
│   ├── tci_utils.py          # Integration utilities
│   ├── qtt_utils.py          # QTTEncoder for quantized TT
│   ├── physics_models.py     # Physical models (Gaussian, etc.)
│   ├── aci_core.py           # Adaptive Cross Interpolation
│   └── tt_core_tci.py        # TT-Core construction experiments
├── tests/                     # Test suite
│   ├── test_fix.py
│   └── test_stable_integral.py
├── scripts/                   # Diagnostic & experimental scripts
│   ├── high_rank_test.py     # High-rank TCI experiments
│   ├── debug_integral.py
│   └── diagnose_qtt.py
├── docs/                      # Documentation
│   └── DECISION_LOG.md       # Physics decision log
└── .antigravity/             # Development tools
    └── scripts/
        └── physics_audit.py  # Automated physics validation
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/XiaoJiang-Phy/tci-polaron.git
cd tci-polaron

# Create conda environment
conda env create -f environment.yml
conda activate tci_env
```

### Run the Demo

```bash
python main.py
```

**Expected Output:**
```
--- 模式 1: 普通网格 TCI ---
普通网格积分: 5.568059 (理论: 5.56832, 误差: 0.00%)

--- 模式 2: QTT 融合比特 TCI (rank=10) ---
QTT 积分: 5.643566 (理论: 5.56832, 误差: 1.35%)

--- 模式 3: 高秩 QTT TCI (rank=50) ---
高秩 QTT 积分: 5.694345 (理论: 5.56832, 误差: 2.26%)
```

### Using the Library

```python
import numpy as np
from src import TCIFitter, QTTEncoder, compute_tci_integral, vectorized_gaussian

# Mode 1: Standard TCI on regular grid
grid = [np.linspace(-3, 3, 64) for _ in range(3)]
solver = TCIFitter(vectorized_gaussian, grid, rank=1)
solver.build_cores()
result = compute_tci_integral(solver, dx_vol=(6/64)**3)
print(f"Standard TCI: {result:.6f}")

# Mode 2: QTT for ultra-high resolution
encoder = QTTEncoder(num_vars=3, num_bits=20, bounds=[(-3, 3)]*3)
def qtt_func(idx): return vectorized_gaussian(encoder.decode(idx))
domain = [np.arange(encoder.d, dtype=int)] * encoder.R

solver = TCIFitter(qtt_func, domain, rank=10)
solver.build_cores(anchors=encoder.get_anchors())
dx_vol = (6.0**3) / (encoder.d ** encoder.R)
result = compute_tci_integral(solver, dx_vol=dx_vol)
print(f"QTT TCI (2^60 resolution): {result:.6f}")
```

## 📊 Performance Comparison

| Method | Grid Size | Error | Notes |
|--------|-----------|-------|-------|
| Standard TCI (Rank=1) | 64³ | <0.01% | Perfect for separable functions |
| QTT TCI (Rank=10) | 2⁶⁰ | ~1-2% | MC sampling, ultra-high resolution |
| QTT TCI (Rank=50) | 2⁶⁰ | ~1-3% | Higher rank, same sampling limitation |

### When to Use QTT vs Standard TCI

| Scenario | Recommended |
|----------|-------------|
| Low dimension (D ≤ 5) | Standard TCI |
| Separable functions | Standard TCI (Rank=1 sufficient) |
| High dimension (D > 10) | QTT |
| Ultra-high resolution needed | QTT |
| Multi-scale functions | QTT |

> ⚠️ **Note**: The Gaussian integral example is not ideal for demonstrating QTT advantages because:
> 1. It's a fully separable function: $e^{-x^2} \cdot e^{-y^2} \cdot e^{-z^2}$
> 2. It lacks multi-scale structure
> 3. Low dimension (D=3) doesn't benefit from QTT compression
>
> QTT excels in problems with **scale separation** and **high dimensionality**. See `docs/DECISION_LOG.md` for detailed analysis.

## 🛠️ Technology Stack

- **Core Logic:** Python, NumPy, SciPy
- **Linear Algebra:** QR decomposition, SVD, pseudo-inverse
- **Algorithms:** TCI, QTT, MaxVol, DMRG-like sweeps

## 📅 Roadmap

- [x] **Phase 1 (MVP):** Implement TCI for high-dimensional Gaussian function ✅
- [x] **Phase 1.5:** Fix QTT numerical stability issues ✅
- [x] **Phase 1.6:** Implement Adaptive Cross Interpolation (ACI) ✅
- [ ] **Phase 2:** Apply TCI to 1st/2nd order Feynman diagrams (Holstein model)
- [ ] **Phase 3:** Benchmark against DiagMC
- [ ] **Phase 4:** Optimize for higher-order expansions

## 📖 Background

The interaction between electrons and lattice vibrations (phonons) is fundamental to understanding superconductivity and charge transport. Standard numerical methods struggle with the "curse of dimensionality." This project aims to demonstrate that **Tensor Networks** offer a superior alternative for these calculations, bridging the gap between Quantum Many-Body Physics and modern Data Compression algorithms.

## 📚 References

- Oseledets, I. (2011). Tensor-Train Decomposition. *SIAM J. Sci. Comput.*
- Savostyanov, D. & Oseledets, I. (2011). Fast Adaptive Interpolation of Multi-dimensional Arrays. *LNCS*
- Ritter, M. et al. (2024). Quantics Tensor Cross Interpolation. *arXiv:2303.11819*

---

*Author: Xiao Jiang*  
*Contact: jiangxiao199412@gmail.com*