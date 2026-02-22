# tci-polaron

**Fast & Scalable Polaron Solver using Tensor Cross Interpolation (TCI)**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Status](https://img.shields.io/badge/status-active--development-green)

## 🚀 Overview

**tci-polaron** is a computational framework for solving the **Electron-Phonon interaction (Polaron)** problem using tensor network methods.

Unlike traditional **Diagrammatic Monte Carlo (DiagMC)** methods which suffer from the sign problem and slow convergence in high-order expansion, this project leverages **Tensor Cross Interpolation (TCI)** and **Tensor Train (TT/MPS)** decomposition to compress the high-dimensional integrals found in Feynman diagrams.

**Current focus**: 1D Holstein polaron self-energy via perturbative Feynman diagram expansion.

## ✨ Features

### Core Algorithms

| Module | Description |
|--------|-------------|
| `TCIFitter` | Tensor Cross Interpolation with bidirectional sweeping and MaxVol pivot selection |
| `QTTEncoder` | Quantized Tensor Train encoder for ultra-high resolution grids ($2^{60}$ virtual points) |
| `AdaptiveTCI` | Adaptive Cross Interpolation with automatic rank determination |
| `HolsteinParams` | 1D Holstein model parameter container |

### Holstein Polaron Self-Energy

- **2nd-order self-energy** $\Sigma^{(2)}$: Single phonon exchange (rainbow diagram)
- **4th-order self-energy** $\Sigma^{(4)}$: Two-phonon exchange with full 4D summation
- **Brute-force, vectorized, and TCI** implementations for comparison
- Bare electron/phonon Green's functions in Matsubara formalism

### Integration Methods

- **Standard TCI Integral**: Rank-1 separable approximation for low-dimensional problems
- **Vectorized Matsubara Summation**: Efficient frequency sum with NumPy broadcasting
- **QTT Guided Sampling**: Monte Carlo integration guided by TCI pivots

## 📁 Project Structure

```
tci-polaron/
├── main.py                    # Main entry point (Gaussian + Holstein demos)
├── src/                       # Core library
│   ├── __init__.py           # Package exports
│   ├── tci_core.py           # TCIFitter - core TCI algorithm
│   ├── tci_utils.py          # Integration utilities
│   ├── qtt_utils.py          # QTTEncoder for quantized TT
│   ├── physics_models.py     # Propagators: G₀, D₀, ε(k), Matsubara freqs
│   ├── holstein.py           # Holstein polaron Σ(2) and Σ(4)
│   ├── aci_core.py           # Adaptive Cross Interpolation
│   └── tt_core_tci.py        # TT-Core construction experiments
├── tests/                     # Test suite (10 tests)
│   ├── test_holstein.py      # Holstein self-energy tests
│   ├── test_fix.py           # QTT regression tests
│   └── test_stable_integral.py
├── scripts/                   # Benchmarking & experiments
│   ├── benchmark.py          # Σ(2) speed/convergence/dispersion benchmarks
│   └── high_rank_test.py     # High-rank TCI experiments
└── environment.yml            # Conda environment
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/XiaoJiang-Phy/tci-polaron.git
cd tci-polaron

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

--- 模式 4: Holstein Polaron 2阶自能 ---
参数: t=1.0, ω₀=0.5, g=0.3, β=10.0
网格: N_k=64, N_ν=128

计算暴力求和...
Σ(2) 暴力求和: -0.00000000-0.01662247j
  Re[Σ] = -0.00000000, Im[Σ] = -0.01662247

计算 TCI 加速...
Σ(2) TCI (rank=5): -0.00000000-0.01662247j

相对误差: 0.00%
```

### Run Tests

```bash
python tests/test_holstein.py
```

### Run Benchmarks

```bash
python scripts/benchmark.py
```

### Using the Library

```python
import numpy as np
from src import HolsteinParams, compute_sigma2_brute_force, compute_sigma2_tci

# Define Holstein model parameters
params = HolsteinParams(t=1.0, omega0=0.5, g=0.3, beta=10.0, N_k=64, N_nu=128)

# Compute 2nd-order self-energy
sigma_bf = compute_sigma2_brute_force(params, k_ext=0.0, n_ext=0)
sigma_tci = compute_sigma2_tci(params, k_ext=0.0, n_ext=0, rank=5)

print(f"Σ(2) brute force: {sigma_bf:.8f}")
print(f"Σ(2) TCI:         {sigma_tci:.8f}")
```

## 📊 Results

### Holstein Polaron Self-Energy ($t=1, \omega_0=0.5, g=0.3, \beta=10$)

| Order | Value | Scaling | $|\Sigma|/g^n$ |
|-------|-------|---------|----------------|
| $\Sigma^{(2)}$ | $-0.01662j$ | $\propto g^2$ | 0.184694 |
| $\Sigma^{(4)}$ | $-0.000926$ | $\propto g^4$ | 0.114387 |
| Ratio | $|\Sigma^{(4)}|/|\Sigma^{(2)}| = 0.050$ | Perturbative ✅ | |

### Speed Comparison (Σ(4), N_k=16, N_ν=32)

| Method | Time | Speedup |
|--------|------|---------|
| Brute force (4 nested loops) | 3.70s | 1× |
| Vectorized (Matsubara sum first) | 0.27s | **13.9×** |

### Σ(2) Matsubara Convergence

| $N_\nu$ | $\text{Im}[\Sigma^{(2)}]$ | $\Delta\Sigma$ |
|---------|---------------------------|----------------|
| 16 | -0.01663398 | — |
| 64 | -0.01662263 | 1.3×10⁻⁶ |
| 128 | -0.01662247 | 1.6×10⁻⁷ |
| 512 | -0.01662245 | 2.5×10⁻⁹ |

## 🛠️ Technology Stack

- **Core Logic:** Python, NumPy, SciPy
- **Linear Algebra:** QR decomposition, SVD, pseudo-inverse
- **Algorithms:** TCI, QTT, MaxVol, DMRG-like sweeps, Matsubara formalism

## 📅 Roadmap

- [x] **Phase 1:** TCI for high-dimensional Gaussian integral ✅
- [x] **Phase 1.5:** QTT numerical stability fixes ✅
- [x] **Phase 1.6:** Adaptive Cross Interpolation (ACI) ✅
- [x] **Phase 2:** Holstein polaron 2nd-order self-energy (Feynman diagrams) ✅
- [x] **Phase 3:** 4th-order self-energy, benchmarking ✅
- [ ] **Phase 4:** Direct 4D TCI on Feynman diagram integrands (TT-core construction)
- [ ] **Phase 5:** Higher-order expansions & comparison with DiagMC

## 📖 Background

The interaction between electrons and lattice vibrations (phonons) is fundamental to understanding superconductivity and charge transport. The **Holstein model** describes an electron coupled to dispersionless optical phonons:

$$H = -t \sum_{\langle i,j \rangle} c_i^\dagger c_j + \omega_0 \sum_i a_i^\dagger a_i + g \sum_i c_i^\dagger c_i (a_i^\dagger + a_i)$$

Standard numerical methods struggle with the "curse of dimensionality" at higher perturbation orders. This project explores **Tensor Cross Interpolation** as an alternative for compressing Feynman diagram integrands.

## 📚 References

- Oseledets, I. (2011). Tensor-Train Decomposition. *SIAM J. Sci. Comput.*
- Savostyanov, D. & Oseledets, I. (2011). Fast Adaptive Interpolation of Multi-dimensional Arrays. *LNCS*
- Ritter, M. et al. (2024). Quantics Tensor Cross Interpolation. *arXiv:2303.11819*
- Shinaoka, H. et al. (2023). Multiscale Space-Time Ansatz for Correlation Functions of Quantum Systems. *Phys. Rev. X*

---

*Author: Xiao Jiang*  
*Contact: jiangxiao199412@gmail.com*