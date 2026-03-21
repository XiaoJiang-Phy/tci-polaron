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
- **Direct 4D TCI** for $\Sigma^{(4)}$: No dimensional pre-reduction — TCI discovers the tensor structure directly
- **Imaginary-time τ representation**: $G_0(\tau) = -e^{-\varepsilon\tau} n_F(\varepsilon)$, with **first-moment tail subtraction** for $O(1/N_\tau^2)$ Fourier convergence
- **Brute-force, vectorized, and TCI** implementations for comparison
- Bare electron/phonon Green's functions in both Matsubara and imaginary-time formalism

### Integration Methods

- **CUR-based TT Integration**: Multi-bond CUR decomposition with SVD-regularized inversion and median averaging
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
├── tests/                     # Test suite (16 tests)
│   ├── test_holstein.py      # Holstein self-energy tests (incl. direct 4D TCI)
│   ├── test_fix.py           # QTT regression tests
│   └── test_stable_integral.py
├── scripts/                   # Benchmarking & experiments
│   ├── benchmark.py          # Σ(2) speed/convergence/dispersion benchmarks
│   ├── benchmark_sigma4.py   # Σ(4) direct-4D-TCI benchmarks
│   └── high_rank_test.py     # High-rank TCI experiments
├── docs/                      # Documentation
│   └── DECISION_LOG.md       # Physics decision log & technical issues
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

**Expected Output (selected):**
```
--- Mode 1: Standard Grid TCI ---
Standard grid integral: 5.568059 (theory: 5.56832, error: 0.00%)

--- Mode 4: Holstein Polaron 2nd-Order Self-Energy ---
Sigma(2) TCI (rank=5): -0.00000000-0.01662247j  Relative error: 0.00%

--- Mode 5: Sigma(4) Direct 4D TCI (no dim reduction) ---
Sigma(4) TCI r=20: -0.00092653  Relative error: 0.00%

--- Mode 6: Sigma(4) Imaginary-Time tau Representation ---
Sigma(4) tau-BF (exact):  -0.00157287  error 0.0000%
tau-TCI (tail subtraction) N_tau=128: error 0.0367%
```

### Run Tests

```bash
python tests/test_holstein.py
```

### Run Benchmarks

```bash
python scripts/benchmark.py        # Σ(2) benchmarks
python scripts/benchmark_sigma4.py  # Σ(4) direct 4D TCI benchmarks
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

### Direct 4D TCI for Σ(4) (N_k=16, N_ν=32)

| Method | Time | Error |
|--------|------|-------|
| Vectorized brute force | 0.61s | — (reference) |
| Direct 4D TCI, rank=5 | 0.16s | 32.6% |
| Direct 4D TCI, rank=10 | 0.25s | 15.4% |
| Direct 4D TCI, rank=20 | 0.61s | **0.00%** |

### Imaginary-Time τ Representation ($N_k=8, N_\nu=16$)

| Method | $N_\tau$ | Error vs Matsubara |
|--------|----------|--------------------|
| τ brute-force (Matsubara $h$) | — | **0.00%** (exact) |
| τ-TCI (no tail subtraction) | 128 | 5.80% |
| τ-TCI (tail subtraction) | 32 | 0.008% |
| τ-TCI (tail subtraction) | 128 | **0.037%** |

> **Key insight:** First-moment tail subtraction $G_0^{\text{reg}}(\tau) = G_0(\tau) + 1/2$ restores true anti-periodicity, improving Fourier convergence from $O(1/N_\tau)$ to $O(1/N_\tau^2)$.

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
- [x] **Phase 4:** Direct 4D TCI on full integrand — CUR-based TT integration ✅

### Phase 5: Precision & Representation Optimization (Near-term)

- [x] **5a: Imaginary-time τ representation** — $G_0(\tau)$ propagator + first-moment tail subtraction + analytic $D_0^{\text{FT}}(i\omega')$, $O(1/N_\tau^2)$ convergence ✅
- [ ] **5b: Coordinate rotation to eliminate diagonal coupling** — $(q_1, q_2) \to (Q, q_-)$ makes momentum conservation $q_1+q_2=Q$ a direct-product structure, improving CUR convergence
- [ ] **5c: 6th-order self-energy Σ(6)** — 6D integral, truly demonstrating TCI complexity advantage over brute force: $O(r \cdot N^3)$ vs $O(N^6)$
- [ ] **5d: Comparison with TCI.jl** — Julia's TensorCrossInterpolation.jl provides explicit TT-cores, enabling standard TT integration to verify CUR approximation bottlenecks

### Phase 6: Physics Extensions (Mid-term)

- [ ] **6a: Dispersive phonons** — $D_0(q, \nu)$ with momentum-dependent $\omega_q = \omega_0 \sqrt{1 + \alpha \cos q}$, introducing non-trivial dispersion renormalization to $\Sigma(k)$
- [ ] **6b: DiagMC comparison** — Implement Diagrammatic Monte Carlo for direct accuracy/speed/scaling comparison with TCI
- [ ] **6c: 2D Holstein** — 2D model, momentum integral 1D → 2D, self-energy dimensions 4D → 6D


## 📖 Background

The interaction between electrons and lattice vibrations (phonons) is fundamental to understanding superconductivity and charge transport. The **Holstein model** describes an electron coupled to dispersionless optical phonons:

$$H = -t \sum_{\langle i,j \rangle} c_i^\dagger c_j + \omega_0 \sum_i a_i^\dagger a_i + g \sum_i c_i^\dagger c_i (a_i^\dagger + a_i)$$

Standard numerical methods struggle with the "curse of dimensionality" at higher perturbation orders. This project explores **Tensor Cross Interpolation** as an alternative for compressing Feynman diagram integrands.

## 📚 References

- Oseledets, I. (2011). Tensor-Train Decomposition. *SIAM J. Sci. Comput.*
- Savostyanov, D. & Oseledets, I. (2011). Fast Adaptive Interpolation of Multi-dimensional Arrays. *LNCS*
- Ritter, M. et al. (2024). Quantics Tensor Cross Interpolation. *arXiv:2303.11819*
- Shinaoka, H. et al. (2023). Multiscale Space-Time Ansatz for Correlation Functions of Quantum Systems. *Phys. Rev. X*
- Boehnke, L. et al. (2011). Orthogonal polynomial representation of imaginary-time Green's functions. *Phys. Rev. B* 84, 075145

---

*Author: Xiao Jiang*  
*Contact: jiangxiao199412@gmail.com*