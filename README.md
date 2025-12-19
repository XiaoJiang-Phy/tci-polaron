# tci-polaron

**Fast & Scalable Polaron Solver using Tensor Cross Interpolation (TCI)**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Status](https://img.shields.io/badge/status-work--in--progress-orange)

## 🚀 Overview

**tci-polaron** is a computational framework designed to solve the **Electron-Phonon interaction (Polaron)** problem with exponential efficiency. 

Unlike traditional **Diagrammatic Monte Carlo (DiagMC)** methods which suffer from the sign problem and slow convergence in high-order expansion, this project leverages **Tensor Cross Interpolation (TCI)** and **Tensor Train (TT/MPS)** decomposition to compress the high-dimensional integrals found in Feynman diagrams.

## ✨ Key Features (Planned)

- **Tensor Cross Interpolation (TCI):** Replaces stochastic Monte Carlo sampling with deterministic tensor interpolation, significantly reducing noise.
- **Dimensionality Reduction:** Compresses the phonon bath degrees of freedom using Tensor Train (TT) format.
- **High-Performance:** Optimized for calculating Green's functions and self-energy in complex many-body systems.
- **Python/C++ Hybrid:** Core algorithms prototyped in Python with critical tensor contractions accelerated via C++/backend.

## 🛠️ Technology Stack

- **Core Logic:** Python, NumPy, SciPy
- **Tensor Libraries:** (Plan to integrate) TeNPy / ITensor / TensorNetwork
- **Algorithm:** Multidimensional Tensor Interpolation, SVD, DMRG-like sweeps

## 📅 Roadmap

- [ ] **Phase 1 (MVP):** Implement TCI for a simple high-dimensional function (Gaussian/Rational) to validate convergence.
- [ ] **Phase 2:** Apply TCI to calculate the 1st and 2nd order Feynman diagrams for the Holstein model.
- [ ] **Phase 3:** Benchmark speed/accuracy against standard DiagMC results.
- [ ] **Phase 4:** Optimize memory usage for higher-order expansions.

## 📖 Background

The interaction between electrons and lattice vibrations (phonons) is fundamental to understanding superconductivity and charge transport. Standard numerical methods struggle with the "curse of dimensionality." This project aims to demonstrate that **Tensor Networks** offer a superior alternative for these calculations, bridging the gap between Quantum Many-Body Physics and modern Data Compression algorithms.

---
*Author: Xiao Jiang*
*Contact: jiangxiao199412@gmail.com*