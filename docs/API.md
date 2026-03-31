# tci-polaron API Reference

> **Scope**: This document covers all public symbols exported by the `src` package.
> See [`__init__.py`](../src/__init__.py) for the full `__all__` list.

---

## Table of Contents

- [Data Classes](#data-classes)
  - [`HolsteinParams`](#holsteinparams)
- [Physics Propagators (`physics_models`)](#physics-propagators)
  - [`epsilon_k`](#epsilon_k)
  - [`bare_electron_gf`](#bare_electron_gf)
  - [`bare_phonon_gf`](#bare_phonon_gf)
  - [`bare_electron_gf_tau`](#bare_electron_gf_tau)
  - [`bare_phonon_gf_tau`](#bare_phonon_gf_tau)
  - [`matsubara_freq_fermion`](#matsubara_freq_fermion)
  - [`matsubara_freq_boson`](#matsubara_freq_boson)
  - [`vectorized_gaussian`](#vectorized_gaussian)
- [2nd-Order Self-Energy (`holstein`)](#2nd-order-self-energy)
  - [`compute_sigma2_brute_force`](#compute_sigma2_brute_force)
  - [`compute_sigma2_tci`](#compute_sigma2_tci)
  - [`compute_sigma2_tau`](#compute_sigma2_tau)
  - [`sigma_tau_to_matsubara`](#sigma_tau_to_matsubara)
- [4th-Order Self-Energy (`holstein`)](#4th-order-self-energy)
  - [`compute_sigma4_brute_force`](#compute_sigma4_brute_force)
  - [`compute_sigma4_vectorized`](#compute_sigma4_vectorized)
  - [`compute_sigma4_tci`](#compute_sigma4_tci)
  - [`compute_sigma4_direct_tci`](#compute_sigma4_direct_tci)
  - [`compute_sigma4_tau_brute_force`](#compute_sigma4_tau_brute_force)
  - [`compute_sigma4_tau_tci`](#compute_sigma4_tau_tci)
- [TCI Engine (`tci_core`)](#tci-engine)
  - [`TCIFitter`](#tcifitter)
- [TCI Integration Utilities (`tci_utils`)](#tci-integration-utilities)
  - [`compute_tci_integral`](#compute_tci_integral)
  - [`compute_tci_integral_reference`](#compute_tci_integral_reference)
- [QTT Encoder (`qtt_utils`)](#qtt-encoder)
  - [`QTTEncoder`](#qttencoder)
- [Adaptive TCI (`aci_core`)](#adaptive-tci)
  - [`AdaptiveTCI`](#adaptivetci)
- [TT-Core TCI (`tt_core_tci`)](#tt-core-tci)
  - [`TTCoreTCI`](#ttcoretci)

---

## Data Classes

### `HolsteinParams`

```python
from src import HolsteinParams
```

Parameter container for the 1D Holstein polaron model.

| Field    | Type    | Default | Description |
|----------|---------|---------|-------------|
| `t`      | `float` | `1.0`   | Hopping amplitude |
| `omega0` | `float` | `0.5`   | Einstein phonon frequency $\omega_0$ |
| `g`      | `float` | `0.3`   | Electron-phonon coupling constant |
| `beta`   | `float` | `10.0`  | Inverse temperature $\beta = 1/T$ |
| `N_k`    | `int`   | `64`    | Momentum grid size ($q \in [0, 2\pi)$, uniform) |
| `N_nu`   | `int`   | `128`   | Bosonic Matsubara count (sum runs $m = -N_\nu \ldots N_\nu - 1$) |

**Example:**
```python
params = HolsteinParams(t=1.0, omega0=0.5, g=0.3, beta=10.0, N_k=16, N_nu=32)
```

---

## Physics Propagators

Source: [`src/physics_models.py`](../src/physics_models.py)

All functions accept both scalar and NumPy array arguments (broadcasting).

---

### `epsilon_k`

```python
epsilon_k(k, t=1.0) -> float | ndarray
```

1D tight-binding dispersion relation.

$$\varepsilon(k) = -2t \cos(k)$$

| Param | Type | Description |
|-------|------|-------------|
| `k`   | `float \| ndarray` | Crystal momentum |
| `t`   | `float` | Hopping amplitude |

---

### `bare_electron_gf`

```python
bare_electron_gf(k, iwn, t=1.0) -> complex | ndarray
```

Non-interacting electron Matsubara Green's function.

$$G_0(k, i\omega_n) = \frac{1}{i\omega_n - \varepsilon_k}$$

| Param | Type | Description |
|-------|------|-------------|
| `k`   | `float \| ndarray` | Crystal momentum |
| `iwn` | `float \| ndarray` | Fermionic Matsubara frequency $\omega_n$ (real-valued, **not** $i\omega_n$) |
| `t`   | `float` | Hopping amplitude |

> **Convention**: The argument `iwn` is the *real-valued* frequency $\omega_n$; the factor $i$ is applied internally.

---

### `bare_phonon_gf`

```python
bare_phonon_gf(inu_m, omega0) -> float | ndarray
```

Non-interacting phonon Matsubara Green's function (dispersionless Holstein / Einstein phonons).

$$D_0(i\nu_m) = \frac{-2\omega_0}{\nu_m^2 + \omega_0^2}$$

| Param    | Type | Description |
|----------|------|-------------|
| `inu_m`  | `float \| ndarray` | Bosonic Matsubara frequency $\nu_m$ (real-valued) |
| `omega0` | `float` | Phonon frequency |

---

### `bare_electron_gf_tau`

```python
bare_electron_gf_tau(k, tau, beta, t=1.0) -> float | ndarray
```

Electron Green's function in imaginary time.

$$G_0(k, \tau) = -\frac{e^{-\varepsilon_k \tau}}{1 + e^{-\beta \varepsilon_k}}, \quad \tau \in [0, \beta)$$

| Param  | Type | Description |
|--------|------|-------------|
| `k`    | `float \| ndarray` | Crystal momentum |
| `tau`  | `float \| ndarray` | Imaginary time $\tau \in [0, \beta)$ |
| `beta` | `float` | Inverse temperature |
| `t`    | `float` | Hopping amplitude |

---

### `bare_phonon_gf_tau`

```python
bare_phonon_gf_tau(tau, beta, omega0) -> float | ndarray
```

Phonon Green's function in imaginary time.

$$D_0(\tau) = -\frac{\cosh[\omega_0(\beta/2 - \tau)]}{\sinh(\omega_0\beta/2)}, \quad \tau \in [0, \beta)$$

| Param    | Type | Description |
|----------|------|-------------|
| `tau`    | `float \| ndarray` | Imaginary time |
| `beta`   | `float` | Inverse temperature |
| `omega0` | `float` | Phonon frequency |

---

### `matsubara_freq_fermion`

```python
matsubara_freq_fermion(n, beta) -> float | ndarray
```

$$\omega_n = \frac{(2n+1)\pi}{\beta}$$

---

### `matsubara_freq_boson`

```python
matsubara_freq_boson(m, beta) -> float | ndarray
```

$$\nu_m = \frac{2m\pi}{\beta}$$

---

### `vectorized_gaussian`

```python
vectorized_gaussian(coords) -> ndarray
```

$N$-dimensional Gaussian test function $f(\mathbf{x}) = e^{-\|\mathbf{x}\|^2}$.

| Param    | Type | Description |
|----------|------|-------------|
| `coords` | `ndarray (M, N)` | Batch of $N$-dimensional coordinates |

**Returns**: `ndarray (M,)` — function values.

---

## 2nd-Order Self-Energy

Source: [`src/holstein.py`](../src/holstein.py)

Diagram: *rainbow* (single phonon exchange).

$$\Sigma^{(2)}(k, i\omega_n) = -\frac{g^2}{N_k \beta} \sum_{q,m} G_0(k{-}q,\, i\omega_n{-}i\nu_m)\, D_0(i\nu_m)$$

---

### `compute_sigma2_brute_force`

```python
compute_sigma2_brute_force(params, k_ext, n_ext) -> complex
```

Direct nested-loop evaluation: $O(N_k \times 2N_\nu)$.

| Param   | Type | Description |
|---------|------|-------------|
| `params` | `HolsteinParams` | Model parameters |
| `k_ext` | `float` | External momentum $k$ |
| `n_ext` | `int` | Fermionic Matsubara index ($\omega_n = (2n{+}1)\pi/\beta$) |

**Returns**: `complex` — $\Sigma^{(2)}(k, i\omega_n)$.

---

### `compute_sigma2_tci`

```python
compute_sigma2_tci(params, k_ext, n_ext, rank=10, verbose=False) -> complex
```

TCI-accelerated computation. The Matsubara sum is performed analytically per $q$-point (vectorized), then the 1D $q$-sum is evaluated with TCI decomposition.

| Param     | Type | Description |
|-----------|------|-------------|
| `params`  | `HolsteinParams` | Model parameters |
| `k_ext`   | `float` | External momentum |
| `n_ext`   | `int` | Fermionic Matsubara index |
| `rank`    | `int` | TCI bond dimension |
| `verbose` | `bool` | Print diagnostics |

> **Note**: For 1D, TCI provides no speedup over direct summation. This function validates the TCI framework before scaling to higher dimensions.

---

### `compute_sigma2_tau`

```python
compute_sigma2_tau(params, k_ext, N_tau=256) -> tuple[ndarray, ndarray]
```

Imaginary-time representation — no frequency summation.

$$\Sigma^{(2)}(k, \tau) = -g^2 \frac{1}{N_k}\sum_q G_0(k{-}q,\, \tau)\, D_0(\tau)$$

| Param   | Type | Description |
|---------|------|-------------|
| `params` | `HolsteinParams` | Model parameters |
| `k_ext` | `float` | External momentum |
| `N_tau` | `int` | Number of $\tau$ grid points |

**Returns**: `(tau_grid, sigma_tau)` — both `ndarray (N_tau,)`.

---

### `sigma_tau_to_matsubara`

```python
sigma_tau_to_matsubara(tau_grid, sigma_tau, beta, n_ext) -> complex
```

Discrete Fourier transform $\Sigma(\tau) \to \Sigma(i\omega_n)$.

$$\Sigma(i\omega_n) = \int_0^\beta d\tau\, e^{i\omega_n \tau} \Sigma(\tau) \;\approx\; \Delta\tau \sum_j e^{i\omega_n \tau_j} \Sigma(\tau_j)$$

| Param       | Type | Description |
|-------------|------|-------------|
| `tau_grid`  | `ndarray` | $\tau$ grid points |
| `sigma_tau` | `ndarray` | $\Sigma(\tau)$ values |
| `beta`      | `float` | Inverse temperature |
| `n_ext`     | `int` | Fermionic Matsubara index |

---

## 4th-Order Self-Energy

Source: [`src/holstein.py`](../src/holstein.py)

Diagram: nested *rainbow²* (two-phonon exchange).

$$\Sigma^{(4)}(k, i\omega_n) = \frac{g^4}{N_k^2 \beta^2} \sum_{q_1, q_2} \sum_{m_1, m_2} G_0 \cdot D_0 \cdot G_0 \cdot D_0$$

---

### `compute_sigma4_brute_force`

```python
compute_sigma4_brute_force(params, k_ext, n_ext) -> complex
```

Four nested loops. Complexity: $O(N_k^2 \times (2N_\nu)^2)$.

---

### `compute_sigma4_vectorized`

```python
compute_sigma4_vectorized(params, k_ext, n_ext) -> complex
```

Vectorized inner Matsubara sum via NumPy broadcasting. Complexity: $O(N_k^2 \times N_\nu)$.

---

### `compute_sigma4_tci`

```python
compute_sigma4_tci(params, k_ext, n_ext, rank=5, verbose=False) -> complex
```

Row-by-row vectorized Matsubara sum with potential TCI compression on the $q_1$ axis.

---

### `compute_sigma4_direct_tci`

```python
compute_sigma4_direct_tci(params, k_ext, n_ext, rank=10, n_sweeps=4, verbose=False) -> complex
```

**Direct 4D TCI** — applies TCI decomposition to the full 4D integrand $F(q_1, q_2, \nu_{m_1}, \nu_{m_2})$ without any dimensional pre-reduction. Integration is performed by multi-bond CUR decomposition with SVD-regularized inversion and median averaging across all TT bonds.

| Param      | Type | Description |
|------------|------|-------------|
| `params`   | `HolsteinParams` | Model parameters |
| `k_ext`    | `float` | External momentum |
| `n_ext`    | `int` | Fermionic Matsubara index |
| `rank`     | `int` | TCI bond dimension |
| `n_sweeps` | `int` | Number of bidirectional sweeps |
| `verbose`  | `bool` | Print diagnostics |

**Returns**: `complex` — $\Sigma^{(4)}(k, i\omega_n)$.

> **Key result**: At `rank=20`, achieves **0.00% error** vs brute-force reference on the `(N_k=16, N_nu=32)` grid.

---

### `compute_sigma4_tau_brute_force`

```python
compute_sigma4_tau_brute_force(params, k_ext, n_ext, N_tau=256) -> complex
```

Reference implementation using factorized inner Matsubara sum. Computes the auxiliary function:

$$h(p, i\omega') = \frac{1}{\beta}\sum_m G_0(p,\, i\omega'{-}i\nu_m)\, D_0(i\nu_m)$$

exactly in Matsubara space, then performs the outer double sum. Complexity: $O(N_k^2 \times N_\nu^2)$.

| Param   | Type | Description |
|---------|------|-------------|
| `N_tau` | `int` | Unused (API compatibility); actual computation is in Matsubara space |

---

### `compute_sigma4_tau_tci`

```python
compute_sigma4_tau_tci(params, k_ext, n_ext, N_tau=64, rank=10, n_sweeps=4, verbose=False) -> complex
```

**Imaginary-time with first-moment tail subtraction** — the key Phase 5a result.

**Algorithm**:

1. Decompose $G_0(p, \tau) = G_0^{\mathrm{reg}}(p, \tau) + G_0^{\mathrm{tail}}(\tau)$
   - $G_0^{\mathrm{reg}}(p, \tau) = G_0(p, \tau) + \frac{1}{2}$ — smooth and truly anti-periodic
   - $G_0^{\mathrm{tail}}(\tau) = -\frac{1}{2}$ — constant tail
2. Compute $h = h_{\mathrm{reg}} + h_{\mathrm{tail}}$:
   - $h_{\mathrm{reg}} = \Delta\tau \sum_j [G_0(p, \tau_j) + \tfrac{1}{2}]\, D_0(\tau_j)\, e^{i\omega'\tau_j}$ — converges as $O(1/N_\tau^2)$
   - $h_{\mathrm{tail}} = (-\tfrac{1}{2}) \cdot D_0^{\mathrm{FT}}(i\omega')$ — analytical, where $D_0^{\mathrm{FT}}(i\omega') = -2i\omega' \coth(\omega_0\beta/2) / (\omega_0^2 + \omega'^2)$

| Param      | Type | Description |
|------------|------|-------------|
| `params`   | `HolsteinParams` | Model parameters |
| `k_ext`    | `float` | External momentum |
| `n_ext`    | `int` | Fermionic Matsubara index |
| `N_tau`    | `int` | Number of $\tau$ grid points for Fourier integration |
| `rank`     | `int` | Reserved for future TCI compression (currently unused) |
| `n_sweeps` | `int` | Reserved (unused) |
| `verbose`  | `bool` | Print diagnostics |

**Returns**: `complex` — $\Sigma^{(4)}(k, i\omega_n)$.

> **Convergence**: tail subtraction improves DFT error from $O(1/N_\tau)$ to $O(1/N_\tau^2)$. At $N_\tau = 32$, error is **0.008%** vs Matsubara reference.

---

## TCI Engine

Source: [`src/tci_core.py`](../src/tci_core.py)

### `TCIFitter`

```python
from src import TCIFitter

solver = TCIFitter(func, domain, rank=10)
solver.build_cores(anchors=None, n_sweeps=3, verbose=False)
```

Core Tensor Cross Interpolation engine with bidirectional DMRG-like sweeps and MaxVol pivot selection.

#### Constructor

| Param    | Type | Description |
|----------|------|-------------|
| `func`   | `Callable[[ndarray], ndarray]` | Target function. Input: `(M, D)` index array → Output: `(M,)` values |
| `domain` | `list[ndarray]` | List of $D$ index arrays, one per dimension |
| `rank`   | `int` | Maximum TCI bond dimension (number of pivot paths) |

#### `build_cores`

```python
solver.build_cores(anchors=None, n_sweeps=3, verbose=False)
```

Runs the bidirectional sweep algorithm.

| Param      | Type | Description |
|------------|------|-------------|
| `anchors`  | `ndarray (N, D) \| None` | Strategic seed points (indices). The anchor with the largest $|f|$ initializes all pivots |
| `n_sweeps` | `int` | Number of full sweeps (each sweep = forward + backward pass) |
| `verbose`  | `bool` | Print per-sweep convergence info |

**Side effects**: Populates `solver.pivot_paths: ndarray (rank, D)` — optimized pivot index paths.

#### Attributes

| Attribute      | Type | Description |
|----------------|------|-------------|
| `pivot_paths`  | `ndarray (rank, D)` | Optimized multi-index pivot paths |
| `func`         | `Callable` | Stored target function |
| `domain`       | `list[ndarray]` | Stored domain arrays |
| `n_dims`       | `int` | Number of dimensions $D$ |
| `rank`         | `int` | Bond dimension |

---

## TCI Integration Utilities

Source: [`src/tci_utils.py`](../src/tci_utils.py)

### `compute_tci_integral`

```python
compute_tci_integral(solver, dx_vol=1.0) -> float
```

Computes $\int f\, d\mathbf{x}$ using TCI pivot structure.

- **Standard mode** ($D \leq 5$ or non-uniform domains): Rank-1 separable approximation with log-space product accumulation.
- **QTT mode** ($D > 5$, uniform small domains): Pivot-guided Monte Carlo importance sampling.

| Param    | Type | Description |
|----------|------|-------------|
| `solver` | `TCIFitter` | Fitted TCI solver instance |
| `dx_vol` | `float` | Volume element per grid point |

---

### `compute_tci_integral_reference`

```python
compute_tci_integral_reference(solver, dx_vol=1.0, n_samples=100000) -> float
```

Monte Carlo reference integral for validation.

---

## QTT Encoder

Source: [`src/qtt_utils.py`](../src/qtt_utils.py)

### `QTTEncoder`

```python
from src import QTTEncoder

encoder = QTTEncoder(num_vars=3, num_bits=20, bounds=[(-3, 3)] * 3)
```

Quantized Tensor Train encoder following [Ritter et al., arXiv:2303.11819].

Maps $N$-variable continuous coordinates into a fused-bit TT index space, enabling $2^{R}$ virtual grid points per variable with only $R$ TT sites of physical dimension $d = 2^{N_{\mathrm{vars}}}$.

#### Constructor

| Param      | Type | Description |
|------------|------|-------------|
| `num_vars` | `int` | Number of physical variables $N$ |
| `num_bits` | `int` | Number of QTT layers $R$ (resolution: $2^R$ points per variable) |
| `bounds`   | `list[tuple]` | `[(min, max)]` physical range per variable |

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `decode` | `(qtt_indices: ndarray(M, R)) -> ndarray(M, N)` | QTT indices → physical coordinates |
| `encode` | `(physical_coords: ndarray(M, N)) -> ndarray(M, R)` | Physical coordinates → QTT indices |
| `get_anchors` | `() -> ndarray(5, R)` | Strategic anchor points (origin, boundaries, probes) |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_vars`  | `int` | Number of physical variables |
| `R`       | `int` | Number of QTT bit layers |
| `d`       | `int` | Physical site dimension $= 2^{N_{\mathrm{vars}}}$ |
| `bounds`  | `list` | Physical coordinate ranges |

---

## Adaptive TCI

Source: [`src/aci_core.py`](../src/aci_core.py)

### `AdaptiveTCI`

```python
from src import AdaptiveTCI

aci = AdaptiveTCI(func, domain, max_rank=100, tolerance=1e-6,
                  max_pivots_per_sweep=10, n_test_samples=1000)
final_rank = aci.build_adaptive(anchors=None, verbose=True)
result = aci.compute_integral(dx_vol)
```

Automatic rank adaptation with error-driven stopping.

#### Constructor

| Param                 | Type | Default | Description |
|-----------------------|------|---------|-------------|
| `func`                | `Callable` | — | Target function |
| `domain`              | `list[ndarray]` | — | Index domain per dimension |
| `max_rank`            | `int` | `100` | Upper bound on rank |
| `tolerance`           | `float` | `1e-6` | Relative error stopping threshold |
| `max_pivots_per_sweep`| `int` | `10` | Max pivots added per iteration |
| `n_test_samples`      | `int` | `1000` | Sampling budget for error estimation |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `build_adaptive(anchors, verbose)` | `int` | Adaptively add pivots until convergence; returns final rank |
| `compute_integral(dx_vol)` | `float` | Importance-sampling integral using discovered pivots |

#### Convergence History

```python
aci.history['rank']          # list[int]   — rank at each iteration
aci.history['max_residual']  # list[float] — max relative residual
aci.history['mean_error']    # list[float] — mean relative error
```

---

## TT-Core TCI

Source: [`src/tt_core_tci.py`](../src/tt_core_tci.py)

### `TTCoreTCI`

```python
from src.tt_core_tci import TTCoreTCI

tt = TTCoreTCI(func, domain, max_rank=50, tolerance=1e-8)
tt.build_cores_from_sampling(n_samples_per_dim=100)
result = tt.compute_integral(dx_vol)
```

Explicit TT-core construction with SVD-based adaptive rank truncation. Computes integrals via standard TT-core contraction: $\int f \approx \prod_d \left(\sum_{i_d} G_d[:, i_d, :]\right)$.

> **Status**: Experimental. The sampling-based core construction does not yet achieve the accuracy of the pivot-based `TCIFitter` + CUR integration pipeline.

#### Constructor

| Param       | Type | Default | Description |
|-------------|------|---------|-------------|
| `func`      | `Callable` | — | Target function |
| `domain`    | `list[ndarray]` | — | Index domain |
| `max_rank`  | `int` | `50` | Maximum bond dimension |
| `tolerance` | `float` | `1e-8` | SVD truncation tolerance |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `cores`   | `list[ndarray \| None]` | TT-cores; `cores[d]` has shape $(r_{d-1}, n_d, r_d)$ |
| `ranks`   | `list[int]` | Bond dimensions $[r_0=1, r_1, \ldots, r_D=1]$ |

---

## Usage Patterns

### Minimal: 2nd-order self-energy

```python
from src import HolsteinParams, compute_sigma2_brute_force

params = HolsteinParams(t=1.0, omega0=0.5, g=0.3, beta=10.0, N_k=64, N_nu=128)
sigma = compute_sigma2_brute_force(params, k_ext=0.0, n_ext=0)
print(f"Σ(2) = {sigma}")  # ~ -0.01662j
```

### Cross-validation: Matsubara vs imaginary-time

```python
from src import (HolsteinParams,
                 compute_sigma4_tau_brute_force,
                 compute_sigma4_tau_tci)

params = HolsteinParams(t=1.0, omega0=0.5, g=0.3, beta=10.0, N_k=8, N_nu=16)

ref = compute_sigma4_tau_brute_force(params, k_ext=0.0, n_ext=0)
tci = compute_sigma4_tau_tci(params, k_ext=0.0, n_ext=0, N_tau=128)

print(f"Reference:  {ref:.8f}")
print(f"τ-TCI:      {tci:.8f}")
print(f"Rel. error: {abs(tci - ref) / abs(ref) * 100:.4f}%")
```

### TCI on a custom function

```python
import numpy as np
from src import TCIFitter
from src.tci_utils import compute_tci_integral

def my_func(coords):
    coords = np.atleast_2d(coords)
    return np.exp(-np.sum(coords**2, axis=1))

domain = [np.linspace(-3, 3, 64)] * 3
solver = TCIFitter(my_func, domain, rank=15)
solver.build_cores(n_sweeps=4, verbose=True)

dx = (6.0 / 64) ** 3
integral = compute_tci_integral(solver, dx_vol=dx)
print(f"∫ exp(-r²) d³r ≈ {integral:.4f}  (exact: {np.pi**1.5:.4f})")
```
