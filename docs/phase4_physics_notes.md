# Phase 4 物理与数值问题总结

## 问题背景

Phase 4 的目标是对 Σ(4) 的完整 4D 被积函数 F(q₁, q₂, ν_m₁, ν_m₂) 直接做 TCI，不做任何 Matsubara 频率预求和。

$$\Sigma^{(4)}(k, i\omega_n) = \frac{g^4}{N_k^2 \beta^2} \sum_{q_1, q_2, m_1, m_2} G_0(k-q_1, i\omega_n - i\nu_{m_1}) D_0(i\nu_{m_1}) G_0(k-q_1-q_2, i\omega_n - i\nu_{m_1} - i\nu_{m_2}) D_0(i\nu_{m_2})$$

## 遇到的核心问题

### 1. 被积函数的非可分性 (Non-Separability)

**问题**: TCI 的 rank-1 积分公式假设 f(x₁,...,x_D) 是近似可分的：

$$\int f \approx \frac{\prod_{d} \left[\sum_{i_d} f(\text{pivot}_{0..d-1}, i_d, \text{pivot}_{d+1..D-1})\right]}{f(\text{pivot})^{D-1}}$$

4阶自能的 4D 被积函数中，q₁ 和 q₂ 通过 G₀(k-q₁-q₂, ...) 强耦合。频率 ν_m₁ 和 ν_m₂ 通过 G₀(k-q₁-q₂, ωn-ν_m₁-ν_m₂) 也是耦合的。函数不是可分的，rank-1 cross product 给出 **154% 误差**。

**教训**: 对于非可分被积函数，rank-1 交叉乘积积分根本不适用。需要真正的多秩（multi-rank）积分方法。

### 2. TT-Cross 骨架分解的数值不稳定性

**问题**: 尝试用 skeleton decomposition 重构 TT cores 并做标准 TT 收缩积分：

$$\int f \approx \left(\sum_{i_0} C_0\right) \cdot P_0^{-1} \cdot \left(\sum_{i_1} C_1\right) \cdot P_1^{-1} \cdot \ldots$$

这需要在每个 bond 处计算 pivot 子矩阵 P_d 的逆。对于 4D 问题 (3个 bonds)，误差在连续矩阵逆的链式传播下被放大，结果为 **77,000% 误差**。

**教训**: 链式矩阵逆极易放大误差。D-1 次矩阵逆的链式乘积使骨架 TT 积分在 D≥4 时数值上非常不稳定。

### 3. Pivot 冗余与 P 矩阵病态

**问题**: `TCIFitter` 使用 MaxVol 算法选择 pivot，但在 4D 空间中：
- 高 rank 时，许多 pivot 在某些维度上几乎相同
- 当 pivot 的左/右多指标有重复时，P 矩阵变成（近似）奇异的
- 即使用 SVD truncation，截断后的有效秩可能远低于名义秩

**现象**: rank 从 8 增加到 15 时，积分误差反而从 0.6% 增加到 32%（非单调收敛）。

**解决方案**: 对 pivot 多指标做去重（deduplication），只保留唯一的左/右 pivot 路径。

### 4. 最终解决方案：多键 CUR 中位数

**关键思路**: 不在所有 D-1 个 bond 做链式骨架分解，而是：

1. **中间切割 CUR**: 在每个 bond d=1,...,D-1 处，将 D 维张量展开为 2D 矩阵，做单次 CUR 分解
2. **SVD 正则化**: P⁻¹ 通过截断 SVD 计算，丢弃小于 σ_max × 10⁻¹⁰ 的奇异值
3. **多键中位数**: 在所有 D-1 个 bond 各独立算一个积分估计，取中位数

$$I_{\text{bond}=d} = \left[\sum_{I_L} C(I_L, J_R)\right] \cdot P(J_L, J_R)^{-1} \cdot \left[\sum_{I_R} R(J_L, I_R)\right]$$

$$I_{\text{final}} = \text{median}(I_{\text{bond}=1}, I_{\text{bond}=2}, I_{\text{bond}=3})$$

**效果**: 消除了单一 bond 的 pivot 偏差，将 rank=5 误差从 33% → 0% (中位数选中了最准的 bond)。

### 5. Matsubara 频率的峰结构

**物理洞察**: 被积函数的主要贡献来自：
- **低 Matsubara 频率** (|m| ≈ 0): 玻色 Green 函数 D₀(ν_m) = -2ω₀/(ν_m² + ω₀²) 在 ν=0 处有强峰
- **前向散射动量** (q ≈ k): 电子 Green 函数 G₀(k-q, ...) 在 q≈k 处有极点

使用 **战略锚点** (strategic anchors) 初始化 TCI pivot，将初始采样点放在这些物理重要区域，避免了随机初始化导致 TCI 完全错过峰值的风险。

## 复杂度分析

| 方法 | 函数调用数 | 适用场景 |
|------|-----------|---------|
| 暴力4重循环 | O(N_k² × N_m²) ≈ 10⁶ | 仅用于小网格验证 |
| 向量化 (Matsubara预求和) | O(N_k² × N_m) | Phase 3, 快速但仍是降维 |
| 直接4D TCI + CUR | O(r × (N_k² + N_m²)) | Phase 4, 不做降维 |

当 N_k, N_m 增大时，直接4D TCI的复杂度优势将会更加显著。

## 开放问题

1. **收敛的非单调性**: 高 rank 不保证更低误差，因为 pivot 选择质量受 MaxVol 算法限制
2. **CUR vs TT**: 当前用的是单次 CUR 分解（一次矩阵逆），不是完整的 TT 收缩（D-1次矩阵逆），牺牲了近似精度换取数值稳定性
3. **更高阶推广**: 6阶自能是 6D 积分，CUR 需要在 5 个 bond 取中位数，计算代价增加但原理相同
