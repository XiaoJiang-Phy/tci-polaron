import numpy as np

def vectorized_gaussian(coords):
    """
    coords: (M, N) 物理坐标
    """
    coords = np.atleast_2d(coords)
    # 简单的 Gaussian 变体：exp(-sum(x^2))
    # 极化子计算中通常对应 exp(-lambda * x^2)
    return np.exp(-np.sum(coords**2, axis=1))

def vectorized_holstein(coords):
    """
    Placeholder for Polaron model
    """
    pass