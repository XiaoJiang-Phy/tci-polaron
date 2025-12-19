# physics_models.py
import numpy as np

def vectorized_gaussian(coords):
    """
    coords: 形状可以是 (N,) 或 (M, N)
    """
    # 强制转换为 2D，确保 axis=1 始终有效
    coords = np.atleast_2d(coords) 
    return np.exp(-np.sum(coords**2, axis=1))

def vectorized_holstein(coords):
    """
    未来的极化子函数也需要支持 axis=1 操作
    """
    # ... 实现逻辑 ...
    pass