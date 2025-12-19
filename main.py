from tci_core import TCIFitter
from physics_models import vectorized_gaussian
import numpy as np

def run_phase1_mvp():
    # 定义 3D 网格
    grid = [np.linspace(-3, 3, 50) for _ in range(3)]
    
    # 初始化 Fitter (注意此时 func 已经支持批量处理)
    solver = TCIFitter(vectorized_gaussian, grid, rank=1)
    
    print("开始 TCI 扫频...")
    solver.build_cores()
    
    # 验证
    integral = solver.get_tci_integral()
    print(f"向量化 TCI 积分结果: {integral}")
    print(f"理论误差: {abs(integral - 5.56832)}")

if __name__ == "__main__":
    run_phase1_mvp()