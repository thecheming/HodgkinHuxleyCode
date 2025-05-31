import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
from tqdm import tqdm
from matplotlib.ticker import MultipleLocator, FixedLocator


def hh_model(t, state, xi0, tau, T):
    # 重写HH模型函数，将温度T作为变量参数
    V, a_d, a_r, a_sd, a_sr = state
    
    # 固定参数
    C_M = 1.0       # μF/cm²
    T0 = 25.0       # °C (参考温度)
    tau0 = 10.0     # °C
    
    # 温度相关参数
    rho = 1.3**((T - T0)/tau0)  # 温度因子ρ
    phi = 3.0**((T - T0)/tau0)  # 温度因子φ
    
    # 反转电位 (mV)
    E_d = 50.0
    E_r = -90.0
    E_sd = 50.0
    E_sr = -90.0
    E_l = -60.0    # 泄漏电流反转电位
    
    # 最大电导 (μS/cm²)
    g_d = 1.69     # 固定为文档中的值
    g_r = 2.0
    g_sd = 0.25
    g_sr = 0.4
    g_l = 0.1      # 泄漏电导
    
    # 激活函数参数
    s_d = 0.25     # mV⁻¹
    s_r = 0.25     # mV⁻¹
    s_sd = 0.09    # mV⁻¹
    V0_d = -25.0   # mV
    V0_r = -25.0   # mV
    V0_sd = -40.0  # mV
    
    tau_d = 0.05   # ms
    tau_r = 2.0    # ms
    tau_sd = 10.0  # ms
    tau_sr = 20.0  # ms
    eta = 0.012    # cm²/μA
    gamma = 0.17   # 无单位
    
    # 稳态激活值
    a_d_inf = 1 / (1 + np.exp(-s_d * (V - V0_d)))
    a_r_inf = 1 / (1 + np.exp(-s_r * (V - V0_r)))
    a_sd_inf = 1 / (1 + np.exp(-s_sd * (V - V0_sd)))
    
    # 离子电流计算
    I_d = rho * g_d * a_d * (V - E_d)
    I_r = rho * g_r * a_r * (V - E_r)
    I_sd = rho * g_sd * a_sd * (V - E_sd)
    I_sr = rho * g_sr * a_sr * (V - E_sr)
    I_l = g_l * (V - E_l)
    
    # 外部周期扰动
    xi = 0.5 * xi0 * (1 + np.sin(2 * np.pi * t / tau))
    
    # 膜电位微分方程
    dVdt = (-I_d - I_r - I_sd - I_sr - I_l + xi) / C_M
    
    # 激活变量微分方程
    da_ddt = (phi / tau_d) * (a_d_inf - a_d)
    da_rdt = (phi / tau_r) * (a_r_inf - a_r)
    da_sddt = (phi / tau_sd) * (a_sd_inf - a_sd)
    da_srdt = (phi / tau_sr) * (-eta * I_sd - gamma * a_sr)
    
    return [dVdt, da_ddt, da_rdt, da_sddt, da_srdt]

# 数值积分参数
transient_time = 5000  # 瞬态时间
record_time = 4000     # 记录时间
total_time = transient_time + record_time
dt = 0.01              # 步长
t_eval = np.arange(0, total_time, dt)

# 检测尖峰间隔（ISI）
def compute_isi(V):
    spikes = np.where((V[:-1] < 0) & (V[1:] >= 0))[0]  # 阈值0mV
    if len(spikes) < 2:
        return []
    return np.diff(spikes) * dt  # 转换为ms

# 计算单个温度值的分岔点
def compute_temperature_bifurcation(T, xi0, tau):
    state0 = [-60.0, 0.01, 0.01, 0.01, 0.01]
    sol = solve_ivp(
        hh_model, [0, total_time], state0,
        args=(xi0, tau, T),  # 传递xi0、tau和温度T
        t_eval=t_eval,
        method='RK45', rtol=1e-5, atol=1e-5
    )
    
    # 提取稳态数据
    idx = int(transient_time / dt)
    V_record = sol.y[0, idx:]
    return T, compute_isi(V_record)

# 主计算流程
def main():
    # 分岔参数扫描：温度从10°C到30°C（生理相关范围）
    temperature_values = np.linspace(10, 30, 1000)  # 500个温度点
    fixed_xi0 = 1.            # 固定ξ₀=0.91（文档图2中有趣区域）
    fixed_tau = 2520            # 固定τ=2520 ms
    
    total_points = len(temperature_values)
    start_total = time.time()
    
    print("\n开始计算温度分岔图（ISI）...")
    print(f"参数范围：温度 ∈ [{temperature_values.min():.1f}°C, {temperature_values.max():.1f}°C], 点数={total_points}")
    print(f"ξ₀值: {fixed_xi0}")
    print(f"τ值: {fixed_tau} ms")
    print("=" * 60)
    
    # 使用tqdm显示进度条
    with tqdm(total=total_points, desc="温度扫描", unit="点", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        bifurcation_data = []
        for T in temperature_values:
            T, isi = compute_temperature_bifurcation(T, fixed_xi0, fixed_tau)
            bifurcation_data.extend([(T, val) for val in isi])
            pbar.update(1)
    
    # 绘制分岔图
    Ts, isis = zip(*bifurcation_data) if bifurcation_data else ([], [])
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if Ts:
        ax.scatter(Ts, isis, s=1.5, alpha=0.8, color='black')  # 蓝色小点
        ax.set_title(r'Temperature-ISI Bifurcation ($\xi_0$ = '+f'{fixed_xi0})', fontsize=14)
        ax.set_xlabel('Temperature (°C)', fontsize=12)
        ax.set_ylabel('ISI (ms)', fontsize=12)
        
        # 设置坐标轴范围和刻度
        ax.set_xlim(temperature_values.min(), temperature_values.max())
        ax.xaxis.set_major_locator(MultipleLocator(2))  # 主刻度间隔2°C
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))  # 小刻度间隔0.5°C
        
        ax.grid(linestyle='--', alpha=0.5, zorder=0)
        
        # ax.legend()
    else:
        ax.text(0.5, 0.5, '未检测到跨膜电位活动', ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'HHV-dV\Temperature_ISI_Bifurcation_xi0_{fixed_xi0}.png', dpi=600)
    plt.show()
    
    # 输出耗时
    total_elapsed = time.time() - start_total
    print(f"\n计算完成！总耗时: {total_elapsed:.2f}秒")

if __name__ == "__main__":
    main()
