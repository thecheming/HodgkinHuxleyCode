import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
from tqdm import tqdm
from matplotlib.ticker import MultipleLocator, FixedLocator


def hh_model(t, state, xi0, tau, g_d_value):
    # 重写HH模型函数，ξ₀作为变量参数，g_d固定
    V, a_d, a_r, a_sd, a_sr = state
    
    # 固定参数（文档补充材料）
    C_M = 1.0       # μF/cm²
    T = 13.0        # °C
    T0 = 25.0       # °C
    tau0 = 10.0     # °C
    rho = 1.3**((T - T0)/tau0)  # 温度因子ρ
    phi = 3.0**((T - T0)/tau0)  # 温度因子φ
    
    # 反转电位 (mV)
    E_d = 50.0
    E_r = -90.0
    E_sd = 50.0
    E_sr = -90.0
    E_l = -60.0    # 泄漏电流反转电位
    
    # 最大电导 (μS/cm²) - g_d作为参数传入
    g_d = g_d_value  # 可以调整的参数
    g_r = 2.0
    g_sd = 0.25
    g_sr = 0.4
    g_l = 0.1      # 泄漏电导
    
    # 激活函数参数（文档）
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
    
    # 外部周期扰动（ξ₀作为变量参数）
    xi = 0.5 * xi0 * (1 + np.sin(2 * np.pi * t / tau))
    
    # 膜电位微分方程
    dVdt = (-I_d - I_r - I_sd - I_sr - I_l + xi) / C_M
    
    # 激活变量微分方程
    da_ddt = (phi / tau_d) * (a_d_inf - a_d)
    da_rdt = (phi / tau_r) * (a_r_inf - a_r)
    da_sddt = (phi / tau_sd) * (a_sd_inf - a_sd)
    da_srdt = (phi / tau_sr) * (-eta * I_sd - gamma * a_sr)
    
    return [dVdt, da_ddt, da_rdt, da_sddt, da_srdt]

# 数值积分参数（文档补充材料）
transient_time = 5000
record_time = 8000
total_time = transient_time + record_time
dt = 0.01
t_eval = np.arange(0, total_time, dt)

# 分岔参数扫描：ξ₀从0.88到0.93（覆盖文档图2范围）
xi0_values = np.linspace(0.88, 0.93, 1000)
fixed_tau = 2520

# 检测尖峰间隔（ISI）
def compute_isi(V):
    spikes = np.where((V[:-1] < 0) & (V[1:] >= 0))[0]  # 阈值0mV（文档图1）
    if len(spikes) < 2:
        return []
    return np.diff(spikes) * dt

# 计算单个ξ₀的分岔点
def compute_xi0_bifurcation(xi0, g_d_value):
    state0 = [-60.0, 0.01, 0.01, 0.01, 0.01]
    sol = solve_ivp(
        hh_model, [0, total_time], state0,
        args=(xi0, fixed_tau, g_d_value),  # 传递xi0、固定tau和g_d值
        t_eval=t_eval,
        method='RK45', rtol=1e-5, atol=1e-5
    )
    
    idx = int(transient_time / dt)
    V_record = sol.y[0, idx:]
    return xi0, compute_isi(V_record)

# 主计算流程（带进度条）
total_points = len(xi0_values)
start_total = time.time()

# 可以调整的g_d值
g_d_value = 1.69

print("\n开始计算ξ₀分岔图（ISI）...")
print(f"参数范围：ξ₀ ∈ [{xi0_values.min():.2f}, {xi0_values.max():.2f}], 点数={total_points}")
print(f"g_d 值: {g_d_value}")
print("=" * 60)

# 使用tqdm显示进度条
with tqdm(total=total_points, desc="ξ₀扫描", unit="点", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
    bifurcation_data = []
    for xi0 in xi0_values:
        xi0, isi = compute_xi0_bifurcation(xi0, g_d_value)
        bifurcation_data.extend([(xi0, val) for val in isi])
        pbar.update(1)

# 绘制分岔图（修改绘图部分）
xis, isis = zip(*bifurcation_data) if bifurcation_data else ([], [])
fig, ax = plt.subplots(figsize=(10, 6))

if xis:
    ax.scatter(xis, isis, s=1.5, alpha=0.8, color='black')
    ax.set_title(r'ISI- $ξ_0$ bifurcation ($g_d$ = '+'{g_d_value})'.format(g_d_value=g_d_value), fontsize=14)
    ax.set_xlabel(r'$\xi_0$ ($\mu$A/$cm^2$)', fontsize=12)
    ax.set_ylabel('ISI (ms)', fontsize=12)
    ax.set_xlim(0.88, 0.93)
    ax.set_xticks([0.88, 0.89, 0.90, 0.91, 0.92, 0.93])
    ax.xaxis.set_minor_locator(MultipleLocator(0.001))
    
    ax.grid(linestyle='--', alpha=0.5, zorder=0)
else:
    ax.text(0.5, 0.5, '未检测到跨膜电位活动', ha='center', va='center', fontsize=12)

plt.tight_layout()
plt.savefig(f'ISI分岔图(gd_{g_d_value},t瞬态_{transient_time}ms,采样{len(xi0_values)}点),采样时间{record_time}ms.png', dpi=600)
plt.show()

# 输出耗时
total_elapsed = time.time() - start_total
print(f"\n计算完成！总耗时: {total_elapsed:.2f}秒")
