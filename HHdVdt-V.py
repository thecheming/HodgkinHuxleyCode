import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 改进热敏HH模型（保持原文件中的定义不变）
def hh_model(t, state, xi0, tau):
    V, a_d, a_r, a_sd, a_sr = state
    
    # 固定参数（来自文档表1和正文）
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
    
    # 最大电导 (μS/cm²)
    g_d = 1.5
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
    
    # 计算稳态激活值
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
    
    # 膜电位微分方程 [公式(1)]
    dVdt = (-I_d - I_r - I_sd - I_sr - I_l + xi) / C_M
    
    # 激活变量微分方程 [公式(3)]
    da_ddt = (phi / tau_d) * (a_d_inf - a_d)
    da_rdt = (phi / tau_r) * (a_r_inf - a_r)
    da_sddt = (phi / tau_sd) * (a_sd_inf - a_sd)
    da_srdt = (phi / tau_sr) * (-eta * I_sd - gamma * a_sr)
    
    return [dVdt, da_ddt, da_rdt, da_sddt, da_srdt]

# 计算膜电位的导数 (dV/dt)
def calculate_dVdt(V, a_d, a_r, a_sd, a_sr, t, xi0, tau):
    # 固定参数（与hh_model保持一致）
    C_M = 1.0
    T = 13.0
    T0 = 25.0
    tau0 = 10.0
    rho = 1.3**((T - T0)/tau0)
    phi = 3.0**((T - T0)/tau0)
    
    E_d = 50.0
    E_r = -90.0
    E_sd = 50.0
    E_sr = -90.0
    E_l = -60.0
    
    g_d = 1.5
    g_r = 2.0
    g_sd = 0.25
    g_sr = 0.4
    g_l = 0.1
    
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
    return dVdt

# 绘制相图
def plot_phase_portrait(ax, xi0, tau, title, point_count=20):
    # 初始状态
    state0 = [-60.0, 0.01, 0.01, 0.01, 0.01]
    
    # 计算足够长时间以达到稳态
    transient_time = 5000  # ms
    record_time = 4000     # ms
    t_total = transient_time + record_time
    t_eval = np.linspace(0, t_total, int(t_total*50)+1)
    
    # 求解微分方程,RK45
    sol = solve_ivp(
        hh_model,
        [0, t_total],
        state0,
        args=(xi0, tau),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-6
    )
    
    # 提取稳态数据
    idx_transient = int(transient_time * 10)
    V = sol.y[0, idx_transient:]
    a_d = sol.y[1, idx_transient:]
    a_r = sol.y[2, idx_transient:]
    a_sd = sol.y[3, idx_transient:]
    a_sr = sol.y[4, idx_transient:]
    t = sol.t[idx_transient:]
    
    # 计算dV/dt
    dVdt_values = []
    for i in range(len(V)):
        dVdt = calculate_dVdt(V[i], a_d[i], a_r[i], a_sd[i], a_sr[i], t[i], xi0, tau)
        dVdt_values.append(dVdt)
    ax.plot(V, dVdt_values, 'b-', alpha=0.7, linewidth=1.5)
    
    
    # 添加零倾线 (dV/dt = 0)
    V_range = np.linspace(-100, 50, 200)
    zero_isocline = np.zeros_like(V_range)
    ax.plot(V_range, zero_isocline, 'r--', alpha=0.5, label='dV/dt = 0')
    
    ax.set_xlabel('Membrane Potential V (mV)')
    ax.set_ylabel('dV/dt (mV/ms)')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    
    return ax

if __name__ == '__main__':
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axs = axs.flatten()

    conditions = [
        (0.0, 1000, r'(a) $\xi_0=0$ (No stimulation)'),
        (0.15, 1200, r'(b) $\xi_0=0.15$, $\tau=1200$ ms'),
        (1.0, 1500, r'(c) $\xi_0=1.0$, $\tau=1500$ ms'),
        (0.1, 2000, r'(d) $\xi_0=0.1$, $\tau=2000$ ms')
    ]

    for i, (xi0, tau, title) in enumerate(conditions):
        ax = axs[i]
        plot_phase_portrait(ax, xi0, tau, title)

    fig.suptitle('Phase Portrait of Membrane Potential (V vs dV/dt)', fontsize=14, y=1.02)
    plt.savefig('figure_phase_portrait.png', dpi=300, bbox_inches='tight')
    plt.show()
