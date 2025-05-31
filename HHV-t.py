# 该文件为主文件
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 改进热敏HH模型
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

# 绘制单条件下的膜电位和扰动电流
def plot_condition(ax, xi0, tau, title, idx=0):
    # 初始状态 (V, a_d, a_r, a_sd, a_sr)
    state0 = [-60.0, 0.01, 0.01, 0.01, 0.01]
    
    transient_time = 000  # ms 瞬态时间
    record_time = 10000     # ms 记录时间
    t_total = transient_time + record_time
    t_eval = np.linspace(0, t_total, int(t_total*100)+1)
    
    # 求解微分方程，RK45方法
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
    idx_transient = int(transient_time * 100)
    t = sol.t[idx_transient:] / 1000  # 转换为秒
    V = sol.y[0, idx_transient:]
    xi = 0.5 * xi0 * (1 + np.sin(2 * np.pi * sol.t[idx_transient:]/tau))  # 计算扰动电流
    
    # 绘制膜电位（主y轴）
    ax.plot(t, V, color='k', label='Membrane Potential (V)', zorder=1)
    ax.set_ylabel('V (mV)', color='k')
    ax.tick_params('y', colors='k')
    
    # 创建双y轴绘制扰动电流
    ax2 = ax.twinx()
    ax2.plot(t, xi, color='r', label=r'$\xi(t)$', zorder=2)
    ax2.set_ylabel(r'$\xi$ ($\mu$A/cm²)', color='r')
    ax2.tick_params('y', colors='r')
    
    # 图形设置
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_xlim(t[0], t[-1])
    
    return ax, ax2

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
        plot_condition(ax, xi0, tau, title)

    fig.suptitle('Dynamical Behavior of Membrane Potential under Periodic Perturbation', 
                fontsize=14, y=1.02)

    plt.savefig('figure1_hh_model2.png', dpi=300, bbox_inches='tight')
    plt.show()
