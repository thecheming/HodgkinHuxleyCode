import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch  # 用于创建图例
from scipy.integrate import solve_ivp
from tqdm import tqdm
import time

# ---------------------- 参数配置 ---------------------- #
class ModelParams:
    """参数模块化管理"""
    C_M = 1.0; T = 13.0; T0 = 25.0; tau0 = 10.0
    g_d = 1.7; g_r = 2.0; g_sd = 0.25; g_sr = 0.4; g_l = 0.1
    s_d = 0.25; s_r = 0.25; s_sd = 0.09
    V0_d = -25.0; V0_r = -25.0; V0_sd = -40.0
    tau_d = 0.05; tau_r = 2.0; tau_sd = 10.0; tau_sr = 20.0
    eta = 0.012; gamma = 0.17; threshold = 0.0
    transient = 5e5; record = 2e3; dt = 0.01
    tau_range = np.linspace(2450, 2650, 100)
    xi0_range = np.linspace(0.80, 0.93, 100)

# ---------------------- 核心模型 ---------------------- #
def hh_model(t, state, xi0, tau):
    """Hodgkin-Huxley模型微分方程"""
    V, a_d, a_r, a_sd, a_sr = state
    p = ModelParams  # 引用参数类
    
    # 温度因子计算
    rho = 1.3**((p.T - p.T0)/p.tau0)
    phi = 3.0**((p.T - p.T0)/p.tau0)
    
    # 激活函数
    a_d_inf = 1/(1 + np.exp(-p.s_d*(V - p.V0_d)))
    a_r_inf = 1/(1 + np.exp(-p.s_r*(V - p.V0_r)))
    a_sd_inf = 1/(1 + np.exp(-p.s_sd*(V - p.V0_sd)))
    
    # 离子电流
    I_d = rho*p.g_d*a_d*(V - 50.0)
    I_r = rho*p.g_r*a_r*(V + 90.0)
    I_sd = rho*p.g_sd*a_sd*(V - 50.0)
    I_sr = rho*p.g_sr*a_sr*(V + 90.0)
    I_l = p.g_l*(V + 60.0)
    
    # 周期扰动
    xi = 0.5*xi0*(1 + np.sin(2*np.pi*t/tau))
    
    # 微分方程
    dVdt = (-I_d - I_r - I_sd - I_sr - I_l + xi)/p.C_M
    da_ddt = (phi/p.tau_d)*(a_d_inf - a_d)
    da_rdt = (phi/p.tau_r)*(a_r_inf - a_r)
    da_sddt = (phi/p.tau_sd)*(a_sd_inf - a_sd)
    da_srdt = (phi/p.tau_sr)*(-p.eta*I_sd - p.gamma*a_sr)
    
    return [dVdt, da_ddt, da_rdt, da_sddt, da_srdt]

# ---------------------- 计算模块 ---------------------- #
def compute_spike_counts():
    """参数扫描并计算尖峰数"""
    p = ModelParams
    total = len(p.xi0_range)*len(p.tau_range)
    spike_counts = np.zeros((len(p.tau_range), len(p.xi0_range)))  # 交换
    
    # 时间参数
    total_time = p.transient + p.record
    t_eval = np.arange(0, total_time, p.dt)
    record_start = int(p.transient/p.dt)
    
    with tqdm(total=total, desc="参数扫描") as pbar:
        # 交换循环顺序
        for j, tau in enumerate(p.tau_range):
            for i, xi0 in enumerate(p.xi0_range):
                # 正确初始化状态：5个变量 [V, a_d, a_r, a_sd, a_sr]
                initial_state = [-60.0, 0.01, 0.01, 0.01, 0.01]

                sol = solve_ivp(
                    hh_model, [0, total_time], initial_state,
                    args=(xi0, tau), t_eval=t_eval, method='RK45',
                    rtol=1e-5, atol=1e-5, vectorized=True
                )
                
                # 提取稳态数据
                V = sol.y[0, record_start:]
                spikes = np.where((V[:-1] < p.threshold) & (V[1:] >= p.threshold))[0].size
                spike_counts[j, i] = spikes / (p.record/tau) if tau else 0  # 注意索引顺序
                
                pbar.update(1)
    return spike_counts

# ---------------------- 绘图模块 ---------------------- #
def plot_bifurcation(spike_counts):
    """绘制分岔图（应用特殊颜色区间并添加图例）"""
    p = ModelParams
    fig, ax = plt.subplots(figsize=(10, 7))  # 增加高度以容纳图例
    
    # 使用jet颜色映射绘制基础图像
    im = ax.imshow(
        spike_counts,
        extent=[p.xi0_range.min(), p.xi0_range.max(),
                p.tau_range.min(), p.tau_range.max()],
        origin='lower', aspect='auto',
        cmap='jet',
        interpolation='nearest',
        vmin=0, vmax=100
    )
    '''
    # 创建特殊颜色的覆盖层
    overlay = np.zeros((*spike_counts.shape, 4))
    color_map = {
        (0, 1): '#ff69b4',    # 粉红
        (10, 11): '#000000',  # 纯黑
        (21, 22): '#a9a9a9',  # 灰色
        (32, 33): '#4b0082',  # 靛蓝
        (43, 44): '#FFFFFF'   # 纯白
    }
    
    # 为特殊区间填充颜色
    for i in range(spike_counts.shape[0]):
        for j in range(spike_counts.shape[1]):
            val = spike_counts[i, j]
            for (low, high), color in color_map.items():
                if low <= val <= high:
                    rgba = list(mcolors.to_rgba(color))
                    rgba[3] = 1.0  # 完全不透明
                    overlay[i, j] = rgba
                    break
    
    # 绘制特殊颜色覆盖层
    ax.imshow(
        overlay,
        extent=[p.xi0_range.min(), p.xi0_range.max(),
                p.tau_range.min(), p.tau_range.max()],
        origin='lower', aspect='auto',
        interpolation='nearest'
    )
    '''
    # 图表修饰
    ax.set_title(f'Isospikes Diagram (g_d={p.g_d} μS/cm²)', fontsize=14)
    ax.set_xlabel(r'$\xi_0$ ($\mu$A/cm²)', fontsize=12)
    ax.set_ylabel(r'$\tau$ (ms)', fontsize=12)
    
    # 添加颜色条
    cbar = plt.colorbar(im, label='Spikes per Period (n)', fraction=0.022, pad=0.04)
    '''
    # 创建特殊区间的图例
    legend_elements = [
        Patch(facecolor='#ff69b4', edgecolor='k', label='0-1 Spikes'),
        Patch(facecolor='#000000', edgecolor='k', label='10-11 Spikes'),
        Patch(facecolor='#a9a9a9', edgecolor='k', label='21-22 Spikes'),
        Patch(facecolor='#4b0082', edgecolor='k', label='32-33 Spikes'),
        Patch(facecolor='#FFFFFF', edgecolor='k', label='43-44 Spikes')
    ]
    
    # 在图表右上角添加图例
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    '''
    ax.grid(linestyle='--', alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig('isospikes图.png', dpi=600)
    plt.show()


# ---------------------- 主程序 ---------------------- #
if __name__ == "__main__":
    start = time.time()
    print("开始分岔图计算...")
    
    spike_counts = compute_spike_counts()
    plot_bifurcation(spike_counts)
    
    print(f"计算完成！总耗时: {time.time()-start:.2f}秒")
