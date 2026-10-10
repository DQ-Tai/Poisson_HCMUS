import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma  
import matplotlib.ticker as ticker

# ==========================================
# 0. ACADEMIC STYLE SETUP
# ==========================================
plt.rcdefaults()
plt.rcParams.update({
    "font.family": "serif",           
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "xtick.direction": "in",          
    "ytick.direction": "in",
    "xtick.top": True,                
    "ytick.right": True,              
    "axes.formatter.use_mathtext": True
})

# Means from experimental data
means = {
    'Background': 6.83,
    'Ra-226 ($d=950$ mm)': 102.39,
    'Ra-226 ($d=765$ mm, shielded)': 44.22
}

# ==========================================
# 1. MONTE CARLO SIMULATION (Section 3.1)
# ==========================================
def run_and_plot_simulation(title, exp_mean, save_name):
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=100)
    tries = 1000  
    
    # Generate 1000 random events following Poisson distribution
    sim_data = np.random.poisson(lam=exp_mean, size=tries)
    
    min_val, max_val = min(sim_data), max(sim_data)
    bins_edges = np.arange(min_val - 0.5, max_val + 1.5, 1)
    bin_centers = np.arange(min_val, max_val + 1)
    
    sim_counts, _ = np.histogram(sim_data, bins=bins_edges)
    
    # Plot Simulation (Red dots as in Fig 3 of manual)
    ax.plot(bin_centers, sim_counts, 'r.', markersize=7, label=f'Monte Carlo ($N={tries}$)')
    
    # Plot Theoretical Poisson scaled to 1000 tries
    x_smooth = np.linspace(min_val, max_val, 500)
    y_poisson_smooth = (np.exp(-exp_mean) * (exp_mean**x_smooth) / gamma(x_smooth + 1)) * tries
    ax.plot(x_smooth, y_poisson_smooth, color='#2E7D32', linestyle='-', linewidth=2, label='Theoretical Poisson')
    
    # Aesthetics
    ax.set_title(f"Monte Carlo Simulation: {title}", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Counts', fontsize=12)
    ax.set_ylabel('Number of events', fontsize=12) 
    ax.legend(loc='upper left', frameon=True, edgecolor='black')
    ax.grid(axis='both', linestyle='-', alpha=0.15)
    
    # Statistics Box
    sim_mean = np.mean(sim_data)
    sim_var = np.var(sim_data, ddof=1)
    stats_text = f"Sim Mean: {sim_mean:.2f}\nSim Var: {sim_var:.2f}\nRef Mean ($\\mu$): {exp_mean:.2f}"
    
    ax.text(0.96, 0.96, stats_text, transform=ax.transAxes, fontsize=10, va='top', ha='right', 
            bbox=dict(facecolor='white', edgecolor='black', alpha=1.0))
    
    tick_step = 1 if (max_val - min_val) <= 20 else 5
    ax.set_xticks(np.arange(min_val, max_val + 1, tick_step))
    
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoLocator())
    ax.tick_params(which='minor', direction='in', top=True, right=True)
    
    plt.tight_layout()
    fig.savefig(f"{save_name}.pdf", bbox_inches='tight')
    plt.close(fig)
    
    # Print results to terminal
    print(f"\n[Simulation: {title}]")
    print("-" * 40)
    print(f"-> Input Mean (mu) : {exp_mean:.2f}")
    print(f"-> Simulated Mean  : {sim_mean:.2f}")
    print(f"-> Simulated Var   : {sim_var:.2f}")

# ==========================================
# 2. EXECUTE SIMULATIONS
# ==========================================
print(">>> RUNNING 1000-TRIAL MONTE CARLO SIMULATIONS...")

run_and_plot_simulation('Background', means['Background'], 'Fig_3_Sim_BG')
run_and_plot_simulation('Ra-226 ($d=950$ mm)', means['Ra-226 ($d=950$ mm)'], 'Fig_3_Sim_Ra950')
run_and_plot_simulation('Ra-226 ($d=765$ mm)', means['Ra-226 ($d=765$ mm, shielded)'], 'Fig_3_Sim_Ra765')

print("\n>>> All simulations completed and plots saved.")