import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
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

# ==========================================
# 1. EXPERIMENTAL DATA (100 RUNS)
# ==========================================
bg_counts = np.array([6, 6, 6, 4, 4, 3, 4, 5, 5, 7, 9, 7, 12, 7, 6, 7, 7, 7, 8, 5, 2, 9, 9, 2, 5, 10, 
                      3, 9, 6, 10, 9, 10, 6, 7, 4, 6, 13, 9, 9, 10, 13, 7, 6, 3, 5, 12, 4, 4, 7, 7, 6, 8, 11, 8, 10, 
                      11, 7, 3, 6, 3, 16, 6, 6, 7, 7, 8, 7, 3, 6, 8, 1, 5, 11, 4, 5, 6, 8, 9, 7, 8, 4, 10, 
                      6, 11, 4, 5, 7, 6, 2, 3, 11, 11, 6, 16, 7, 2, 2, 4, 9, 5])

ra_950mm = np.array([109, 95, 92, 92, 93, 110, 104, 83, 109, 101, 110, 99, 99, 108, 102, 97, 88, 105, 97, 105, 92, 109, 94, 126, 79, 84, 
                     103, 108, 115, 106, 97, 107, 104, 85, 91, 89, 97, 93, 113, 98, 94, 105, 123, 118, 96, 96, 111, 97, 106, 114, 95, 95, 96, 118, 90, 
                     120, 104, 88, 108, 93, 102, 106, 104, 123, 89, 92, 112, 82, 103, 101, 98, 111, 107, 117, 99, 95, 111, 126, 101, 120, 99, 99, 
                     105, 111, 107, 93, 108, 96, 96, 115, 100, 120, 97, 107, 96, 105, 120, 101, 99, 111])

ra_765mm = np.array([47, 51, 44, 54, 49, 42, 50, 49, 58, 49, 53, 54, 45, 32, 48, 52, 55, 50, 43, 59, 47, 48, 30, 34, 38, 39, 
                     49, 46, 42, 45, 45, 38, 45, 53, 44, 39, 59, 63, 47, 35, 37, 31, 42, 48, 46, 59, 45, 45, 40, 32, 49, 41, 46, 39, 55, 
                     51, 49, 30, 35, 38, 43, 40, 35, 41, 45, 34, 41, 52, 38, 42, 34, 48, 49, 38, 42, 56, 40, 42, 42, 40, 32, 45, 
                     54, 39, 46, 46, 54, 43, 42, 41, 58, 48, 35, 41, 37, 43, 40, 33, 35, 42])

# ==========================================
# 2. RUNNING AVERAGE PLOTS (Section 2.3.1)
# ==========================================
def plot_running_average(data, title, save_name):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=100)
    j_vals = np.arange(1, len(data) + 1)
    r_c = np.cumsum(data) / j_vals  
    
    ax.plot(j_vals, r_c, color='#1565C0', marker='.', markersize=4, label='Running Avg $r_c(j)$')
    ax.axhline(np.mean(data), color='red', linestyle='--', linewidth=1.5, label=f'Final Mean ({np.mean(data):.2f})')
    
    ax.set_title(f"Convergence of Mean: {title}", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Measurement number ($j$)', fontsize=12)
    ax.set_ylabel('Running Average (Counts)', fontsize=12)
    ax.legend(frameon=True, edgecolor='black')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    fig.savefig(f"{save_name}.pdf", bbox_inches='tight')
    plt.close(fig)

# ==========================================
# 3. FREQUENCY DISTRIBUTIONS (Section 2.3.3 & 4.2)
# ==========================================
def calculate_chi2(obs_freq, expected_freq):
    valid = obs_freq > 0
    return np.sum(((obs_freq[valid] - expected_freq[valid])**2) / obs_freq[valid])

def plot_experiment_vs_theory(data, title, save_name, is_large_mu=False):
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=100)
    
    N_total = len(data)
    exp_mean = np.mean(data) 
    exp_var = np.var(data, ddof=1)
    
    min_val, max_val = min(data), max(data)
    bins_edges = np.arange(min_val - 0.5, max_val + 1.5, 1)
    bin_centers = np.arange(min_val, max_val + 1)
    counts_obs, _ = np.histogram(data, bins=bins_edges)
    errors_obs = np.sqrt(counts_obs)
    valid = counts_obs > 0
    
    # Experimental Data
    ax.errorbar(bin_centers[valid], counts_obs[valid], yerr=errors_obs[valid], 
                fmt='o', color='black', markersize=4, capsize=2, elinewidth=1.2, label='Experimental Data')
    
    # Theoretical Poisson
    x_smooth = np.linspace(min_val, max_val, 500)
    y_poisson_smooth = (np.exp(-exp_mean) * (exp_mean**x_smooth) / gamma(x_smooth + 1)) * N_total
    ax.plot(x_smooth, y_poisson_smooth, color='#2E7D32', linestyle='-', linewidth=2, label='Theoretical Poisson')
    
    y_poisson_discrete = (np.exp(-exp_mean) * (exp_mean**bin_centers) / gamma(bin_centers + 1)) * N_total
    chi2_poisson = calculate_chi2(counts_obs, y_poisson_discrete)
    
    stats_text = (f"Entries $N$ = {N_total}\n"
                  f"Exp. Mean $\\bar{{x}}$ = {exp_mean:.2f}\n"
                  f"Exp. Var $s^2$ = {exp_var:.2f}\n"
                  f"$\\chi^2_{{Poisson}}$ = {chi2_poisson:.2f}")

    # Theoretical Gaussian
    if is_large_mu:
        sigma_theory = np.sqrt(exp_mean) 
        y_gauss_smooth = norm.pdf(x_smooth, loc=exp_mean, scale=sigma_theory) * N_total
        ax.plot(x_smooth, y_gauss_smooth, color='#1565C0', linestyle='--', linewidth=2, label='Theoretical Gaussian')
        
        y_gauss_discrete = norm.pdf(bin_centers, loc=exp_mean, scale=sigma_theory) * N_total
        chi2_gauss = calculate_chi2(counts_obs, y_gauss_discrete)
        stats_text += f"\n$\\chi^2_{{Gauss}}$ = {chi2_gauss:.2f}"
    
    # Aesthetics
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Counts', fontsize=12)
    ax.set_ylabel('Number of events', fontsize=12) 
    
    ax.text(0.96, 0.96, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', 
            bbox=dict(boxstyle='square,pad=0.5', facecolor='white', edgecolor='black', alpha=1.0))
    
    ax.set_ylim(0, np.max(counts_obs[valid] + errors_obs[valid]) * 1.35) 
    tick_step = 1 if (max_val - min_val) <= 20 else 5
    ax.set_xticks(np.arange(min_val, max_val + 1, tick_step))
    
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoLocator())
    ax.tick_params(which='minor', direction='in', top=True, right=True)
    ax.legend(loc='upper left', frameon=True, edgecolor='black')
    ax.grid(axis='both', linestyle='-', alpha=0.15)
    
    plt.tight_layout()
    fig.savefig(f"{save_name}.pdf", bbox_inches='tight')
    plt.close(fig)

# ==========================================
# 4. TERMINAL REPORTING
# ==========================================
def print_final_analysis(data, name, ref_100s):
    mean_val = np.mean(data)
    var_val = np.var(data, ddof=1)
    std_val = np.std(data, ddof=1)
    sem = std_val / np.sqrt(len(data))
    
    print(f"\n[{name}] DATA ANALYSIS REPORT:")
    print("-" * 50)
    
    print(f"-> Exp Mean = {mean_val:.2f} | 100s Ref = {ref_100s}")
    diff = abs(mean_val - ref_100s)
    print(f"-> Difference = {diff:.2f} counts (~{diff/sem:.2f} standard errors).")
    print(f"-> Exp Variance (s^2) = {var_val:.2f} | Theo Variance (mu) = {mean_val:.2f}")
    
    if name == "Background":
        max_dev = max(abs(np.max(data) - mean_val), abs(np.min(data) - mean_val))
        print(f"-> Max deviation from mean = {max_dev:.2f} (~{max_dev/std_val:.1f} sigma).")
        print("-> Conclusion: System operates normally (within expected statistical limits).")

# ==========================================
# 5. EXECUTE 
# ==========================================
print(">>> GENERATING EXPERIMENTAL PLOTS...")
plot_running_average(bg_counts, 'Background', 'Fig_1_Conv_BG')
plot_running_average(ra_950mm, 'Ra-226 ($d=950$ mm)', 'Fig_1_Conv_Ra950')
plot_running_average(ra_765mm, 'Ra-226 ($d=765$ mm)', 'Fig_1_Conv_Ra765')

plot_experiment_vs_theory(bg_counts, 'Background', 'Fig_2_Dist_BG', is_large_mu=False)
plot_experiment_vs_theory(ra_950mm, 'Ra-226 ($d=950$ mm)', 'Fig_2_Dist_Ra950', is_large_mu=True)
plot_experiment_vs_theory(ra_765mm, 'Ra-226 ($d=765$ mm, shielded)', 'Fig_2_Dist_Ra765', is_large_mu=True)

print_final_analysis(bg_counts, "Background", 8.0)
print_final_analysis(ra_950mm, "Ra-226 (950mm)", 96.6)
print_final_analysis(ra_765mm, "Ra-226 (765mm)", 46.1)

# ==========================================
# 4. TERMINAL & TEXT FILE REPORTING
# ==========================================
def print_final_analysis(data, name, ref_100s, file):
    mean_val = np.mean(data)
    var_val = np.var(data, ddof=1)
    std_val = np.std(data, ddof=1)
    sem = std_val / np.sqrt(len(data))
    
    # Tạo nội dung báo cáo
    report_text = f"\n[{name}] DATA ANALYSIS REPORT:\n"
    report_text += "-" * 50 + "\n"
    report_text += f"2.3.2) Exp Mean = {mean_val:.2f} | 100s Ref = {ref_100s}\n"
    
    diff = abs(mean_val - ref_100s)
    report_text += f"       Difference = {diff:.2f} counts (~{diff/sem:.2f} standard errors from mean).\n"
    report_text += f"4.2)   Exp Variance (s^2) = {var_val:.2f} | Theo Variance (sigma^2 = mu) = {mean_val:.2f}\n"
    
    if name == "Background":
        max_dev = max(abs(np.max(data) - mean_val), abs(np.min(data) - mean_val))
        report_text += f"2.3.4) Max deviation from mean = {max_dev:.2f}\n"
        report_text += f"       This is ~{max_dev/std_val:.1f} standard deviations (sigma).\n"
        report_text += "       Conclusion: Within 100 samples, a 3-sigma event is statistically expected. System is stable.\n"
    
    # In ra Terminal
    print(report_text, end="")
    # Viết vào file txt
    file.write(report_text)
