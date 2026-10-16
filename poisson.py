import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import norm, poisson
from scipy.special import gamma

# ==============================================================================
# 0. ĐỒ THỊ
# ==============================================================================
plt.rcdefaults()
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.formatter.use_mathtext": True,
    "savefig.dpi": 300
})

# 
np.random.seed(42)

# ==============================================================================
# 1. DỮ LIỆU THỰC NGHIỆM (100 LẦN ĐO MỖI CẤU HÌNH, CHU KỲ ĐO dt = 10s)
# ==============================================================================
dt = 10.0  # Thời gian mỗi lần đo (giây)

bg_counts = np.array([
    6, 6, 6, 4, 4, 3, 4, 5, 5, 7, 9, 7, 12, 7, 6, 7, 7, 7, 8, 5,
    2, 9, 9, 2, 5, 10, 3, 9, 6, 10, 9, 10, 6, 7, 4, 6, 13, 9, 9, 10,
    13, 7, 6, 3, 5, 12, 4, 4, 7, 7, 6, 8, 11, 8, 10, 11, 7, 3, 6, 3,
    16, 6, 6, 7, 7, 8, 7, 3, 6, 8, 1, 5, 11, 4, 5, 6, 8, 9, 7, 8,
    4, 10, 6, 11, 4, 5, 7, 6, 2, 3, 11, 11, 6, 16, 7, 2, 2, 4, 9, 5
])

ra_950mm = np.array([
    109, 95, 92, 92, 93, 110, 104, 83, 109, 101, 110, 99, 99, 108, 102, 97, 88, 105, 97, 105,
    92, 109, 94, 126, 79, 84, 103, 108, 115, 106, 97, 107, 104, 85, 91, 89, 97, 93, 113, 98,
    94, 105, 123, 118, 96, 96, 111, 97, 106, 114, 95, 95, 96, 118, 90, 120, 104, 88, 108, 93,
    102, 106, 104, 123, 89, 92, 112, 82, 103, 101, 98, 111, 107, 117, 99, 95, 111, 126, 101, 120,
    99, 99, 105, 111, 107, 93, 108, 96, 96, 115, 100, 120, 97, 107, 96, 105, 120, 101, 99, 111
])

ra_765mm = np.array([
    47, 51, 44, 54, 49, 42, 50, 49, 56, 49, 53, 54, 45, 32, 48, 52, 55, 50, 43, 59,
    47, 48, 30, 34, 38, 39, 49, 46, 42, 45, 45, 38, 45, 53, 44, 39, 59, 63, 47, 35,
    37, 31, 42, 48, 46, 59, 45, 45, 40, 32, 49, 41, 46, 39, 55, 51, 49, 30, 35, 38,
    43, 40, 35, 41, 45, 34, 41, 52, 38, 42, 34, 48, 49, 38, 42, 56, 40, 42, 42, 40,
    32, 45, 54, 39, 46, 46, 54, 43, 42, 41, 58, 48, 35, 41, 37, 43, 40, 33, 35, 42
])

# Số đếm thực tế ghi nhận trong 1 lần đo kéo dài 100s
counts_100s_raw = {
    "Background": 80,
    "Ra-226 (950mm)": 966,
    "Ra-226 (765mm)": 461
}

# ==============================================================================
# 2. HÀM VẼ ĐỒ THỊ HỘI TỤ GIÁ TRỊ TRUNG BÌNH r_c(j) 
# ==============================================================================
def plot_running_average(data, title, save_name):
    fig, ax1 = plt.subplots(figsize=(7.2, 4.5))
    j_vals = np.arange(1, len(data) + 1)
    
    # Trung bình số đếm tích lũy trên mỗi chu kỳ 10s
    r_c_10s = np.cumsum(data) / j_vals
    mean_final = np.mean(data)
    std_final = np.std(data, ddof=1)
    
    # Vẽ dải sai số chuẩn hội tụ +/- sigma / sqrt(j)
    se_band = std_final / np.sqrt(j_vals)
    ax1.fill_between(j_vals, mean_final - se_band, mean_final + se_band,
                     color='#90CAF9', alpha=0.35, label='Dải hội tụ $\\bar{x} \\pm s/\\sqrt{j}$')
    
    ax1.plot(j_vals, r_c_10s, color='#1565C0', marker='o', markersize=3.5,
             linewidth=1.4, label='Trung bình tích lũy $r_c(j)$')
    ax1.axhline(mean_final, color='#D32F2F', linestyle='--', linewidth=1.6,
                label=f'Giá trị hội tụ $\\mu = {mean_final:.2f}$ (counts/10s)')
    
    ax1.set_title(f"Sự hội tụ của tốc độ đếm trung bình: {title}", fontweight='bold', pad=10)
    ax1.set_xlabel('Số lần đo tích lũy ($j$)')
    ax1.set_ylabel('Số đếm trung bình trên 10s ($r_c \\cdot \\Delta t$)')
    ax1.set_xlim(0, 102)
    
    # Trục phụ bên phải hiển thị tốc độ đếm tính theo giây (counts/s) đúng công thức (3)
    ax2 = ax1.twinx()
    y1_min, y1_max = ax1.get_ylim()
    ax2.set_ylim(y1_min / dt, y1_max / dt)
    ax2.set_ylabel('Tốc độ đếm $r_c(j)$ (counts/s)')
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.legend(loc='best', frameon=True, edgecolor='black')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    fig.savefig(f"{save_name}.pdf", bbox_inches='tight')
    fig.savefig(f"{save_name}.png", bbox_inches='tight')
    plt.close(fig)

# ==============================================================================
# 3. HÀM TÍNH CHI-SQUARE & VẼ PHÂN BỐ TẦN SUẤT
# ==============================================================================
def calculate_chi2(obs_freq, expected_freq):
    """
    Tính Chi-square theo đúng Công thức (5) trong tài liệu:
    chi^2 = sum( (x_i - E_i)^2 / sigma_i^2 ) với sigma_i = sqrt(x_i)
    (chỉ tính trên các giá trị số đếm có xuất hiện trong thực nghiệm: obs_freq > 0)
    """
    valid = obs_freq > 0
    return np.sum(((obs_freq[valid] - expected_freq[valid]) ** 2) / obs_freq[valid]), np.sum(valid)

def plot_distribution(data, title, save_name, is_large_mu=False, is_mc=False):
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    
    N_total = len(data)
    exp_mean = np.mean(data)
    exp_var = np.var(data, ddof=1)
    exp_std = np.std(data, ddof=1)
    
    min_val, max_val = int(np.min(data)), int(np.max(data))
    margin = max(2, int((max_val - min_val) * 0.1))
    plot_min = max(0, min_val - margin)
    plot_max = max_val + margin
    
    bins_edges = np.arange(min_val - 0.5, max_val + 1.5, 1)
    bin_centers = np.arange(min_val, max_val + 1)
    counts_obs, _ = np.histogram(data, bins=bins_edges)
    errors_obs = np.sqrt(counts_obs)
    valid = counts_obs > 0
    
    # 1. Vẽ dữ liệu thực nghiệm / mô phỏng kèm thanh sai số
    data_label = f'Mô phỏng Monte Carlo ($N={N_total}$)' if is_mc else f'Thực nghiệm ($N={N_total}$)'
    point_color = '#C62828' if is_mc else 'black'
    ax.errorbar(bin_centers[valid], counts_obs[valid], yerr=errors_obs[valid],
                fmt='o', color=point_color, markersize=4.5, capsize=2.5,
                elinewidth=1.2, label=data_label, zorder=3)
    
    # 2. Vẽ đường lý thuyết Poisson liên tục (dùng hàm Gamma mở rộng cho giai thừa)
    x_smooth = np.linspace(plot_min, plot_max, 600)
    y_poisson_smooth = (np.exp(-exp_mean) * (exp_mean ** x_smooth) / gamma(x_smooth + 1)) * N_total
    ax.plot(x_smooth, y_poisson_smooth, color='#2E7D32', linestyle='-',
            linewidth=2.0, label='Lý thuyết Poisson', zorder=2)
    
    # Tính Chi-square cho Poisson tại các điểm rời rạc
    y_poisson_discrete = poisson.pmf(bin_centers, exp_mean) * N_total
    chi2_poisson, n_bins = calculate_chi2(counts_obs, y_poisson_discrete)
    
    stats_text = (
        f"Số lần thử $N = {N_total}$\n"
        f"Trung bình $\\bar{{x}} = {exp_mean:.2f}$\n"
        f"Phương sai $s^2 = {exp_var:.2f}$\n"
        f"Độ lệch chuẩn $s = {exp_std:.2f}$\n"
        f"$\\sigma_{{lt}} = \\sqrt{{\\bar{{x}}}} = {np.sqrt(exp_mean):.2f}$\n"
        f"$\\chi^2_{{Poisson}} = {chi2_poisson:.2f}$ ({n_bins} bins)"
    )
    
    chi2_gauss = None
    # 3. Vẽ đường lý thuyết Gaussian (Công thức 4: trung bình mu, phương sai mu)
    if is_large_mu:
        sigma_theory = np.sqrt(exp_mean)
        y_gauss_smooth = norm.pdf(x_smooth, loc=exp_mean, scale=sigma_theory) * N_total
        ax.plot(x_smooth, y_gauss_smooth, color='#1565C0', linestyle='--',
                linewidth=2.0, label='Lý thuyết Gaussian', zorder=2)
        
        y_gauss_discrete = norm.pdf(bin_centers, loc=exp_mean, scale=sigma_theory) * N_total
        chi2_gauss, _ = calculate_chi2(counts_obs, y_gauss_discrete)
        stats_text += f"\n$\\chi^2_{{Gauss}} = {chi2_gauss:.2f}$ ({n_bins} bins)"
    
    # Định dạng trục và hộp thông tin
    ax.set_title(title, fontweight='bold', pad=12)
    ax.set_xlabel('Số đếm trong 10s (Counts)')
    ax.set_ylabel('Số lần xuất hiện (Number of events)')
    
    ax.text(0.97, 0.96, stats_text, transform=ax.transAxes, fontsize=9.5,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='square,pad=0.5', facecolor='white', edgecolor='black', alpha=0.95))
    
    max_y = np.max(counts_obs[valid] + errors_obs[valid])
    ax.set_ylim(0, max(max_y, np.max(y_poisson_smooth)) * 1.38)
    ax.set_xlim(plot_min, plot_max)
    
    span = plot_max - plot_min
    tick_step = 2 if span <= 25 else (5 if span <= 50 else 10)
    ax.set_xticks(np.arange((plot_min // tick_step) * tick_step, plot_max + 1, tick_step))
    
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which='minor', direction='in', top=True, right=True)
    ax.legend(loc='upper left', frameon=True, edgecolor='black')
    ax.grid(axis='both', linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    fig.savefig(f"{save_name}.pdf", bbox_inches='tight')
    fig.savefig(f"{save_name}.png", bbox_inches='tight')
    plt.close(fig)
    
    return {
        "N": N_total,
        "mean": exp_mean,
        "var": exp_var,
        "std": exp_std,
        "theo_std": np.sqrt(exp_mean),
        "chi2_pois": chi2_poisson,
        "chi2_gauss": chi2_gauss,
        "n_bins": n_bins
    }

# ==============================================================================
# 4. HÀM MÔ PHỎNG MONTE CARLO 
# ==============================================================================
def run_monte_carlo_simulation(mu_val, n_trials, label_title, save_prefix, is_large_mu=False):
    """
    Mô phỏng phân bố Poisson bằng phương pháp Monte Carlo với n_trials = 1000 lần thử,
    xuất file CSV giống hàm poissonsim(mu, N) và vẽ đồ thị so sánh.
    """
    sim_data = np.random.poisson(lam=mu_val, size=n_trials)
    
    # Lưu file CSV gồm 2 cột: Số đếm (Counts) và Tần suất (Frequency)
    vals, freqs = np.unique(sim_data, return_counts=True)
    csv_data = np.column_stack((vals, freqs))
    np.savetxt(f"{save_prefix}.csv", csv_data, delimiter=",", fmt="%d",
               header="Counts,Frequency", comments="")
    
    mc_stats = plot_distribution(
        sim_data,
        title=f"Mô phỏng Monte Carlo ($N={n_trials}$, $\\mu={mu_val:.2f}$): {label_title}",
        save_name=save_prefix,
        is_large_mu=is_large_mu,
        is_mc=True
    )
    return mc_stats

# ==============================================================================
# 5. HÀM PHÂN TÍCH THỐNG KÊ
# ==============================================================================
def full_statistical_analysis(data, name, count_100s, dist_stats, mc_stats):
    N = len(data)
    mean_val = dist_stats["mean"]
    var_val = dist_stats["var"]
    std_val = dist_stats["std"]
    sem_val = std_val / np.sqrt(N)
    theo_std = dist_stats["theo_std"]
    
    # Quy đổi phép đo 100s về chu kỳ 10s
    ref_mean_10s = count_100s / 10.0
    # Độ lệch chuẩn lý thuyết của 1 lần đo 10s ước lượng từ phép đo 100s
    ref_single_std = np.sqrt(ref_mean_10s)
    # Sai số chuẩn của phép đo 100s khi quy về 10s: sqrt(N_100s) / 10
    ref_sem_10s = np.sqrt(count_100s) / 10.0
    
    # Độ lệch giữa trung bình 100 lần đo (10s) và phép đo đối chứng (100s)
    diff_mean = abs(mean_val - ref_mean_10s)
    diff_in_s = diff_mean / std_val
    diff_in_sem = diff_mean / sem_val
    combined_se = np.sqrt(sem_val**2 + ref_sem_10s**2)
    diff_in_comb_se = diff_mean / combined_se
    
    lines = []
    lines.append(f"\n======================================================================")
    lines.append(f" BÁO CÁO PHÂN TÍCH CHI TIẾT: {name.upper()}")
    lines.append(f"======================================================================")
    lines.append(f"1. ĐẶC TRƯNG THỐNG KÊ THỰC NGHIỆM (N = {N} lần đo, dt = 10s):")
    lines.append(f"   - Giá trị trung bình thực nghiệm (x_bar)  : {mean_val:.2f} counts/10s (tốc độ đếm r = {mean_val/10:.3f} cps)")
    lines.append(f"   - Phương sai thực nghiệm (s^2)            : {var_val:.2f}")
    lines.append(f"   - Phương sai lý thuyết Poisson (sigma^2)  : {mean_val:.2f} (Tỉ số s^2 / x_bar = {var_val/mean_val:.3f})")
    lines.append(f"   - Độ lệch chuẩn thực nghiệm (s)           : {std_val:.2f}")
    lines.append(f"   - Độ lệch chuẩn lý thuyết (sqrt(x_bar))   : {theo_std:.2f}")
    lines.append(f"   - Sai số chuẩn của trị trung bình (SE)    : {sem_val:.3f}")
    
    lines.append(f"\n2. SO SÁNH VỚI PHÉP ĐO ĐỐI CHỨNG 100 GIÂY (Mục 2.3.2):")
    lines.append(f"   - Tổng số đếm đo trong 100s               : {count_100s} counts")
    lines.append(f"   - Trung bình quy đổi về 10s (mu_100s)     : {ref_mean_10s:.2f} counts/10s")
    lines.append(f"   - Độ lệch chuẩn 1 lần đo 10s (sqrt(mu))   : {ref_single_std:.2f} (so với s = {std_val:.2f})")
    lines.append(f"   - Sai số chuẩn của mu_100s (sqrt(N)/10)   : {ref_sem_10s:.2f}")
    lines.append(f"   - Độ lệch tuyệt đối |x_bar - mu_100s|     : {diff_mean:.2f} counts")
    lines.append(f"     + Tính theo độ lệch chuẩn mẫu (s)       : {diff_in_s:.2f} s  (< 1 sigma -> Rất phù hợp)")
    lines.append(f"     + Tính theo sai số chuẩn tổng hợp       : {diff_in_comb_se:.2f} sigma_diff")
    lines.append(f"     + Tính theo SE của mẫu (s/sqrt(100))    : {diff_in_sem:.2f} SE")
    
    if name == "Đo phông":
        max_val = np.max(data)
        min_val = np.min(data)
        max_dev = max(abs(max_val - mean_val), abs(min_val - mean_val))
        # Xác suất Poisson để có số đếm >= 16 khi mu = 6.83
        prob_single = 1.0 - poisson.cdf(max_val - 1, mean_val)
        expected_in_100 = N * prob_single
        prob_at_least_one = 1.0 - (1.0 - prob_single) ** N
        
        lines.append(f"\n3. KIỂM TRA ĐỘ LỆCH LỚN NHẤT - MAX DEVIATION (Mục 2.3.4):")
        lines.append(f"   - Giá trị cực đại quan sát được (x_max)   : {max_val} counts (xuất hiện {np.sum(data == max_val)} lần)")
        lines.append(f"   - Độ lệch lớn nhất |x_max - x_bar|        : {max_dev:.2f} counts")
        lines.append(f"   - Quy đổi theo độ lệch chuẩn thực nghiệm  : {max_dev / std_val:.2f} s (~ 3.1 sigma)")
        lines.append(f"   - Quy đổi theo độ lệch chuẩn lý thuyết    : {max_dev / theo_std:.2f} sqrt(mu)")
        lines.append(f"   - Xác suất P(X >= {max_val}) trong 1 lần đo      : {prob_single:.5f} ({prob_single*100:.3f}%)")
        lines.append(f"   - Số lần kỳ vọng trong 100 cơ hội         : {expected_in_100:.2f} lần")
        lines.append(f"   - Xác suất xuất hiện >= 1 lần trong 100 đo: {prob_at_least_one*100:.2f}%")
        lines.append(f"   => KẾT LUẬN: Trong 100 phép đo, độ lệch ~3 sigma hoàn toàn nằm trong kỳ vọng thống kê.")
        lines.append(f"                Hệ đếm Geiger-Muller hoạt động ổn định, bình thường.")
        
    lines.append(f"\n4. KIỂM ĐỊNH CHI-SQUARE & SO SÁNH POISSON vs GAUSSIAN (Mục 2.3.5 & 4.3):")
    lines.append(f"   - Số mức số đếm xuất hiện (Số bins)       : {dist_stats['n_bins']}")
    lines.append(f"   - Chi-square Poisson (chi^2_Pois)         : {dist_stats['chi2_pois']:.2f}")
    if dist_stats["chi2_gauss"] is not None:
        lines.append(f"   - Chi-square Gaussian (chi^2_Gauss)       : {dist_stats['chi2_gauss']:.2f}")
        better = "Poisson" if dist_stats["chi2_pois"] < dist_stats["chi2_gauss"] else "Gaussian"
        lines.append(f"   => ĐÁNH GIÁ: Phân bố {better} có giá trị Chi-square nhỏ hơn ({min(dist_stats['chi2_pois'], dist_stats['chi2_gauss']):.2f} vs {max(dist_stats['chi2_pois'], dist_stats['chi2_gauss']):.2f}),")
        lines.append(f"                do đó phù hợp với dữ liệu thực nghiệm tốt hơn.")
        
    lines.append(f"\n5. SO SÁNH VỚI MÔ PHỎNG MONTE CARLO (N_sim = 1000 - Mục 3.1 & 4.4):")
    lines.append(f"   - Trung bình Monte Carlo (x_bar_MC)       : {mc_stats['mean']:.2f} (Thực nghiệm: {mean_val:.2f})")
    lines.append(f"   - Phương sai Monte Carlo (s^2_MC)         : {mc_stats['var']:.2f} (Thực nghiệm: {var_val:.2f})")
    lines.append(f"   - Độ lệch chuẩn Monte Carlo (s_MC)        : {mc_stats['std']:.2f} (Lý thuyết sqrt(mu): {theo_std:.2f})")
    lines.append(f"   - Chi-square Poisson (Monte Carlo)        : {mc_stats['chi2_pois']:.2f} ({mc_stats['n_bins']} bins)")
    
    return "\n".join(lines), {
        "mean": mean_val, "var": var_val, "std": std_val, "theo_std": theo_std,
        "sem": sem_val, "ref_100s": ref_mean_10s, "ref_single_std": ref_single_std,
        "diff": diff_mean, "diff_s": diff_in_s, "diff_sem": diff_in_sem,
        "chi2_p": dist_stats["chi2_pois"], "chi2_g": dist_stats["chi2_gauss"],
        "mc_mean": mc_stats["mean"], "mc_var": mc_stats["var"], "mc_std": mc_stats["std"]
    }

# ==============================================================================
# 6. CHẠY TOÀN BỘ CHƯƠNG TRÌNH VÀ XUẤT KẾT QUẢ
# ==============================================================================
if __name__ == "__main__":
    print(">>> 1. Đang vẽ các đồ thị hội tụ r_c(j) (Mục 2.3.1)...")
    plot_running_average(bg_counts, "Đo phông (Background)", "Fig_1_Conv_BG")
    plot_running_average(ra_950mm, "Nguồn Ra-226 ($d = 950$ mm)", "Fig_1_Conv_Ra950")
    plot_running_average(ra_765mm, "Nguồn Ra-226 ($d = 765$ mm, chắn mica)", "Fig_1_Conv_Ra765")

    print(">>> 2. Đang vẽ các đồ thị phân bố thực nghiệm vs Lý thuyết (Mục 2.3.3 & 2.3.5)...")
    stats_bg = plot_distribution(bg_counts, "Phân bố tần suất số đếm: Đo phông", "Fig_2_Dist_BG", is_large_mu=False)
    stats_950 = plot_distribution(ra_950mm, "Phân bố tần suất số đếm: Ra-226 ($d = 950$ mm)", "Fig_2_Dist_Ra950", is_large_mu=True)
    stats_765 = plot_distribution(ra_765mm, "Phân bố tần suất số đếm: Ra-226 ($d = 765$ mm, chắn mica)", "Fig_2_Dist_Ra765", is_large_mu=True)

    print(">>> 3. Đang chạy mô phỏng Monte Carlo N = 1000 lần thử (Mục 3.1 & 4.4)...")
    mc_bg = run_monte_carlo_simulation(stats_bg["mean"], 1000, "Đo phông", "Fig_3_MC_BG", is_large_mu=False)
    mc_950 = run_monte_carlo_simulation(stats_950["mean"], 1000, "Ra-226 ($d = 950$ mm)", "Fig_3_MC_Ra950", is_large_mu=True)
    mc_765 = run_monte_carlo_simulation(stats_765["mean"], 1000, "Ra-226 ($d = 765$ mm)", "Fig_3_MC_Ra765", is_large_mu=True)

    print(">>> 4. Đang tổng hợp báo cáo thống kê và xuất mã bảng LaTeX...")
    rep_bg, sum_bg = full_statistical_analysis(bg_counts, "Đo phông", counts_100s_raw["Background"], stats_bg, mc_bg)
    rep_950, sum_950 = full_statistical_analysis(ra_950mm, "Ra-226 (950mm)", counts_100s_raw["Ra-226 (950mm)"], stats_950, mc_950)
    rep_765, sum_765 = full_statistical_analysis(ra_765mm, "Ra-226 (765mm)", counts_100s_raw["Ra-226 (765mm)"], stats_765, mc_765)

    latex_table = f"""
% =====================================================================
% MÃ LATEX BẢNG 2 ĐÃ CẬP NHẬT ĐẦY ĐỦ CÁC ĐẠI LƯỢNG THỐNG KÊ, CHI^2 & MC
% =====================================================================
\\begin{{table}}[H]
    \\centering
    \\scalebox{{0.92}}{{
        \\begin{{tabular}}{{|p{{7.2cm}}|c|c|c|}}
            \\hline
            \\textbf{{Đại lượng thống kê}} & \\textbf{{Đo phông}} & \\textbf{{Ra-226 (950mm)}} & \\textbf{{Ra-226 (765mm)}} \\\\
            \\hline
            Giá trị trung bình thực nghiệm ($\\bar{{x}} = \\mu$) & {sum_bg['mean']:.2f} & {sum_950['mean']:.2f} & {sum_765['mean']:.2f} \\\\
            Phương sai thực nghiệm ($s^2$) & {sum_bg['var']:.2f} & {sum_950['var']:.2f} & {sum_765['var']:.2f} \\\\
            Phương sai lý thuyết Poisson ($\\sigma^2 = \\mu$) & {sum_bg['mean']:.2f} & {sum_950['mean']:.2f} & {sum_765['mean']:.2f} \\\\
            Độ lệch chuẩn thực nghiệm ($s = \\sqrt{{s^2}}$) & {sum_bg['std']:.2f} & {sum_950['std']:.2f} & {sum_765['std']:.2f} \\\\
            Độ lệch chuẩn lý thuyết ($\\sigma_{{lt}} = \\sqrt{{\\mu}}$) & {sum_bg['theo_std']:.2f} & {sum_950['theo_std']:.2f} & {sum_765['theo_std']:.2f} \\\\
            Sai số chuẩn của trị trung bình ($SE = s/\\sqrt{{100}}$) & {sum_bg['sem']:.2f} & {sum_950['sem']:.2f} & {sum_765['sem']:.2f} \\\\
            \\hline
            Số đếm đối chứng (đo 100s, quy đổi về 10s: $\\mu_{{100s}}$) & {sum_bg['ref_100s']:.2f} & {sum_950['ref_100s']:.2f} & {sum_765['ref_100s']:.2f} \\\\
            Độ lệch chuẩn phép đo đối chứng ($\\sigma_{{100s}} = \\sqrt{{\\mu_{{100s}}}}$) & {sum_bg['ref_single_std']:.2f} & {sum_950['ref_single_std']:.2f} & {sum_765['ref_single_std']:.2f} \\\\
            Độ lệch tuyệt đối ($\\Delta x = |\\bar{{x}} - \\mu_{{100s}}|$) & {sum_bg['diff']:.2f} & {sum_950['diff']:.2f} & {sum_765['diff']:.2f} \\\\
            Độ lệch tương đối theo độ lệch chuẩn mẫu ($\\Delta x / s$) & {sum_bg['diff_s']:.2f}$s$ & {sum_950['diff_s']:.2f}$s$ & {sum_765['diff_s']:.2f}$s$ \\\\
            Độ lệch tương đối theo sai số chuẩn ($\\Delta x / SE$) & {sum_bg['diff_sem']:.2f} SE & {sum_950['diff_sem']:.2f} SE & {sum_765['diff_sem']:.2f} SE \\\\
            \\hline
            Độ lệch cực đại so với $\\bar{{x}}$ (Max deviation) & 9.17 ($3.06s$) & -- & -- \\\\
            \\hline
            Hệ số kiểm định $\\chi^2$ (Phân bố Poisson) & {sum_bg['chi2_p']:.2f} & {sum_950['chi2_p']:.2f} & {sum_765['chi2_p']:.2f} \\\\
            Hệ số kiểm định $\\chi^2$ (Phân bố Gaussian) & -- & {sum_950['chi2_g']:.2f} & {sum_765['chi2_g']:.2f} \\\\
            \\hline
            Mô phỏng Monte Carlo ($N=1000$): $\\bar{{x}}_{{MC}} \\pm s_{{MC}}$ & ${sum_bg['mc_mean']:.2f} \\pm {sum_bg['mc_std']:.2f}$ & ${sum_950['mc_mean']:.2f} \\pm {sum_950['mc_std']:.2f}$ & ${sum_765['mc_mean']:.2f} \\pm {sum_765['mc_std']:.2f}$ \\\\
            \\hline
        \\end{{tabular}}
    }}
    \\vspace{{0.2cm}}
    \\caption{{Bảng tổng hợp các đại lượng thống kê phân tích, kiểm định $\\chi^2$ và mô phỏng Monte Carlo}}
\\end{{table}}
"""

    full_output = rep_bg + "\n" + rep_950 + "\n" + rep_765 + "\n\n" + latex_table
    print(full_output)

    with open("Bao_cao_Thong_ke.txt", "w", encoding="utf-8") as f:
        f.write("=== BÁO CÁO PHÂN TÍCH THỐNG KÊ PHÂN RÃ PHÓNG XẠ ===\n")
        f.write(full_output)

    print("\n>>> HOÀN TẤT! Đã xuất 9 file đồ thị (.pdf & .png), 3 file .csv mô phỏng và file 'Bao_cao_Thong_ke.txt'.")
