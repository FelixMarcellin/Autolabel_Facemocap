# -*- coding: utf-8 -*-
"""
VERSION 6.2 - COMPLETE & JOURNAL READY
- Restore FULL .txt report (All metrics)
- Restore FULL .csv export (All files)
- Keep High-Visibility Figure (XL Fonts)
"""

import os
import numpy as np
import pandas as pd
from tkinter import Tk, filedialog
from scipy import stats
import matplotlib.pyplot as plt

try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
except:
    pass

# --- CONFIGURATION GRAPHIQUE (Journal Style) ---
FONT_SIZE = 18
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.labelsize': FONT_SIZE + 2,
    'axes.titlesize': FONT_SIZE + 4,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'axes.linewidth': 2,
    'savefig.dpi': 300,
    'font.family': 'sans-serif'
})

def compute_icc(x, y):
    data = np.vstack([x, y]).T
    n, k = data.shape
    ms_between = (k * np.sum((np.mean(data, axis=1) - np.mean(data))**2)) / (n - 1)
    ms_error = (np.sum((data - np.mean(data))**2) - (ms_between * (n - 1))) / (n * (k - 1))
    return (ms_between - ms_error) / (ms_between + (k - 1) * ms_error)

def create_publication_dashboard(nx_s, py_s, file_rmse_data, res, pct_outliers):
    diff = nx_s - py_s
    rmse_values = [x['RMSE_mm'] for x in file_rmse_data]
    m_rmse, s_rmse = np.mean(rmse_values), np.std(rmse_values, ddof=1)
    
    fig = plt.figure(figsize=(20, 14))
    
    # A. DISTRIBUTION
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(diff, bins=40, color='#a1c9f4', edgecolor='black', lw=1.5)
    ax1.set_title('A. Distribution')

    # B. QQ-PLOT
    ax2 = plt.subplot(2, 3, 2)
    stats.probplot(diff, dist="norm", plot=ax2)
    ax2.set_title('B. Q-Q Plot')

    # C. BLAND-ALTMAN (Annotations XL)
    ax3 = plt.subplot(2, 3, 3)
    avg = (nx_s + py_s) / 2
    ax3.axhline(res['mean_diff'], color='red', lw=4)
    ax3.axhline(res['loa'][0], color='red', ls='--', lw=3)
    ax3.axhline(res['loa'][1], color='red', ls='--', lw=3)
    ax3.scatter(avg, diff, color='#08519c', alpha=0.4, s=25)
    
    bbox = dict(boxstyle="round,pad=0.2", fc="white", ec="red", lw=2)
    ax3.text(avg.max()*1.05, res['mean_diff'], f"Bias: {res['mean_diff']:.2f}", color='red', fontweight='bold', bbox=bbox)
    ax3.text(avg.max()*1.05, res['loa'][1], f"ULoA: {res['loa'][1]:.1f}", color='red', bbox=bbox)
    ax3.text(avg.max()*1.05, res['loa'][0], f"LLoA: {res['loa'][0]:.1f}", color='red', bbox=bbox)
    ax3.set_title('C. Bland-Altman')

    # D. RMSE BOXPLOT
    ax4 = plt.subplot(2, 3, 4)
    bp = ax4.boxplot(rmse_values, patch_artist=True, widths=0.5, medianprops=dict(lw=4, color='red'))
    plt.setp(bp['boxes'], facecolor='#deebf7')
    med, q1, q3 = np.median(rmse_values), np.percentile(rmse_values, 25), np.percentile(rmse_values, 75)
    ax4.text(1.3, med, f'Med: {med:.1f}', color='red', fontweight='bold')
    ax4.text(1.3, q1, f'Q1: {q1:.1f}', color='#003366')
    ax4.text(1.3, q3, f'Q3: {q3:.1f}', color='#003366')
    ax4.set_title(f'D. RMSE\n{m_rmse:.2f} ± {s_rmse:.2f} mm')
    ax4.set_xlim(0.6, 2.0); ax4.set_xticks([])

    # E. CORRELATION
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(nx_s, py_s, alpha=0.3, s=25, color='#08519c')
    lims = [min(nx_s.min(), py_s.min()), max(nx_s.max(), py_s.max())]
    ax5.plot(lims, lims, 'r--', lw=3)
    ax5.set_title(f"E. r = {res['corr']:.4f}")

    # F. SIGNIFICANCE (Labels XL)
    ax6 = plt.subplot(2, 3, 6)
    names = ['t-test', 'Wilcox', 'Cohen |d|']
    vals = [res['p_t'], res['p_w'], abs(res['cohen'])]
    colors = ['#4daf4a' if v > 0.05 or (i==2 and v < 0.2) else '#e41a1c' for i, v in enumerate(vals)]
    bars = ax6.bar(names, [max(0.015, v) for v in vals], color=colors, edgecolor='black', lw=2)
    ax6.axhline(0.05, color='black', ls='--', lw=2)
    ax6.set_ylim(0, max(1.2, abs(res['cohen'])+0.2))
    for bar, v in zip(bars, vals):
        label = f"{v:.3f}" if v > 0.001 else "<.001"
        ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02, label, ha='center', fontweight='bold')

    plt.tight_layout(pad=3.0)
    plt.savefig('Scientific_Figure_300DPI.png', bbox_inches='tight')
    plt.close()

def main():
    root = Tk(); root.withdraw()
    path_n = filedialog.askdirectory(title="Select Nexus Folder")
    path_p = filedialog.askdirectory(title="Select Python Folder")
    if not path_n or not path_p: return

    all_nx, all_py, file_rmse_list = [], [], []
    total_raw, total_excl = 0, 0

    files = [f for f in os.listdir(path_n) if f.lower().endswith(".csv")]
    for f in files:
        p_p = os.path.join(path_p, f.replace(".csv", "_labeled.csv"))
        if os.path.exists(p_p):
            try:
                dn = pd.read_csv(os.path.join(path_n, f), skiprows=7).iloc[:, 2:]
                dp = pd.read_csv(p_p, skiprows=7).iloc[:, 2:]
                n, p = np.nan_to_num(dn.to_numpy()).flatten(), np.nan_to_num(dp.to_numpy()).flatten()
                ml = min(len(n), len(p)); n, p = n[:ml], p[:ml]
                mask = np.abs(n - p) < 100.0; nc, pc = n[mask], p[mask]
                if len(nc) > 0:
                    rmse = np.sqrt(np.mean((nc - pc)**2))
                    file_rmse_list.append({"Filename": f, "RMSE_mm": rmse})
                    all_nx.append(nc); all_py.append(pc)
                    total_raw += len(n); total_excl += (len(n)-len(nc))
            except: continue

    if not all_nx: return

    nx_full, py_full = np.concatenate(all_nx), np.concatenate(all_py)
    idx = np.random.choice(len(nx_full), min(5000, len(nx_full)), replace=False)
    nx_s, py_s = nx_full[idx], py_full[idx]
    
    _, pt = stats.ttest_rel(nx_s, py_s); _, pw = stats.wilcoxon(nx_s[:2000], py_s[:2000]); r, _ = stats.pearsonr(nx_s, py_s)
    bias, sd_diff = np.mean(nx_s - py_s), np.std(nx_s - py_s, ddof=1)
    res = {'p_t': pt, 'p_w': pw, 'corr': r, 'icc': compute_icc(nx_s, py_s), 'cohen': (bias/sd_diff),
           'mean_diff': bias, 'std_diff': sd_diff, 'loa': (bias - 1.96*sd_diff, bias + 1.96*sd_diff)}

    # --- RESTAURATION DES EXPORTS ---
    rmse_v = [x['RMSE_mm'] for x in file_rmse_list]
    pd.DataFrame(file_rmse_list).to_csv("summary_rmse.csv", index=False, sep=";") # CSV RESTAURÉ
    
    with open("statistical_report.txt", "w") as f_rep: # RAPPORT COMPLET RESTAURÉ
        f_rep.write("STATISTICAL VALIDATION REPORT\n=============================\n\n")
        f_rep.write(f"RMSE STATS (n={len(file_rmse_list)} files):\n")
        f_rep.write(f"- Mean:   {np.mean(rmse_v):.4f} mm\n- Median: {np.median(rmse_v):.4f} mm\n")
        f_rep.write(f"- Q1:     {np.percentile(rmse_v, 25):.4f} mm\n- Q3:     {np.percentile(rmse_v, 75):.4f} mm\n")
        f_rep.write(f"- SD:     {np.std(rmse_v, ddof=1):.4f} mm\n\n")
        f_rep.write(f"AGREEMENT & CORRELATION:\n")
        f_rep.write(f"- Bias:   {res['mean_diff']:.4f} mm\n- Upper LoA: {res['loa'][1]:.4f} mm\n")
        f_rep.write(f"- Lower LoA: {res['loa'][0]:.4f} mm\n- ICC:    {res['icc']:.4f}\n")
        f_rep.write(f"- Pearson r: {res['corr']:.4f}\n- Cohen's d: {res['cohen']:.4f}\n\n")
        f_rep.write(f"P-VALUES:\n- t-test: {res['p_t']:.4e}\n- Wilcoxon: {res['p_w']:.4e}\n")

    create_publication_dashboard(nx_s, py_s, file_rmse_list, res, (total_excl/total_raw)*100)
    print(f"✅ Terminé. Tout est dans : {os.getcwd()}")

if __name__ == "__main__":
    main()