# -*- coding: utf-8 -*-
"""
Statistical comparison between automated (Python) and manual (Nexus)
facial marker labeling.

Computes:
- Labeling accuracy (frame-wise comparison with reference, threshold 20 mm)
- RMSE per file
- Pearson correlation, ICC(2,1), Cohen's d
- Bland-Altman analysis
- Statistical tests (t-test, Wilcoxon)
- Publication-ready figure

Author: Félix Marcellin
"""

import os
import numpy as np
import pandas as pd
from tkinter import Tk, filedialog
from scipy import stats
import matplotlib.pyplot as plt
import logging

try:
    import pingouin as pg
    PINGOUIN_AVAILABLE = True
except ImportError:
    PINGOUIN_AVAILABLE = False
    print("Warning: pingouin not installed. ICC(2,1) will not be computed.")

# =====================================================
# OUTPUT DIRECTORY (fixed path)
# =====================================================
OUTPUT_DIR = r"C:\Users\felima\Documents\Python Scripts\auto label nexus\Stat\stat v3 sain patient"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "statistical_analysis.log"), mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =====================================================
# REPRODUCIBILITY
# =====================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =====================================================
# GRAPH CONFIGURATION (Journal Style)
# =====================================================
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

# =====================================================
# PARAMETERS
# =====================================================
# Distance threshold (mm) between Nexus and Python coordinates.
# Differences above this threshold are considered outliers and excluded
# from statistical analysis (physiologically implausible facial motion).
OUTLIER_THRESHOLD_MM = 100.0

# Distance threshold (mm) for a marker to be considered "correctly labeled"
# in the frame-wise accuracy computation. Corresponds to approximately
# twice the marker diameter.
LABELING_CORRECT_THRESHOLD_MM = 5

# Subsample sizes for statistical analysis (computational efficiency).
SUBSAMPLE_SIZE = 5000
WILCOXON_SUBSAMPLE_SIZE = 2000


# =====================================================
# ICC(2,1) — TWO-WAY RANDOM EFFECTS, ABSOLUTE AGREEMENT
# =====================================================
def compute_icc_2_1(x, y):
    """
    Compute ICC(2,1) using pingouin if available.
    Falls back to a manual implementation if pingouin is not installed.

    ICC(2,1) is the appropriate model for assessing absolute agreement
    between two methods of measurement (Shrout & Fleiss, 1979).
    """
    if PINGOUIN_AVAILABLE:
        n = len(x)
        df = pd.DataFrame({
            'target': np.repeat(np.arange(n), 2),
            'rater': np.tile(['Nexus', 'Python'], n),
            'rating': np.column_stack([x, y]).flatten()
        })
        icc_table = pg.intraclass_corr(data=df, targets='target',
                                       raters='rater', ratings='rating')
        icc_value = icc_table.loc[icc_table['Type'] == 'ICC2', 'ICC'].values[0]
        return icc_value, 'ICC(2,1) via pingouin'
    else:
        data = np.vstack([x, y]).T
        n, k = data.shape
        grand_mean = np.mean(data)
        ms_rows = k * np.sum((np.mean(data, axis=1) - grand_mean)**2) / (n - 1)
        ms_cols = n * np.sum((np.mean(data, axis=0) - grand_mean)**2) / (k - 1)
        ms_error = (np.sum((data - np.mean(data, axis=1, keepdims=True) -
                            np.mean(data, axis=0, keepdims=True) + grand_mean)**2)
                    / ((n - 1) * (k - 1)))
        icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error +
                                       k * (ms_cols - ms_error) / n)
        return icc, 'ICC(2,1) manual fallback'


# =====================================================
# PUBLICATION DASHBOARD
# =====================================================
def create_publication_dashboard(nx_s, py_s, file_rmse_data, res, pct_outliers,
                                 labeling_accuracy, total_compared, total_correct):
    diff = nx_s - py_s
    rmse_values = [x['RMSE_mm'] for x in file_rmse_data]
    m_rmse, s_rmse = np.mean(rmse_values), np.std(rmse_values, ddof=1)

    fig = plt.figure(figsize=(20, 14))

    # A. DISTRIBUTION
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(diff, bins=40, color='#a1c9f4', edgecolor='black', lw=1.5)
    ax1.set_title('A. Distribution of Differences')
    ax1.set_xlabel('Difference (Nexus − Python) [mm]')
    ax1.set_ylabel('Frequency')

    # B. QQ-PLOT
    ax2 = plt.subplot(2, 3, 2)
    stats.probplot(diff, dist="norm", plot=ax2)
    ax2.set_title('B. Normal Q–Q Plot')
    ax2.set_xlabel('Theoretical Quantiles')
    ax2.set_ylabel('Sample Quantiles')

    # C. BLAND-ALTMAN
    ax3 = plt.subplot(2, 3, 3)
    avg = (nx_s + py_s) / 2
    ax3.axhline(res['mean_diff'], color='red', lw=4)
    ax3.axhline(res['loa'][0], color='red', ls='--', lw=3)
    ax3.axhline(res['loa'][1], color='red', ls='--', lw=3)
    ax3.scatter(avg, diff, color='#08519c', alpha=0.4, s=25)

    bbox = dict(boxstyle="round,pad=0.2", fc="white", ec="red", lw=2)
    ax3.text(avg.max()*1.05, res['mean_diff'],
             f"Bias: {res['mean_diff']:.2f}",
             color='red', fontweight='bold', bbox=bbox)
    ax3.text(avg.max()*1.05, res['loa'][1],
             f"Upper LoA: {res['loa'][1]:.1f}",
             color='red', bbox=bbox)
    ax3.text(avg.max()*1.05, res['loa'][0],
             f"Lower LoA: {res['loa'][0]:.1f}",
             color='red', bbox=bbox)

    ax3.set_title('C. Bland–Altman Plot')
    ax3.set_xlabel('Mean of Measurements (Nexus & Python) [mm]')
    ax3.set_ylabel('Difference (Nexus − Python) [mm]')

    # D. RMSE BOXPLOT
    ax4 = plt.subplot(2, 3, 4)
    bp = ax4.boxplot(rmse_values, patch_artist=True, widths=0.5,
                     medianprops=dict(lw=4, color='red'))
    plt.setp(bp['boxes'], facecolor='#deebf7')

    med = np.median(rmse_values)
    q1 = np.percentile(rmse_values, 25)
    q3 = np.percentile(rmse_values, 75)

    ax4.text(1.3, med, f'Median: {med:.1f}', color='red', fontweight='bold')
    ax4.text(1.3, q1, f'Q1: {q1:.1f}', color='#003366')
    ax4.text(1.3, q3, f'Q3: {q3:.1f}', color='#003366')

    ax4.set_title(f'D. RMSE Distribution\nMean = {m_rmse:.2f} ± {s_rmse:.2f} mm')
    ax4.set_ylabel('RMSE [mm]')
    ax4.set_xlim(0.6, 2.0)
    ax4.set_xticks([])

    # E. CORRELATION
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(nx_s, py_s, alpha=0.3, s=25, color='#08519c')
    lims = [min(nx_s.min(), py_s.min()), max(nx_s.max(), py_s.max())]
    ax5.plot(lims, lims, 'r--', lw=3)
    ax5.set_title(f'E. Correlation (r = {res["corr"]:.4f})')
    ax5.set_xlabel('Nexus Measurements [mm]')
    ax5.set_ylabel('Python Measurements [mm]')

    # F. SIGNIFICANCE
    ax6 = plt.subplot(2, 3, 6)
    names = ['t-test', 'Wilcoxon', '|Cohen\'s d|']
    vals = [res['p_t'], res['p_w'], abs(res['cohen'])]

    colors = ['#4daf4a' if v > 0.05 or (i == 2 and v < 0.2)
              else '#e41a1c' for i, v in enumerate(vals)]

    bars = ax6.bar(names, [max(0.015, v) for v in vals],
                   color=colors, edgecolor='black', lw=2)
    ax6.axhline(0.05, color='black', ls='--', lw=2)
    ax6.set_ylim(0, max(1.2, abs(res['cohen']) + 0.2))

    for bar, v in zip(bars, vals):
        label = f"{v:.3f}" if v > 0.001 else "<0.001"
        ax6.text(bar.get_x() + bar.get_width()/2.,
                 bar.get_height() + 0.02,
                 label, ha='center', fontweight='bold')

    ax6.set_title('F. Statistical Significance')
    ax6.set_ylabel('p-value / Effect Size')

    plt.tight_layout(pad=3.0)
    fig_path = os.path.join(OUTPUT_DIR, 'Scientific_Figure_300DPI.png')
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

    logger.info(f"Figure saved: {fig_path}")
    logger.info(f"Labeling accuracy (threshold {LABELING_CORRECT_THRESHOLD_MM} mm): "
                f"{labeling_accuracy:.2%} "
                f"({total_correct}/{total_compared})")


# =====================================================
# MAIN
# =====================================================
def main():
    root = Tk()
    root.withdraw()
    path_n = filedialog.askdirectory(title="Select Nexus Folder")
    path_p = filedialog.askdirectory(title="Select Python Folder")
    if not path_n or not path_p:
        return

    all_nx, all_py, file_rmse_list = [], [], []
    total_raw, total_excl = 0, 0
    total_compared, total_correct = 0, 0
    ignored_files = []

    files = [f for f in os.listdir(path_n) if f.lower().endswith(".csv")]
    processed, failed = 0, []

    for f in files:
        p_p = os.path.join(path_p, f.replace(".csv", "_labeled.csv"))

        if not os.path.exists(p_p):
            ignored_files.append(f)
            continue

        try:
            dn = pd.read_csv(os.path.join(path_n, f), skiprows=7).iloc[:, 2:]
            dp = pd.read_csv(p_p, skiprows=7).iloc[:, 2:]

            n = np.nan_to_num(dn.to_numpy()).flatten()
            p = np.nan_to_num(dp.to_numpy()).flatten()

            ml = min(len(n), len(p))
            n, p = n[:ml], p[:ml]

            # Exclude outliers (difference > OUTLIER_THRESHOLD_MM)
            mask = np.abs(n - p) < OUTLIER_THRESHOLD_MM
            nc, pc = n[mask], p[mask]

            if len(nc) > 0:
                rmse = np.sqrt(np.mean((nc - pc)**2))
                file_rmse_list.append({"Filename": f, "RMSE_mm": rmse})

                all_nx.append(nc)
                all_py.append(pc)

                total_raw += len(n)
                total_excl += (len(n) - len(nc))

                # Frame-wise labeling accuracy
                correct = np.sum(np.abs(nc - pc) < LABELING_CORRECT_THRESHOLD_MM)
                total_correct += correct
                total_compared += len(nc)

            processed += 1

        except Exception as e:
            logger.error(f"Error processing {f}: {e}")
            failed.append((f, str(e)))
            continue

    if not all_nx:
        logger.error("No valid data found. Exiting.")
        return

    nx_full = np.concatenate(all_nx)
    py_full = np.concatenate(all_py)

    # Subsample for statistical analysis (reproducible)
    n_sub = min(SUBSAMPLE_SIZE, len(nx_full))
    idx = np.random.choice(len(nx_full), n_sub, replace=False)
    nx_s, py_s = nx_full[idx], py_full[idx]

    # Statistical tests
    _, pt = stats.ttest_rel(nx_s, py_s)
    n_w = min(WILCOXON_SUBSAMPLE_SIZE, len(nx_s))
    _, pw = stats.wilcoxon(nx_s[:n_w], py_s[:n_w])
    r, _ = stats.pearsonr(nx_s, py_s)

    bias = np.mean(nx_s - py_s)
    sd_diff = np.std(nx_s - py_s, ddof=1)

    icc_value, icc_method = compute_icc_2_1(nx_s, py_s)

    res = {
        'p_t': pt,
        'p_w': pw,
        'corr': r,
        'icc': icc_value,
        'icc_method': icc_method,
        'cohen': (bias / sd_diff),
        'mean_diff': bias,
        'std_diff': sd_diff,
        'loa': (bias - 1.96 * sd_diff, bias + 1.96 * sd_diff)
    }

    labeling_accuracy = total_correct / total_compared if total_compared else 0

    # ---- Save RMSE per file ----
    rmse_csv_path = os.path.join(OUTPUT_DIR, "summary_rmse.csv")
    pd.DataFrame(file_rmse_list).to_csv(rmse_csv_path, index=False, sep=";")
    logger.info(f"RMSE summary saved: {rmse_csv_path}")

    # ---- Save statistical report ----
    report_path = os.path.join(OUTPUT_DIR, "statistical_report.txt")
    with open(report_path, "w", encoding="utf-8") as f_rep:
        f_rep.write("STATISTICAL VALIDATION REPORT\n")
        f_rep.write("=" * 50 + "\n\n")
        f_rep.write(f"Random seed: {RANDOM_SEED}\n")
        f_rep.write(f"Nexus folder: {path_n}\n")
        f_rep.write(f"Python folder: {path_p}\n\n")
        f_rep.write(f"Files processed: {processed}\n")
        f_rep.write(f"Files failed: {len(failed)}\n")
        f_rep.write(f"Files ignored (no Python counterpart): {len(ignored_files)}\n\n")
        f_rep.write(f"Total raw comparisons: {total_raw}\n")
        f_rep.write(f"Total excluded (>{OUTLIER_THRESHOLD_MM} mm): {total_excl} "
                    f"({total_excl/total_raw*100:.2f}%)\n")
        f_rep.write(f"Total compared: {total_compared}\n\n")
        f_rep.write(f"Labeling accuracy (threshold {LABELING_CORRECT_THRESHOLD_MM} mm): "
                    f"{labeling_accuracy:.4f} ({labeling_accuracy*100:.2f}%)\n")
        f_rep.write(f"  Correct: {total_correct}\n")
        f_rep.write(f"  Total: {total_compared}\n\n")
        f_rep.write(f"Subsample size (statistical tests): {n_sub}\n")
        f_rep.write(f"Wilcoxon subsample size: {n_w}\n\n")
        f_rep.write(f"Pearson r: {r:.6f}\n")
        f_rep.write(f"ICC: {icc_value:.6f} ({icc_method})\n")
        f_rep.write(f"Cohen's d: {res['cohen']:.6f}\n")
        f_rep.write(f"Mean difference (bias): {bias:.4f} mm\n")
        f_rep.write(f"SD of differences: {sd_diff:.4f} mm\n")
        f_rep.write(f"Limits of agreement: [{res['loa'][0]:.4f}, {res['loa'][1]:.4f}] mm\n")
        f_rep.write(f"t-test p-value: {pt:.6f}\n")
        f_rep.write(f"Wilcoxon p-value: {pw:.6f}\n")
        f_rep.write(f"Mean RMSE: {np.mean([x['RMSE_mm'] for x in file_rmse_list]):.4f} mm\n")
        f_rep.write(f"SD RMSE: {np.std([x['RMSE_mm'] for x in file_rmse_list], ddof=1):.4f} mm\n")

    logger.info(f"Report saved: {report_path}")

    # ---- Save failed files list ----
    if failed:
        fail_path = os.path.join(OUTPUT_DIR, "failed_files.txt")
        with open(fail_path, "w", encoding="utf-8") as f_fail:
            for fname, err in failed:
                f_fail.write(f"{fname}\t{err}\n")
        logger.warning(f"Failed files list saved: {fail_path}")

    # ---- Save ignored files list ----
    if ignored_files:
        ignored_path = os.path.join(OUTPUT_DIR, "ignored_files.txt")
        with open(ignored_path, "w", encoding="utf-8") as f_ign:
            for fname in ignored_files:
                f_ign.write(f"{fname}\n")
        logger.warning(f"Ignored files list saved: {ignored_path} "
                       f"({len(ignored_files)} files without Python counterpart)")

    # ---- Generate figure ----
    create_publication_dashboard(nx_s, py_s, file_rmse_list, res,
                                 (total_excl / total_raw) * 100,
                                 labeling_accuracy, total_compared, total_correct)

    logger.info(f"All outputs saved to: {OUTPUT_DIR}")
    logger.info("Done.")


if __name__ == "__main__":
    main()