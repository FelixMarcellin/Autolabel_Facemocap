# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:22:47 2026

@author: felima
"""

# -*- coding: utf-8 -*-
"""
analyse_stat_V4.3.py

Evaluation méthodologique de l'autolabeling facial MoCap :
comparaison des coordonnées automatiquement labellisées (Python)
avec les coordonnées de référence manuellement labellisées (Vicon Nexus).

V4.3 - Version révisée pour l'article scientifique (post-relecture).

Auteur : Félix Marcellin
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from tkinter import Tk, filedialog, messagebox
from scipy import stats

try:
    import pingouin as pg
    PINGOUIN_AVAILABLE = True
except ImportError:
    PINGOUIN_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR_NAME = "Stat_V4.3_JNER"

SKIP_ROWS = 7
NON_COORD_COLUMNS = 2

# ============================================================
# SEUILS — DISTINCTION IMPORTANTE
# ============================================================
# Le seuil d'exclusion Nexus-Python (OUTLIER_THRESHOLD_MM) N'EST PLUS
# appliqué à l'analyse principale depuis la V4.3. Il est uniquement
# utilisé pour les analyses de sensibilité.

OUTLIER_THRESHOLD_MM = 100.0

# Seuil définissant les "gross labeling discrepancies".
GROSS_DISCREPANCY_MM = 100.0

# Seuils pour l'accuracy spatiale 3D.
ACCURACY_THRESHOLDS_MM = [1, 2, 3, 5, 10, 15, 20, 30]

# Sensibilité du filtre d'exclusion.
OUTLIER_SENSITIVITY_MM = [50, 75, 100, 150]

RANDOM_SEED = 42

ENABLE_DIRECT_VS_NON_DIRECT = True

FONT_SIZE = 15

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 2,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "axes.linewidth": 1.5,
    "savefig.dpi": 300,
    "font.family": "sans-serif"
})


# ============================================================
# OUTILS
# ============================================================

def safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else np.nan


def safe_std(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1)) if len(x) > 1 else np.nan


def safe_median(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def safe_percentile(x, q):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) else np.nan


def bootstrap_ci_mean(x, n_boot=2000, alpha=0.05, seed=42):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)

    for i in range(n_boot):
        sample = rng.choice(x, size=len(x), replace=True)
        means[i] = np.mean(sample)

    return (
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2)))
    )

def cluster_bootstrap_ci_mean(
    metrics_df,
    value_col,
    participant_col="Participant_ID",
    n_boot=2000,
    alpha=0.05,
    seed=42
):
    """
    Bootstrap par participant.

    Chaque réplication tire des participants avec remise,
    puis conserve toutes les acquisitions appartenant aux
    participants tirés.

    Cela tient compte du fait que plusieurs acquisitions
    appartiennent au même participant.
    """
    df = metrics_df[[participant_col, value_col]].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[participant_col, value_col])

    participants = df[participant_col].unique()

    if len(participants) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)

    for i in range(n_boot):

        sampled_participants = rng.choice(
            participants,
            size=len(participants),
            replace=True
        )

        sampled_values = []

        for participant in sampled_participants:
            values = df.loc[
                df[participant_col] == participant,
                value_col
            ].to_numpy()

            sampled_values.extend(values)

        boot_means[i] = np.mean(sampled_values)

    return (
        float(np.percentile(
            boot_means,
            100 * alpha / 2
        )),
        float(np.percentile(
            boot_means,
            100 * (1 - alpha / 2)
        ))
    )

def parse_marker_columns(df):
    coord_cols = list(df.columns[NON_COORD_COLUMNS:])
    n_complete = len(coord_cols) // 3
    coord_cols = coord_cols[:n_complete * 3]

    markers = []

    for i in range(n_complete):
        cols = coord_cols[i * 3:(i + 1) * 3]
        markers.append({
            "marker_index": i + 1,
            "name": f"M{i + 1}",
            "columns": cols
        })

    return markers


def read_mocap_csv(path):
    df = pd.read_csv(path, skiprows=SKIP_ROWS)

    if df.shape[1] <= NON_COORD_COLUMNS:
        raise ValueError(
            f"Pas assez de colonnes coordonnées dans {os.path.basename(path)}"
        )

    return df


def extract_xyz(df, markers):
    n_frames = len(df)
    n_markers = len(markers)

    arr = np.full((n_frames, n_markers, 3), np.nan, dtype=float)

    for j, marker in enumerate(markers):
        for k, col in enumerate(marker["columns"]):
            arr[:, j, k] = pd.to_numeric(
                df[col], errors="coerce"
            ).to_numpy(dtype=float)

    return arr


def find_matching_file(folder, filename):
    candidates = [
        filename,
        filename.replace(".csv", "_labeled.csv"),
    ]

    for name in candidates:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path

    return None


def extract_python_array_by_position(df, n_markers):
    coord_cols = list(df.columns[NON_COORD_COLUMNS:])
    n_complete = min(len(coord_cols) // 3, n_markers)

    arr = np.full((len(df), n_complete, 3), np.nan, dtype=float)

    for j in range(n_complete):
        cols = coord_cols[j * 3:(j + 1) * 3]
        for k, col in enumerate(cols):
            arr[:, j, k] = pd.to_numeric(
                df[col], errors="coerce"
            ).to_numpy(dtype=float)

    return arr


def calculate_direct_vs_nondirect(nexus_arr, python_final_arr, python_raw_arr):
    nf = min(nexus_arr.shape[0], python_final_arr.shape[0], python_raw_arr.shape[0])
    nm = min(nexus_arr.shape[1], python_final_arr.shape[1], python_raw_arr.shape[1])

    n = nexus_arr[:nf, :nm]
    pf = python_final_arr[:nf, :nm]
    pr = python_raw_arr[:nf, :nm]

    nexus_valid = np.all(np.isfinite(n), axis=2)
    final_valid = np.all(np.isfinite(pf), axis=2)
    raw_direct = np.all(np.isfinite(pr), axis=2)

    paired_final = nexus_valid & final_valid
    direct_mask = paired_final & raw_direct
    nondirect_mask = paired_final & ~raw_direct

    diff = n - pf
    e3d = np.sqrt(np.sum(diff ** 2, axis=2))

    def metrics(mask):
        m = mask & np.isfinite(e3d)
        e = e3d[m]
        return {
            "N": int(len(e)),
            "RMSE_mm": float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan,
            "MAE_mm": safe_mean(np.abs(e)),
            "Median_mm": safe_median(e),
            "P95_mm": safe_percentile(e, 95),
            "Agreement_3mm": float(np.mean(e < 3)) if len(e) else np.nan,
            "Agreement_5mm": float(np.mean(e < 5)) if len(e) else np.nan,
            "Agreement_10mm": float(np.mean(e < 10)) if len(e) else np.nan,
        }

    total_final = int(np.sum(paired_final))
    direct_n = int(np.sum(direct_mask))
    nondirect_n = int(np.sum(nondirect_mask))

    direct = metrics(direct_mask)
    nondirect = metrics(nondirect_mask)

    return {
        "Final_paired_N": total_final,
        "Direct_N": direct_n,
        "NonDirect_N": nondirect_n,
        "NonDirect_percent": 100 * nondirect_n / total_final if total_final else np.nan,
        "Direct_RMSE_3D_mm": direct["RMSE_mm"],
        "Direct_MAE_3D_mm": direct["MAE_mm"],
        "Direct_Median_error_3D_mm": direct["Median_mm"],
        "Direct_P95_error_3D_mm": direct["P95_mm"],
        "Direct_Agreement_3mm": direct["Agreement_3mm"],
        "Direct_Agreement_5mm": direct["Agreement_5mm"],
        "Direct_Agreement_10mm": direct["Agreement_10mm"],
        "NonDirect_RMSE_3D_mm": nondirect["RMSE_mm"],
        "NonDirect_MAE_3D_mm": nondirect["MAE_mm"],
        "NonDirect_Median_error_3D_mm": nondirect["Median_mm"],
        "NonDirect_P95_error_3D_mm": nondirect["P95_mm"],
        "NonDirect_Agreement_3mm": nondirect["Agreement_3mm"],
        "NonDirect_Agreement_5mm": nondirect["Agreement_5mm"],
        "NonDirect_Agreement_10mm": nondirect["Agreement_10mm"],
    }


def calculate_icc_2_1(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan

    if PINGOUIN_AVAILABLE:
        try:
            n = len(x)
            data = pd.DataFrame({
                "target": np.repeat(np.arange(n), 2),
                "rater": np.tile(["Nexus", "Python"], n),
                "rating": np.column_stack([x, y]).flatten()
            })

            table = pg.intraclass_corr(
                data=data,
                targets="target",
                raters="rater",
                ratings="rating"
            )

            row = table.loc[table["Type"] == "ICC2"]

            if len(row):
                return float(row["ICC"].iloc[0])

        except Exception:
            pass

    data = np.column_stack([x, y])
    n, k = data.shape

    if n < 3:
        return np.nan

    grand_mean = np.mean(data)

    ms_rows = (
        k * np.sum((np.mean(data, axis=1) - grand_mean) ** 2)
        / (n - 1)
    )

    ms_cols = (
        n * np.sum((np.mean(data, axis=0) - grand_mean) ** 2)
        / (k - 1)
    )

    residual = (
        data
        - np.mean(data, axis=1, keepdims=True)
        - np.mean(data, axis=0, keepdims=True)
        + grand_mean
    )

    ms_error = np.sum(residual ** 2) / ((n - 1) * (k - 1))

    denominator = (
        ms_rows
        + (k - 1) * ms_error
        + k * (ms_cols - ms_error) / n
    )

    if denominator == 0:
        return np.nan

    return float((ms_rows - ms_error) / denominator)

def extract_participant_id(filename):
    """
    Extrait l'identifiant participant à partir du nom du fichier.

    Formats attendus :
        YYYYMMDD_ID_Mx_XX
        YYYYMMDD_ID_Mx
        YYYYMMDD_Mx_XX
        Mx_XX

    Exemples :
        20230412_CH_M1_01       -> CH
        20250408_MA_M4_03       -> MA
        20250123_M5_03          -> 20250123
        M2_03                   -> UNKNOWN
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("_")

    # Recherche du mouvement M1 à M9
    movement_idx = None

    for i, part in enumerate(parts):
        if re.fullmatch(r"M[1-9]", part):
            movement_idx = i
            break

    if movement_idx is None:
        return "UNKNOWN"

    # Cas standard : YYYYMMDD_ID_Mx_XX
    if movement_idx >= 2:
        candidate = parts[movement_idx - 1]

        # On évite de prendre la date comme participant
        if not re.fullmatch(r"\d{8}", candidate):
            return candidate

        # Si le bloc précédent est uniquement une date,
        # alors pas d'identifiant participant explicite
        return candidate

    # Cas YYYYMMDD_Mx_XX
    if movement_idx == 1:
        return parts[0]

    # Cas Mx_XX
    return "UNKNOWN"

# ============================================================
# ANALYSE D'UNE ACQUISITION
# ============================================================

def analyze_recording(nexus_arr, python_arr, filename):
    n_frames = min(nexus_arr.shape[0], python_arr.shape[0])
    n_markers = min(nexus_arr.shape[1], python_arr.shape[1])

    nexus_arr = nexus_arr[:n_frames, :n_markers]
    python_arr = python_arr[:n_frames, :n_markers]

    diff = nexus_arr - python_arr

    nexus_valid_xyz = np.all(np.isfinite(nexus_arr), axis=2)
    python_valid_xyz = np.all(np.isfinite(python_arr), axis=2)

    both_valid_xyz = nexus_valid_xyz & python_valid_xyz

    n_nexus_missing = int(np.sum(~nexus_valid_xyz))
    n_python_missing = int(np.sum(~python_valid_xyz))

    n_both_valid = int(np.sum(both_valid_xyz))

    error_3d = np.sqrt(np.sum(diff ** 2, axis=2))

    valid_3d = np.isfinite(error_3d) & both_valid_xyz

    error_valid = error_3d[valid_3d]

    error_primary = error_valid.copy()

    n_3d_raw = int(np.sum(valid_3d))
    n_3d_gross = int(np.sum(
        valid_3d & (error_3d >= GROSS_DISCREPANCY_MM)
    ))

    analysis_mask_3d = valid_3d & (
        error_3d < OUTLIER_THRESHOLD_MM
    )
    error_sensitivity = error_3d[analysis_mask_3d]

    coord_valid = np.isfinite(nexus_arr) & np.isfinite(python_arr)
    coord_diff_primary = diff[coord_valid]

    coord_analysis_mask = (
        coord_valid
        & (np.abs(diff) < OUTLIER_THRESHOLD_MM)
    )
    coord_diff_sensitivity = diff[coord_analysis_mask]

    rmse_3d = (
        float(np.sqrt(np.mean(error_primary ** 2)))
        if len(error_primary) else np.nan
    )

    mae_3d = safe_mean(np.abs(error_primary))
    median_3d = safe_median(error_primary)
    p95_3d = safe_percentile(error_primary, 95)

    rmse_3d_sens = (
        float(np.sqrt(np.mean(error_sensitivity ** 2)))
        if len(error_sensitivity) else np.nan
    )

    mae_3d_sens = safe_mean(np.abs(error_sensitivity))
    median_3d_sens = safe_median(error_sensitivity)
    p95_3d_sens = safe_percentile(error_sensitivity, 95)

    rmse_coord = (
        float(np.sqrt(np.mean(coord_diff_primary ** 2)))
        if len(coord_diff_primary) else np.nan
    )

    bias_x = safe_mean(diff[:, :, 0][coord_valid[:, :, 0]])
    bias_y = safe_mean(diff[:, :, 1][coord_valid[:, :, 1]])
    bias_z = safe_mean(diff[:, :, 2][coord_valid[:, :, 2]])

    accuracy = {}

    for threshold in ACCURACY_THRESHOLDS_MM:
        if len(error_primary):
            accuracy[threshold] = float(
                np.mean(error_primary < threshold)
            )
        else:
            accuracy[threshold] = np.nan

    flat_nexus = nexus_arr[coord_valid]
    flat_python = python_arr[coord_valid]

    icc = calculate_icc_2_1(flat_nexus, flat_python)

    total_marker_frames = n_frames * n_markers

    availability_nexus = (
        np.sum(nexus_valid_xyz) / total_marker_frames
        if total_marker_frames else np.nan
    )

    availability_python = (
        np.sum(python_valid_xyz) / total_marker_frames
        if total_marker_frames else np.nan
    )

    agreement_availability = (
        np.sum(both_valid_xyz) / total_marker_frames
        if total_marker_frames else np.nan
    )

    result = {
        "Filename": filename,
        "Frames": n_frames,
        "Markers": n_markers,

        "Nexus_valid_marker_frames": int(np.sum(nexus_valid_xyz)),
        "Python_valid_marker_frames": int(np.sum(python_valid_xyz)),
        "Both_valid_marker_frames": n_both_valid,

        "Nexus_missing_marker_frames": n_nexus_missing,
        "Python_missing_marker_frames": n_python_missing,

        "Nexus_availability": availability_nexus,
        "Python_availability": availability_python,
        "Paired_availability": agreement_availability,

        "Valid_3D_comparisons_raw": n_3d_raw,
        "Gross_discrepancies_3D": n_3d_gross,
        "Gross_discrepancies_3D_percent": (
            100 * n_3d_gross / n_3d_raw if n_3d_raw else np.nan
        ),

        "RMSE_3D_mm": rmse_3d,
        "MAE_3D_mm": mae_3d,
        "Median_error_3D_mm": median_3d,
        "P95_error_3D_mm": p95_3d,

        "RMSE_3D_sensitivity_mm": rmse_3d_sens,
        "MAE_3D_sensitivity_mm": mae_3d_sens,
        "Median_error_3D_sensitivity_mm": median_3d_sens,
        "P95_error_3D_sensitivity_mm": p95_3d_sens,

        "RMSE_coordinate_mm": rmse_coord,

        "Bias_X_mm": bias_x,
        "Bias_Y_mm": bias_y,
        "Bias_Z_mm": bias_z,

        "ICC_2_1_secondary": icc,

        "BA_mean_nexus_mm": (
            float(np.mean(flat_nexus)) if len(flat_nexus) else np.nan
        ),
        "BA_mean_python_mm": (
            float(np.mean(flat_python)) if len(flat_python) else np.nan
        ),
        "BA_bias_mm": (
            float(np.mean(flat_nexus - flat_python))
            if len(flat_nexus) else np.nan
        ),
    }

    for threshold in ACCURACY_THRESHOLDS_MM:
        result[f"Accuracy_3D_{threshold}mm"] = accuracy[threshold]

    details = {
        "error_3d": error_primary,
        "error_3d_raw": error_valid,
        "error_3d_sensitivity": error_sensitivity,
        "coord_diff": coord_diff_primary,
        "nexus_coord": flat_nexus,
        "python_coord": flat_python,
        "diff_matrix": diff,
        "coord_analysis_mask": coord_analysis_mask,
        "valid_3d_mask": valid_3d,
        "analysis_3d_mask": analysis_mask_3d,
        "coord_valid": coord_valid,
    }

    return result, details


# ============================================================
# ANALYSE PAR MARQUEUR ET PAR AXE
# ============================================================

def analyze_marker_level(nexus_arr, python_arr, filename, markers):
    rows_marker = []
    rows_axis = []

    n_frames = min(nexus_arr.shape[0], python_arr.shape[0])
    n_markers = min(nexus_arr.shape[1], python_arr.shape[1])

    nexus_arr = nexus_arr[:n_frames, :n_markers]
    python_arr = python_arr[:n_frames, :n_markers]

    diff = nexus_arr - python_arr

    for j in range(n_markers):

        marker_name = (
            markers[j]["name"]
            if j < len(markers)
            else f"M{j + 1}"
        )

        d = diff[:, j, :]
        e3d = np.sqrt(np.sum(d ** 2, axis=1))

        valid3d = (
            np.all(np.isfinite(nexus_arr[:, j, :]), axis=1)
            & np.all(np.isfinite(python_arr[:, j, :]), axis=1)
            & np.isfinite(e3d)
        )

        ev = e3d[valid3d]

        analysis_sens = valid3d & (e3d < OUTLIER_THRESHOLD_MM)
        ev_sens = e3d[analysis_sens]

        row = {
            "Filename": filename,
            "Marker": marker_name,
            "N_valid_3D": int(np.sum(valid3d)),
            "N_gross_discrepancies": int(np.sum(
                valid3d & (e3d >= GROSS_DISCREPANCY_MM)
            )),
            "RMSE_3D_mm": (
                float(np.sqrt(np.mean(ev ** 2)))
                if len(ev) else np.nan
            ),
            "MAE_3D_mm": safe_mean(np.abs(ev)),
            "Median_error_3D_mm": safe_median(ev),
            "P95_error_3D_mm": safe_percentile(ev, 95),
            "RMSE_3D_sensitivity_mm": (
                float(np.sqrt(np.mean(ev_sens ** 2)))
                if len(ev_sens) else np.nan
            ),
            "MAE_3D_sensitivity_mm": safe_mean(np.abs(ev_sens)),
            "Median_error_3D_sensitivity_mm": safe_median(ev_sens),
            "P95_error_3D_sensitivity_mm": safe_percentile(ev_sens, 95),
        }

        for threshold in ACCURACY_THRESHOLDS_MM:
            row[f"Accuracy_3D_{threshold}mm"] = (
                float(np.mean(ev < threshold))
                if len(ev) else np.nan
            )

        rows_marker.append(row)

        for axis_index, axis_name in enumerate(["X", "Y", "Z"]):

            axis_n = nexus_arr[:, j, axis_index]
            axis_p = python_arr[:, j, axis_index]

            valid = np.isfinite(axis_n) & np.isfinite(axis_p)

            axis_diff = axis_n - axis_p

            dv = axis_diff[valid]

            rows_axis.append({
                "Filename": filename,
                "Marker": marker_name,
                "Axis": axis_name,
                "N_valid": int(np.sum(valid)),
                "Bias_mm": safe_mean(dv),
                "MAE_mm": safe_mean(np.abs(dv)),
                "RMSE_mm": (
                    float(np.sqrt(np.mean(dv ** 2)))
                    if len(dv) else np.nan
                ),
                "Median_abs_error_mm": safe_median(np.abs(dv)),
                "P95_abs_error_mm": safe_percentile(
                    np.abs(dv), 95
                ),
                "Accuracy_3mm": (
                    float(np.mean(np.abs(dv) < 3))
                    if len(dv) else np.nan
                ),
                "Accuracy_5mm": (
                    float(np.mean(np.abs(dv) < 5))
                    if len(dv) else np.nan
                ),
                "Accuracy_10mm": (
                    float(np.mean(np.abs(dv) < 10))
                    if len(dv) else np.nan
                ),
            })

    return rows_marker, rows_axis


# ============================================================
# SENSIBILITÉ DU SEUIL D'EXCLUSION
# ============================================================

def calculate_outlier_sensitivity(all_raw_errors):
    rows = []

    if not all_raw_errors:
        return pd.DataFrame()

    errors = np.concatenate(all_raw_errors)
    errors = errors[np.isfinite(errors)]

    if len(errors) == 0:
        return pd.DataFrame()

    rows.append({
        "Exclusion_threshold_mm": "None (primary)",
        "N_raw": len(errors),
        "N_included": len(errors),
        "N_excluded": 0,
        "Percent_excluded": 0.0,
        "RMSE_3D_mm": float(np.sqrt(np.mean(errors ** 2))),
        "MAE_3D_mm": safe_mean(np.abs(errors)),
        "Median_error_3D_mm": safe_median(errors),
        "P95_error_3D_mm": safe_percentile(errors, 95),
        "Accuracy_3mm": float(np.mean(errors < 3)),
        "Accuracy_5mm": float(np.mean(errors < 5)),
        "Accuracy_10mm": float(np.mean(errors < 10)),
    })

    for threshold in OUTLIER_SENSITIVITY_MM:

        valid = errors < threshold
        e = errors[valid]

        rows.append({
            "Exclusion_threshold_mm": threshold,
            "N_raw": len(errors),
            "N_included": len(e),
            "N_excluded": int(np.sum(~valid)),
            "Percent_excluded": (
                100 * np.mean(~valid)
                if len(errors) else np.nan
            ),
            "RMSE_3D_mm": (
                float(np.sqrt(np.mean(e ** 2)))
                if len(e) else np.nan
            ),
            "MAE_3D_mm": safe_mean(np.abs(e)),
            "Median_error_3D_mm": safe_median(e),
            "P95_error_3D_mm": safe_percentile(e, 95),
            "Accuracy_3mm": (
                float(np.mean(e < 3))
                if len(e) else np.nan
            ),
            "Accuracy_5mm": (
                float(np.mean(e < 5))
                if len(e) else np.nan
            ),
            "Accuracy_10mm": (
                float(np.mean(e < 10))
                if len(e) else np.nan
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# STATISTIQUES AU NIVEAU ACQUISITION
# ============================================================

def recording_level_statistics(metrics_df):
    out = {}

    rmse = metrics_df["RMSE_3D_mm"].dropna().to_numpy()

    out["N_recordings_RMSE"] = len(rmse)
    out["RMSE3D_mean_mm"] = safe_mean(rmse)
    out["RMSE3D_sd_mm"] = safe_std(rmse)
    out["RMSE3D_median_mm"] = safe_median(rmse)
    out["RMSE3D_Q1_mm"] = safe_percentile(rmse, 25)
    out["RMSE3D_Q3_mm"] = safe_percentile(rmse, 75)
    ci_rmse = bootstrap_ci_mean(rmse, seed=RANDOM_SEED)
    ci_rmse_cluster = cluster_bootstrap_ci_mean(
    metrics_df,
    "RMSE_3D_mm",
    seed=RANDOM_SEED
)
    out["RMSE3D_mean_CI95_lower_mm"] = ci_rmse[0]
    out["RMSE3D_mean_CI95_upper_mm"] = ci_rmse[1]
    out["RMSE3D_mean_CI95_cluster_lower_mm"] = ci_rmse_cluster[0]
    out["RMSE3D_mean_CI95_cluster_upper_mm"] = ci_rmse_cluster[1]

    acc3 = metrics_df["Accuracy_3D_3mm"].dropna().to_numpy()
    out["Accuracy3D_3mm_mean_recording_percent"] = (
        safe_mean(acc3) * 100 if len(acc3) else np.nan
    )
    ci_acc3 = bootstrap_ci_mean(acc3, seed=RANDOM_SEED)
    ci_acc3_cluster = cluster_bootstrap_ci_mean(
    metrics_df,
    "Accuracy_3D_3mm",
    seed=RANDOM_SEED
)
    out["Accuracy3D_3mm_mean_recording_CI95_lower_percent"] = (
        ci_acc3[0] * 100 if np.isfinite(ci_acc3[0]) else np.nan
    )
    out["Accuracy3D_3mm_mean_recording_CI95_upper_percent"] = (
        ci_acc3[1] * 100 if np.isfinite(ci_acc3[1]) else np.nan
    )
    
    out["Accuracy3D_3mm_mean_recording_CI95_cluster_lower_percent"] = (
    ci_acc3_cluster[0] * 100
    if np.isfinite(ci_acc3_cluster[0]) else np.nan
)

    out["Accuracy3D_3mm_mean_recording_CI95_cluster_upper_percent"] = (
        ci_acc3_cluster[1] * 100
        if np.isfinite(ci_acc3_cluster[1]) else np.nan
    )

    for threshold in ACCURACY_THRESHOLDS_MM:
        col = f"Accuracy_3D_{threshold}mm"
        vals = pd.to_numeric(metrics_df[col], errors="coerce").dropna().to_numpy()
        ci = bootstrap_ci_mean(vals, seed=RANDOM_SEED) if len(vals) else (np.nan, np.nan)
        out[f"Accuracy3D_{threshold}mm_mean_recording_percent"] = (
            safe_mean(vals) * 100 if len(vals) else np.nan
        )
        out[f"Accuracy3D_{threshold}mm_mean_recording_CI95_lower_percent"] = (
            ci[0] * 100 if np.isfinite(ci[0]) else np.nan
        )
        out[f"Accuracy3D_{threshold}mm_mean_recording_CI95_upper_percent"] = (
            ci[1] * 100 if np.isfinite(ci[1]) else np.nan
        )

    rmse_sens = metrics_df["RMSE_3D_sensitivity_mm"].dropna().to_numpy()

    out["RMSE3D_sensitivity_mean_mm"] = safe_mean(rmse_sens)
    out["RMSE3D_sensitivity_sd_mm"] = safe_std(rmse_sens)
    out["RMSE3D_sensitivity_median_mm"] = safe_median(rmse_sens)
    out["RMSE3D_sensitivity_Q1_mm"] = safe_percentile(rmse_sens, 25)
    out["RMSE3D_sensitivity_Q3_mm"] = safe_percentile(rmse_sens, 75)

    bias_cols = ["Bias_X_mm", "Bias_Y_mm", "Bias_Z_mm"]

    for col in bias_cols:
        x = metrics_df[col].dropna().to_numpy()

        out[f"{col}_mean"] = safe_mean(x)
        out[f"{col}_sd"] = safe_std(x)

        if len(x) >= 2:
            try:
                out[f"{col}_ttest_p"] = float(
                    stats.ttest_1samp(x, 0).pvalue
                )
            except Exception:
                out[f"{col}_ttest_p"] = np.nan

            try:
                out[f"{col}_wilcoxon_p"] = float(
                    stats.wilcoxon(x).pvalue
                )
            except Exception:
                out[f"{col}_wilcoxon_p"] = np.nan
        else:
            out[f"{col}_ttest_p"] = np.nan
            out[f"{col}_wilcoxon_p"] = np.nan

    all_bias = []

    for _, row in metrics_df.iterrows():
        vals = [
            row["Bias_X_mm"],
            row["Bias_Y_mm"],
            row["Bias_Z_mm"]
        ]
        vals = [v for v in vals if np.isfinite(v)]

        if vals:
            all_bias.append(np.mean(vals))

    all_bias = np.asarray(all_bias, dtype=float)

    if len(all_bias) > 1:
        sd = np.std(all_bias, ddof=1)

        out["d_z_mean_coordinate_bias"] = (
            float(np.mean(all_bias) / sd)
            if sd > 0 else np.nan
        )

        out["mean_coordinate_bias_mm"] = float(np.mean(all_bias))

    else:
        out["d_z_mean_coordinate_bias"] = np.nan
        out["mean_coordinate_bias_mm"] = np.nan

    return out


# ============================================================
# BLAND-ALTMAN
# ============================================================

def bland_altman_stats(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return {}

    diff = x - y
    mean_values = (x + y) / 2

    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))

    loa_low = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd

    ci_bias = bootstrap_ci_mean(diff, seed=RANDOM_SEED)

    return {
        "N": len(diff),
        "Bias_mm": bias,
        "SD_difference_mm": sd,
        "LoA_lower_mm": float(loa_low),
        "LoA_upper_mm": float(loa_high),
        "Bias_CI95_lower_mm": ci_bias[0],
        "Bias_CI95_upper_mm": ci_bias[1],
        "Mean_measurement_min": float(np.min(mean_values)),
        "Mean_measurement_max": float(np.max(mean_values)),
    }


# ============================================================
# BLAND-ALTMAN AU NIVEAU ACQUISITION
# ============================================================

def recording_level_bland_altman(metrics_df):
    required = [
        "BA_mean_nexus_mm",
        "BA_mean_python_mm",
        "BA_bias_mm"
    ]

    if any(c not in metrics_df.columns for c in required):
        return {}

    df = metrics_df[required].dropna()

    if len(df) < 2:
        return {}

    mean_measurement = df["BA_mean_nexus_mm"].to_numpy(float)
    mean_python = df["BA_mean_python_mm"].to_numpy(float)
    diff = df["BA_bias_mm"].to_numpy(float)

    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    loa_low = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd

    ci_bias = bootstrap_ci_mean(diff, seed=RANDOM_SEED)

    rng = np.random.default_rng(RANDOM_SEED)
    n_boot = 2000
    boot_bias = np.empty(n_boot)
    boot_low = np.empty(n_boot)
    boot_high = np.empty(n_boot)

    for i in range(n_boot):
        sample = rng.choice(diff, size=len(diff), replace=True)
        b = np.mean(sample)
        sd_b = np.std(sample, ddof=1) if len(sample) > 1 else 0.0
        boot_bias[i] = b
        boot_low[i] = b - 1.96 * sd_b
        boot_high[i] = b + 1.96 * sd_b

    return {
        "N_recordings": int(len(diff)),
        "Bias_mm": bias,
        "SD_recording_bias_mm": sd,
        "LoA_lower_mm": float(loa_low),
        "LoA_upper_mm": float(loa_high),
        "Bias_CI95_lower_mm": float(np.percentile(boot_bias, 2.5)),
        "Bias_CI95_upper_mm": float(np.percentile(boot_bias, 97.5)),
        "LoA_lower_CI95_lower_mm": float(np.percentile(boot_low, 2.5)),
        "LoA_lower_CI95_upper_mm": float(np.percentile(boot_low, 97.5)),
        "LoA_upper_CI95_lower_mm": float(np.percentile(boot_high, 2.5)),
        "LoA_upper_CI95_upper_mm": float(np.percentile(boot_high, 97.5)),
        "mean_measurement_min_mm": float(np.min(mean_measurement)),
        "mean_measurement_max_mm": float(np.max(mean_measurement)),
        "recording_data": df.copy()
    }


# ============================================================
# FIGURE PRINCIPALE DE PUBLICATION
# ============================================================

def create_publication_figure(
    pooled_nexus,
    pooled_python,
    pooled_errors,
    metrics_df,
    accuracy_global,
    ba_recording,
    output_path
):
    """
    Figure principale (6 panels).

    IMPORTANT (V4.3) :
    Le panel A (distribution de l'erreur 3D) inclut TOUTES les
    observations valides, sans exclusion.
    """

    fig = plt.figure(figsize=(18, 13))

    # --------------------------------------------------------
    # A - Erreur spatiale 3D (SANS exclusion)
    # --------------------------------------------------------
    ax1 = plt.subplot(2, 3, 1)

    if len(pooled_errors):
        visible = pooled_errors[pooled_errors <= 30]
        weights = np.ones(len(visible)) * 100.0 / len(pooled_errors)
        ax1.hist(
            visible,
            bins=50,
            weights=weights,
            edgecolor="black",
            linewidth=0.6
        )

        median_val = np.median(pooled_errors)
        ax1.axvline(
            median_val,
            linestyle="--",
            linewidth=2,
            color="red",
            label=f"Median = {median_val:.2f} mm"
        )

        pct_gt30 = 100 * np.mean(pooled_errors > 30)
        ax1.text(
            0.98, 0.95,
            f">30 mm: {pct_gt30:.2f}%",
            transform=ax1.transAxes,
            ha="right", va="top",
            fontsize=11,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="gray",
                alpha=0.9
            )
        )

    ax1.set_xlim(0, 30)
    ax1.set_title("A. 3D spatial error (no exclusion)")
    ax1.set_xlabel("3D error [mm]")
    ax1.set_ylabel("Observations [%]")

    # Légende déplacée en bas à droite pour éviter le chevauchement
    ax1.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02)
    )

    # --------------------------------------------------------
    # B - Accord en fonction du seuil (SANS exclusion)
    # --------------------------------------------------------
    ax2 = plt.subplot(2, 3, 2)

    thresholds = list(accuracy_global.keys())
    values = []
    lower = []
    upper = []

    for t in thresholds:
        col = f"Accuracy_3D_{t}mm"
        vals = pd.to_numeric(metrics_df[col], errors="coerce").dropna().to_numpy()
        ci = bootstrap_ci_mean(vals, seed=RANDOM_SEED) if len(vals) else (np.nan, np.nan)
        values.append(safe_mean(vals) * 100 if len(vals) else np.nan)
        lower.append(ci[0] * 100 if np.isfinite(ci[0]) else np.nan)
        upper.append(ci[1] * 100 if np.isfinite(ci[1]) else np.nan)

    ax2.plot(
        thresholds,
        values,
        marker="o",
        linewidth=2,
        label="Mean across recordings"
    )

    ax2.fill_between(
        thresholds,
        lower,
        upper,
        alpha=0.15,
        linewidth=0
    )

    ax2.set_xlabel("3D error threshold [mm]")
    ax2.set_ylabel("Agreement [%]")
    ax2.set_ylim(0, 105)
    ax2.set_title("B. 3D agreement vs threshold (no exclusion)")
    ax2.grid(alpha=0.2)
    ax2.legend(frameon=False, loc="lower right")

    # --------------------------------------------------------
    # C - Bland-Altman (niveau acquisition)
    # --------------------------------------------------------
    ax3 = plt.subplot(2, 3, 3)

    if ba_recording and "recording_data" in ba_recording:
        ba_plot = ba_recording["recording_data"]

        ax3.scatter(
            ba_plot["BA_mean_nexus_mm"],
            ba_plot["BA_bias_mm"],
            s=24,
            alpha=0.65
        )

    if ba_recording:
        ax3.axhline(
            ba_recording["Bias_mm"],
            linewidth=2,
            label=f"Bias = {ba_recording['Bias_mm']:.2f} mm"
        )
        ax3.axhline(
            ba_recording["LoA_lower_mm"],
            linestyle="--",
            linewidth=1.5
        )
        ax3.axhline(
            ba_recording["LoA_upper_mm"],
            linestyle="--",
            linewidth=1.5
        )

    ax3.axhline(0, linewidth=1)

    ax3.set_title("C. Bland-Altman - recording level")
    ax3.set_xlabel("Mean of Nexus and Python [mm]")
    ax3.set_ylabel("Nexus - Python [mm]")
    ax3.legend(frameon=False, loc="upper right")

    # --------------------------------------------------------
    # D - RMSE par acquisition (SANS exclusion)
    # --------------------------------------------------------
    ax4 = plt.subplot(2, 3, 4)

    rmse = metrics_df["RMSE_3D_mm"].dropna().to_numpy()

    if len(rmse):
        ax4.boxplot(
            rmse,
            widths=0.5,
            showmeans=True
        )

        ax4.scatter(
            np.ones(len(rmse)),
            rmse,
            alpha=0.20,
            s=10
        )

        ax4.text(
            1.12, np.median(rmse),
            f"Median {np.median(rmse):.2f} mm",
            va="center",
            fontsize=11
        )
        ax4.text(
            1.12, np.mean(rmse),
            f"Mean {np.mean(rmse):.2f} mm",
            va="center",
            fontsize=11
        )

    ax4.set_title("D. 3D RMSE per recording (no exclusion)")
    ax4.set_ylabel("RMSE [mm]")
    ax4.set_xticks([])

    # --------------------------------------------------------
    # E - Nexus vs Python
    # --------------------------------------------------------
    ax5 = plt.subplot(2, 3, 5)

    if len(pooled_nexus):

        plot_n = min(len(pooled_nexus), 30000)

        if len(pooled_nexus) > plot_n:
            rng = np.random.default_rng(RANDOM_SEED)
            idx = rng.choice(
                len(pooled_nexus),
                plot_n,
                replace=False
            )
            x = pooled_nexus[idx]
            y = pooled_python[idx]
        else:
            x = pooled_nexus
            y = pooled_python

        ax5.scatter(
            x,
            y,
            s=5,
            alpha=0.25
        )

        low = min(np.min(x), np.min(y))
        high = max(np.max(x), np.max(y))

        ax5.plot(
            [low, high],
            [low, high],
            linestyle="--",
            linewidth=2
        )

    ax5.set_title("E. Nexus versus Python")
    ax5.set_xlabel("Nexus [mm]")
    ax5.set_ylabel("Python [mm]")

    # --------------------------------------------------------
    # F - Biais par axe (SANS exclusion)
    # --------------------------------------------------------
    ax6 = plt.subplot(2, 3, 6)

    axes = ["X", "Y", "Z"]

    bias_values = [
        safe_mean(metrics_df["Bias_X_mm"]),
        safe_mean(metrics_df["Bias_Y_mm"]),
        safe_mean(metrics_df["Bias_Z_mm"])
    ]

    ax6.bar(axes, bias_values)
    ax6.axhline(0, linewidth=1)

    ax6.set_title("F. Mean bias by axis (no exclusion)")
    ax6.set_ylabel("Bias [mm]")

    plt.tight_layout()

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# NOUVELLE FIGURE : SENSIBILITÉ AU SEUIL D'EXCLUSION
# ============================================================

def create_sensitivity_figure(sensitivity_df, output_path):
    if sensitivity_df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    df_num = sensitivity_df[
        sensitivity_df["Exclusion_threshold_mm"] != "None (primary)"
    ].copy()
    df_num["Exclusion_threshold_mm"] = pd.to_numeric(
        df_num["Exclusion_threshold_mm"]
    )

    primary_row = sensitivity_df[
        sensitivity_df["Exclusion_threshold_mm"] == "None (primary)"
    ]

    ax = axes[0]
    ax.plot(
        df_num["Exclusion_threshold_mm"],
        df_num["RMSE_3D_mm"],
        marker="o",
        linewidth=2,
        label="With exclusion"
    )

    if not primary_row.empty:
        ax.axhline(
            primary_row["RMSE_3D_mm"].iloc[0],
            linestyle="--",
            linewidth=2,
            color="red",
            label="Primary (no exclusion)"
        )

    ax.set_xlabel("Exclusion threshold [mm]")
    ax.set_ylabel("Pooled RMSE 3D [mm]")
    ax.set_title("A. RMSE vs exclusion threshold")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(
        df_num["Exclusion_threshold_mm"],
        df_num["MAE_3D_mm"],
        marker="o",
        linewidth=2,
        label="With exclusion"
    )

    if not primary_row.empty:
        ax.axhline(
            primary_row["MAE_3D_mm"].iloc[0],
            linestyle="--",
            linewidth=2,
            color="red",
            label="Primary (no exclusion)"
        )

    ax.set_xlabel("Exclusion threshold [mm]")
    ax.set_ylabel("Pooled MAE 3D [mm]")
    ax.set_title("B. MAE vs exclusion threshold")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    ax = axes[2]
    ax.plot(
        df_num["Exclusion_threshold_mm"],
        df_num["Accuracy_3mm"] * 100,
        marker="o",
        linewidth=2,
        label="With exclusion"
    )

    if not primary_row.empty:
        ax.axhline(
            primary_row["Accuracy_3mm"].iloc[0] * 100,
            linestyle="--",
            linewidth=2,
            color="red",
            label="Primary (no exclusion)"
        )

    ax.set_xlabel("Exclusion threshold [mm]")
    ax.set_ylabel("Agreement < 3 mm [%]")
    ax.set_title("C. Agreement < 3 mm vs exclusion threshold")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# RAPPORT TEXTE
# ============================================================

def write_report(
    output_path,
    path_n,
    path_p,
    metrics_df,
    global_accuracy,
    ba_recording,
    ba_pooled,
    sensitivity_df,
    inferential,
    pooled_rmse_3d,
    direct_available
):

    with open(output_path, "w", encoding="utf-8") as f:

        f.write("STATISTICAL VALIDATION REPORT - V4.3 JNER\n")
        f.write("=" * 70 + "\n\n")

        f.write(
            "This report evaluates automated facial marker labeling "
            "against manual Vicon Nexus labeling.\n"
        )
        f.write(
            "The autolabeling algorithm itself is not modified by this script.\n\n"
        )

        f.write(
            "IMPORTANT CHANGE (V4.3)\n"
            "------------------------\n"
        )
        f.write(
            "The PRIMARY analysis includes ALL valid paired observations, "
            "without exclusion based on the magnitude of the Nexus-Python "
            "discrepancy. Observations exceeding the gross discrepancy "
            "threshold (100 mm) are RETAINED in the primary analysis and "
            "reported separately as potential labeling failures.\n\n"
        )
        f.write(
            "Exclusion thresholds (50, 75, 100, 150 mm) are used ONLY for "
            "SENSITIVITY analyses, which are reported separately.\n\n"
        )

        f.write(
            "IMPORTANT DISTINCTION\n"
            "----------------------\n"
        )
        f.write(
            "The 100 mm threshold used in the sensitivity analyses is an "
            "exclusion criterion for Nexus-vs-Python comparison. It is "
            "distinct from the 100 mm trajectory outlier threshold used "
            "inside the autolabeling pipeline relative to the static reference.\n\n"
        )

        f.write(f"Nexus folder: {path_n}\n")
        f.write(f"Python folder: {path_p}\n\n")

        f.write(f"Number of recordings analysed: {len(metrics_df)}\n\n")

        f.write("DATA AVAILABILITY\n" "-----------------\n")

        f.write(
            f"Mean Nexus availability: "
            f"{metrics_df['Nexus_availability'].mean()*100:.2f}%\n"
        )
        f.write(
            f"Mean Python availability: "
            f"{metrics_df['Python_availability'].mean()*100:.2f}%\n"
        )
        f.write(
            f"Mean paired availability: "
            f"{metrics_df['Paired_availability'].mean()*100:.2f}%\n"
        )
        f.write(
            f"Direct/non-direct sensitivity available: {direct_available}\n\n"
        )

        f.write(
            "PRIMARY SPATIAL AGREEMENT (NO EXCLUSION)\n"
            "------------------------------------------\n"
        )

        f.write(
            f"Mean RMSE 3D: "
            f"{inferential.get('RMSE3D_mean_mm', np.nan):.3f} mm\n"
        )
        f.write(
            f"SD RMSE 3D: "
            f"{inferential.get('RMSE3D_sd_mm', np.nan):.3f} mm\n"
        )
        f.write(
            f"Median RMSE 3D: "
            f"{inferential.get('RMSE3D_median_mm', np.nan):.3f} mm\n"
        )
        f.write(
            f"Q1 RMSE 3D: "
            f"{inferential.get('RMSE3D_Q1_mm', np.nan):.3f} mm\n"
        )
        f.write(
            f"Q3 RMSE 3D: "
            f"{inferential.get('RMSE3D_Q3_mm', np.nan):.3f} mm\n"
        )
        f.write(
            f"95% CI of mean RMSE 3D: "
            f"[{inferential.get('RMSE3D_mean_CI95_lower_mm', np.nan):.3f}, "
            f"{inferential.get('RMSE3D_mean_CI95_upper_mm', np.nan):.3f}] mm\n"
        )
        f.write(
            "Pooled RMSE 3D across ALL valid observations (no exclusion): "
            f"{pooled_rmse_3d:.3f} mm\n"
        )
        f.write(
            "Mean recording-level agreement < 3 mm: "
            f"{inferential.get('Accuracy3D_3mm_mean_recording_percent', np.nan):.3f}% "
            f"(95% bootstrap CI "
            f"[{inferential.get('Accuracy3D_3mm_mean_recording_CI95_lower_percent', np.nan):.3f}, "
            f"{inferential.get('Accuracy3D_3mm_mean_recording_CI95_upper_percent', np.nan):.3f}%)\n\n"
        )

        total_gross = metrics_df['Gross_discrepancies_3D'].sum()
        total_valid = metrics_df['Valid_3D_comparisons_raw'].sum()
        percent_gross = (
            100 * total_gross / total_valid if total_valid else np.nan
        )

        f.write(
            "GROSS LABELING DISCREPANCIES\n"
            "-----------------------------\n"
        )
        f.write(f"Threshold: {GROSS_DISCREPANCY_MM} mm\n")
        f.write(
            f"Total gross discrepancies: {int(total_gross)} / "
            f"{int(total_valid)} ({percent_gross:.3f}%)\n"
        )
        f.write(
            "These observations were RETAINED in the primary analysis "
            "but are described separately as potential labeling failures.\n\n"
        )

        f.write("3D AGREEMENT - RECORDING-LEVEL PRIMARY SUMMARY (NO EXCLUSION)\n")
        f.write("-------------------------------------------------------------\n")
        for threshold in ACCURACY_THRESHOLDS_MM:
            mean_rec = inferential.get(
                f"Accuracy3D_{threshold}mm_mean_recording_percent", np.nan
            )
            lo_rec = inferential.get(
                f"Accuracy3D_{threshold}mm_mean_recording_CI95_lower_percent", np.nan
            )
            hi_rec = inferential.get(
                f"Accuracy3D_{threshold}mm_mean_recording_CI95_upper_percent", np.nan
            )
            pooled = (
                global_accuracy.get(threshold, np.nan) * 100
                if np.isfinite(global_accuracy.get(threshold, np.nan))
                else np.nan
            )
            f.write(
                f"< {threshold} mm: recording-level mean = {mean_rec:.3f}% "
                f"(95% CI [{lo_rec:.3f}, {hi_rec:.3f}%]); "
                f"pooled descriptive = {pooled:.3f}%\n"
            )

        f.write("\n")

        f.write(
            "SENSITIVITY ANALYSIS (EXCLUSION THRESHOLDS)\n"
            "--------------------------------------------\n"
        )
        f.write(
            f"Mean RMSE 3D with 100 mm exclusion: "
            f"{inferential.get('RMSE3D_sensitivity_mean_mm', np.nan):.3f} mm\n"
        )
        f.write(
            f"Median RMSE 3D with 100 mm exclusion: "
            f"{inferential.get('RMSE3D_sensitivity_median_mm', np.nan):.3f} mm\n"
        )
        f.write(
            "Full sensitivity table is provided in "
            "'06_outlier_sensitivity.csv'.\n\n"
        )

        f.write(
            "COORDINATE-WISE BIAS - SECONDARY (NO EXCLUSION)\n"
            "-----------------------------------------------\n"
        )

        for axis in ["X", "Y", "Z"]:
            key = f"Bias_{axis}_mm"

            f.write(
                f"{axis}: mean = "
                f"{inferential.get(key + '_mean', np.nan):.4f} mm; "
                f"SD = "
                f"{inferential.get(key + '_sd', np.nan):.4f} mm; "
                f"paired t-test p = "
                f"{inferential.get(key + '_ttest_p', np.nan):.6f}; "
                f"Wilcoxon p = "
                f"{inferential.get(key + '_wilcoxon_p', np.nan):.6f}\n"
            )

        f.write("\n")

        f.write(
            "STANDARDIZED PAIRED DIFFERENCE - SECONDARY\n"
            "--------------------------------------------\n"
        )
        f.write(
            "d_z is reported only as a secondary standardized measure "
            "of the mean acquisition-level coordinate bias. It is not "
            "a test of equivalence.\n"
        )
        f.write(
            f"d_z = "
            f"{inferential.get('d_z_mean_coordinate_bias', np.nan):.6f}\n\n"
        )

        f.write(
            "BLAND-ALTMAN - PRIMARY RECORDING-LEVEL ANALYSIS (NO EXCLUSION)\n"
            "---------------------------------------------------------------\n"
        )

        if ba_recording:
            f.write(f"N recordings = {ba_recording['N_recordings']}\n")
            f.write(f"Bias = {ba_recording['Bias_mm']:.4f} mm\n")
            f.write(
                f"95% CI bias = "
                f"[{ba_recording['Bias_CI95_lower_mm']:.4f}, "
                f"{ba_recording['Bias_CI95_upper_mm']:.4f}] mm\n"
            )
            f.write(
                f"Lower LoA = {ba_recording['LoA_lower_mm']:.4f} mm; "
                f"bootstrap 95% CI = "
                f"[{ba_recording['LoA_lower_CI95_lower_mm']:.4f}, "
                f"{ba_recording['LoA_lower_CI95_upper_mm']:.4f}] mm\n"
            )
            f.write(
                f"Upper LoA = {ba_recording['LoA_upper_mm']:.4f} mm; "
                f"bootstrap 95% CI = "
                f"[{ba_recording['LoA_upper_CI95_lower_mm']:.4f}, "
                f"{ba_recording['LoA_upper_CI95_upper_mm']:.4f}] mm\n"
            )
            f.write(
                "Interpretation: one observation per recording; this "
                "analysis describes the recording-level mean coordinate "
                "offset and is intended to avoid pseudoreplication.\n"
            )
            f.write(
                "IMPORTANT: axis-specific biases may partially cancel when "
                "averaged. The global Bland-Altman analysis should be "
                "interpreted in conjunction with coordinate-wise bias "
                "analysis and 3D spatial error metrics.\n"
            )

        f.write("\nSECONDARY POOLED OBSERVATION-LEVEL BLAND-ALTMAN\n")
        f.write("----------------------------------------------\n")
        if ba_pooled:
            f.write(f"N coordinate observations = {ba_pooled['N']}\n")
            f.write(f"Bias = {ba_pooled['Bias_mm']:.4f} mm\n")
            f.write(
                f"95% CI bias = [{ba_pooled['Bias_CI95_lower_mm']:.4f}, "
                f"{ba_pooled['Bias_CI95_upper_mm']:.4f}] mm\n"
            )
            f.write(f"Lower LoA = {ba_pooled['LoA_lower_mm']:.4f} mm\n")
            f.write(f"Upper LoA = {ba_pooled['LoA_upper_mm']:.4f} mm\n")
            f.write(
                "This pooled analysis is retained for transparency but is "
                "not used as the primary inferential evidence because "
                "repeated frames/markers are not independent observations.\n"
            )

        f.write("\n")

        f.write(
            "DIRECT VS NON-DIRECT PYTHON OBSERVATIONS\n"
            "----------------------------------------\n"
        )
        if direct_available:
            cols = [
                "Direct_N", "NonDirect_N", "NonDirect_percent",
                "Direct_RMSE_3D_mm", "Direct_Agreement_3mm",
                "NonDirect_RMSE_3D_mm", "NonDirect_Agreement_3mm"
            ]
            direct_df = metrics_df[cols].dropna(how="all")
            for col in cols:
                if col in direct_df:
                    val = direct_df[col].mean()
                    f.write(f"Mean {col}: {val:.4f}\n")
            f.write(
                "Note: 'non-direct' means the final Python XYZ sample was not "
                "already fully available in the pre-interpolation output. "
                "Without an explicit interpolation mask, this category may also "
                "contain values affected by prior outlier removal.\n"
            )
        else:
            f.write(
                "No pre-interpolation Python dataset was supplied. Direct vs "
                "non-direct agreement could therefore not be assessed.\n"
            )

        f.write("\n")

        f.write(
            "OUTLIER SENSITIVITY - FULL TABLE\n"
            "--------------------------------\n"
        )

        if not sensitivity_df.empty:
            f.write(
                sensitivity_df.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}"
                )
            )

        f.write("\n\n")

        f.write(
            "INTERPRETATION NOTE\n"
            "-------------------\n"
        )

        f.write(
            "High Pearson correlation or ICC does not by itself "
            "demonstrate spatial equivalence. The primary evidence "
            "for spatial agreement should be based on 3D error, "
            "bias, limits of agreement and per-recording variability.\n"
        )
        f.write(
            "No formal equivalence claim is made unless an a priori "
            "equivalence margin is independently defined and tested.\n"
        )
        f.write(
            "The primary analysis includes ALL valid paired observations. "
            "The sensitivity analyses demonstrate the influence of extreme "
            "discrepancies on aggregate error estimates.\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    root = Tk()
    root.withdraw()

    messagebox.showinfo(
        "Analyse statistique V4.3",
        "Sélectionne d'abord le dossier Nexus, puis le dossier Python.\n\n"
        "IMPORTANT (V4.3) :\n"
        "L'analyse principale n'exclut AUCUNE observation. Les seuils "
        "d'exclusion (50/75/100/150 mm) sont réservés aux analyses de "
        "sensibilité."
    )

    path_n = filedialog.askdirectory(
        title="Select Nexus Folder"
    )

    if not path_n:
        root.destroy()
        return

    path_p = filedialog.askdirectory(
        title="Select Python Folder"
    )

    if not path_p:
        root.destroy()
        return

    path_raw = None
    if ENABLE_DIRECT_VS_NON_DIRECT:
        use_raw = messagebox.askyesno(
            "Données pré-interpolation",
            "Disposez-vous d'un dossier Python contenant les données AVANT interpolation ?\n\n"
            "Oui : sélectionnez ce dossier pour distinguer les observations directes des données non-directes.\n"
            "Non : l'analyse principale sera réalisée normalement, sans cette sensibilité."
        )
        if use_raw:
            path_raw = filedialog.askdirectory(
                title="Sélectionner le dossier Python AVANT interpolation"
            )
            if not path_raw:
                messagebox.showwarning(
                    "Dossier non sélectionné",
                    "Aucun dossier pré-interpolation sélectionné. L'analyse direct/non-direct sera désactivée."
                )

    output_dir = os.path.join(path_p, OUTPUT_DIR_NAME)

    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "statistical_analysis_V4.3.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ],
        force=True
    )

    logger = logging.getLogger(__name__)

    logger.info("Starting V4.3 JNER analysis")

    nexus_files = sorted([
        f for f in os.listdir(path_n)
        if f.lower().endswith(".csv")
    ])

    metrics = []
    marker_rows = []
    axis_rows = []

    pooled_nexus = []
    pooled_python = []
    pooled_errors = []
    raw_errors = []

    failed = []
    ignored = []

    for filename in nexus_files:

        python_filename = filename.replace(".csv", "_labeled.csv")

        nexus_path = os.path.join(path_n, filename)
        python_path = os.path.join(path_p, python_filename)

        if not os.path.exists(python_path):
            ignored.append(filename)
            logger.warning(f"No Python counterpart: {filename}")
            continue

        try:

            nexus_df = read_mocap_csv(nexus_path)
            python_df = read_mocap_csv(python_path)

            markers = parse_marker_columns(nexus_df)

            python_coord_cols = list(python_df.columns[NON_COORD_COLUMNS:])
            n_complete_python = len(python_coord_cols) // 3

            n_markers_common = min(len(markers), n_complete_python)

            if n_markers_common <= 0:
                raise ValueError("Aucun groupe XYZ commun.")

            markers_common = markers[:n_markers_common]

            nexus_arr = extract_xyz(nexus_df, markers_common)

            python_arr = np.full(
                (len(python_df), n_markers_common, 3),
                np.nan,
                dtype=float
            )

            for j in range(n_markers_common):
                cols = python_coord_cols[j * 3:(j + 1) * 3]
                for k, col in enumerate(cols):
                    python_arr[:, j, k] = pd.to_numeric(
                        python_df[col], errors="coerce"
                    ).to_numpy(dtype=float)

            result, details = analyze_recording(
                nexus_arr, python_arr, filename
            )
            
            result["Participant_ID"] = extract_participant_id(filename)

            direct_result = {
                "Final_paired_N": np.nan,
                "Direct_N": np.nan,
                "NonDirect_N": np.nan,
                "NonDirect_percent": np.nan,
                "Direct_RMSE_3D_mm": np.nan,
                "Direct_MAE_3D_mm": np.nan,
                "Direct_Median_error_3D_mm": np.nan,
                "Direct_P95_error_3D_mm": np.nan,
                "Direct_Agreement_3mm": np.nan,
                "Direct_Agreement_5mm": np.nan,
                "Direct_Agreement_10mm": np.nan,
                "NonDirect_RMSE_3D_mm": np.nan,
                "NonDirect_MAE_3D_mm": np.nan,
                "NonDirect_Median_error_3D_mm": np.nan,
                "NonDirect_P95_error_3D_mm": np.nan,
                "NonDirect_Agreement_3mm": np.nan,
                "NonDirect_Agreement_5mm": np.nan,
                "NonDirect_Agreement_10mm": np.nan,
            }

            if path_raw:
                raw_path = find_matching_file(path_raw, filename)
                if raw_path is not None:
                    try:
                        raw_df = read_mocap_csv(raw_path)
                        raw_arr = extract_python_array_by_position(
                            raw_df, n_markers_common
                        )
                        direct_result = calculate_direct_vs_nondirect(
                            nexus_arr, python_arr, raw_arr
                        )
                    except Exception as raw_exc:
                        logger.warning(
                            f"Direct/non-direct impossible for {filename}: {raw_exc}"
                        )
                else:
                    logger.warning(
                        f"No pre-interpolation counterpart for {filename}"
                    )

            result.update(direct_result)
            metrics.append(result)

            m_rows, a_rows = analyze_marker_level(
                nexus_arr, python_arr, filename, markers_common
            )

            marker_rows.extend(m_rows)
            axis_rows.extend(a_rows)

            if len(details["error_3d"]):
                pooled_errors.append(details["error_3d"])

            if len(details["error_3d_raw"]):
                raw_errors.append(details["error_3d_raw"])

            if len(details["nexus_coord"]):
                pooled_nexus.append(details["nexus_coord"])
                pooled_python.append(details["python_coord"])

            logger.info(
                f"Processed: {filename} | "
                f"RMSE 3D (no exclusion) = {result['RMSE_3D_mm']:.3f} mm | "
                f"RMSE 3D (sensitivity) = "
                f"{result['RMSE_3D_sensitivity_mm']:.3f} mm | "
                f"P95 (no exclusion) = {result['P95_error_3D_mm']:.3f} mm"
            )

        except Exception as exc:
            failed.append((filename, str(exc)))
            logger.exception(f"Error processing {filename}")

    if not metrics:
        messagebox.showerror(
            "Erreur",
            "Aucune acquisition valide n'a été analysée."
        )
        root.destroy()
        return

    metrics_df = pd.DataFrame(metrics)

    pooled_errors_array = (
        np.concatenate(pooled_errors)
        if pooled_errors else np.array([], dtype=float)
    )

    pooled_raw_errors_array = (
        np.concatenate(raw_errors)
        if raw_errors else np.array([], dtype=float)
    )

    pooled_nexus_array = (
        np.concatenate(pooled_nexus)
        if pooled_nexus else np.array([], dtype=float)
    )

    pooled_python_array = (
        np.concatenate(pooled_python)
        if pooled_python else np.array([], dtype=float)
    )

    pooled_rmse_3d = (
        float(np.sqrt(np.mean(pooled_errors_array ** 2)))
        if len(pooled_errors_array) else np.nan
    )

    global_accuracy = {}

    for threshold in ACCURACY_THRESHOLDS_MM:
        if len(pooled_errors_array):
            global_accuracy[threshold] = float(
                np.mean(pooled_errors_array < threshold)
            )
        else:
            global_accuracy[threshold] = np.nan

    ba_pooled = bland_altman_stats(
        pooled_nexus_array, pooled_python_array
    )

    ba_recording = recording_level_bland_altman(metrics_df)

    inferential = recording_level_statistics(metrics_df)

    sensitivity_df = calculate_outlier_sensitivity(raw_errors)

    metrics_path = os.path.join(
        output_dir, "02_metrics_per_recording.csv"
    )
    metrics_df.to_csv(
        metrics_path, index=False, sep=";", encoding="utf-8-sig"
    )

    marker_df = pd.DataFrame(marker_rows)
    marker_path = os.path.join(
        output_dir, "03_metrics_per_marker.csv"
    )
    marker_df.to_csv(
        marker_path, index=False, sep=";", encoding="utf-8-sig"
    )

    if not marker_df.empty:
        marker_summary_rows = []
        for marker_name, g in marker_df.groupby("Marker"):
            row = {
                "Marker": marker_name,
                "N_recordings": int(g["Filename"].nunique()),
                "N_valid_3D": int(g["N_valid_3D"].sum()),
                "N_gross_discrepancies": int(g["N_gross_discrepancies"].sum()),
            }
            for col in [
                "RMSE_3D_mm", "MAE_3D_mm", "Median_error_3D_mm",
                "P95_error_3D_mm", "Accuracy_3D_1mm",
                "Accuracy_3D_2mm", "Accuracy_3D_3mm",
                "Accuracy_3D_5mm", "Accuracy_3D_10mm"
            ]:
                vals = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy()
                row[f"{col}_mean_recording"] = safe_mean(vals)
                row[f"{col}_median_recording"] = safe_median(vals)
                row[f"{col}_sd_recording"] = safe_std(vals)
            marker_summary_rows.append(row)

        marker_summary_df = pd.DataFrame(marker_summary_rows)
        marker_summary_df.to_csv(
            os.path.join(output_dir, "03b_metrics_per_marker_summary.csv"),
            index=False, sep=";", encoding="utf-8-sig"
        )

    axis_df = pd.DataFrame(axis_rows)
    axis_path = os.path.join(
        output_dir, "04_metrics_per_axis.csv"
    )
    axis_df.to_csv(
        axis_path, index=False, sep=";", encoding="utf-8-sig"
    )

    accuracy_df = pd.DataFrame([
        {
            "Threshold_3D_mm": t,
            "Agreement_percent": (
                global_accuracy[t] * 100
                if np.isfinite(global_accuracy[t])
                else np.nan
            ),
            "N_3D_comparisons": len(pooled_errors_array)
        }
        for t in ACCURACY_THRESHOLDS_MM
    ])

    accuracy_path = os.path.join(
        output_dir, "05_accuracy_thresholds.csv"
    )
    accuracy_df.to_csv(
        accuracy_path, index=False, sep=";", encoding="utf-8-sig"
    )

    sensitivity_path = os.path.join(
        output_dir, "06_outlier_sensitivity.csv"
    )
    sensitivity_df.to_csv(
        sensitivity_path, index=False, sep=";", encoding="utf-8-sig"
    )

    ba_recording_export = {
        k: v for k, v in (ba_recording or {}).items()
        if k != "recording_data"
    }
    ba_df = pd.DataFrame(
        [ba_recording_export] if ba_recording_export else []
    )
    ba_path = os.path.join(
        output_dir, "07_bland_altman_recording_level.csv"
    )
    ba_df.to_csv(
        ba_path, index=False, sep=";", encoding="utf-8-sig"
    )

    if ba_recording and "recording_data" in ba_recording:
        ba_recording["recording_data"].to_csv(
            os.path.join(output_dir, "07b_bland_altman_recording_points.csv"),
            index=False, sep=";", encoding="utf-8-sig"
        )

    ba_pooled_df = pd.DataFrame([ba_pooled] if ba_pooled else [])
    ba_pooled_df.to_csv(
        os.path.join(output_dir, "07c_bland_altman_pooled_observations.csv"),
        index=False, sep=";", encoding="utf-8-sig"
    )

    summary = {
        "N_recordings": len(metrics_df),
        "N_failed": len(failed),
        "N_ignored": len(ignored),

        "Analysis_version": "V4.3 - primary without exclusion",

        "RMSE_3D_mean_mm": inferential.get("RMSE3D_mean_mm", np.nan),
        "RMSE_3D_SD_mm": inferential.get("RMSE3D_sd_mm", np.nan),
        "RMSE_3D_median_mm": inferential.get("RMSE3D_median_mm", np.nan),
        "RMSE_3D_Q1_mm": inferential.get("RMSE3D_Q1_mm", np.nan),
        "RMSE_3D_Q3_mm": inferential.get("RMSE3D_Q3_mm", np.nan),
        "RMSE_3D_mean_CI95_lower_mm": inferential.get(
            "RMSE3D_mean_CI95_lower_mm", np.nan
        ),
        "RMSE_3D_mean_CI95_upper_mm": inferential.get(
            "RMSE3D_mean_CI95_upper_mm", np.nan
        ),
        
        "RMSE_3D_mean_CI95_cluster_lower_mm": inferential.get(
            "RMSE3D_mean_CI95_cluster_lower_mm", np.nan
        ),
        "RMSE_3D_mean_CI95_cluster_upper_mm": inferential.get(
            "RMSE3D_mean_CI95_cluster_upper_mm", np.nan
        ),
        
        "RMSE_3D_pooled_all_observations_mm": pooled_rmse_3d,

        "RMSE_3D_sensitivity_mean_mm": inferential.get(
            "RMSE3D_sensitivity_mean_mm", np.nan
        ),
        "RMSE_3D_sensitivity_median_mm": inferential.get(
            "RMSE3D_sensitivity_median_mm", np.nan
        ),

        "Gross_discrepancies_3D_total": int(
            metrics_df['Gross_discrepancies_3D'].sum()
        ),
        "Gross_discrepancies_3D_percent": (
            100 * metrics_df['Gross_discrepancies_3D'].sum()
            / metrics_df['Valid_3D_comparisons_raw'].sum()
            if metrics_df['Valid_3D_comparisons_raw'].sum() else np.nan
        ),

        "BlandAltman_recording_N": ba_recording.get("N_recordings", np.nan) if ba_recording else np.nan,
        "BlandAltman_recording_bias_mm": ba_recording.get("Bias_mm", np.nan) if ba_recording else np.nan,
        "BlandAltman_recording_LoA_lower_mm": ba_recording.get("LoA_lower_mm", np.nan) if ba_recording else np.nan,
        "BlandAltman_recording_LoA_upper_mm": ba_recording.get("LoA_upper_mm", np.nan) if ba_recording else np.nan,
        "BlandAltman_recording_bias_CI95_lower_mm": ba_recording.get("Bias_CI95_lower_mm", np.nan) if ba_recording else np.nan,
        "BlandAltman_recording_bias_CI95_upper_mm": ba_recording.get("Bias_CI95_upper_mm", np.nan) if ba_recording else np.nan,
        "Accuracy_3D_3mm_mean_recording_percent":
            inferential.get("Accuracy3D_3mm_mean_recording_percent", np.nan),
        "Accuracy_3D_3mm_mean_recording_CI95_lower_percent":
            inferential.get("Accuracy3D_3mm_mean_recording_CI95_lower_percent", np.nan),
        "Accuracy_3D_3mm_mean_recording_CI95_upper_percent":
            inferential.get("Accuracy3D_3mm_mean_recording_CI95_upper_percent", np.nan),
        "Accuracy_3D_3mm_mean_recording_CI95_cluster_lower_percent":
            inferential.get(
        "Accuracy3D_3mm_mean_recording_CI95_cluster_lower_percent",
        np.nan
    ),
        "Accuracy_3D_3mm_mean_recording_CI95_cluster_upper_percent":
            inferential.get(
        "Accuracy3D_3mm_mean_recording_CI95_cluster_upper_percent",
        np.nan
    ),
        "MAE_3D_mean_across_recordings": metrics_df["MAE_3D_mm"].mean(),
        "Median_error_3D_mean_across_recordings": metrics_df["Median_error_3D_mm"].mean(),
        "P95_error_3D_mean_across_recordings": metrics_df["P95_error_3D_mm"].mean(),
        "P95_error_3D_median_across_recordings": metrics_df["P95_error_3D_mm"].median(),

        "Bias_X_mean_mm": metrics_df["Bias_X_mm"].mean(),
        "Bias_Y_mean_mm": metrics_df["Bias_Y_mm"].mean(),
        "Bias_Z_mean_mm": metrics_df["Bias_Z_mm"].mean(),

        "ICC_2_1_mean": metrics_df["ICC_2_1_secondary"].mean(),

        "Paired_availability_mean_percent": metrics_df["Paired_availability"].mean() * 100,
        "Direct_nonDirect_analysis_available": bool(path_raw),
    }

    for threshold in ACCURACY_THRESHOLDS_MM:
        summary[f"Global_accuracy_3D_{threshold}mm_percent"] = (
            global_accuracy[threshold] * 100
            if np.isfinite(global_accuracy[threshold])
            else np.nan
        )
        summary[f"Recording_accuracy_3D_{threshold}mm_percent"] = inferential.get(
            f"Accuracy3D_{threshold}mm_mean_recording_percent", np.nan
        )
        summary[f"Recording_accuracy_3D_{threshold}mm_CI95_lower_percent"] = inferential.get(
            f"Accuracy3D_{threshold}mm_mean_recording_CI95_lower_percent", np.nan
        )
        summary[f"Recording_accuracy_3D_{threshold}mm_CI95_upper_percent"] = inferential.get(
            f"Accuracy3D_{threshold}mm_mean_recording_CI95_upper_percent", np.nan
        )

    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(
        output_dir, "01_summary_global.csv"
    )
    summary_df.to_csv(
        summary_path, index=False, sep=";", encoding="utf-8-sig"
    )

    report_path = os.path.join(
        output_dir, "08_statistical_report.txt"
    )

    write_report(
        report_path,
        path_n,
        path_p,
        metrics_df,
        global_accuracy,
        ba_recording,
        ba_pooled,
        sensitivity_df,
        inferential,
        pooled_rmse_3d,
        path_raw is not None
    )

    if failed:
        failed_path = os.path.join(output_dir, "failed_files.txt")
        with open(failed_path, "w", encoding="utf-8") as f:
            for filename, error in failed:
                f.write(f"{filename}\t{error}\n")

    if ignored:
        ignored_path = os.path.join(output_dir, "ignored_files.txt")
        with open(ignored_path, "w", encoding="utf-8") as f:
            for filename in ignored:
                f.write(filename + "\n")

    figure_path = os.path.join(
        output_dir, "10_publication_figure_V4.3_JNER.png"
    )

    create_publication_figure(
        pooled_nexus_array,
        pooled_python_array,
        pooled_errors_array,
        metrics_df,
        global_accuracy,
        ba_recording,
        figure_path
    )

    sensitivity_figure_path = os.path.join(
        output_dir, "11_sensitivity_to_exclusion_threshold.png"
    )

    create_sensitivity_figure(
        sensitivity_df,
        sensitivity_figure_path
    )

    print("\n")
    print("=" * 70)
    print("ANALYSE STATISTIQUE V4.3 — JNER")
    print("=" * 70)
    print(f"Acquisitions analysées : {len(metrics_df)}")
    print(f"Échecs : {len(failed)}")
    print(f"Fichiers sans équivalent Python : {len(ignored)}")
    print("-" * 70)

    print("ANALYSE PRINCIPALE (SANS EXCLUSION)")
    print("-" * 70)
    print(
        f"RMSE 3D moyen : "
        f"{inferential.get('RMSE3D_mean_mm', np.nan):.3f} mm"
    )
    print(
        f"RMSE 3D médian : "
        f"{inferential.get('RMSE3D_median_mm', np.nan):.3f} mm"
    )
    print(f"RMSE 3D poolé : {pooled_rmse_3d:.3f} mm")
    print(
        f"IC95% moyenne RMSE : "
        f"[{inferential.get('RMSE3D_mean_CI95_lower_mm', np.nan):.3f}, "
        f"{inferential.get('RMSE3D_mean_CI95_upper_mm', np.nan):.3f}] mm"
    )
    print(
        f"Gross discrepancies (>= 100 mm) : "
        f"{metrics_df['Gross_discrepancies_3D'].sum()} "
        f"({summary['Gross_discrepancies_3D_percent']:.3f}%)"
    )

    print("-" * 70)
    print("ANALYSE DE SENSIBILITÉ (EXCLUSION 100 MM)")
    print("-" * 70)
    print(
        f"RMSE 3D moyen (sensibilité) : "
        f"{inferential.get('RMSE3D_sensitivity_mean_mm', np.nan):.3f} mm"
    )
    print(
        f"RMSE 3D médian (sensibilité) : "
        f"{inferential.get('RMSE3D_sensitivity_median_mm', np.nan):.3f} mm"
    )

    print("-" * 70)
    for threshold in ACCURACY_THRESHOLDS_MM:
        value = inferential.get(
            f"Accuracy3D_{threshold}mm_mean_recording_percent",
            np.nan
        )
        print(
            f"Accord 3D < {threshold:>2} mm (SANS exclusion, moyenne/acquisition) : "
            f"{value:.2f}%"
        )

    print("-" * 70)
    print(f"Biais X : {inferential.get('Bias_X_mm_mean', np.nan):.4f} mm")
    print(f"Biais Y : {inferential.get('Bias_Y_mm_mean', np.nan):.4f} mm")
    print(f"Biais Z : {inferential.get('Bias_Z_mm_mean', np.nan):.4f} mm")

    if ba_recording:
        print("-" * 70)
        print(
            f"Bland-Altman recording-level bias : "
            f"{ba_recording['Bias_mm']:.4f} mm"
        )
        print(
            f"Recording-level LoA : "
            f"[{ba_recording['LoA_lower_mm']:.4f}, "
            f"{ba_recording['LoA_upper_mm']:.4f}] mm"
        )

    print("-" * 70)
    print("Résultats enregistrés dans :")
    print(output_dir)
    print("=" * 70)

    messagebox.showinfo(
        "Analyse terminée",
        "La V4.3 JNER est terminée.\n\n"
        f"Résultats :\n{output_dir}\n\n"
        "Fichiers clés :\n"
        "- 01_summary_global.csv\n"
        "- 02_metrics_per_recording.csv\n"
        "- 06_outlier_sensitivity.csv\n"
        "- 10_publication_figure_V4.3_JNER.png\n"
        "- 11_sensitivity_to_exclusion_threshold.png"
    )

    root.destroy()


if __name__ == "__main__":
    main()