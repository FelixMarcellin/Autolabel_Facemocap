# -*- coding: utf-8 -*-
"""
Automated non-rigid facial marker labeling pipeline.

This script implements the pipeline described in the Methods section:
1. Marker set definition and static reference extraction
2. Initial frame-wise marker assignment (Hungarian + voting)
3. Temporal propagation of marker identities
4. Data refinement and output generation (C3D + CSV)

Author: Félix Marcellin
"""

import os
import ezc3d
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from collections import Counter, defaultdict
import csv
from tkinter import filedialog, Tk
import time
import logging

# =====================================================
# LOGGING CONFIGURATION
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("autolabel_pipeline.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =====================================================
# GLOBAL PARAMETERS (documented in Methods section)
# =====================================================
MAX_DISTANCE = 50
OUTLIER_THRESHOLD = 100
LABEL_VOTE_THRESHOLD = 0.4
TRACKING_THRESHOLD = 0.6
SKIP_FRAMES = 15
N_FRAMES_INITIAL_LABELING = 5

GROUND_MARGIN_MM = 20
MIN_VOTE_FOR_LABEL = 0.6
MAX_FRAME_JUMP_MM = 40


# =====================================================
# MARKER SET LOADING (CSV)
# =====================================================
def load_marker_set():
    """Load the expected marker set from a CSV file."""
    logger.info("Loading marker set CSV file...")
    root = Tk()
    root.withdraw()
    csv_path = filedialog.askopenfilename(
        title="Select marker set CSV file",
        filetypes=[("CSV files", "*.csv")]
    )
    root.destroy()

    if not csv_path:
        raise RuntimeError("No marker set file selected")

    df = pd.read_csv(csv_path, header=None)

    first_cell = str(df.iloc[0, 0]).strip().lower()
    if not any(char.isdigit() for char in first_cell):
        df = df.iloc[1:]

    markers = df.iloc[:, 0].astype(str).str.strip().tolist()

    logger.info(f"{len(markers)} markers loaded from: {csv_path}")
    return markers


# =====================================================
# C3D UTILITIES
# =====================================================
def load_c3d(filepath):
    logger.info(f"Loading C3D file: {os.path.basename(filepath)}")
    c3d = ezc3d.c3d(filepath)

    all_points = c3d['data']['points'][:3, :, :]
    n_frames = all_points.shape[2]

    if n_frames > SKIP_FRAMES:
        points = all_points[:, :, SKIP_FRAMES:]
        logger.info(f"Skipped {SKIP_FRAMES} initial frames ({SKIP_FRAMES}/{n_frames})")
    else:
        points = all_points
        logger.warning(f"File too short ({n_frames} frames), no frames skipped")

    labels = [str(l) for l in c3d['parameters']['POINT']['LABELS']['value']]
    rate = c3d['parameters']['POINT']['RATE']['value'][0]

    return {'c3d': c3d, 'points': points, 'labels': labels, 'rate': rate}


# =====================================================
# ASSIGNMENT UTILITY (Hungarian with rejection)
# =====================================================
def hungarian_with_rejection(cost_matrix, max_distance):
    n_rows, n_cols = cost_matrix.shape
    n = max(n_rows, n_cols)

    augmented = np.full((n, n), max_distance, dtype=float)
    augmented[:n_rows, :n_cols] = cost_matrix

    row_ind, col_ind = linear_sum_assignment(augmented)

    pairs = []
    for i, j in zip(row_ind, col_ind):
        if i < n_rows and j < n_cols and cost_matrix[i, j] < max_distance:
            pairs.append((i, j))
    return pairs


# =====================================================
# INITIAL MARKER ASSIGNMENT (Hungarian + voting)
# =====================================================
def match_markers(static, movement, frame_indices, filename):
    logger.info(f"Initial labeling for {os.path.basename(filename)}...")

    static_pos = np.nanmean(static['points'], axis=2).T
    static_labels = np.array(static['labels'])
    valid_static = ~np.isnan(static_pos).any(axis=1)

    static_clean = static_pos[valid_static]
    labels_clean = static_labels[valid_static]

    z_floor = np.nanmin(static_clean[:, 2]) - GROUND_MARGIN_MM

    logger.info(f"  [DIAG] Static markers: {len(static_clean)}")
    logger.info(f"  [DIAG] Static Z range: [{np.nanmin(static_clean[:,2]):.1f}, "
                f"{np.nanmax(static_clean[:,2]):.1f}] mm")
    logger.info(f"  [DIAG] Ground filter threshold (z_floor): {z_floor:.1f} mm")
    logger.info(f"  [DIAG] Selected frames for voting: {list(frame_indices)}")

    votes = {label: [] for label in labels_clean}
    total_assignments, successful = 0, 0
    per_frame_stats = []

    for idx, f in enumerate(frame_indices):
        move = movement['points'][:, :, f].T
        valid_move = ~np.isnan(move).any(axis=1)
        move_clean = move[valid_move]
        move_idx_clean = np.where(valid_move)[0]

        n_detected = len(move_clean)

        if len(move_clean) == 0:
            per_frame_stats.append((f, n_detected, 0, 0))
            continue

        above_floor = move_clean[:, 2] > z_floor
        n_above = int(np.sum(above_floor))
        move_clean = move_clean[above_floor]
        move_idx_clean = move_idx_clean[above_floor]

        if len(move_clean) == 0:
            per_frame_stats.append((f, n_detected, 0, 0))
            continue

        cost = cdist(move_clean, static_clean)
        pairs = hungarian_with_rejection(cost, MAX_DISTANCE)
        total_assignments += len(move_clean)

        per_frame_stats.append((f, n_detected, n_above, len(pairs)))

        for i, j in pairs:
            votes[labels_clean[j]].append(move_idx_clean[i])
            successful += 1

    logger.info("  [DIAG] Per-frame stats (frame, detected, above_floor, assigned):")
    for row in per_frame_stats:
        logger.info(f"           {row}")

    n_zero = sum(1 for v in votes.values() if len(v) == 0)
    n_below = sum(1 for v in votes.values()
                  if 0 < len(v) < MIN_VOTE_FOR_LABEL * len(frame_indices))
    n_ok = sum(1 for v in votes.values()
               if len(v) >= MIN_VOTE_FOR_LABEL * len(frame_indices))
    logger.info(f"  [DIAG] Vote distribution: "
                f"no_vote={n_zero}, below_threshold={n_below}, OK={n_ok}")
    logger.info(f"  [DIAG] Threshold = {MIN_VOTE_FOR_LABEL} "
                f"× {len(frame_indices)} frames = "
                f"{MIN_VOTE_FOR_LABEL * len(frame_indices):.1f} votes needed")

    success_rate = successful / total_assignments if total_assignments else 0

    final_labels = {}
    vote_scores = {}

    for label, idxs in votes.items():
        if idxs:
            chosen, count = Counter(idxs).most_common(1)[0]
            vote_score = count / len(frame_indices)
            vote_scores[label] = vote_score
            if vote_score >= MIN_VOTE_FOR_LABEL:
                final_labels[chosen] = label
        else:
            vote_scores[label] = 0.0

    logger.info(
        f"Initial labeling completed: {success_rate:.1%} success rate, "
        f"{len(final_labels)}/{len(labels_clean)} markers confirmed"
    )
    return final_labels, success_rate, vote_scores, total_assignments, successful


# =====================================================
# TEMPORAL PROPAGATION
# =====================================================
def propagate_labels(movement, initial_labels, filename,
                     max_jump_mm=MAX_FRAME_JUMP_MM):
    logger.info(f"Temporal propagation for {os.path.basename(filename)}...")

    n_frames = movement['points'].shape[2]
    history = [{} for _ in range(n_frames)]
    history[0] = initial_labels.copy()

    confs = []
    for f in range(1, n_frames):
        new = {}
        for idx, lab in history[f - 1].items():
            pt_prev = movement['points'][:3, idx, f - 1]
            pt_curr = movement['points'][:3, idx, f]

            if np.isnan(pt_curr).any():
                continue

            if not np.isnan(pt_prev).any():
                jump = np.linalg.norm(pt_curr - pt_prev)
                if jump > max_jump_mm:
                    continue

            new[idx] = lab
        denom = len(initial_labels) if initial_labels else 1
        confs.append(len(new) / denom)
        history[f] = new

    tracking_conf = float(np.mean(confs)) if confs else 0.0
    logger.info(f"Temporal propagation completed: mean confidence = {tracking_conf:.3f}")
    return history, tracking_conf


# =====================================================
# TRACKING STABILITY ANALYSIS
# =====================================================
def analyze_tracking_stability(labels_history, movement):
    n_frames = movement['points'].shape[2]
    presence = Counter()

    for f in range(n_frames):
        for idx, lab in labels_history[f].items():
            if not np.isnan(movement['points'][:, idx, f]).any():
                presence[lab] += 1

    return {lab: presence[lab] / n_frames for lab in presence}


# =====================================================
# OUTLIER REMOVAL + INTERPOLATION
# =====================================================
def remove_outliers(points, labels, static):
    logger.info("Removing outliers...")
    static_pos = np.nanmean(static['points'], axis=2).T
    total_outliers = 0
    for idx, (label, _) in enumerate(labels.items()):
        ref_idx = np.where(np.array(static['labels']) == label)[0]
        if not len(ref_idx):
            continue
        ref = static_pos[ref_idx[0]]
        traj = points[:3, idx, :].T
        dists = np.linalg.norm(traj - ref, axis=1)
        outlier_count = np.sum(dists > OUTLIER_THRESHOLD)
        if outlier_count > 0:
            logger.info(f"  {label}: {outlier_count} outliers detected")
            total_outliers += outlier_count
            points[:3, idx, dists > OUTLIER_THRESHOLD] = np.nan
    logger.info(f"Total outliers removed: {total_outliers}")
    return points


def interpolate(points):
    logger.info("Interpolating missing data...")
    total_interpolated = 0
    for m in range(points.shape[1]):
        for c in range(3):
            data = points[c, m]
            mask = ~np.isnan(data)
            if np.sum(mask) > 1:
                nan_count = np.sum(~mask)
                if nan_count > 0:
                    total_interpolated += nan_count
                    data[~mask] = np.interp(
                        np.flatnonzero(~mask),
                        np.flatnonzero(mask),
                        data[mask]
                    )
                    points[c, m] = data
    logger.info(f"{total_interpolated} points interpolated")
    return points


# =====================================================
# OUTPUT FILE GENERATION (C3D + CSV)
# =====================================================
def create_files(movement, labels_per_frame, output_dir, filename,
                 static, desired_order, votes, tracking):
    """
    Generate labeled C3D and formatted CSV output files.

    IMPORTANT: unlabeled markers are written as EMPTY CELLS in the CSV
    (not as 0.0, 0.0, 0.0), to avoid spurious large deviations when the
    data are later analyzed or compared.
    """
    logger.info(f"Creating output files for {os.path.basename(filename)}...")

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(filename)[0]

    out_c3d = os.path.join(output_dir, f"{base}_labeled.c3d")
    out_csv = os.path.join(output_dir, f"{base}_labeled.csv")

    n_frames = movement['points'].shape[2]
    n_markers = len(desired_order)
    rate = movement['rate']

    # Build the (4, n_markers, n_frames) array with NaN by default.
    points = np.full((4, n_markers, n_frames), np.nan)
    lab_idx = {l: i for i, l in enumerate(desired_order)}

    # Fill the points array from the label history.
    for f, lm in enumerate(labels_per_frame):
        for idx, lab in lm.items():
            if lab in lab_idx:
                i = lab_idx[lab]
                pt = movement['points'][:, idx, f]
                if not np.isnan(pt).any():
                    points[:3, i, f] = pt
                    points[3, i, f] = 1

    # Remove outliers and interpolate BEFORE writing CSV, so that the
    # interpolated values appear in the CSV and no 0.0 is written for
    # genuinely missing data.
    points = interpolate(remove_outliers(points, lab_idx, static))

    # ---- C3D output ----
    out = ezc3d.c3d()
    out['data']['points'] = points
    out['parameters']['POINT']['LABELS']['value'] = desired_order
    out['parameters']['POINT']['USED']['value'] = [n_markers]
    out['parameters']['POINT']['RATE']['value'] = [rate]
    out.write(out_c3d)

    # ---- CSV output ----
    header_lines = [
        f"Filename,{base}_labeled",
        f"Sampling rate,{rate}",
        f"Nb Frames,{n_frames}",
        f"Nb markers,{n_markers}"
    ]

    csv_header = ["Frame", "Time (s)"]
    for marker in desired_order:
        csv_header.extend([f"{marker}_X", f"{marker}_Y", f"{marker}_Z"])

    time_step = 1.0 / rate

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line + '\n')
        f.write(",".join(csv_header) + '\n')

        writer = csv.writer(f)
        for frame_i in range(n_frames):
            row = [frame_i, frame_i * time_step]
            for m_i in range(n_markers):
                x = points[0, m_i, frame_i]
                y = points[1, m_i, frame_i]
                z = points[2, m_i, frame_i]
                if np.isnan(x) or np.isnan(y) or np.isnan(z):
                    # Empty cells instead of 0.0,0.0,0.0
                    row.extend(["", "", ""])
                else:
                    row.extend([float(x), float(y), float(z)])
            writer.writerow(row)

    logger.info(f"Files created: {os.path.basename(out_c3d)}, {os.path.basename(out_csv)}")


# =====================================================
# MAIN PIPELINE
# =====================================================
def process_root(root_folder, desired_order):
    logger.info("=" * 70)
    logger.info("STARTING AUTOMATED MARKER LABELING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Root folder: {root_folder}")
    logger.info(f"Number of markers: {len(desired_order)}")

    global_summary = {}
    global_votes = defaultdict(list)
    global_tracks = defaultdict(list)
    processed_files = 0
    failed_files = []

    subfolders = [sub for sub in os.listdir(root_folder)
                  if os.path.isdir(os.path.join(root_folder, sub))]

    for sub_idx, sub in enumerate(subfolders, 1):
        logger.info(f"\nSubfolder {sub_idx}/{len(subfolders)}: {sub}")
        subfolder = os.path.join(root_folder, sub)

        static_files = [f for f in os.listdir(subfolder)
                        if f.lower().endswith("statique.c3d")]

        if not static_files:
            logger.warning(f"No static file found in {sub}")
            continue

        static_file = static_files[0]
        logger.info(f"Static file: {static_file}")
        static = load_c3d(os.path.join(subfolder, static_file))

        output_dir = os.path.join(subfolder, "labeled_output")

        dyn_files = [f for f in os.listdir(subfolder)
                     if f.lower().endswith(".c3d") and "statique" not in f.lower()]

        logger.info(f"Dynamic files to process: {len(dyn_files)}")

        for file_idx, file in enumerate(dyn_files, 1):
            logger.info(f"\n  Processing {file_idx}/{len(dyn_files)}: {file}")
            start_time = time.time()

            try:
                mov = load_c3d(os.path.join(subfolder, file))

                valid = np.sum(~np.isnan(mov['points'][0]), axis=0)
                frames = np.argsort(valid)[-N_FRAMES_INITIAL_LABELING:]

                logger.info("  >> Calling match_markers...")
                result = match_markers(static, mov, frames, file)
                logger.info(f"  >> match_markers returned type: {type(result)}, "
                            f"length: {len(result) if result is not None else 'None'}")

                if result is None:
                    raise RuntimeError("match_markers returned None!")

                labels, init_conf, votes, total_assign, success_assign = result
                logger.info(f"  >> Labels: {len(labels)} markers")

                logger.info("  >> Calling propagate_labels...")
                prop_result = propagate_labels(mov, labels, file)
                logger.info(f"  >> propagate_labels returned type: {type(prop_result)}")

                if prop_result is None:
                    raise RuntimeError("propagate_labels returned None!")

                history, track_conf = prop_result
                logger.info("  >> Calling analyze_tracking_stability...")
                tracking = analyze_tracking_stability(history, mov)

                logger.info("  >> Calling create_files...")
                create_files(mov, history, output_dir, file, static,
                             desired_order, votes, tracking)

                processed_files += 1
                global_summary[file] = {
                    "subfolder": sub,
                    "init_conf": init_conf,
                    "track_conf": track_conf,
                    "total_assignments": total_assign,
                    "successful_assignments": success_assign,
                    "confirmed_labels": len(labels),
                }
                for lab, v in votes.items():
                    global_votes[lab].append(v)
                for lab, t in tracking.items():
                    global_tracks[lab].append(t)

            except Exception as e:
                import traceback
                logger.error(f"  Error processing {file}: {e}")
                logger.error(f"  Traceback:\n{traceback.format_exc()}")
                failed_files.append((sub, file, str(e)))
                continue

    logger.info("\n" + "=" * 70)
    logger.info("GENERATING GLOBAL REPORTS")
    logger.info("=" * 70)

    summary_csv = os.path.join(root_folder, "global_tracking_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["File", "Subfolder", "Init_conf", "Track_conf",
                    "Total_assignments", "Successful_assignments",
                    "Confirmed_labels"])
        for name, vals in global_summary.items():
            w.writerow([name, vals["subfolder"], vals["init_conf"],
                        vals["track_conf"], vals["total_assignments"],
                        vals["successful_assignments"],
                        vals["confirmed_labels"]])

    marker_csv = os.path.join(root_folder, "global_marker_stability.csv")
    with open(marker_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Marker", "Mean_Labeling_Stability", "Mean_Tracking_Stability",
                    "Global_Stability", "Files_Count", "Status"])
        for lab in desired_order:
            lv = np.mean(global_votes.get(lab, [0]))
            tv = np.mean(global_tracks.get(lab, [0]))
            gs = (lv + tv) / 2
            count = max(len(global_votes.get(lab, [])),
                        len(global_tracks.get(lab, [])))
            status = "UNSTABLE" if (lv < LABEL_VOTE_THRESHOLD or
                                    tv < TRACKING_THRESHOLD) else "OK"
            w.writerow([lab, round(lv, 3), round(tv, 3), round(gs, 3),
                        count, status])

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Subfolders processed: {len(subfolders)}")
    logger.info(f"Files successfully processed: {processed_files}")
    logger.info(f"Files failed: {len(failed_files)}")
    if failed_files:
        logger.warning("Failed files:")
        for sub, f, err in failed_files:
            logger.warning(f"  [{sub}] {f}: {err}")


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    desired_order = load_marker_set()

    root = Tk()
    root.withdraw()
    root_folder = filedialog.askdirectory(title="Select root folder")
    root.destroy()

    if root_folder:
        process_root(root_folder, desired_order)
    else:
        logger.warning("No folder selected. Exiting.")