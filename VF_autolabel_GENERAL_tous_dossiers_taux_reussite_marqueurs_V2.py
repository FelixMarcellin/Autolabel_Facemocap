# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:36:33 2026

@author: felima
"""

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
# Maximum distance (mm) for a valid marker assignment during initial labeling.
# Assignments above this threshold are considered spurious.
MAX_DISTANCE = 50

# Distance threshold (mm) relative to static reference position.
# Trajectories exceeding this distance are considered tracking outliers
# (physiologically implausible facial motion).
OUTLIER_THRESHOLD = 100

# Minimum vote score for a marker to be considered stably labeled during
# the initial voting step.
LABEL_VOTE_THRESHOLD = 0.4

# Minimum tracking stability for a marker to be considered reliably tracked
# across the whole sequence.
TRACKING_THRESHOLD = 0.6

# Number of initial frames to discard at the beginning of each recording.
# These frames are affected by an acquisition artifact in Vicon software
# resulting in incomplete marker trajectories. They do not contain any
# movement of interest.
SKIP_FRAMES = 15

# Number of frames used for initial marker assignment (the N frames
# with the highest number of valid markers are selected).
N_FRAMES_INITIAL_LABELING = 5

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

    # Remove potential text header
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
    """
    Load a C3D file and skip the first SKIP_FRAMES frames.

    The first SKIP_FRAMES frames are discarded because they are affected
    by an acquisition artifact in Vicon, resulting in incomplete marker
    trajectories. These frames do not contain any movement of interest.
    """
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
# INITIAL MARKER ASSIGNMENT (Hungarian + voting)
# =====================================================
def match_markers(static, movement, frame_indices, filename):
    """
    Perform initial marker assignment.

    For each of the selected frames, marker assignment is formulated as a
    linear sum assignment problem minimizing Euclidean distance between
    detected markers and static reference positions. The Hungarian algorithm
    is used to compute the optimal one-to-one assignment.

    A voting mechanism is then applied across the selected frames: for each
    static marker label, the dynamic marker index most frequently assigned
    to it is retained.

    Returns:
        final_labels: dict {dynamic_index: marker_label}
        success_rate: proportion of successful assignments (distance < MAX_DISTANCE)
        vote_scores: dict {marker_label: vote_score}
        total_assignments: total number of assignment attempts
        successful_assignments: number of successful assignments
    """
    logger.info(f"Initial labeling for {os.path.basename(filename)}...")

    static_pos = np.nanmean(static['points'], axis=2).T
    static_labels = np.array(static['labels'])
    valid_static = ~np.isnan(static_pos).any(axis=1)

    static_clean = static_pos[valid_static]
    labels_clean = static_labels[valid_static]

    votes = {label: [] for label in labels_clean}
    total_assignments, successful = 0, 0

    for idx, f in enumerate(frame_indices):
        move = movement['points'][:, :, f].T
        valid_move = ~np.isnan(move).any(axis=1)
        move_clean = move[valid_move]
        if len(move_clean) == 0:
            continue

        cost = cdist(move_clean, static_clean)
        row_ind, col_ind = linear_sum_assignment(cost)
        total_assignments += len(row_ind)

        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < MAX_DISTANCE:
                votes[labels_clean[j]].append(np.where(valid_move)[0][i])
                successful += 1

        if len(frame_indices) > 1:
            progress = (idx + 1) / len(frame_indices) * 100
            logger.debug(f"[{idx+1}/{len(frame_indices)}] Frame {f} - {progress:.0f}%")

    success_rate = successful / total_assignments if total_assignments else 0

    final_labels = {}
    vote_scores = {}

    for label, idxs in votes.items():
        if idxs:
            chosen, count = Counter(idxs).most_common(1)[0]
            final_labels[chosen] = label
            vote_scores[label] = count / len(frame_indices)
        else:
            vote_scores[label] = 0.0

    logger.info(f"Initial labeling completed: {success_rate:.1%} success rate")
    return final_labels, success_rate, vote_scores, total_assignments, successful


# =====================================================
# TEMPORAL PROPAGATION
# =====================================================
def propagate_labels(movement, initial_labels, filename):
    """
    Propagate marker identities frame by frame.

    For each frame, a marker identity is retained only if its 3D position
    is valid (not NaN) in that frame. Tracking continuity is quantified as
    the mean proportion of successfully tracked markers per frame across
    the whole sequence.
    """
    logger.info(f"Temporal propagation for {os.path.basename(filename)}...")

    n_frames = movement['points'].shape[2]
    history = [{} for _ in range(n_frames)]
    history[0] = initial_labels.copy()
    current = initial_labels.copy()
    confs = []

    for f in range(1, n_frames):
        new = {}
        valid = 0
        for idx, lab in current.items():
            if not np.isnan(movement['points'][:, idx, f]).any():
                new[idx] = lab
                valid += 1
        confs.append(valid / len(current) if current else 0)
        history[f] = new
        current = new.copy()

    tracking_conf = np.mean(confs) if confs else 0
    logger.info(f"Temporal propagation completed: mean confidence = {tracking_conf:.3f}")
    return history, tracking_conf


# =====================================================
# TRACKING STABILITY ANALYSIS
# =====================================================
def analyze_tracking_stability(labels_history, movement):
    """Compute per-marker tracking stability across the sequence."""
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
    """
    Remove tracking outliers.

    For each marker, trajectories exceeding OUTLIER_THRESHOLD (100 mm) from
    their static reference position are considered tracking artifacts and
    set to NaN.
    """
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
    """Linear interpolation of missing data along each coordinate axis."""
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
    """Generate labeled C3D and formatted CSV output files."""
    logger.info(f"Creating output files for {os.path.basename(filename)}...")

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(filename)[0]

    out_c3d = os.path.join(output_dir, f"{base}_labeled.c3d")
    out_csv = os.path.join(output_dir, f"{base}_labeled.csv")

    n_frames = movement['points'].shape[2]
    n_markers = len(desired_order)
    rate = movement['rate']

    points = np.full((4, n_markers, n_frames), np.nan)
    lab_idx = {l: i for i, l in enumerate(desired_order)}

    header_lines = [
        f"Filename,{base}_labeled",
        f"Sampling rate,{rate}",
        f"Nb Frames,{n_frames}",
        f"Nb markers,{n_markers}"
    ]

    csv_header = ["Frame", "Time (s)"]
    for marker in desired_order:
        csv_header.extend([f"{marker}_X", f"{marker}_Y", f"{marker}_Z"])

    csv_data = []
    time_step = 1.0 / rate

    for f, lm in enumerate(labels_per_frame):
        row = [f, f * time_step]
        frame_data = {marker: [0.0, 0.0, 0.0] for marker in desired_order}

        for idx, lab in lm.items():
            if lab in lab_idx:
                i = lab_idx[lab]
                points[:3, i, f] = movement['points'][:, idx, f]
                points[3, i, f] = 1
                frame_data[lab] = list(movement['points'][:, idx, f])

        for marker in desired_order:
            row.extend(frame_data[marker])

        csv_data.append(row)

    points = interpolate(remove_outliers(points, lab_idx, static))

    # Write C3D
    out = ezc3d.c3d()
    out['data']['points'] = points
    out['parameters']['POINT']['LABELS']['value'] = desired_order
    out['parameters']['POINT']['USED']['value'] = [n_markers]
    out['parameters']['POINT']['RATE']['value'] = [rate]
    out.write(out_c3d)

    # Write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line + '\n')
        f.write(",".join(csv_header) + '\n')
        writer = csv.writer(f)
        writer.writerows(csv_data)

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

                # Select the N frames with the highest number of valid markers
                valid = np.sum(~np.isnan(mov['points'][0]), axis=0)
                frames = np.argsort(valid)[-N_FRAMES_INITIAL_LABELING:]

                labels, init_conf, votes, total_assign, success_assign = match_markers(
                    static, mov, frames, file
                )

                history, track_conf = propagate_labels(mov, labels, file)
                tracking = analyze_tracking_stability(history, mov)

                create_files(mov, history, output_dir, file, static,
                             desired_order, votes, tracking)

                base = os.path.splitext(file)[0]
                global_summary[base] = {
                    "subfolder": sub,
                    "init_conf": round(init_conf, 3),
                    "track_conf": round(track_conf, 3),
                    "total_assignments": total_assign,
                    "successful_assignments": success_assign
                }

                for k, v in votes.items():
                    global_votes[k].append(v)
                for k, v in tracking.items():
                    global_tracks[k].append(v)

                elapsed = time.time() - start_time
                logger.info(f"  Completed in {elapsed:.1f}s")
                logger.info(f"  Initial confidence: {init_conf:.3f}, "
                            f"Tracking confidence: {track_conf:.3f}")
                processed_files += 1

            except Exception as e:
                logger.error(f"  Error processing {file}: {e}")
                failed_files.append((sub, file, str(e)))
                continue

    logger.info("\n" + "=" * 70)
    logger.info("GENERATING GLOBAL REPORTS")
    logger.info("=" * 70)

    # Global per-file summary
    summary_csv = os.path.join(root_folder, "global_tracking_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["File", "Subfolder", "Init_conf", "Track_conf",
                    "Total_assignments", "Successful_assignments"])
        for name, vals in global_summary.items():
            w.writerow([name, vals["subfolder"], vals["init_conf"],
                        vals["track_conf"], vals["total_assignments"],
                        vals["successful_assignments"]])

    # Global per-marker stability
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