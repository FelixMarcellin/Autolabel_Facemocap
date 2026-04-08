# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 11:08:27 2026

@author: felima
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 10:47:38 2026

@author: felima
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

# =====================================================
# PARAMÈTRES GLOBAUX
# =====================================================
MAX_DISTANCE = 50
OUTLIER_THRESHOLD = 100
LABEL_VOTE_THRESHOLD = 0.4
TRACKING_THRESHOLD = 0.6
SKIP_FRAMES = 15  # Nombre de frames à ignorer au début

# =====================================================
# CHARGEMENT DU SET DE MARQUEURS (CSV)
# =====================================================
def load_marker_set():
    print("📂 Chargement du fichier CSV des marqueurs...")
    root = Tk()
    root.withdraw()
    csv_path = filedialog.askopenfilename(
        title="Sélectionner le fichier CSV des marqueurs",
        filetypes=[("CSV files", "*.csv")]
    )
    root.destroy()

    if not csv_path:
        raise RuntimeError("❌ Aucun fichier marqueurs sélectionné")

    df = pd.read_csv(csv_path, header=None)

    # Suppression d'un éventuel header texte
    first_cell = str(df.iloc[0, 0]).strip().lower()
    if not any(char.isdigit() for char in first_cell):
        df = df.iloc[1:]

    markers = df.iloc[:, 0].astype(str).str.strip().tolist()

    print(f"✅ {len(markers)} marqueurs chargés depuis :")
    print(f"   {csv_path}")
    print("-" * 60)
    
    return markers


# =====================================================
# UTILITAIRES C3D
# =====================================================
def load_c3d(filepath):
    print(f"   ↳ Chargement du fichier C3D: {os.path.basename(filepath)}")
    c3d = ezc3d.c3d(filepath)
    
    # Sauter les 15 premières frames
    all_points = c3d['data']['points'][:3, :, :]
    n_frames = all_points.shape[2]
    
    if n_frames > SKIP_FRAMES:
        points = all_points[:, :, SKIP_FRAMES:]
        print(f"     → Sauté {SKIP_FRAMES} premières frames ({SKIP_FRAMES}/{n_frames})")
    else:
        points = all_points
        print(f"     → Avertissement: fichier trop court ({n_frames} frames)")
    
    labels = [str(l) for l in c3d['parameters']['POINT']['LABELS']['value']]
    rate = c3d['parameters']['POINT']['RATE']['value'][0]
    
    return {'c3d': c3d, 'points': points, 'labels': labels, 'rate': rate}


# =====================================================
# MATCH LABELLISATION INITIALE
# =====================================================
def match_markers(static, movement, frame_indices, filename):
    print(f"   ↳ Labellisation initiale pour {os.path.basename(filename)}...")
    
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
        
        # Progression de la labellisation
        if len(frame_indices) > 1:
            progress = (idx + 1) / len(frame_indices) * 100
            print(f"     [{idx+1}/{len(frame_indices)}] Frame {f} - {progress:.0f}%", end='\r')

    if len(frame_indices) > 1:
        print()  # Nouvelle ligne après la barre de progression

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

    print(f"   ✅ Labellisation terminée: {success_rate:.1%} de succès")
    return final_labels, success_rate, vote_scores


# =====================================================
# TRACKING TEMPOREL
# =====================================================
def propagate_labels(movement, initial_labels, filename):
    print(f"   ↳ Tracking temporel pour {os.path.basename(filename)}...")
    
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
        
        # Afficher la progression toutes les 100 frames
        if f % 100 == 0 or f == n_frames - 1:
            progress = (f + 1) / n_frames * 100
            print(f"     Frame {f+1}/{n_frames} - {progress:.0f}%", end='\r')

    print()  # Nouvelle ligne après la barre de progression
    
    tracking_conf = np.mean(confs) if confs else 0
    print(f"   ✅ Tracking terminé: confiance moyenne = {tracking_conf:.3f}")
    return history, tracking_conf


# =====================================================
# STABILITÉ TRACKING
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
# OUTLIERS + INTERPOLATION
# =====================================================
def remove_outliers(points, labels, static):
    print("   ↳ Nettoyage des outliers...")
    static_pos = np.nanmean(static['points'], axis=2).T
    for idx, (label, _) in enumerate(labels.items()):
        ref_idx = np.where(np.array(static['labels']) == label)[0]
        if not len(ref_idx):
            continue
        ref = static_pos[ref_idx[0]]
        traj = points[:3, idx, :].T
        dists = np.linalg.norm(traj - ref, axis=1)
        outlier_count = np.sum(dists > OUTLIER_THRESHOLD)
        if outlier_count > 0:
            print(f"     {label}: {outlier_count} outliers détectés")
            points[:3, idx, dists > OUTLIER_THRESHOLD] = np.nan
    return points


def interpolate(points):
    print("   ↳ Interpolation des données manquantes...")
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
    print(f"   ✅ {total_interpolated} points interpolés")
    return points


# =====================================================
# CRÉATION DES FICHIERS LABELLISÉS (C3D + CSV)
# =====================================================
def create_files(movement, labels_per_frame, output_dir, filename, static, desired_order, votes, tracking):
    print(f"   ↳ Création des fichiers labellisés...")
    
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(filename)[0]
    
    # Noms des fichiers de sortie
    out_c3d = os.path.join(output_dir, f"{base}_labeled.c3d")
    out_csv = os.path.join(output_dir, f"{base}_labeled.csv")
    
    n_frames = movement['points'].shape[2]
    n_markers = len(desired_order)
    rate = movement['rate']

    points = np.full((4, n_markers, n_frames), np.nan)
    lab_idx = {l: i for i, l in enumerate(desired_order)}

    # STRUCTURE DU CSV EXACTEMENT COMME L'EXEMPLE
    # Lignes d'en-tête spécifiques
    header_lines = [
        f"Filename,{base}_labeled",
        f"Sampling rate,{rate}",
        f"Nb Frames,{n_frames}",
        f"Nb markers,{n_markers}"
    ]
    
    # En-tête des colonnes (très long)
    csv_header = ["Frame", "Time (s)"]
    for marker in desired_order:
        csv_header.extend([f"{marker}_X", f"{marker}_Y", f"{marker}_Z"])
    
    # Données pour chaque frame
    csv_data = []
    time_step = 1.0 / rate
    
    for f, lm in enumerate(labels_per_frame):
        # Créer une ligne avec tous les marqueurs
        row = [f, f * time_step]  # Frame et temps
        
        # Initialiser toutes les positions à 0.0 (comme dans l'exemple)
        frame_data = {marker: [0.0, 0.0, 0.0] for marker in desired_order}
        
        # Remplir avec les données disponibles
        for idx, lab in lm.items():
            if lab in lab_idx:
                i = lab_idx[lab]
                points[:3, i, f] = movement['points'][:, idx, f]
                points[3, i, f] = 1
                frame_data[lab] = list(movement['points'][:, idx, f])
        
        # Ajouter les données au CSV dans l'ordre
        for marker in desired_order:
            row.extend(frame_data[marker])
        
        csv_data.append(row)

    # Nettoyage et interpolation
    points = interpolate(remove_outliers(points, lab_idx, static))
    
    # 1. ÉCRIRE LE FICHIER C3D
    out = ezc3d.c3d()
    out['data']['points'] = points
    out['parameters']['POINT']['LABELS']['value'] = desired_order
    out['parameters']['POINT']['USED']['value'] = [n_markers]
    out['parameters']['POINT']['RATE']['value'] = [rate]
    out.write(out_c3d)
    
    # 2. ÉCRIRE LE FICHIER CSV AVEC LE FORMAT EXACT
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        # Écrire les 4 premières lignes d'en-tête
        for line in header_lines:
            f.write(line + '\n')
        
        # Écrire l'en-tête des colonnes (une seule ligne avec toutes les colonnes)
        f.write(",".join(csv_header) + '\n')
        
        # Écrire les données
        writer = csv.writer(f)
        writer.writerows(csv_data)
    
    print(f"   ✅ Fichiers créés:")
    print(f"     - {os.path.basename(out_c3d)}")
    print(f"     - {os.path.basename(out_csv)} (format: {n_frames} frames, {n_markers} marqueurs)")


# =====================================================
# PIPELINE PRINCIPAL
# =====================================================
def process_root(root_folder, desired_order):
    print("=" * 70)
    print("🚀 DÉMARRAGE DU TRAITEMENT")
    print("=" * 70)
    print(f"Dossier racine: {root_folder}")
    print(f"Nombre de marqueurs à traiter: {len(desired_order)}")
    print("-" * 70)

    global_summary = {}
    global_votes = defaultdict(list)
    global_tracks = defaultdict(list)
    
    subfolders = [sub for sub in os.listdir(root_folder) 
                  if os.path.isdir(os.path.join(root_folder, sub))]
    
    print(f"Nombre de sous-dossiers à traiter: {len(subfolders)}")
    print("-" * 70)

    for sub_idx, sub in enumerate(subfolders, 1):
        print(f"\n📁 Sous-dossier {sub_idx}/{len(subfolders)}: {sub}")
        print("-" * 50)
        
        subfolder = os.path.join(root_folder, sub)
        
        # Chercher les fichiers statiques
        static_files = [f for f in os.listdir(subfolder) 
                       if f.lower().endswith("statique.c3d")]
        
        if not static_files:
            print(f"   ⚠️  Aucun fichier statique trouvé dans {sub}")
            continue
        
        static_file = static_files[0]
        print(f"   📄 Fichier statique: {static_file}")
        static = load_c3d(os.path.join(subfolder, static_file))
        
        output_dir = os.path.join(subfolder, "labeled_output")
        
        # Lister les fichiers dynamiques
        dyn_files = [f for f in os.listdir(subfolder) 
                    if f.lower().endswith(".c3d") and "statique" not in f.lower()]
        
        print(f"   📊 Fichiers dynamiques à traiter: {len(dyn_files)}")
        
        for file_idx, file in enumerate(dyn_files, 1):
            print(f"\n   🔄 Traitement {file_idx}/{len(dyn_files)}: {file}")
            print("   " + "-" * 40)
            
            start_time = time.time()
            
            # Chargement du fichier dynamique
            mov = load_c3d(os.path.join(subfolder, file))
            
            # Sélection des frames pour la labellisation
            valid = np.sum(~np.isnan(mov['points'][0]), axis=0)
            frames = np.argsort(valid)[-5:]
            
            # Labellisation initiale
            labels, init_conf, votes = match_markers(static, mov, frames, file)
            
            # Tracking temporel
            history, track_conf = propagate_labels(mov, labels, file)
            
            # Analyse de stabilité
            tracking = analyze_tracking_stability(history, mov)
            
            # Création des fichiers (C3D + CSV)
            create_files(mov, history, output_dir, file, static, desired_order, votes, tracking)
            
            # Stockage des résultats globaux
            base = os.path.splitext(file)[0]
            global_summary[base] = {
                "subfolder": sub,
                "init_conf": round(init_conf, 3),
                "track_conf": round(track_conf, 3)
            }

            for k, v in votes.items():
                global_votes[k].append(v)
            for k, v in tracking.items():
                global_tracks[k].append(v)
            
            elapsed = time.time() - start_time
            print(f"   ✅ Traitement terminé en {elapsed:.1f}s")
            print(f"   📈 Confiances - Initiale: {init_conf:.3f}, Tracking: {track_conf:.3f}")

    print("\n" + "=" * 70)
    print("📊 GÉNÉRATION DES RAPPORTS GLOBAUX")
    print("=" * 70)

    # ==========================
    # CSV PAR FICHIER (global)
    # ==========================
    summary_csv = os.path.join(root_folder, "global_tracking_summary.csv")
    print(f"📄 Création du fichier global: {summary_csv}")
    
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Fichier", "Sous-dossier", "Init_conf", "Track_conf"])
        for name, vals in global_summary.items():
            w.writerow([name, vals["subfolder"], vals["init_conf"], vals["track_conf"]])
    
    print(f"✅ Rapport global par fichier généré: {len(global_summary)} entrées")

    # ==========================
    # CSV PAR MARQUEUR (global)
    # ==========================
    marker_csv = os.path.join(root_folder, "global_marker_stability.csv")
    print(f"📄 Création du fichier global: {marker_csv}")
    
    with open(marker_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Marker",
            "Mean_Labeling_Stability",
            "Mean_Tracking_Stability",
            "Global_Stability",
            "Files_Count",
            "Status"
        ])

        for lab in desired_order:
            lv = np.mean(global_votes.get(lab, [0]))
            tv = np.mean(global_tracks.get(lab, [0]))
            gs = (lv + tv) / 2
            count = max(len(global_votes.get(lab, [])), len(global_tracks.get(lab, [])))
            status = "UNSTABLE" if (lv < LABEL_VOTE_THRESHOLD or tv < TRACKING_THRESHOLD) else "OK"

            w.writerow([lab, round(lv, 3), round(tv, 3), round(gs, 3), count, status])
            
            # Afficher le statut des marqueurs instables
            if status == "UNSTABLE":
                print(f"   ⚠️  Marqueur globalement instable: {lab} (Label: {lv:.3f}, Track: {tv:.3f})")
    
    print(f"✅ Rapport global par marqueur généré: {len(desired_order)} marqueurs analysés")

    print("\n" + "=" * 70)
    print("🎉 TRAITEMENT TERMINÉ AVEC SUCCÈS")
    print("=" * 70)
    print(f"📁 Sous-dossiers traités: {len(subfolders)}")
    print(f"📊 Fichiers traités: {len(global_summary)}")
    print(f"📈 Marqueurs analysés: {len(desired_order)}")
    print(f"\n💾 FICHIERS GÉNÉRÉS PAR DOSSIER:")
    print(f"   Pour chaque fichier traité:")
    print(f"     • [nom]_labeled.c3d (fichier C3D labellisé)")
    print(f"     • [nom]_labeled.csv (format identique à l'exemple)")
    print(f"\n💾 RAPPORTS GLOBAUX:")
    print(f"   • {summary_csv} (résumé par fichier)")
    print(f"   • {marker_csv} (stabilité des marqueurs)")
    print("=" * 70)


# =====================================================
# LANCEMENT
# =====================================================
if __name__ == "__main__":
    print("🔄 Initialisation du programme...")
    
    # Chargement des marqueurs
    desired_order = load_marker_set()

    # Sélection du dossier racine
    root = Tk()
    root.withdraw()
    root_folder = filedialog.askdirectory(title="Sélectionner le dossier racine")
    root.destroy()

    if root_folder:
        process_root(root_folder, desired_order)
    else:
        print("❌ Aucun dossier sélectionné. Programme terminé.")