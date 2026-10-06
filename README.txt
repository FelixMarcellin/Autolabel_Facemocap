# Autolabel_Facemocap

Pipeline automatisé d'étiquetage de marqueurs faciaux non rigides pour données de capture de mouvement (MoCap) basées sur marqueurs.

Ce dépôt accompagne l'article :

> **Automated non-rigid facial marker labeling enables reliable and scalable analysis of facial motion**
> Félix Marcellin, Eder Rodriguez, Myriam Raoult, Anne-Gaëlle Le Moing, Emilien Colin, François-Régis Sarhan, Stéphanie Dakpé
> *INSERM UA-21 CHIMERE, Université de Picardie Jules Verne ; Institut Faire Faces, Amiens, France*
> Essai clinique enregistré : NCT05581680

Le pipeline permet d'attribuer automatiquement des étiquettes anatomiques prédéfinies à des trajectoires de marqueurs faciaux reconstruites, sans étiquetage manuel image par image, et de comparer ces résultats à un étiquetage de référence réalisé sous Vicon Nexus.

---

## Table des matières

- [Contexte](#contexte)
- [Fonctionnalités](#fonctionnalités)
- [Structure du dépôt](#structure-du-dépôt)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Données d'entrée](#données-dentrée)
- [Utilisation](#utilisation)
  - [1. Étiquetage automatique](#1-étiquetage-automatique)
  - [2. Analyse statistique](#2-analyse-statistique)
- [Description du pipeline](#description-du-pipeline)
- [Sorties](#sorties)
- [Métriques d'évaluation](#métriques-dévaluation)
- [Reproductibilité](#reproductibilité)
- [Résultats principaux](#résultats-principaux)
- [Limitations connues](#limitations-connues)
- [Citation](#citation)
- [Financements](#financements)
- [Licence et disponibilité des données](#licence-et-disponibilité-des-données)
- [Contact](#contact)

---

## Contexte

L'analyse quantitative du mouvement facial est de plus en plus utilisée en neuroréhabilitation, chirurgie maxillo-faciale et évaluation des troubles moteurs. La capture de mouvement à base de marqueurs fournit des données tridimensionnelles haute résolution, mais reste limitée par l'étiquetage manuel des marqueurs, chronophage, opérateur-dépendant et difficile à passer à l'échelle.

Les solutions automatisées développées pour le corps rigide ne sont pas directement applicables au visage, dont les déformations sont localement non rigides. Ce pipeline comble ce vide méthodologique en proposant une approche entièrement automatisée, robuste et reproductible.

---

## Fonctionnalités

- Extraction d'une **référence statique** par enregistrement (position moyenne des marqueurs en position neutre).
- **Étiquetage initial** par appariement de distances euclidiennes et **algorithme hongrois** (problème d'affectation linéaire un-à-un).
- **Mécanisme de vote** sur cinq trames d'initialisation pour réduire les erreurs transitoires.
- **Propagation temporelle** des identités de marqueurs sans ré-optimisation à chaque trame.
- **Détection d'aberrants spatiaux** (> 100 mm par rapport à la référence statique) et **interpolation linéaire** des données manquantes.
- Génération de fichiers **C3D étiquetés** et de fichiers **CSV structurés** (X, Y, Z par trame et par marqueur).
- **Analyse statistique** comparative (RMSE, corrélation de Pearson, ICC(2,1), Bland-Altman, tests appariés, taille d'effet).

---

## Structure du dépôt
Autolabel_Facemocap/
├── VF_autolabel_GENERAL_tous_dossiers_taux_reussite_marqueurs_V2.py
├── analyse_stat_V4.2.py
└── README.md

text

| Fichier | Rôle |
|---|---|
| `VF_autolabel_GENERAL_tous_dossiers_taux_reussite_marqueurs_V2.py` | Pipeline principal d'étiquetage automatique. Traite un ou plusieurs dossiers de fichiers C3D, effectue l'étiquetage et calcule le taux de réussite par marqueur. |
| `analyse_stat_V3.5_outlier.py` | Analyse statistique comparative entre les trajectoires automatiques (Python) et manuelles (Nexus). Gère l'exclusion des aberrants, le sous-échantillonnage et le calcul des métriques d'accord. |

> **Note** : ce dépôt est en cours de documentation. Les sections Installation et Utilisation ci-dessous sont à adapter aux arguments réels des scripts.

---

## Prérequis

- **Python** ≥ 3.11.7 (version utilisée dans l'étude)
- Système d'exploitation : Linux, macOS ou Windows
- Données de capture de mouvement au format **C3D**
- Fichiers d'étiquetage manuel de référence (issus de Vicon Nexus) pour l'analyse comparative

### Bibliothèques Python

Les scripts s'appuient vraisemblablement sur les bibliothèques suivantes :
numpy
scipy
pandas
ezc3d # ou c3d pour la lecture/écriture C3D
matplotlib
scikit-learn # optionnel

text

Pour l'analyse statistique : `scipy.stats` (Shapiro-Wilk, Wilcoxon, t-test), `pingouin` ou implémentation manuelle pour l'ICC(2,1).

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/FelixMarcellin/Autolabel_Facemocap.git
cd Autolabel_Facemocap

# 2. (Recommandé) Créer un environnement virtuel
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
