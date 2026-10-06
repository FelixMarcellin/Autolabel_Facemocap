Autolabel_Facemocap
Pipeline automatisé d'étiquetage de marqueurs faciaux non rigides pour données de capture de mouvement (MoCap) basées sur marqueurs.

Ce dépôt accompagne l'article :

Automated non-rigid facial marker labeling enables reliable and scalable analysis of facial motion
Félix Marcellin, Eder Rodriguez, Myriam Raoult, Anne-Gaëlle Le Moing, Emilien Colin, François-Régis Sarhan, Stéphanie Dakpé
INSERM UA-21 CHIMERE, Université de Picardie Jules Verne ; Institut Faire Faces, Amiens, France
Essai clinique enregistré : NCT05581680

Le pipeline permet d'attribuer automatiquement des étiquettes anatomiques prédéfinies à des trajectoires de marqueurs faciaux reconstruites, sans étiquetage manuel image par image, et de comparer ces résultats à un étiquetage de référence réalisé sous Vicon Nexus.

Table des matières
Contexte

Fonctionnalités

Structure du dépôt

Prérequis

Installation

Données d'entrée

Utilisation

1. Étiquetage automatique

2. Analyse statistique

Description du pipeline

Sorties

Métriques d'évaluation

Reproductibilité

Résultats principaux

Limitations connues

Citation

Financements

Licence et disponibilité des données

Contact

Contexte
L'analyse quantitative du mouvement facial est de plus en plus utilisée en neuroréhabilitation, chirurgie maxillo-faciale et évaluation des troubles moteurs. La capture de mouvement à base de marqueurs fournit des données tridimensionnelles haute résolution, mais reste limitée par l'étiquetage manuel des marqueurs, chronophage, opérateur-dépendant et difficile à passer à l'échelle.

Les solutions automatisées développées pour le corps rigide ne sont pas directement applicables au visage, dont les déformations sont localement non rigides. Ce pipeline comble ce vide méthodologique en proposant une approche entièrement automatisée, robuste et reproductible.

Fonctionnalités
Extraction d'une référence statique par enregistrement (position moyenne des marqueurs en position neutre).

Étiquetage initial par appariement de distances euclidiennes et algorithme hongrois (problème d'affectation linéaire un-à-un).

Mécanisme de vote sur cinq trames d'initialisation pour réduire les erreurs transitoires.

Propagation temporelle des identités de marqueurs sans ré-optimisation à chaque trame.

Détection d'aberrants spatiaux (> 100 mm par rapport à la référence statique) et interpolation linéaire des données manquantes.

Génération de fichiers C3D étiquetés et de fichiers CSV structurés (X, Y, Z par trame et par marqueur).

Analyse statistique comparative (RMSE, corrélation de Pearson, ICC(2,1), Bland-Altman, tests appariés, taille d'effet).

Structure du dépôt
text
Autolabel_Facemocap/
├── VF_autolabel_GENERAL_tous_dossiers_taux_reussite_marqueurs_V2.py
├── analyse_stat_V4.3.py
├── requirements.txt
└── README.md
Fichier	Rôle
VF_autolabel_GENERAL_tous_dossiers_taux_reussite_marqueurs_V2.py	Pipeline principal d'étiquetage automatique. Traite un ou plusieurs dossiers de fichiers C3D, effectue l'étiquetage et calcule le taux de réussite par marqueur.
analyse_stat_V4.3.py	Analyse statistique comparative entre les trajectoires automatiques (Python) et manuelles (Nexus). Gère l'exclusion des aberrants, le sous-échantillonnage et le calcul des métriques d'accord.
requirements.txt	Liste des dépendances Python nécessaires à l'exécution des scripts.
Note : ce dépôt est en cours de documentation. Les sections Installation et Utilisation ci-dessous sont à adapter aux arguments réels des scripts.

Prérequis
Python ≥ 3.11.7 (version utilisée dans l'étude)

Système d'exploitation : Linux, macOS ou Windows

Données de capture de mouvement au format C3D

Fichiers d'étiquetage manuel de référence (issus de Vicon Nexus) pour l'analyse comparative

Bibliothèques Python
Les scripts s'appuient sur les bibliothèques suivantes (voir requirements.txt) :

text
numpy
scipy
pandas
ezc3d  # ou c3d pour la lecture/écriture C3D
matplotlib
scikit-learn  # optionnel
Pour l'analyse statistique : scipy.stats (Shapiro-Wilk, Wilcoxon, t-test), pingouin ou implémentation manuelle pour l'ICC(2,1).

Installation
bash
# 1. Cloner le dépôt
git clone https://github.com/FelixMarcellin/Autolabel_Facemocap.git
cd Autolabel_Facemocap

# 2. (Recommandé) Créer un environnement virtuel
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
Données d'entrée
Le pipeline nécessite :

Fichiers C3D bruts issus du système de capture Vicon, contenant les trajectoires reconstruites des marqueurs faciaux.

Un fichier CSV listant les étiquettes anatomiques prédéfinies des marqueurs attendus.

Des fichiers C3D étiquetés manuellement sous Vicon Nexus, utilisés comme référence opérationnelle pour l'analyse comparative.

Chaque enregistrement doit comprendre :

Une acquisition statique (visage neutre) pour établir la configuration de référence.

Une ou plusieurs acquisitions dynamiques contenant les mouvements faciaux prédéfinis.

Utilisation
1. Étiquetage automatique
bash
python VF_autolabel_GENERAL_tous_dossiers_taux_reussite_marqueurs_V2.py --input <dossier_donnees> --output <dossier_sortie>
Le script parcourt les dossiers de fichiers C3D, applique le pipeline d'étiquetage et génère :

Un fichier C3D étiqueté par enregistrement.

Un fichier CSV structuré par enregistrement.

Un rapport du taux de réussite par marqueur.

Les arguments exacts (--input, --output, etc.) sont à vérifier directement dans le script, la documentation étant en cours de finalisation.

2. Analyse statistique
bash
python analyse_stat_V4.3.py --auto <dossier_python> --manual <dossier_nexus> --output <dossier_resultats>
Le script compare les trajectoires automatiques et manuelles et calcule :

RMSE 3D par enregistrement.

MAE, erreur médiane, P95.

Proportions d'observations sous différents seuils (1, 2, 3, 5, 10, 15, 20, 30 mm).

Biais coordonnée par coordonnée (X, Y, Z).

Analyse de Bland-Altman (niveau enregistrement et niveau observation poolée).

Coefficients de corrélation de Pearson et ICC(2,1).

Tests appariés (t-test, Wilcoxon) et taille d'effet standardisée.

Les arguments exacts sont à vérifier directement dans le script.

Description du pipeline
Le pipeline d'étiquetage automatique se décompose en quatre étapes principales :

Extraction de la référence statique

Chargement de la liste prédéfinie des étiquettes de marqueurs faciaux (CSV).

Calcul de la position 3D de référence de chaque marqueur comme la moyenne sur toutes les trames statiques valides.

Étiquetage initial

Sélection automatique des cinq trames contenant le plus grand nombre de marqueurs reconstruits valides.

Calcul de la distance euclidienne entre chaque marqueur dynamique valide et chaque position de référence statique.

Résolution du problème d'affectation un-à-un par l'algorithme hongrois.

Une affectation n'est valide que si la distance euclidienne est < 50 mm.

Vote sur les cinq trames : l'index de marqueur dynamique le plus fréquemment assigné à chaque étiquette est retenu comme identité initiale.

Propagation temporelle

Les identités initialement assignées sont conservées sur les trames suivantes, sans ré-optimisation.

Si la position d'un marqueur assigné est indisponible, l'observation est considérée comme manquante (pas de réassignation spatiale).

La continuité de suivi est quantifiée comme la proportion moyenne d'identités de marqueurs actives restant associées à des trajectoires 3D valides à chaque trame.

Raffinement et génération des sorties

Les positions dépassant 100 mm par rapport à la référence statique sont considérées comme des artefacts et mises à NaN.

Les échantillons manquants sont interpolés linéairement le long de chaque axe.

Les trajectoires sont réordonnées selon la liste anatomique prédéfinie.

Sorties : fichier C3D étiqueté + fichier CSV structuré (X, Y, Z par marqueur et par trame).

Aucun filtre de lissage temporel (moyenne mobile ou autre) n'est appliqué par l'implémentation actuelle.

Sorties
Pour chaque enregistrement traité, le pipeline génère :

Sortie	Description
Fichier C3D étiqueté	Trajectoires reconstruites dans l'ordre prédéfini des marqueurs.
Fichier CSV structuré	Coordonnées X, Y, Z par trame et par marqueur.
Rapport de taux de réussite	Proportion d'assignations valides par marqueur.
L'analyse statistique génère :

Tableaux de métriques d'accord (RMSE, MAE, médiane, P95) par enregistrement.

Proportions d'observations sous seuils.

Biais coordonnée par coordonnée.

Graphiques de Bland-Altman et distributions d'erreurs.

Métriques d'évaluation
Métrique	Description
RMSE 3D	Erreur quadratique moyenne au niveau enregistrement.
MAE 3D	Erreur absolue moyenne.
Erreur médiane et P95	Distribution des erreurs 3D.
Accord par seuil	Proportion d'observations sous 1, 2, 3, 5, 10, 15, 20, 30 mm.
Biais coordonné	Différence moyenne par axe (X, Y, Z).
Bland-Altman	Biais moyen et limites d'agrément à 95 %.
ICC(2,1)	Coefficient de corrélation intraclasse pour l'accord absolu.
d_z	Différence appariée standardisée.
Important : les analyses principales sont agrégées au niveau enregistrement pour tenir compte de la structure de mesures répétées (pseudoréplication). Les analyses poolées au niveau observation sont secondaires et descriptives.

Reproductibilité
Langage : Python ≥ 3.11.7

Bibliothèques : voir requirements.txt

Données : les fichiers C3D ne sont pas inclus dans ce dépôt. Ils peuvent être obtenus auprès de l'auteur correspondant sur demande raisonnable.

Code : les scripts d'étiquetage et d'analyse statistique sont fournis tels quels.

La reproductibilité complète des résultats de l'article nécessite l'accès aux données C3D brutes, qui ne sont pas hébergées sur ce dépôt.

Résultats principaux
Sur les 100 enregistrements analysés :

RMSE 3D moyen au niveau enregistrement : 8,84 ± 13,98 mm (médiane : 4,52 mm ; IQR : 1,89–9,08 mm).

Proportion moyenne d'observations dans 3 mm : 87,46 % (IC bootstrap 95 % : 82,18–91,94 %).

Biais moyen (Bland-Altman au niveau enregistrement) : −0,60 mm (IC 95 % : −1,44 à 0,12 mm).

Limites d'agrément à 95 % : −8,61 à 7,42 mm.

Biais coordonnée par coordonnée : X = −0,26 mm ; Y = 0,95 mm ; Z = −2,52 mm.

Seuil de comparaison principal : 100 mm (6,18 % d'observations brutes appariées exclues).

Ces résultats indiquent un accord spatial substantiel entre l'étiquetage automatique et l'étiquetage manuel Vicon Nexus, avec la plupart des observations présentant des écarts de quelques millimètres.

Limitations connues
Référence manuelle imparfaite : l'étiquetage Nexus est une référence opérationnelle, pas une vérité terrain absolue.

Distribution hétérogène des erreurs : quelques enregistrements contribuent de manière disproportionnée au RMSE global.

Dépendance à la configuration statique : les changements importants de configuration faciale peuvent challenger l'appariement initial.

Différences axiales résiduelles : notamment en Z, possiblement liées à l'alignement du référentiel.

Diversité clinique limitée : peu de phénotypes pathologiques sévères dans le jeu de validation.

Sensibilité au seuil de comparaison : les métriques agrégées dépendent du seuil d'exclusion choisi (50, 75, 100, 150 mm).

Interpolation non isolée : pas de version pré-interpolation disponible pour quantifier sa contribution spécifique.

Temps de traitement non standardisé : pas d'étude temps-mouvement formelle pour quantifier la réduction de charge opérateur.

Données non fournies : les fichiers C3D ne sont pas inclus dans le dépôt, limitant la reproductibilité complète.

Citation
Si vous utilisez ce code ou ces résultats, merci de citer :

bibtex
@article{marcellin2026automated,
  title={Automated non-rigid facial marker labeling enables reliable and scalable analysis of facial motion},
  author={Marcellin, Félix and Rodriguez, Eder and Raoult, Myriam and Le Moing, Anne-Gaëlle and Colin, Emilien and Sarhan, François-Régis and Dakpé, Stéphanie},
  journal={[À compléter]},
  year={2026},
  note={Essai clinique enregistré : NCT05581680}
}
Financements
Cette recherche a été financée par :

Fondation des Gueules Cassées (Projet AstreFace (43-2023 et 51-2024) et Projet IMMOUV (Convention d'aide financière 2024-2027)).

L'État français via les programmes "Investissements d'Avenir" et "France 2030" (EquipEx FiGuRes (ANR-10-EQPX-0001)).

Licence et disponibilité des données
Code source : disponible dans ce dépôt GitHub.

Données : les jeux de données C3D ne sont pas inclus dans ce dépôt. Ils peuvent être obtenus auprès de l'auteur correspondant sur demande raisonnable.

Pour toute question relative à la licence du code, veuillez contacter l'auteur correspondant.

Contact
Félix Marcellin
INSERM UA-21 CHIMERE, Université de Picardie Jules Verne (UPJV), Amiens, France
Institut Faire Faces, Amiens, France
Email : felix.marcellin@u-picardie.fr
