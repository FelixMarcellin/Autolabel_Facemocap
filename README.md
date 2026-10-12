# Automated Non-Rigid Facial Marker Labeling for Motion Capture

This repository contains a Python pipeline for automatically assigning
and propagating identities of reflective facial markers in marker-based
motion-capture (MoCap) recordings. It was developed to reduce the manual
workload associated with labeling non-rigid facial motion while
retaining a reproducible, geometry-based workflow.

The pipeline uses a **manually labeled static acquisition as the spatial
reference**. It is therefore an automated labeling pipeline initialized
from a manually labeled reference, rather than a fully automated process
requiring no manual input.

Repository: https://github.com/FelixMarcellin/Autolabel_Facemocap

## Overview

The pipeline processes C3D motion-capture recordings and generates
labeled coordinate outputs. Its main stages are:

1.  **Static reference extraction:** obtain marker positions from the
    manually labeled static acquisition.
2.  **Initial marker assignment:** compare candidate marker positions
    with the static reference using Euclidean distances and one-to-one
    Hungarian assignment; reject implausible matches and use voting
    across selected frames to determine initial identities.
3.  **Temporal propagation:** propagate marker identities through the
    dynamic recording while checking trajectory continuity.
4.  **Trajectory refinement and export:** flag large deviations from the
    static reference, represent missing positions as `NaN`, interpolate
    missing coordinate values linearly, and export the labeled
    trajectories.

For initialization, the first 15 frames of each dynamic recording are
excluded. Of the remaining frames, the five frames containing the
largest number of valid reconstructed markers are selected for initial
labeling.

## Main parameters

The current pipeline uses the following principal parameters:

  ------------------------------------------------------------------------
  Parameter                                    Value Purpose
  --------------------- ---------------------------- ---------------------
  Initial dynamic                                 15 Avoid the initial
  frames excluded                                    acquisition period
                                                     with incomplete or
                                                     unreliable
                                                     trajectories

  Frames used for                                  5 Select frames with
  initial labeling                                   the greatest number
                                                     of valid
                                                     reconstructed markers

  Maximum assignment                           50 mm Limit candidate
  distance                                           matches during
                                                     initial assignment

  Minimum vote                            0.60 (60%) Require sufficient
  proportion for a                                   agreement across
  label                                              initialization frames

  Ground/floor margin                          20 mm Apply the configured
                                                     floor-related
                                                     filtering criterion

  Maximum                                      40 mm Reject implausible
  frame-to-frame jump                                trajectory jumps
                                                     during temporal
                                                     propagation

  Static-reference                            100 mm Flag positions
  outlier threshold                                  displaced by more
                                                     than this distance
                                                     from the static
                                                     reference
  ------------------------------------------------------------------------

These parameters are defined in the source code and can be changed for
other acquisition setups. Any change should be documented and validated
before being used for scientific analyses.

**Interpolation note:** missing or rejected positions are represented as
`NaN` during processing and are linearly interpolated coordinate-wise in
the pipeline output. The pipeline does not apply a moving-average filter
or other temporal smoothing.

## Requirements and setup

-   Python 3.11.7 or a compatible Python 3.11 environment
-   Dependencies listed in [`requirements.txt`](requirements.txt)

Clone the repository and create an environment:

``` bash
git clone https://github.com/FelixMarcellin/Autolabel_Facemocap.git
cd Autolabel_Facemocap

python -m venv .venv
```

Activate the environment:

**Windows (PowerShell)**

``` powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**

``` bash
source .venv/bin/activate
```

Install dependencies:

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Input data and outputs

The workflow is designed for marker-based motion-capture C3D recordings
and the corresponding manually labeled static reference acquisition. The
static reference must be labeled consistently with the marker template
used by the pipeline.

The processing workflow produces labeled coordinate data, including CSV
outputs with the suffix `_labeled.csv`; the pipeline may also export C3D
outputs depending on the selected script/configuration. Check the
relevant script's configuration before running it, including input
folders, output folders, static-reference filename conventions, and
marker-template assumptions.

**Important:** this repository describes the processing code;
study-specific raw recordings and participant data are not included in
this README. Ensure that you have the necessary permissions to process
and share any motion-capture data.

## Validation and statistical analysis

The repository includes the statistical analysis script
`analyse_stat_V4.3.py`, which compares the automated labeled CSV outputs
against the corresponding manually labeled Nexus reference data.

The validation analysis follows these principles:

-   Only coordinate pairs that are finite and valid in both the
    automated and reference datasets are used for the corresponding
    metric.
-   Missing or non-finite values are excluded pairwise for validation;
    no additional imputation is performed during validation.
-   The **primary analysis retains all valid paired 3D observations**.
    The 100-mm gross-discrepancy threshold does not remove observations
    from the primary accuracy analysis; threshold-based exclusions are
    used only for sensitivity analyses.
-   Recording-level summaries are used to avoid treating all
    marker-frame observations as independent recordings. Confidence
    intervals are estimated using bootstrap resampling of recordings.
-   A participant-level cluster bootstrap is used as a sensitivity
    analysis to account for multiple recordings from the same
    participant.
-   The bootstrap random seed is fixed at 42 for reproducibility.

The 100-mm threshold is an operational criterion for identifying gross
discrepancies. It should not be interpreted as a guarantee that every
marker identity is anatomically correct.

## Validation dataset and reported results

In the evaluation described in the associated manuscript, the pipeline
was assessed on 100 C3D recordings from 12 participants: nine healthy
participants and three participants with facial palsy. Each participant
contributed multiple dynamic recordings.

The reported overall results were:

  Metric                                         Result
  ------------------------------------ ----------------
  Mean 3D RMSE                           5.96 ± 4.46 mm
  Median 3D error                               4.49 mm
  Valid 3D observations within 3 mm              87.35%
  Valid 3D observations within 5 mm              91.53%
  Valid 3D observations within 10 mm             93.95%

These values summarize the reported evaluation dataset and should not be
interpreted as guaranteed performance for other marker layouts,
acquisition systems, populations, or protocols. The 100-mm threshold was
not used to exclude observations from the primary analysis.

## Reproducibility

For reproducible use:

1.  Keep the raw C3D recordings and manually labeled static references
    unchanged.
2.  Record the Python version and installed package versions.
3.  Record the exact scripts, parameter values, and repository commit
    used.
4.  Preserve the generated `_labeled.csv` outputs and the logs/results
    from the analysis script.
5.  If changing thresholds or marker templates, report those changes and
    rerun validation.
6.  Do not compare outputs unless the recordings, coordinate
    conventions, marker definitions, and reference labels are
    compatible.

The scripts and parameter settings are research software and should be
validated on the intended acquisition setup before clinical or other
high-stakes use.

## Associated manuscript

**Automated non-rigid facial marker labeling enables reliable and
scalable analysis of facial motion.**

Please consult the associated manuscript for the full methodological
description, validation design, statistical analyses, limitations, and
interpretation of the results. If the manuscript is published, replace
this note with the final citation and DOI.

## Citation and contact

If you use this code in academic work, cite the associated manuscript
when its final bibliographic details become available and cite this
repository with the version or commit used in your analysis.

For questions about the implementation, open an issue in the GitHub
repository.

## License

Please refer to the license file in the repository, if present, for the
terms governing reuse and redistribution.
