# SUFE VOS Leaderboard Colab

This project is scoped only for the `SUFE_DL_FinalProject_Vision` leaderboard submission workflow. It intentionally does not cover the course report, single-frame image demo, self-recorded video demo, or presentation material.

## Current Status

Implemented:

- Colab notebook skeleton: `notebooks/sufe_leaderboard_colab.ipynb`
- Colab dependency list: `requirements_colab.txt`
- Base runtime / strategy config: `configs/leaderboard_colab.yaml`
- Optional backend registry: `configs/optional_backends.yaml`
- Data layout inspection: `src/data/inspect_sufe.py`
- Sample/provisional submission adaptation: `src/data/submission.py`
- Strict SAM 3.1 Object Multiplex adapter: `src/trackers/sam3_tracker_optional.py`
- Native SAM 3.1 runner: `scripts/run_sam31_vos.py`
- SAM 3.1 Colab workflow: `notebooks/sufe_sam31_colab.ipynb`
- Frozen MOSEv2 split generator: `src/eval/mosev2_split.py`
- Object-level recovery state and memory gate: `src/vos/recovery.py`

The reported SAM2.1 control remains `J&F_new=58.66`. The pseudo-anchor result
`58.58` is rejected as a replacement baseline. The current mainline is a pure
SAM 3.1 baseline; recovery remains disabled until it improves an external
holdout by at least 0.4 points.

## SAM 3.1 Mainline

Use `notebooks/sufe_sam31_colab.ipynb`. It performs runtime/auth checks, a
single-object and multi-object 5-frame smoke test, optional MOSEv2 calibration
and holdout runs, full SUFE inference, and final submission validation.

Recommended runtime:

- H100 80GB
- A100 80GB or 40GB
- T4 is unsupported for the full run

Required runtime: Python 3.12+, PyTorch 2.7+, CUDA 12.6+, and a BF16 GPU.
The notebook defaults to `research21/sam3.1`; `HF_TOKEN` is only needed if you
switch `SAM31_HF_REPO_ID` or `--hf-repo-id` back to a gated repository.

```bash
python scripts/run_sam31_vos.py \
  --data-root /content/sufe_data/video_dataset \
  --sample-submission /content/drive/MyDrive/sufe_vos_inputs/sample_submission.zip \
  --output-dir /content/sufe_runs \
  --experiment-id sam31_native_full \
  --hf-repo-id research21/sam3.1 \
  --sam3-repo-dir /content/sam3 \
  --prompt-mode mask \
  --original-resolution \
  --save-native-scores \
  --save-overlays sample \
  --make-submission
```

The runner requires the official `build_sam3_multiplex_video_model`,
`add_new_masks`, and `propagate_in_video` interfaces. It jointly initializes
all first-frame objects from their complete masks, copies frame zero exactly,
keeps object IDs fixed, saves native presence/IoU fields when exposed, and
never falls back to points or SAM2.

Before a full SAM 3.1 run, probe the official full predictor state flow on one
short real video:

```bash
python scripts/debug_sam31_api.py \
  --data-root /content/sufe_data/video_dataset \
  --output-dir /content/sufe_runs \
  --experiment-id sam31_api_probe \
  --checkpoint /path/to/sam3.1_multiplex.pt \
  --sam3-repo-dir /content/facebookresearch_sam3 \
  --video-id 2b827e3a \
  --max-frames 5
```

This writes `logs/sam31_api_introspection.json` and
`logs/sam31_state_probe.json` without creating a submission.
The Colab notebook also copies these probe JSON files to
`MyDrive/sufe_vos_review/runs/EXP_ID/` for Codex review.
For a diagnostic smoke rerun after an empty-mask collapse, set
`SAM31_EMPTY_MASK_POLICY=previous` in Colab. This holds the previous object mask
only when SAM3.1 emits an empty object mask. The smoke notebook then defaults
`SAM31_INDEXED_ABSENCE_POLICY` to the same value, which also restores an object
if it disappears only after indexed-mask composition. Both policies record every
event in diagnostics and should not be treated as submission policies without
contact-sheet review.

Create the frozen MOSEv2 split:

```bash
python -m src.eval.mosev2_split \
  --root /content/drive/MyDrive/datasets/MOSEv2 \
  --output outputs/validation/mosev2_seed2026.json \
  --seed 2026 \
  --total 80 \
  --calibration-size 40
```

All thresholds must be selected on the 40-video calibration split and checked
once on the 40-video holdout. Leaderboard video/object IDs are not tuning data.

## Colab Workflow

Do not use Google Drive as the experiment filesystem. Drive is only for inputs,
compact review bundles, and optional single-file archives.

Repository:

```text
https://github.com/yyj11266/SUFE_DL_FinalProject_Vision
```

Recommended layout:

```text
/content/sufe_vos_leaderboard/      # code clone or uploaded code
/content/sufe_data/video_dataset/   # extracted dataset on Colab local disk
/content/sufe_runs/EXP/             # full local experiment outputs
/content/drive/MyDrive/sufe_vos_inputs/
  video_dataset.zip
  sample_submission.zip
/content/drive/MyDrive/sufe_vos_review/runs/EXP/
  artifact_manifest.json
  submission.zip
  sanity_check.json
  data_info.json
  format_spec.json
  logs/
  previews/
/content/drive/MyDrive/sufe_vos_archives/
  EXP.full.tar                      # optional full experiment archive
```

1. Mount Google Drive for inputs and review bundles.
2. Keep code on a local filesystem when possible, e.g. `/content/sufe_vos_leaderboard`.
3. Set experiment output roots to `/content/sufe_runs`.
4. Publish only a compact Codex review bundle back to Drive.

When opening a notebook directly in Colab, run its first setup cell before any
experiment cells. The setup cell clones this repository into
`/content/sufe_vos_leaderboard` when it is not already present. You can also
run the bootstrap manually:

```python
from google.colab import drive
drive.mount("/content/drive")

import pathlib, subprocess

repo_url = "https://github.com/yyj11266/SUFE_DL_FinalProject_Vision.git"
project_root = pathlib.Path("/content/sufe_vos_leaderboard")
if not project_root.exists():
    subprocess.run(["git", "clone", repo_url, str(project_root)], check=True)
```

If you use an existing local clone at another path, set `SUFE_PROJECT_ROOT` to
the folder containing `src/` before rerunning the first notebook cell.

Dataset download is guarded by the notebook and config:

- It downloads only when running inside Colab.
- It downloads or copies the dataset zip to a configured `SUFE_DATA_ZIP`.
- Local runs must not download the dataset.
- If `/content/drive/MyDrive/sufe_vos_inputs/sample_submission.zip` exists, the notebook inspects it and writes an exact `format_spec.json`.
- If the sample submission is absent, the notebook infers a provisional format and marks it as `not_verified_by_sample`.

Dataset URL:

```text
https://drive.google.com/file/d/12PLrZwDvpeO3n-IQbAMgA9FOM0xdOqRr/view?usp=sharing
```

## Target Final Artifacts

Every experiment writes full output locally:

```text
/content/sufe_runs/{experiment_id}/
```

The full local run must produce:

- `format_spec.json`
- `data_info.json`
- `sanity_check.json`
- `submission.zip`
- `logs/`
- `masks/`

After a run completes, publish a compact Codex review bundle:

```bash
python scripts/publish_run_for_codex.py \
  --exp-dir /content/sufe_runs/EXP \
  --publish-dir /content/drive/MyDrive/sufe_vos_review/runs/EXP \
  --data-root /content/sufe_data/video_dataset \
  --archive-dir /content/drive/MyDrive/sufe_vos_archives \
  --make-full-archive
```

Codex can review the bundle without syncing the full image tree:

```bash
python scripts/review_codex_bundle.py \
  --bundle-dir /content/drive/MyDrive/sufe_vos_review/runs/EXP
```

The Drive review bundle contains:

- `submission.zip`
- `sanity_check.json`
- `artifact_manifest.json`
- `data_info.json`
- `format_spec.json`
- `logs/`
- `previews/`

`submission.zip` must match the sample submission structure exactly.

## Data and Submission CLIs

Inspect an extracted dataset:

```bash
PYTHONPATH=/content/sufe_vos_leaderboard python -m src.data.inspect_sufe \
  --root /content/sufe_data/video_dataset \
  --output /content/sufe_runs/EXP/data_info.json
```

Inspect a sample submission:

```bash
PYTHONPATH=/content/sufe_vos_leaderboard python -m src.data.submission inspect-sample \
  --sample /content/drive/MyDrive/sufe_vos_inputs/sample_submission.zip \
  --output /content/sufe_runs/EXP/format_spec.json
```

Infer provisional format when no sample exists:

```bash
PYTHONPATH=/content/sufe_vos_leaderboard python -m src.data.submission infer-provisional \
  --data-info /content/sufe_runs/EXP/data_info.json \
  --output /content/sufe_runs/EXP/format_spec.json
```

Create and validate a submission:

```bash
PYTHONPATH=/content/sufe_vos_leaderboard python -m src.data.submission make \
  --pred-root /content/sufe_runs/EXP/pred_masks \
  --output-zip /content/sufe_runs/EXP/submission.zip \
  --format-spec /content/sufe_runs/EXP/format_spec.json

PYTHONPATH=/content/sufe_vos_leaderboard python -m src.data.submission validate \
  --output-zip /content/sufe_runs/EXP/submission.zip \
  --format-spec /content/sufe_runs/EXP/format_spec.json \
  --data-info /content/sufe_runs/EXP/data_info.json \
  --sanity-output /content/sufe_runs/EXP/sanity_check.json
```

## SAM2.1 Baseline CLI

Run the required SAM2.1 Hiera Large baseline:

```bash
python /content/sufe_vos_leaderboard/scripts/run_baseline_sam2.py \
  --data-root /content/sufe_data/video_dataset \
  --sample-submission /content/drive/MyDrive/sufe_vos_inputs/sample_submission.zip \
  --output-dir /content/sufe_runs \
  --experiment-id EXP \
  --checkpoint sam2.1_hiera_large \
  --model-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --prompt-mode mask_box_points \
  --resize-long-side 0 \
  --save-overlays sample \
  --skip-existing \
  --make-submission
```

Outputs:

- `/content/sufe_runs/EXP/masks/{video}/{frame}.png`
- `/content/sufe_runs/EXP/overlays/{video}/{frame}.jpg`
- `/content/sufe_runs/EXP/logs/per_video_status.json`
- `/content/sufe_runs/EXP/submission.zip`
- `/content/sufe_runs/EXP/sanity_check.json`

## SAM2.1 Pseudo-Anchor Optimization

After a full baseline exists, run the optimization notebook:

```text
notebooks/sufe_sam2_optimization_colab.ipynb
```

Recommended runtime:

- Best: Colab Pro/Pro+ A100 40GB.
- Usually workable: L4 with original resolution and one pseudo-anchor per video.
- Fallback only: T4; set `RESIZE_LONG_SIDE = 1536` if original resolution OOMs, but expect possible score loss.

The optimization script reuses baseline masks, selects high-stability pseudo-anchor frames, reruns SAM2 from those anchors, fuses baseline and anchor masks, applies conservative object postprocess, and validates a new submission:

```bash
python /content/sufe_vos_leaderboard/scripts/run_enhanced_sam2_anchors.py \
  --data-root /content/sufe_data/video_dataset \
  --baseline-exp /content/sufe_runs/BASELINE_EXP \
  --output-dir /content/sufe_runs \
  --experiment-id sam2_anchor_fusion_1x \
  --checkpoint /content/sufe_runs/BASELINE_EXP/checkpoints/sam2.1_hiera_large.pt \
  --model-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --prompt-mode mask \
  --anchor-fractions 0.50 \
  --max-anchor-runs 1 \
  --save-overlays sample \
  --save-anchor-overlays none \
  --make-submission
```

Outputs:

- `/content/sufe_runs/EXP/masks/{video}/{frame}.png`
- `/content/sufe_runs/EXP/anchor_runs/{video}/anchor_{frame}/`
- `/content/sufe_runs/EXP/logs/self_diagnostics.csv`
- `/content/sufe_runs/EXP/logs/fusion_debug.csv`
- `/content/sufe_runs/EXP/submission.zip`
- `/content/sufe_runs/EXP/sanity_check.json`

## Anchor Mining CLI

Mine object-level anchors from an ordered frame directory and first-frame mask:

```bash
PYTHONPATH=/content/sufe_vos_leaderboard python -m src.vos.anchor_mining \
  --frames-dir /content/sufe_runs/EXP/cache/frames/VIDEO \
  --initial-mask /content/sufe_data/video_dataset/Annotations/VIDEO/00000.png \
  --output-dir /content/sufe_runs/EXP \
  --video-id VIDEO \
  --object-id 1
```

Outputs:

- `/content/sufe_runs/EXP/anchors/{video}/{object_id}_anchors.json`
- `/content/sufe_runs/EXP/anchors/{video}/anchor_debug.mp4`

## Main Baseline

The required baseline is SAM2.1 Hiera Large:

- Backend: `sam2`
- Checkpoint: `sam2.1_hiera_large.pt`
- Model config: `configs/sam2.1/sam2.1_hiera_l.yaml`

The official SAM2 repository documents SAM2.1 checkpoints and video predictor usage at:

```text
https://github.com/facebookresearch/sam2
```

## Optional Backends

The SAM 3.1 runner is strict and stops when its package, checkpoint, CUDA
runtime, or native mask API is unavailable. The following supplementary
backends remain optional and are not active in the native baseline:

- SAM 3 visual/text detector for later object-level recovery
- Cutie
- SUTrack
- GroundingDINO
- T-Rex2
- DINOv3

Optional backend settings live in `configs/optional_backends.yaml`.

## Leaderboard Modules

The planned competition modules are configured in `configs/leaderboard_colab.yaml`:

- target type classifier
- anchor mining
- reliability scoring
- drift/lost detection
- prompt fusion
- multi-anchor bidirectional propagation
- SAM2Long-like memory tree
- SAMURAI-like motion gate
- DAM4SAM-like distractor-aware memory policy
- optional model ensemble

## Directory Layout

```text
sufe_vos_leaderboard/
├── notebooks/sufe_leaderboard_colab.ipynb
├── configs/
├── src/data
├── src/prompts
├── src/features
├── src/detectors
├── src/trackers
├── src/vos
├── src/eval
├── src/viz
├── scripts
├── tests
└── README.md
```

## Implementation Rules

- All paths must come from notebook variables or YAML config.
- Do not hard-code local machine paths.
- All scripts must expose `argparse`.
- All modules must use type annotations and docstrings.
- Optional backends must never block the required SAM2.1 Hiera Large baseline.
- Local execution must not download the SUFE test data.
