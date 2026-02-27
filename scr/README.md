# scr

## Structure
- `scr/classification`
- `scr/detection`
- `scr/segmentation`
- outputs: `experiments/classification`, `experiments/detection`, `experiments/segmentation`

## Run All ONNX Experiments
```bash
source /Users/user/tomsk/.venv/bin/activate
python /Users/user/tomsk/scr/split_datasets_for_calibration.py --force
python /Users/user/tomsk/scr/run_all_onnx_experiments.py \
  --python /Users/user/tomsk/.venv/bin/python \
  --providers CUDAExecutionProvider,CPUExecutionProvider \
  --batch-size 8
```

## H1 Smoke Test (No Calibration, Classification)
```bash
python /Users/user/tomsk/scr/classification/smoke_h1_classification_no_calibration.py
```

Custom experiments directory:
```bash
python /Users/user/tomsk/scr/run_all_onnx_experiments.py \
  --python /Users/user/tomsk/.venv/bin/python \
  --experiments-dir /Users/user/tomsk/my_experiments
```

Environment info is saved to:
- `<experiments-dir>/environment.json`
- `environment` section inside `<experiments-dir>/results_summary.json`

Each run creates a subfolder:
- `<experiments-dir>/experiment_<provider>_b<batch-size>/...`

Data split output:
- `data/evaluation` (90%)
- `data/calibration` (10%)
- original source folders are removed by default (`--keep-originals` to keep them)

## Provider Hints
- macOS Apple Silicon: `CoreMLExecutionProvider,CPUExecutionProvider`
- Linux/Windows with NVIDIA: `CUDAExecutionProvider,CPUExecutionProvider`
- CPU only: `CPUExecutionProvider`

Check available providers on current machine:
```bash
python /Users/user/tomsk/scr/list_onnx_providers.py
```

Example:
```bash
python /Users/user/tomsk/scr/run_all_onnx_experiments.py \
  --python /Users/user/tomsk/.venv/bin/python \
  --providers CPUExecutionProvider \
  --batch-size 8
```

## Quick Test
```bash
python /Users/user/tomsk/scr/run_all_onnx_experiments.py \
  --python /Users/user/tomsk/.venv/bin/python \
  --classification-limit 500 \
  --detection-limit 100 \
  --segmentation-limit 100 \
  --skip-classification-export \
  --skip-detection-export \
  --skip-segmentation-export
```
