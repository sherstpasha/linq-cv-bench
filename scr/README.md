# scr

## Structure
- `scr/classification`
- `scr/detection`
- `scr/segmentation`
- outputs: `experiments/classification`, `experiments/detection`, `experiments/segmentation`

## Run All ONNX Experiments
```bash
source /Users/user/tomsk/.venv/bin/activate
python /Users/user/tomsk/scr/run_all_onnx_experiments.py \
  --python /Users/user/tomsk/.venv/bin/python
```

## Provider Hints
- macOS Apple Silicon: `CoreMLExecutionProvider,CPUExecutionProvider`
- Linux/Windows with NVIDIA: `CUDAExecutionProvider,CPUExecutionProvider`
- CPU only: `CPUExecutionProvider`

Example:
```bash
python /Users/user/tomsk/scr/run_all_onnx_experiments.py \
  --python /Users/user/tomsk/.venv/bin/python \
  --classification-providers CPUExecutionProvider \
  --detection-providers CoreMLExecutionProvider,CPUExecutionProvider \
  --segmentation-providers CPUExecutionProvider
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
