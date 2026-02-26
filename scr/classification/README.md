# Classification (ONNX)

```bash
python /Users/user/tomsk/scr/classification/export_resnet50_to_onnx.py
python /Users/user/tomsk/scr/classification/infer_resnet50_onnx.py --providers CPUExecutionProvider
python /Users/user/tomsk/scr/classification/metrics.py
```

# Classification (H1 Quantization)

```bash
python /Users/user/tomsk/scr/classification/run_full_h1_classification.py
```

```bash
python /Users/user/tomsk/scr/classification/run_full_h1_classification.py \
  --experiment-suffix h1tpu \
  --num-calibration-images 512 \
  --calibration-chunk-size 128 \
  --percentile 99.9
```

`run_full_h1_classification.py` also writes run parameters to `experiments/classification/run_params_<suffix>.json`.

```bash
python /Users/user/tomsk/scr/classification/quantize_resnet50_h1.py
python /Users/user/tomsk/scr/classification/compile_resnet50_h1.py --preset O5
python /Users/user/tomsk/scr/classification/infer_resnet50_h1_tpu.py
python /Users/user/tomsk/scr/classification/metrics.py \
  --predictions /Users/user/tomsk/experiments/classification/predictions_h1_tpu.jsonl \
  --output-json /Users/user/tomsk/experiments/classification/metrics_h1_tpu.json
```
