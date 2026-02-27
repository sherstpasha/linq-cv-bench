# Detection (ONNX)

```bash
python /Users/user/tomsk/scr/run_all_onnx_experiments.py \
  --providers CPUExecutionProvider \
  --batch-size 8

python /Users/user/tomsk/scr/run_all_onnx_experiments.py \
  --providers CUDAExecutionProvider,CPUExecutionProvider \
  --batch-size 8
```

```bash
python /Users/user/tomsk/scr/detection/export_yolov5su_to_onnx.py
python /Users/user/tomsk/scr/detection/infer_yolov5_onnx.py --providers CoreMLExecutionProvider,CPUExecutionProvider
python /Users/user/tomsk/scr/detection/metrics.py
```

# Detection (H1 Quantization)

```bash
python /Users/user/tomsk/scr/detection/run_full_h1_detection.py
```

```bash
python /Users/user/tomsk/scr/detection/quantize_yolov5_h1.py
python /Users/user/tomsk/scr/detection/compile_yolov5_h1.py --preset O1 --batch-size 8
python /Users/user/tomsk/scr/detection/infer_yolov5_h1_tpu.py
python /Users/user/tomsk/scr/detection/metrics.py \
  --predictions /Users/user/tomsk/experiments/detection/predictions_h1_tpu.json \
  --output-json /Users/user/tomsk/experiments/detection/metrics_h1_tpu.json
```

`run_full_h1_detection.py` also writes run parameters to `experiments/detection/run_params_<suffix>.json`.
