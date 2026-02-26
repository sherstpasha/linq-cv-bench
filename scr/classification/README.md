# Classification (ONNX)

```bash
python /Users/user/tomsk/scr/classification/export_resnet50_to_onnx.py
python /Users/user/tomsk/scr/classification/infer_resnet50_onnx.py --providers CPUExecutionProvider
python /Users/user/tomsk/scr/classification/metrics.py
```

# Classification (H1 Quantization)

```bash
python /Users/user/tomsk/scr/classification/quantize_resnet50_h1.py
```
