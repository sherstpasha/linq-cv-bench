# Segmentation (ONNX + H1)

```bash
python /Users/user/tomsk/scr/segmentation/export_fcn_resnet50_to_onnx.py
python /Users/user/tomsk/scr/segmentation/infer_fcn_resnet50_onnx.py --providers CoreMLExecutionProvider,CPUExecutionProvider
python /Users/user/tomsk/scr/segmentation/metrics.py

# H1 full pipeline (quantize -> compile -> TPU infer -> metrics)
python /Users/user/tomsk/scr/segmentation/run_full_h1_segmentation.py
```
