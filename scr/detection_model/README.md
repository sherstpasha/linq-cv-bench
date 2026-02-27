# Detection Model (FasterRCNN)

```bash
python /Users/user/tomsk/scr/detection_model/export_fasterrcnn_to_onnx.py
python /Users/user/tomsk/scr/detection_model/infer_fasterrcnn_onnx.py
python /Users/user/tomsk/scr/detection/metrics.py --predictions /Users/user/tomsk/experiments/detection_model/predictions_onnx.json

# Full H1 pipeline
python /Users/user/tomsk/scr/detection_model/run_full_h1_fasterrcnn.py
```
