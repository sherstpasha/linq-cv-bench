# Segmentation (FCN-ResNet50)

Минимальный порядок запуска:

```bash
python scr/segmentation/export_fcn_resnet50_to_onnx.py
python scr/segmentation/infer_fcn_resnet50_onnx.py
python scr/segmentation/metrics.py
```

Полный H1 pipeline:

```bash
python scr/segmentation/run_full_h1_segmentation.py
```
