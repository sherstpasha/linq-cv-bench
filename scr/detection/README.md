# detection

Сейчас в репозитории есть один self-build маршрут для детекции:

1. `YOLOv8s` из `ultralytics`

Логика маршрута:

1. экспорт в `ONNX`
2. `ONNX` reference на COCO
3. квантование и компиляция в свой `.tpu`
4. direct TPU inference на COCO
5. `mAP` через `COCOeval`

`mlperf` для quality детекции сюда пока не включен.

## Что должно быть установлено отдельно

- `pytpu`
- `tpu_framework`
- `tpu_compiler`
- `torch`
- `torchvision`
- `ultralytics`

## YOLOv8s

Используемые пути:

- `data/evaluation/MSCOCO2017/val2017`
- `data/evaluation/MSCOCO2017/annotations/instances_val2017.json`
- `data/calibration/MSCOCO2017/val2017`
- `models/detection/yolov8s.onnx`
- `artifacts/detection/yolov8s`
- `experiments/detection/yolov8s`

Один запуск под ключ:

```bash
python /Users/user/tomsk/scr/detection/run_yolov8.py
```

Что делает сценарий:

- экспортирует `YOLOv8s` в `ONNX`
- считает `ONNX` reference на COCO
- собирает `yolov8s_b1.tpu` и `yolov8s_b8.tpu`
- считает direct TPU качество на COCO
- считает `COCO mAP` для `ONNX` и TPU predictions

Итоговая сводка:

- `experiments/detection/yolov8s/results_summary.json`

Ручной порядок:

```bash
python /Users/user/tomsk/scr/detection/export_yolov8s_to_onnx.py
python /Users/user/tomsk/scr/detection/run_yolov8_onnx.py
python /Users/user/tomsk/scr/detection/build_yolov8_program.py
python /Users/user/tomsk/scr/detection/run_yolov8_tpu.py
python /Users/user/tomsk/scr/detection/metrics.py
```
