# detection

Первый шаг для детекции:

1. взять классический `YOLOv5s` из официального `ultralytics/yolov5`
2. экспортировать его в `ONNX`
3. прогнать reference-inference на фрагменте COCO
4. посчитать `mAP` через COCOeval

Используемые пути:

- `models/detection/yolov5s.pt` - веса
- `models/detection/yolov5s.onnx` - экспортированный ONNX
- `experiments/detection_onnx_reference` - predictions, timing, metrics, summary

## Один запуск под ключ

```bash
python /Users/user/tomsk/scr/detection/run_yolov5_onnx.py \
  --python python \
  --clone-if-missing \
  --limit 100
```

## Ручной порядок

Экспортировать ONNX:

```bash
python /Users/user/tomsk/scr/detection/export_yolov5s_to_onnx.py \
  --clone-if-missing
```

Прогнать reference ONNX inference:

```bash
python /Users/user/tomsk/scr/detection/run_yolov5_onnx_reference.py \
  --model-path /Users/user/tomsk/models/detection/yolov5s.onnx \
  --limit 100
```

Посчитать COCO metrics:

```bash
python /Users/user/tomsk/scr/detection/metrics.py \
  --predictions /Users/user/tomsk/experiments/detection_onnx_reference/predictions.json \
  --limit 100
```

## Замечания

- это именно классический `YOLOv5s`, а не `YOLOv5u`
- export идет через официальный `ultralytics/yolov5` repo `v7.0`
- на первом шаге мы не трогаем TPU; цель - проверить, что модель нормально экспортируется и дает осмысленный reference-result на фрагменте данных
