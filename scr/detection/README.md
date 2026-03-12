# detection

Рабочий контур детекции теперь один:

1. экспорт `RetinaNet ResNet50 FPN` из `torchvision` в `ONNX`
2. `ONNX`-reference на COCO
3. квантование и компиляция в свой `.tpu`
4. direct TPU inference на COCO
5. `mAP` через `COCOeval`

Это именно self-build/reference-контур. `mlperf` для детекции сюда пока не включен.

Используемые пути:

- `data/evaluation/MSCOCO2017/val2017` - изображения для inference
- `data/evaluation/MSCOCO2017/annotations/instances_val2017.json` - COCO annotations
- `data/calibration/MSCOCO2017/val2017` - изображения для калибровки
- `models/detection/retinanet_resnet50_fpn.onnx` - экспортированный ONNX
- `models/detection/retinanet_resnet50_fpn.json` - metadata экспорта
- `artifacts/detection/retinanet` - `.qm`, `.tpu` и build metadata
- `experiments/detection/retinanet` - predictions, metrics и итоговая сводка

## Что должно быть установлено отдельно

- `pytpu`
- `tpu_framework`
- `tpu_compiler`
- `torch`
- `torchvision`

## Один запуск под ключ

```bash
python /Users/user/tomsk/scr/detection/run_retinanet.py
```

Что делает этот сценарий:

- при необходимости экспортирует `models/detection/retinanet_resnet50_fpn.onnx`
- считает `ONNX`-reference на COCO
- собирает `artifacts/detection/retinanet/retinanet_resnet50_fpn_b1.tpu` и `retinanet_resnet50_fpn_b8.tpu`
- считает direct TPU качество на COCO
- считает `COCO mAP` для `ONNX` и TPU predictions

Итоговая сводка:

- `experiments/detection/retinanet/results_summary.json`

## Ручной порядок

Экспортировать ONNX:

```bash
python /Users/user/tomsk/scr/detection/export_retinanet_to_onnx.py
```

Собрать `.tpu`:

```bash
python /Users/user/tomsk/scr/detection/build_retinanet_program.py \
  --model-path /Users/user/tomsk/models/detection/retinanet_resnet50_fpn.onnx
```

Считать `ONNX` reference:

```bash
python /Users/user/tomsk/scr/detection/run_retinanet_onnx.py \
  --model-path /Users/user/tomsk/models/detection/retinanet_resnet50_fpn.onnx
```

Считать direct TPU inference:

```bash
python /Users/user/tomsk/scr/detection/run_retinanet_tpu.py \
  --program-path /Users/user/tomsk/artifacts/detection/retinanet/retinanet_resnet50_fpn_b8.tpu \
  --build-summary /Users/user/tomsk/artifacts/detection/retinanet/build_summary.json
```

Считать COCO metrics:

```bash
python /Users/user/tomsk/scr/detection/metrics.py \
  --predictions /Users/user/tomsk/experiments/detection/retinanet/predictions_tpu.json
```
