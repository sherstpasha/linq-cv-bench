# detection

Стартовый контур для детекции сейчас один:

1. скачать `SSD-MobileNetV1` из `ONNX Model Zoo`
2. привести вход к статическому `1x300x300x3`
3. квантизовать и собрать свой `.tpu`
4. прогнать direct TPU inference на фрагменте COCO

Используемые пути:

- `data/evaluation/MSCOCO2017/val2017` - изображения для запуска
- `data/evaluation/MSCOCO2017/annotations/instances_val2017.json` - COCO annotations
- `data/calibration/MSCOCO2017/val2017` - изображения для калибровки
- `models/detection/ssd_mobilenet_v1_10.onnx` - исходный ONNX
- `artifacts/detection/ssd_mobilenet_v1` - `.qm`, `.tpu`, static ONNX и build metadata
- `experiments/detection/ssd_mobilenet_v1` - predictions и итоговая сводка запуска

## Что должно быть установлено отдельно

- `pytpu`
- `tpu_framework`
- `tpu_compiler`

## Один запуск под ключ

```bash
python /Users/user/tomsk/scr/detection/run_ssd_mobilenet_v1.py
```

По умолчанию этот сценарий:

- скачивает `SSD-MobileNetV1` из `ONNX Model Zoo`, если файла нет
- собирает `ssd_mobilenet_v1_b1.tpu` и `ssd_mobilenet_v1_b8.tpu`
- запускает direct TPU inference на первых `100` изображениях COCO
- сохраняет predictions в COCO JSON

Итоговая сводка:

- `experiments/detection/ssd_mobilenet_v1/results_summary.json`

## Ручной порядок

Скачать ONNX:

```bash
python /Users/user/tomsk/scr/detection/download_ssd_mobilenet_v1_model.py
```

Собрать `.tpu`:

```bash
python /Users/user/tomsk/scr/detection/build_ssd_mobilenet_v1_program.py \
  --model-path /Users/user/tomsk/models/detection/ssd_mobilenet_v1_10.onnx
```

Прогнать direct TPU inference:

```bash
python /Users/user/tomsk/scr/detection/run_ssd_mobilenet_v1_tpu.py \
  --program-path /Users/user/tomsk/artifacts/detection/ssd_mobilenet_v1/ssd_mobilenet_v1_b1.tpu \
  --build-summary /Users/user/tomsk/artifacts/detection/ssd_mobilenet_v1/build_summary.json \
  --img-dir /Users/user/tomsk/data/evaluation/MSCOCO2017/val2017 \
  --ann-file /Users/user/tomsk/data/evaluation/MSCOCO2017/annotations/instances_val2017.json
```

## Замечания

- модель берется из официального `ONNX Model Zoo`
- текущий шаг закрывает только:
  - download
  - build
  - direct TPU run
- `mlperf` и COCO-mAP поверх predictions добавим отдельно, когда подтвердим, что `.tpu` запускается стабильно
