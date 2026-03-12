# detection

Сейчас в репозитории есть два маршрута для детекции:

1. vendor `tiny_yolo3`
2. self-build `YOLOv8s` из `ultralytics`

## Vendor `tiny_yolo3`

Используется готовая vendor-программа:

- `linq_files/tpu_programs/tiny_yolo3_b8_o5_128x128_asic.tpu`

Логика маршрута:

1. quality считается вне `mlperf`:
   - direct TPU inference
   - декодирование YOLO-выходов
   - `COCOeval`
2. performance считается через `mlperf`

Один запуск под ключ:

```bash
python /Users/user/tomsk/scr/detection/run_tiny_yolo3_vendor.py \
  --python /path/to/h1_venv/bin/python \
  --mlperf-binary /path/to/mlperf \
  --program-path /path/to/linq_files/tpu_programs/tiny_yolo3_b8_o5_128x128_asic.tpu
```

Подтвержденный runtime contract для этой vendor-программы:

- `input tensor`: `input_1:0`
- `input shape`: `(8, 416, 416, 3)`
- `input layout`: `NHWC`
- `input range`: `float32` в диапазоне `0..1`

Эти параметры уже стоят в дефолтах скриптов.

Только quality:

```bash
python /Users/user/tomsk/scr/detection/run_tiny_yolo3_accuracy.py \
  --program-path /path/to/linq_files/tpu_programs/tiny_yolo3_b8_o5_128x128_asic.tpu \
  --img-dir /path/to/data/evaluation/MSCOCO2017/val2017 \
  --ann-file /path/to/data/evaluation/MSCOCO2017/annotations/instances_val2017.json
```

Только performance:

```bash
python /Users/user/tomsk/scr/detection/run_tiny_yolo3_performance.py \
  --mlperf-binary /path/to/mlperf \
  --program-path /path/to/linq_files/tpu_programs/tiny_yolo3_b8_o5_128x128_asic.tpu
```

Если vendor TPU использует нестандартные имена тензоров:

```bash
  --input-tensor-name ... \
  --output-tensor-name ...
```

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

`ultralytics` лучше не ставить в тот же `venv`, где живут `onnxruntime`, `tensorflow` и H1 vendor-пакеты.
Практический вариант:

- основной H1 runtime/build env: `numpy==1.26.4`, `onnxruntime~=1.18.0`, `tensorflow==2.14.1`
- отдельный export env только для `YOLOv8s` и `ultralytics`

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
python /Users/user/tomsk/scr/detection/run_yolov8.py \
  --python /path/to/h1_venv/bin/python \
  --export-python /path/to/yolo_export_venv/bin/python
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
/path/to/yolo_export_venv/bin/python /Users/user/tomsk/scr/detection/export_yolov8s_to_onnx.py
/path/to/h1_venv/bin/python /Users/user/tomsk/scr/detection/run_yolov8_onnx.py
/path/to/h1_venv/bin/python /Users/user/tomsk/scr/detection/build_yolov8_program.py
/path/to/h1_venv/bin/python /Users/user/tomsk/scr/detection/run_yolov8_tpu.py
/path/to/h1_venv/bin/python /Users/user/tomsk/scr/detection/metrics.py
```
