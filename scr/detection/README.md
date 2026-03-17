# detection

Сейчас для `tiny_yolo3` есть два отдельных маршрута:

1. `ONNX` baseline для `CPU/CUDA`
2. vendor `TPU` программа

## ONNX `tiny_yolo3`

Для `CPU/CUDA` baseline используется готовый `tiny-yolov3-11.onnx` из `ONNX Model Zoo`.

Один запуск под ключ:

```bash
python /Users/user/tomsk/scr/detection/run_tiny_yolo3_onnx.py \
  --python /path/to/onnx_env/bin/python \
  --provider cuda
```

При первом запуске модель автоматически скачивается в:

- `models/detection/tiny-yolov3-11.onnx`

Итоговый файл:

- `experiments/detection/tiny_yolo3_onnx/results_summary.json`

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

По умолчанию quality считается на `5000` изображениях COCO. Для всего eval-набора можно явно передать:

```bash
  --limit 0
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

## Что должно быть установлено отдельно

- `pytpu`
- `tpu_framework`
- `tpu_compiler`
- `torch`
- `torchvision`
