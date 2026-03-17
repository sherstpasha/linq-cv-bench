# detection

Сейчас для `tiny_yolo3` есть три отдельных маршрута:

1. `ONNX` baseline из `ONNX Model Zoo` для `CPU/CUDA`
2. `ONNX strict` baseline из `pjreddie cfg/weights` в `keras`-совместимом raw-head виде
3. vendor `TPU` программа

## ONNX `tiny_yolo3` from Model Zoo

Для быстрого `CPU/CUDA` baseline используется готовый `tiny-yolov3-11.onnx` из `ONNX Model Zoo`.

```bash
python /Users/user/tomsk/scr/detection/run_tiny_yolo3_onnx.py \
  --python /path/to/onnx_env/bin/python \
  --provider cuda \
  --model-source modelzoo
```

## ONNX `tiny_yolo3` strict

Для более строгого сравнения с vendor `TPU` используется raw-head `tiny_yolo3`, собранный из:

- `yolov3-tiny.cfg` от `pjreddie/darknet`
- `yolov3-tiny.weights` от `pjreddie`

Экспорт в `ONNX` требует отдельный `TensorFlow + tf2onnx` env.

```bash
python /Users/user/tomsk/scr/detection/run_tiny_yolo3_onnx.py \
  --python /path/to/onnx_env/bin/python \
  --export-python /path/to/tf_export_env/bin/python \
  --provider cuda \
  --model-source strict \
  --reexport-model
```

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

## Что должно быть установлено отдельно

Для `ONNX CPU/CUDA`:

- `onnxruntime` или `onnxruntime-gpu`
- `pycocotools`
- `torch`
- `torchvision`

Для strict export:

- `tensorflow`
- `tf2onnx`
- `h5py`

Для `TPU`:

- `pytpu`
- `tpu_framework`
- `tpu_compiler`
- `mlperf`
