# scr

В репозитории сейчас есть два слоя:

- подготовка данных:
  - `download_data_from_yandex_disk.py`
  - `split_datasets_for_calibration.py`
- классификация:
  - `classification/export_resnet50_to_onnx.py`
  - `classification/build_resnet50_program.py`
  - `classification/run_resnet50_accuracy.py`
  - `classification/run_resnet50_performance.py`
  - `classification/run_resnet50.py`
- детекция:
  - `detection/run_tiny_yolo3_accuracy.py`
  - `detection/run_tiny_yolo3_performance.py`
  - `detection/run_tiny_yolo3_vendor.py`
  - `detection/export_yolov8s_to_onnx.py`
  - `detection/build_yolov8_program.py`
  - `detection/run_yolov8_onnx.py`
  - `detection/run_yolov8_tpu.py`
  - `detection/run_yolov8.py`
  - `detection/metrics.py`
- сегментация:
  - `segmentation/export_fcn_resnet50_to_onnx.py`
  - `segmentation/infer_fcn_resnet50_onnx.py`
  - `segmentation/quantize_fcn_resnet50_h1.py`
  - `segmentation/compile_fcn_resnet50_h1.py`
  - `segmentation/infer_fcn_resnet50_h1_tpu.py`
  - `segmentation/metrics.py`
  - `segmentation/run_full_h1_segmentation.py`

## Данные

```bash
python /Users/user/tomsk/scr/download_data_from_yandex_disk.py
python /Users/user/tomsk/scr/split_datasets_for_calibration.py --force
```

Используемые каталоги:

- `data/evaluation/imagenet` - данные для accuracy
- `data/calibration/imagenet` - данные для калибровки

## Классификация

Подробный порядок запуска описан в:

- `scr/classification/README.md`

## Детекция

Подробный порядок запуска описан в:

- `scr/detection/README.md`

## Сегментация

Подробный порядок запуска описан в:

- `scr/segmentation/README.md`

## Внешние зависимости

В `requirements.txt` перечислены только Python-пакеты.
Отдельно должны быть установлены:

- `pytpu`
- `tpu_framework`
- `tpu_compiler`
- `mlperf`
