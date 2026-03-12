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
  - `detection/download_ssd_mobilenet_v1_model.py`
  - `detection/build_ssd_mobilenet_v1_program.py`
  - `detection/run_ssd_mobilenet_v1_tpu.py`
  - `detection/run_ssd_mobilenet_v1.py`

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
- `scr/detection/README.md`

## Внешние зависимости

В `requirements.txt` перечислены только Python-пакеты.
Отдельно должны быть установлены:

- `pytpu`
- `tpu_framework`
- `tpu_compiler`
- `mlperf`
