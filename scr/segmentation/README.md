# segmentation

Рабочих контура сегментации теперь два:

1. `TPU`:
   - экспорт `FCN-ResNet50` из `torchvision` в `ONNX`
   - квантование и компиляция в свой `.tpu`
   - `accuracy` через собственный direct TPU runner с расчетом `pixel_accuracy` и `mean_iou`
   - `performance` через `mlperf` на `batch 1` и `batch 8`
2. `ONNX Runtime`:
   - тот же `ONNX`
   - `accuracy` и `performance` на `CPU` или `CUDA`

Используемые пути:

- `data/evaluation/VOCdevkit/VOC2012` - данные для accuracy
- `data/calibration/VOCdevkit/VOC2012/JPEGImages` - данные для калибровки
- `experiments/segmentation/fcn_resnet50.onnx` - экспортированный ONNX
- `artifacts/segmentation` - `.qm`, `.tpu`, build metadata
- `experiments/segmentation` - predictions, accuracy и итоговая сводка
- `experiments/segmentation_onnx` - ONNX CPU/CUDA accuracy, performance и итоговая сводка

## Что должно быть установлено отдельно

- `pytpu`
- `tpu_framework`
- `tpu_compiler`
- `torch`
- `torchvision`
- `mlperf`
- `onnxruntime`

Для `CUDA` нужен отдельный runtime:

- `onnxruntime-gpu`

Важно:

- `onnxruntime-gpu` ставить в отдельный `venv`, не в `linq_venv311`
- `linq_venv311` оставить под `TPU/mlperf/tpu_framework`
- для `CUDA`-запусков достаточно отдельного `ONNX Runtime`-окружения

## Один запуск под ключ

```bash
python scr/segmentation/run_fcn_resnet50.py
```

Итоговая сводка:

- `experiments/segmentation/results_summary.json`

## ONNX Runtime

Один запуск под ключ:

```bash
python scr/segmentation/run_fcn_resnet50_onnx.py \
  --provider auto
```

Что делает этот сценарий:

- при необходимости экспортирует `fcn_resnet50_b1.onnx` и `fcn_resnet50_b8.onnx`
- считает `accuracy` на `VOC`
- считает `performance` для `batch 1` и `batch 8`
- сохраняет сводку в:
  - `experiments/segmentation_onnx/results_summary.json`

## Ручной порядок

Экспортировать ONNX:

```bash
python scr/segmentation/export_fcn_resnet50_to_onnx.py
```

Собрать `.tpu`:

```bash
python scr/segmentation/build_fcn_resnet50_program.py
```

Считать accuracy своим runner-ом:

```bash
python scr/segmentation/run_fcn_resnet50_accuracy.py
```

Считать performance через `mlperf`:

```bash
python scr/segmentation/run_fcn_resnet50_performance.py \
  --mlperf-binary /path/to/mlperf \
  --batch-size 1

python scr/segmentation/run_fcn_resnet50_performance.py \
  --mlperf-binary /path/to/mlperf \
  --batch-size 8
```

Ручной `ONNX Runtime`:

```bash
python scr/segmentation/run_fcn_resnet50_onnx_accuracy.py \
  --provider cpu

python scr/segmentation/run_fcn_resnet50_onnx_performance.py \
  --provider cpu \
  --batch-size 1

python scr/segmentation/run_fcn_resnet50_onnx_performance.py \
  --provider cpu \
  --batch-size 8
```
