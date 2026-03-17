# classification

Рабочих контура классификации теперь два:

1. `TPU`:
   - экспорт `ResNet-50` из `torchvision` в `ONNX`
   - квантование и компиляция в свой `.tpu`
   - `accuracy` через собственный direct TPU runner
   - `performance` через `mlperf` по `6.7`
2. `ONNX Runtime`:
   - тот же `ONNX`
   - `accuracy` и `performance` на `CPU` или `CUDA`

Используемые пути:

- `data/evaluation/imagenet` - данные для accuracy
- `data/calibration/imagenet` - данные для калибровки
- `models/classification/resnet50.onnx` - экспортированный ONNX
- `models/classification/resnet50.json` - metadata экспорта
- `artifacts/classification` - `.qm`, `.tpu`, build metadata
- `experiments/classification` - accuracy, performance и итоговая сводка
- `experiments/classification_onnx` - ONNX CPU/CUDA accuracy, performance и итоговая сводка

## Что должно быть установлено отдельно

- `mlperf`
- `pytpu`
- `tpu_framework`
- `tpu_compiler`
- `torch`
- `torchvision`
- `onnxruntime`

Для `CUDA` нужен отдельный runtime:

- `onnxruntime-gpu`

## Один запуск под ключ

```bash
python scr/classification/run_resnet50.py \
  --mlperf-binary /path/to/mlperf
```

Что делает этот сценарий:

- при необходимости экспортирует `models/classification/resnet50.onnx`
- собирает `artifacts/classification/resnet50_b1.tpu` и `resnet50_b8.tpu`
- считает accuracy своим runner-ом на `batch 1`
- считает performance через `mlperf`:
  - `batch 1`, `q=500`, `3` прогона
  - `batch 8`, `q=1000`, `3` прогона

Итоговая сводка:

- `experiments/classification/results_summary.json`

## ONNX Runtime

Один запуск под ключ:

```bash
python scr/classification/run_resnet50_onnx.py \
  --provider auto
```

Что делает этот сценарий:

- берет тот же `models/classification/resnet50.onnx`
- считает `accuracy` на `CPU` или `CUDA`
- считает `performance` для `batch 1` и `batch 8`
- сохраняет сводку в:
  - `experiments/classification_onnx/results_summary.json`

Ручной порядок:

```bash
python scr/classification/run_resnet50_onnx_accuracy.py \
  --provider cpu

python scr/classification/run_resnet50_onnx_performance.py \
  --provider cpu \
  --batch-size 1

python scr/classification/run_resnet50_onnx_performance.py \
  --provider cpu \
  --batch-size 8
```

## Ручной порядок

Экспортировать ONNX:

```bash
python scr/classification/export_resnet50_to_onnx.py
```

Собрать `.tpu`:

```bash
python scr/classification/build_resnet50_program.py \
  --model-path models/classification/resnet50.onnx
```

Считать accuracy своим runner-ом:

```bash
python scr/classification/run_resnet50_accuracy.py \
  --program-path artifacts/classification/resnet50_b1.tpu \
  --build-summary artifacts/classification/build_summary.json \
  --dataset-dir data/evaluation/imagenet
```

Считать performance через `mlperf`:

```bash
python scr/classification/run_resnet50_performance.py \
  --mlperf-binary /path/to/mlperf \
  --batch-size 1

python scr/classification/run_resnet50_performance.py \
  --mlperf-binary /path/to/mlperf \
  --batch-size 8
```

## Замечания

- экспорт по умолчанию фиксирован: `NHWC + uint8 + internal normalization`
- калибровка фиксирована: `resize -> center crop -> ImageNet mean/std`
- `run_resnet50_accuracy.py` - это собственный direct TPU/evaluator слой, от которого дальше можно строить другие задачи
- `run_resnet50_performance.py` следует `6.7`:
  - `batch 1 -> q=500`
  - `batch 8 -> q=1000`
  - `3` прогона и среднее по `VALID`
- по умолчанию accuracy идет на весь `val_map.txt`; для быстрой проверки можно дать `--accuracy-samples 100` в `run_resnet50.py` или `--samples 100` в `run_resnet50_accuracy.py`
- `run_resnet50_onnx.py` не использует `mlperf`; это отдельный CPU/CUDA baseline на том же `ONNX`
