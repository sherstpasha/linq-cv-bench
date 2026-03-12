# classification

Рабочий контур классификации теперь один:

1. экспорт `ResNet-50` из `torchvision` в `ONNX`
2. квантование и компиляция в свой `.tpu`
3. `accuracy` через собственный direct TPU runner
4. `performance` через `mlperf` по `6.7`

Используемые пути:

- `data/evaluation/imagenet` - данные для accuracy
- `data/calibration/imagenet` - данные для калибровки
- `models/classification/resnet50.onnx` - экспортированный ONNX
- `models/classification/resnet50.json` - metadata экспорта
- `artifacts/classification` - `.qm`, `.tpu`, build metadata
- `experiments/classification` - accuracy, performance и итоговая сводка

## Что должно быть установлено отдельно

- `mlperf`
- `pytpu`
- `tpu_framework`
- `tpu_compiler`
- `torch`
- `torchvision`

## Один запуск под ключ

```bash
python /Users/user/tomsk/scr/classification/run_resnet50.py \
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

## Ручной порядок

Экспортировать ONNX:

```bash
python /Users/user/tomsk/scr/classification/export_resnet50_to_onnx.py
```

Собрать `.tpu`:

```bash
python /Users/user/tomsk/scr/classification/build_resnet50_program.py \
  --model-path /Users/user/tomsk/models/classification/resnet50.onnx
```

Считать accuracy своим runner-ом:

```bash
python /Users/user/tomsk/scr/classification/run_resnet50_accuracy.py \
  --program-path /Users/user/tomsk/artifacts/classification/resnet50_b1.tpu \
  --build-summary /Users/user/tomsk/artifacts/classification/build_summary.json \
  --dataset-dir /Users/user/tomsk/data/evaluation/imagenet
```

Считать performance через `mlperf`:

```bash
python /Users/user/tomsk/scr/classification/run_resnet50_performance.py \
  --mlperf-binary /path/to/mlperf \
  --batch-size 1

python /Users/user/tomsk/scr/classification/run_resnet50_performance.py \
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
