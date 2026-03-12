# classification

Классификационный сценарий построен вокруг своего `ResNet-50` и `INT8/TPU`.

Используемые пути:

- `data/evaluation/imagenet` - данные для `accuracy`
- `data/calibration/imagenet` - данные для калибровки
- `models/classification/resnet50.onnx` - исходная ONNX-модель
- `models/classification/resnet50.json` - метаданные экспорта из `torchvision`
- `artifacts/classification` - `.qm`, `.tpu`, build metadata
- `experiments/classification` - логи запусков и итоговые JSON

## Что должно быть установлено отдельно

- `mlperf`
- `tpu_framework`
- `tpu_compiler`
- `torch`
- `torchvision`
- `onnxruntime`

## Быстрый запуск

```bash
python /Users/user/tomsk/scr/classification/run_resnet50.py
```

Проверка на первых `100` изображениях:

```bash
python /Users/user/tomsk/scr/classification/run_resnet50.py \
  --accuracy-samples 100 \
  --skip-performance
```

## Ручной порядок

Экспортировать ONNX из `torchvision`:

```bash
python /Users/user/tomsk/scr/classification/export_resnet50_to_onnx.py
```

Экспорт по умолчанию уже делает рабочий контракт `NHWC + uint8 + internal normalization`.
Калибровка для `INT8` фиксирована и всегда использует стандартный `ImageNet normalized` preprocess.

Сборка артефактов:

```bash
python /Users/user/tomsk/scr/classification/build_resnet50_program.py \
  --model-path /Users/user/tomsk/models/classification/resnet50.onnx
```

Accuracy:

```bash
python /Users/user/tomsk/scr/classification/run_resnet50_accuracy.py \
  --program-path /Users/user/tomsk/artifacts/classification/resnet50_b1.tpu
```

Reference-check для текущего `ONNX` без `mlperf` и без `TPU`:

```bash
python /Users/user/tomsk/scr/classification/run_resnet50_onnx_reference.py \
  --model-path /Users/user/tomsk/models/classification/resnet50.onnx \
  --dataset-dir /Users/user/tomsk/data/evaluation/imagenet
```

Reference-check для текущего `.tpu` без `mlperf`:

```bash
python /Users/user/tomsk/scr/classification/run_resnet50_tpu_reference.py \
  --program-path /Users/user/tomsk/artifacts/classification/resnet50_b1.tpu \
  --dataset-dir /Users/user/tomsk/data/evaluation/imagenet
```

Запуск vendor `resnet50_mlperf` строго по ПМИ:

```bash
python /Users/user/tomsk/scr/classification/run_vendor_resnet50_mlperf.py \
  --mlperf-binary /path/to/mlperf \
  --program-b1 /path/to/resnet50_mlperf_b1_*.tpu \
  --program-b8 /path/to/resnet50_mlperf_b8_*.tpu \
  --dataset-dir /Users/user/tomsk/data/evaluation/imagenet
```

Performance:

```bash
python /Users/user/tomsk/scr/classification/run_resnet50_performance.py \
  --batch-size 1

python /Users/user/tomsk/scr/classification/run_resnet50_performance.py \
  --batch-size 8
```

## Замечания

- Если `models/classification/resnet50.onnx` отсутствует, orchestrator сам экспортирует его из `torchvision`.
- По умолчанию экспорт идет с `opset 13`, потому что он безопаснее для vendor-конвертера, чем более новые версии.
- При первом экспорте pretrained-весов `torchvision` нужен доступ в интернет.
- По умолчанию экспортируется модель с рабочим контрактом `NHWC + uint8 + internal normalization`.
- Калибровка в build-контуре фиксирована: resize -> center crop -> `ImageNet mean/std` normalization.
- Если рядом с `ONNX` лежит metadata JSON от старого экспорта, используй `--reexport-model` или другой `--model-path`.
- Для обычного `resnet50` используется свой evaluator `evaluate_resnet50_accuracy.py`, а не vendor `accuracy-imagenet.py`.
- `run_resnet50_onnx_reference.py` нужен только для диагностики: он прогоняет текущий `ONNX` на тех же данных и по тому же `val_map.txt`.
- `run_resnet50_tpu_reference.py` нужен только для диагностики: он прогоняет текущий `.tpu` напрямую через `pytpu`, без `mlperf`.
- `run_vendor_resnet50_mlperf.py` нужен для формального прогона vendor `resnet50_mlperf` по ПМИ: `accuracy` на `1000`, `performance` по `3` повтора для `b1` и `b8`.
- `accuracy` по умолчанию использует весь `data/evaluation/imagenet/val_map.txt`.
- Для быстрой проверки можно уменьшить выборку, например `--accuracy-samples 100` или `run_resnet50_accuracy.py --samples 100`.
- `performance` следует ПМИ: запускается через `mlperf` и использует значения `qps` по умолчанию:
  - `500` для `batch 1`
  - `1000` для `batch 8`
- Итоговая сводка сохраняется в `experiments/classification/results_summary.json`.
