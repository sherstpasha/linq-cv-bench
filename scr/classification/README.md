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

## Быстрый запуск

```bash
python /Users/user/tomsk/scr/classification/run_resnet50.py \
  --export-model-if-missing
```

Быстрая отладка на первых `100` изображениях:

```bash
python /Users/user/tomsk/scr/classification/run_resnet50.py \
  --export-model-if-missing \
  --accuracy-samples 100 \
  --skip-performance
```

## Ручной порядок

Экспортировать ONNX из `torchvision`:

```bash
python /Users/user/tomsk/scr/classification/export_resnet50_to_onnx.py
```

Диагностический экспорт под возможный `mlperf`-контракт `NHWC + uint8`:

```bash
python /Users/user/tomsk/scr/classification/export_resnet50_to_onnx.py \
  --input-layout nhwc \
  --input-value-range uint8
```

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

Performance:

```bash
python /Users/user/tomsk/scr/classification/run_resnet50_performance.py \
  --batch-size 1

python /Users/user/tomsk/scr/classification/run_resnet50_performance.py \
  --batch-size 8
```

## Замечания

- Если `models/classification/resnet50.onnx` отсутствует, orchestrator может экспортировать его из `torchvision`.
- По умолчанию экспорт идет с `opset 13`, потому что он безопаснее для vendor-конвертера, чем более новые версии.
- При первом экспорте pretrained-весов `torchvision` нужен доступ в интернет.
- По умолчанию экспортируется модель с входом `NCHW + normalized`.
- Для диагностики можно попробовать `--export-input-layout nhwc --export-input-value-range uint8` в `run_resnet50.py`.
- Для обычного `resnet50` используется свой evaluator `evaluate_resnet50_accuracy.py`, а не vendor `accuracy-imagenet.py`.
- `accuracy` по умолчанию использует первые `5000` строк из `data/evaluation/imagenet/val_map.txt`.
- Для отладки можно уменьшить выборку, например `--accuracy-samples 100` или `run_resnet50_accuracy.py --samples 100`.
- `performance` следует ПМИ: запускается через `mlperf` и использует значения `qps` по умолчанию:
  - `500` для `batch 1`
  - `1000` для `batch 8`
- Итоговая сводка сохраняется в `experiments/classification/results_summary.json`.
