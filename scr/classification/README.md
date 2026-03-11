# classification

Классификационный сценарий построен вокруг `resnet50_mlperf` и `INT8/TPU`.

Используемые пути:

- `data/evaluation/imagenet` - данные для `accuracy`
- `data/calibration/imagenet` - данные для калибровки
- `models/classification/resnet50.onnx` - исходная ONNX-модель
- `artifacts/classification` - `.qm`, `.tpu`, build metadata
- `experiments/classification` - логи запусков и итоговые JSON

## Что должно быть установлено отдельно

- `mlperf`
- `scr/classification/accuracy-imagenet.py`
- `tpu_framework`
- `tpu_compiler`

## Быстрый запуск

```bash
python /Users/user/tomsk/scr/classification/run_resnet50.py \
  --model-path /Users/user/tomsk/models/classification/resnet50.onnx
```

## Ручной порядок

Сборка артефактов:

```bash
python /Users/user/tomsk/scr/classification/build_resnet50_program.py \
  --model-path /Users/user/tomsk/models/classification/resnet50.onnx
```

Accuracy:

```bash
python /Users/user/tomsk/scr/classification/run_resnet50_accuracy.py \
  --program-path /Users/user/tomsk/artifacts/classification/resnet50_mlperf_b1.tpu
```

Performance:

```bash
python /Users/user/tomsk/scr/classification/run_resnet50_performance.py \
  --batch-size 1

python /Users/user/tomsk/scr/classification/run_resnet50_performance.py \
  --batch-size 8
```

## Замечания

- ONNX-модель считается вашей входной моделью и должна быть подготовлена отдельно.
- `accuracy` использует весь `data/evaluation/imagenet`, если явно не задан `--samples`.
- `performance` следует ПМИ: запускается через `mlperf` и использует значения `qps` по умолчанию:
  - `500` для `batch 1`
  - `1000` для `batch 8`
- Итоговая сводка сохраняется в `experiments/classification/results_summary.json`.
