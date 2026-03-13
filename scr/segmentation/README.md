# segmentation

Рабочий контур сегментации теперь один:

1. экспорт `FCN-ResNet50` из `torchvision` в `ONNX`
2. квантование и компиляция в свой `.tpu`
3. `accuracy` через собственный direct TPU runner с расчетом `pixel_accuracy` и `mean_iou`
4. `performance` через `mlperf` на `batch 1` и `batch 8`

Используемые пути:

- `data/evaluation/VOCdevkit/VOC2012` - данные для accuracy
- `data/calibration/VOCdevkit/VOC2012/JPEGImages` - данные для калибровки
- `experiments/segmentation/fcn_resnet50.onnx` - экспортированный ONNX
- `artifacts/segmentation` - `.qm`, `.tpu`, build metadata
- `experiments/segmentation` - predictions, accuracy и итоговая сводка

## Что должно быть установлено отдельно

## Что должно быть установлено отдельно

- `pytpu`
- `tpu_framework`
- `tpu_compiler`
- `torch`
- `torchvision`
- `mlperf`

## Один запуск под ключ

```bash
python scr/segmentation/run_fcn_resnet50.py
```

Итоговая сводка:

- `experiments/segmentation/results_summary.json`

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
