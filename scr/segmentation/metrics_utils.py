from typing import Dict, List, Tuple

import numpy as np

VOC_CLASSES: List[str] = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def update_confusion(conf: np.ndarray, gt: np.ndarray, pred: np.ndarray, num_classes: int, ignore_index: int) -> None:
    valid = (gt != ignore_index) & (gt >= 0) & (gt < num_classes)
    if not np.any(valid):
        return
    hist = np.bincount(
        num_classes * gt[valid].astype(np.int64) + pred[valid].astype(np.int64),
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    conf += hist


def summarize_confusion(conf: np.ndarray, class_names: List[str]) -> Dict:
    tp = np.diag(conf).astype(np.float64)
    gt_pixels = conf.sum(axis=1).astype(np.float64)
    pred_pixels = conf.sum(axis=0).astype(np.float64)

    union = gt_pixels + pred_pixels - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)

    pixel_acc = float(tp.sum() / max(conf.sum(), 1))
    mean_iou = float(iou[union > 0].mean()) if np.any(union > 0) else 0.0

    per_class: List[Tuple[str, float]] = []
    for name, val in zip(class_names, iou.tolist()):
        per_class.append((name, float(val)))

    return {
        "pixel_accuracy": pixel_acc,
        "mean_iou": mean_iou,
        "per_class_iou": [{"class": n, "iou": v} for n, v in per_class],
    }
