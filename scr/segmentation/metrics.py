import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from metrics_utils import VOC_CLASSES, summarize_confusion, update_confusion

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute VOC segmentation metrics from predicted masks")
    parser.add_argument("--voc-root", type=Path, default=REPO_ROOT / "data/VOCdevkit/VOC2012")
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--predictions-dir", type=Path, default=REPO_ROOT / "experiments/segmentation/predictions")
    parser.add_argument("--output-json", type=Path, default=REPO_ROOT / "experiments/segmentation/metrics.json")
    parser.add_argument("--num-classes", type=int, default=21)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_ids(split_file: Path, limit: int):
    ids = [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    return ids[:limit] if limit > 0 else ids


def main() -> None:
    args = parse_args()
    split_file = args.split_file or (args.voc_root / "ImageSets/Segmentation/val.txt")
    gt_dir = args.voc_root / "SegmentationClass"

    image_ids = load_ids(split_file, args.limit)
    conf = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    missing_preds = []
    resized_preds = 0

    for image_id in image_ids:
        gt_path = gt_dir / f"{image_id}.png"
        pred_path = args.predictions_dir / f"{image_id}.png"
        if not pred_path.exists():
            missing_preds.append(image_id)
            continue
        gt = np.array(Image.open(gt_path), dtype=np.uint8)
        pred_img = Image.open(pred_path)
        pred = np.array(pred_img, dtype=np.uint8)
        if pred.shape != gt.shape:
            pred = np.array(pred_img.resize((gt.shape[1], gt.shape[0]), resample=Image.NEAREST), dtype=np.uint8)
            resized_preds += 1
        update_confusion(conf, gt, pred, args.num_classes, args.ignore_index)

    result = {
        "voc_root": args.voc_root.as_posix(),
        "split_file": split_file.as_posix(),
        "predictions_dir": args.predictions_dir.as_posix(),
        "images_in_split": len(image_ids),
        "missing_predictions": len(missing_preds),
        "missing_predictions_preview": missing_preds[:20],
        "resized_predictions": resized_preds,
        **summarize_confusion(conf, VOC_CLASSES[: args.num_classes]),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
