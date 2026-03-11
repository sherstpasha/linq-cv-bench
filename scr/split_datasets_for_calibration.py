import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split available datasets into evaluation and calibration subsets"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Root data directory")
    parser.add_argument("--ratio", type=float, default=0.1, help="Calibration ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force", action="store_true", help="Overwrite data/evaluation and data/calibration")
    return parser.parse_args()


def ensure_clean_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise RuntimeError(f"Path exists: {path}. Use --force to overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def split_items(items: Sequence[str], ratio: float, rng: random.Random) -> Tuple[List[str], List[str]]:
    shuffled = list(items)
    rng.shuffle(shuffled)
    calibration_count = max(1, int(len(shuffled) * ratio))
    calibration = sorted(shuffled[:calibration_count])
    evaluation = sorted(shuffled[calibration_count:])
    return evaluation, calibration


def split_imagenet(
    data_dir: Path,
    evaluation_root: Path,
    calibration_root: Path,
    ratio: float,
    rng: random.Random,
) -> Dict:
    source_root = data_dir / "imagenet"
    source_map = source_root / "val_map.txt"
    if not source_root.exists() or not source_map.exists():
        raise FileNotFoundError(f"ImageNet source not found: {source_root}")

    rows: List[Tuple[str, str]] = []
    with source_map.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            relative_path, class_id = line.split()
            rows.append((relative_path, class_id))

    keys = [key for key, _ in rows]
    evaluation_keys, calibration_keys = split_items(keys, ratio, rng)
    class_by_key = {key: class_id for key, class_id in rows}

    evaluation_dir = evaluation_root / "imagenet"
    calibration_dir = calibration_root / "imagenet"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    calibration_dir.mkdir(parents=True, exist_ok=True)

    with (evaluation_dir / "val_map.txt").open("w", encoding="utf-8") as file:
        for key in evaluation_keys:
            file.write(f"{key} {class_by_key[key]}\n")
            copy_or_link(source_root / key, evaluation_dir / key)

    with (calibration_dir / "val_map.txt").open("w", encoding="utf-8") as file:
        for key in calibration_keys:
            file.write(f"{key} {class_by_key[key]}\n")
            copy_or_link(source_root / key, calibration_dir / key)

    return {
        "dataset": "imagenet",
        "total": len(rows),
        "evaluation": len(evaluation_keys),
        "calibration": len(calibration_keys),
    }


def split_coco(
    data_dir: Path,
    evaluation_root: Path,
    calibration_root: Path,
    ratio: float,
    rng: random.Random,
) -> Dict:
    source_root = data_dir / "MSCOCO2017"
    image_dir = source_root / "val2017"
    annotation_file = source_root / "annotations/instances_val2017.json"
    if not image_dir.exists() or not annotation_file.exists():
        raise FileNotFoundError(f"MSCOCO2017 source not found: {source_root}")

    coco = json.loads(annotation_file.read_text(encoding="utf-8"))
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    info = coco.get("info")
    licenses = coco.get("licenses")

    image_ids = [str(int(item["id"])) for item in images]
    evaluation_ids, calibration_ids = split_items(image_ids, ratio, rng)
    evaluation_id_set = {int(item) for item in evaluation_ids}
    calibration_id_set = {int(item) for item in calibration_ids}

    def build_subset(destination_root: Path, selected_ids: set[int]) -> Dict:
        destination_image_dir = destination_root / "MSCOCO2017/val2017"
        destination_annotation_dir = destination_root / "MSCOCO2017/annotations"
        destination_image_dir.mkdir(parents=True, exist_ok=True)
        destination_annotation_dir.mkdir(parents=True, exist_ok=True)

        subset_images = [item for item in images if int(item["id"]) in selected_ids]
        subset_annotations = [item for item in annotations if int(item["image_id"]) in selected_ids]

        for image in subset_images:
            copy_or_link(image_dir / image["file_name"], destination_image_dir / image["file_name"])

        subset_json = {
            "images": subset_images,
            "annotations": subset_annotations,
            "categories": categories,
        }
        if info is not None:
            subset_json["info"] = info
        if licenses is not None:
            subset_json["licenses"] = licenses

        destination_annotation_file = destination_annotation_dir / "instances_val2017.json"
        destination_annotation_file.write_text(json.dumps(subset_json), encoding="utf-8")
        return {
            "images": len(subset_images),
            "annotations": len(subset_annotations),
            "annotation_file": destination_annotation_file.as_posix(),
        }

    evaluation_stats = build_subset(evaluation_root, evaluation_id_set)
    calibration_stats = build_subset(calibration_root, calibration_id_set)
    return {
        "dataset": "mscoco2017",
        "total_images": len(images),
        "evaluation_images": evaluation_stats["images"],
        "evaluation_annotations": evaluation_stats["annotations"],
        "calibration_images": calibration_stats["images"],
        "calibration_annotations": calibration_stats["annotations"],
    }


def split_voc(
    data_dir: Path,
    evaluation_root: Path,
    calibration_root: Path,
    ratio: float,
    rng: random.Random,
) -> Dict:
    source_root = data_dir / "VOCdevkit/VOC2012"
    jpeg_dir = source_root / "JPEGImages"
    mask_dir = source_root / "SegmentationClass"
    split_file = source_root / "ImageSets/Segmentation/val.txt"
    if not jpeg_dir.exists() or not mask_dir.exists() or not split_file.exists():
        raise FileNotFoundError(f"VOC2012 source not found: {source_root}")

    image_ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    evaluation_ids, calibration_ids = split_items(image_ids, ratio, rng)

    def build_subset(destination_root: Path, subset_ids: List[str]) -> int:
        base_dir = destination_root / "VOCdevkit/VOC2012"
        destination_jpeg_dir = base_dir / "JPEGImages"
        destination_mask_dir = base_dir / "SegmentationClass"
        destination_split_file = base_dir / "ImageSets/Segmentation/val.txt"

        destination_jpeg_dir.mkdir(parents=True, exist_ok=True)
        destination_mask_dir.mkdir(parents=True, exist_ok=True)
        destination_split_file.parent.mkdir(parents=True, exist_ok=True)

        for image_id in subset_ids:
            copy_or_link(jpeg_dir / f"{image_id}.jpg", destination_jpeg_dir / f"{image_id}.jpg")
            copy_or_link(mask_dir / f"{image_id}.png", destination_mask_dir / f"{image_id}.png")

        destination_split_file.write_text("\n".join(subset_ids) + "\n", encoding="utf-8")
        return len(subset_ids)

    evaluation_count = build_subset(evaluation_root, evaluation_ids)
    calibration_count = build_subset(calibration_root, calibration_ids)
    return {
        "dataset": "voc2012",
        "total": len(image_ids),
        "evaluation": evaluation_count,
        "calibration": calibration_count,
    }


def main() -> None:
    args = parse_args()
    if not (0.0 < args.ratio < 1.0):
        raise RuntimeError("--ratio must be between 0 and 1")

    evaluation_root = args.data_dir / "evaluation"
    calibration_root = args.data_dir / "calibration"
    ensure_clean_dir(evaluation_root, args.force)
    ensure_clean_dir(calibration_root, args.force)

    rng = random.Random(args.seed)
    splitters = [
        ("imagenet", split_imagenet),
        ("mscoco2017", split_coco),
        ("voc2012", split_voc),
    ]

    report = {
        "data_dir": args.data_dir.as_posix(),
        "evaluation_dir": evaluation_root.as_posix(),
        "calibration_dir": calibration_root.as_posix(),
        "ratio": args.ratio,
        "seed": args.seed,
        "splits": {},
        "skipped": [],
    }

    for dataset_name, splitter in splitters:
        try:
            report["splits"][dataset_name] = splitter(
                data_dir=args.data_dir,
                evaluation_root=evaluation_root,
                calibration_root=calibration_root,
                ratio=args.ratio,
                rng=rng,
            )
        except FileNotFoundError as error:
            report["skipped"].append({"dataset": dataset_name, "reason": str(error)})

    if not report["splits"]:
        raise RuntimeError(
            f"No supported source datasets were found under {args.data_dir}. "
            "Expected at least one of: imagenet, MSCOCO2017, VOCdevkit/VOC2012."
        )

    report_path = args.data_dir / "split_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved split report: {report_path}")


if __name__ == "__main__":
    main()
