import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split datasets into evaluation (90%) and calibration (10%) subsets")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Root data directory")
    parser.add_argument("--ratio", type=float, default=0.1, help="Calibration ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force", action="store_true", help="Overwrite data/evaluation and data/calibration if they exist")
    parser.add_argument("--keep-originals", action="store_true", help="Keep original dataset folders after split")
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
    calib_count = max(1, int(len(shuffled) * ratio))
    calib = sorted(shuffled[:calib_count])
    eval_ = sorted(shuffled[calib_count:])
    return eval_, calib


def split_imagenet(data_dir: Path, eval_root: Path, calib_root: Path, ratio: float, rng: random.Random) -> Dict:
    src_root = data_dir / "imagenet"
    src_map = src_root / "val_map.txt"
    if not src_root.exists() or not src_map.exists():
        raise RuntimeError(f"ImageNet source not found: {src_root}")

    rows: List[Tuple[str, str]] = []
    with src_map.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rel, cls = line.split()
            rows.append((rel, cls))
    all_keys = [k for k, _ in rows]
    eval_keys, calib_keys = split_items(all_keys, ratio, rng)
    row_by_key = {k: v for k, v in rows}

    eval_dst = eval_root / "imagenet"
    calib_dst = calib_root / "imagenet"
    eval_dst.mkdir(parents=True, exist_ok=True)
    calib_dst.mkdir(parents=True, exist_ok=True)

    with (eval_dst / "val_map.txt").open("w", encoding="utf-8") as f:
        for key in eval_keys:
            f.write(f"{key} {row_by_key[key]}\n")
            copy_or_link(src_root / key, eval_dst / key)

    with (calib_dst / "val_map.txt").open("w", encoding="utf-8") as f:
        for key in calib_keys:
            f.write(f"{key} {row_by_key[key]}\n")
            copy_or_link(src_root / key, calib_dst / key)

    return {"total": len(rows), "evaluation": len(eval_keys), "calibration": len(calib_keys)}


def split_coco(data_dir: Path, eval_root: Path, calib_root: Path, ratio: float, rng: random.Random) -> Dict:
    src_root = data_dir / "MSCOCO2017"
    img_dir = src_root / "val2017"
    ann_file = src_root / "annotations/instances_val2017.json"
    if not img_dir.exists() or not ann_file.exists():
        raise RuntimeError(f"COCO source not found: {src_root}")

    coco = json.loads(ann_file.read_text(encoding="utf-8"))
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    info = coco.get("info")
    licenses = coco.get("licenses")

    image_ids = [int(x["id"]) for x in images]
    eval_ids, calib_ids = split_items([str(x) for x in image_ids], ratio, rng)
    eval_id_set = {int(x) for x in eval_ids}
    calib_id_set = {int(x) for x in calib_ids}

    def build_subset(dst_root: Path, id_set: set[int]) -> int:
        dst_img_dir = dst_root / "MSCOCO2017/val2017"
        dst_ann_dir = dst_root / "MSCOCO2017/annotations"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_ann_dir.mkdir(parents=True, exist_ok=True)

        subset_images = [img for img in images if int(img["id"]) in id_set]
        subset_ann = [ann for ann in annotations if int(ann["image_id"]) in id_set]

        for img in subset_images:
            copy_or_link(img_dir / img["file_name"], dst_img_dir / img["file_name"])

        subset_json = {"images": subset_images, "annotations": subset_ann, "categories": categories}
        if info is not None:
            subset_json["info"] = info
        if licenses is not None:
            subset_json["licenses"] = licenses
        (dst_ann_dir / "instances_val2017.json").write_text(json.dumps(subset_json), encoding="utf-8")
        return len(subset_ann)

    eval_ann_count = build_subset(eval_root, eval_id_set)
    calib_ann_count = build_subset(calib_root, calib_id_set)
    return {
        "total_images": len(images),
        "evaluation_images": len(eval_id_set),
        "calibration_images": len(calib_id_set),
        "evaluation_annotations": eval_ann_count,
        "calibration_annotations": calib_ann_count,
    }


def split_voc(data_dir: Path, eval_root: Path, calib_root: Path, ratio: float, rng: random.Random) -> Dict:
    src_voc = data_dir / "VOCdevkit/VOC2012"
    jpeg_dir = src_voc / "JPEGImages"
    mask_dir = src_voc / "SegmentationClass"
    split_file = src_voc / "ImageSets/Segmentation/val.txt"
    if not jpeg_dir.exists() or not mask_dir.exists() or not split_file.exists():
        raise RuntimeError(f"VOC source not found: {src_voc}")

    ids = [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    eval_ids, calib_ids = split_items(ids, ratio, rng)

    def build_subset(dst_root: Path, subset_ids: List[str]) -> None:
        base = dst_root / "VOCdevkit/VOC2012"
        dst_jpeg = base / "JPEGImages"
        dst_mask = base / "SegmentationClass"
        dst_split = base / "ImageSets/Segmentation/val.txt"
        dst_jpeg.mkdir(parents=True, exist_ok=True)
        dst_mask.mkdir(parents=True, exist_ok=True)
        dst_split.parent.mkdir(parents=True, exist_ok=True)

        for image_id in subset_ids:
            copy_or_link(jpeg_dir / f"{image_id}.jpg", dst_jpeg / f"{image_id}.jpg")
            copy_or_link(mask_dir / f"{image_id}.png", dst_mask / f"{image_id}.png")

        dst_split.write_text("\n".join(subset_ids) + "\n", encoding="utf-8")

    build_subset(eval_root, eval_ids)
    build_subset(calib_root, calib_ids)
    return {"total": len(ids), "evaluation": len(eval_ids), "calibration": len(calib_ids)}


def remove_original_sources(data_dir: Path) -> List[str]:
    removed: List[str] = []
    targets = [
        data_dir / "imagenet",
        data_dir / "MSCOCO2017",
        data_dir / "VOCdevkit",
    ]
    for path in targets:
        if path.exists():
            shutil.rmtree(path)
            removed.append(path.as_posix())
    return removed


def main() -> None:
    args = parse_args()
    if not (0.0 < args.ratio < 1.0):
        raise RuntimeError("--ratio must be between 0 and 1")

    eval_root = args.data_dir / "evaluation"
    calib_root = args.data_dir / "calibration"
    ensure_clean_dir(eval_root, args.force)
    ensure_clean_dir(calib_root, args.force)

    rng = random.Random(args.seed)
    result = {
        "data_dir": args.data_dir.as_posix(),
        "evaluation_dir": eval_root.as_posix(),
        "calibration_dir": calib_root.as_posix(),
        "ratio": args.ratio,
        "seed": args.seed,
        "splits": {},
    }

    result["splits"]["imagenet"] = split_imagenet(args.data_dir, eval_root, calib_root, args.ratio, rng)
    result["splits"]["mscoco2017"] = split_coco(args.data_dir, eval_root, calib_root, args.ratio, rng)
    result["splits"]["voc2012"] = split_voc(args.data_dir, eval_root, calib_root, args.ratio, rng)
    if args.keep_originals:
        result["removed_originals"] = []
    else:
        result["removed_originals"] = remove_original_sources(args.data_dir)

    out_path = args.data_dir / "split_report.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved split report: {out_path}")


if __name__ == "__main__":
    main()
