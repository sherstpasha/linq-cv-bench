import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FCN-ResNet50 ONNX inference on VOC segmentation split")
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "experiments/segmentation/fcn_resnet50.onnx")
    parser.add_argument("--voc-root", type=Path, default=REPO_ROOT / "data/evaluation/VOCdevkit/VOC2012")
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--predictions-dir", type=Path, default=REPO_ROOT / "experiments/segmentation/predictions")
    parser.add_argument("--timing-out", type=Path, default=REPO_ROOT / "experiments/segmentation/inference_timing.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup-images", type=int, default=5)
    parser.add_argument("--providers", type=str, default=None)
    parser.add_argument("--height", type=int, default=520)
    parser.add_argument("--width", type=int, default=520)
    parser.add_argument("--num-classes", type=int, default=21)
    return parser.parse_args()


def resolve_providers(user_providers: Optional[str]) -> Sequence[str]:
    available = ort.get_available_providers()
    if user_providers:
        selected = [p.strip() for p in user_providers.split(",") if p.strip() in available]
        if not selected:
            raise RuntimeError(f"No requested providers available. available={available}")
        return selected
    return [p for p in ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]


def load_ids(split_file: Path, limit: int) -> List[str]:
    ids = [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    return ids[:limit] if limit > 0 else ids


def preprocess(image: Image.Image, width: int, height: int) -> np.ndarray:
    image = image.resize((width, height), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(arr.transpose(2, 0, 1), axis=0).astype(np.float32)


def logits_to_mask(sample_logits: np.ndarray, num_classes: int) -> np.ndarray:
    if sample_logits.ndim != 3:
        raise RuntimeError(f"Unexpected per-sample logits shape: {sample_logits.shape}")
    if sample_logits.shape[0] == num_classes:
        class_axis = 0
    elif sample_logits.shape[-1] == num_classes:
        class_axis = 2
    else:
        class_axis = int(np.argmin(sample_logits.shape))
    return np.argmax(sample_logits, axis=class_axis).astype(np.uint8)


def main() -> None:
    args = parse_args()
    split_file = args.split_file or (args.voc_root / "ImageSets/Segmentation/val.txt")
    jpeg_dir = args.voc_root / "JPEGImages"
    args.predictions_dir.mkdir(parents=True, exist_ok=True)
    args.timing_out.parent.mkdir(parents=True, exist_ok=True)

    ids = load_ids(split_file, args.limit)
    # Torch import preloads CUDA shared libs in many environments where ORT
    # cannot resolve them from system paths directly.
    try:
        import torch  # type: ignore
        _ = torch.cuda.is_available()
    except Exception:
        pass

    providers = resolve_providers(args.providers)
    session = ort.InferenceSession(args.model_path.as_posix(), providers=list(providers))
    active_providers = session.get_providers()
    print(f"Active providers: {active_providers}")
    input_name = session.get_inputs()[0].name

    infer_time = 0.0
    measured_images = 0

    for i, image_id in enumerate(tqdm(ids, desc="Inference")):
        image = Image.open(jpeg_dir / f"{image_id}.jpg").convert("RGB")
        x = preprocess(image, width=args.width, height=args.height)
        t0 = time.perf_counter()
        logits = session.run(None, {input_name: x})[0]
        t1 = time.perf_counter()
        if i >= args.warmup_images:
            infer_time += (t1 - t0)
            measured_images += 1
        pred = logits_to_mask(np.asarray(logits[0]), num_classes=args.num_classes)
        Image.fromarray(pred).save(args.predictions_dir / f"{image_id}.png")

    timing = {
        "providers": list(active_providers),
        "images": len(ids),
        "warmup_images": args.warmup_images,
        "measured_inference_sec": infer_time,
        "throughput_img_per_sec": measured_images / max(infer_time, 1e-9),
        "predictions_dir": args.predictions_dir.as_posix(),
        "split_file": split_file.as_posix(),
        "input_size": [args.height, args.width],
    }
    with args.timing_out.open("w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
