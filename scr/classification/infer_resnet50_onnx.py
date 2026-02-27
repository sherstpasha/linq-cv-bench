import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    path: Path
    key: str


class ImageDataset(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose):
        self.transform = transform
        self.samples: List[Sample] = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                self.samples.append(Sample(path=path, key=path.relative_to(root).as_posix()))
        if not self.samples:
            raise ValueError(f"No images found under: {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        with Image.open(sample.path) as img:
            tensor = self.transform(img.convert("RGB"))
        return tensor, sample.key


def collate_fn(batch: List[Tuple[object, str]]):
    return [x[0] for x in batch], [x[1] for x in batch]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ONNX ResNet50 inference and save predictions")
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "experiments/classification/resnet50.onnx")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/evaluation/imagenet")
    parser.add_argument("--predictions-out", type=Path, default=REPO_ROOT / "experiments/classification/predictions.jsonl")
    parser.add_argument("--timing-out", type=Path, default=REPO_ROOT / "experiments/classification/inference_timing.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--providers", type=str, default=None)
    return parser.parse_args()


def resolve_providers(user_providers: Optional[str]) -> Sequence[str]:
    available = ort.get_available_providers()
    if user_providers:
        selected = [p.strip() for p in user_providers.split(",") if p.strip() in available]
        if not selected:
            raise RuntimeError(f"No requested providers available. available={available}")
        return selected
    return [p for p in ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]


def main() -> None:
    args = parse_args()
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.timing_out.parent.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset: Dataset = ImageDataset(args.data_dir, transform=transform)
    if args.limit > 0:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    providers = resolve_providers(args.providers)
    session = ort.InferenceSession(args.model_path.as_posix(), providers=list(providers))
    active_providers = session.get_providers()
    print(f"Active providers: {active_providers}")
    input_name = session.get_inputs()[0].name

    infer_time = 0.0
    measured_images = 0
    measured_batches = 0
    total_images = 0

    with args.predictions_out.open("w", encoding="utf-8") as out_file:
        for batch_idx, (images, keys) in enumerate(tqdm(loader, desc="Inference")):
            x = np.stack([img.numpy() for img in images]).astype("float32")
            t0 = time.perf_counter()
            logits = session.run(None, {input_name: x})[0]
            t1 = time.perf_counter()

            if batch_idx >= args.warmup_batches:
                infer_time += (t1 - t0)
                measured_images += len(keys)
                measured_batches += 1

            top5 = np.argpartition(logits, -5, axis=1)[:, -5:]
            top5_scores = np.take_along_axis(logits, top5, axis=1)
            order = np.argsort(-top5_scores, axis=1)
            sorted_top5 = np.take_along_axis(top5, order, axis=1)

            for key, pred in zip(keys, sorted_top5):
                out_file.write(json.dumps({"image": key, "top5": [int(v) for v in pred]}) + "\n")
            total_images += len(keys)

    timing = {
        "providers": list(active_providers),
        "images": total_images,
        "warmup_batches": args.warmup_batches,
        "measured_inference_sec": infer_time,
        "throughput_img_per_sec": measured_images / max(infer_time, 1e-9),
        "avg_batch_latency_ms": (infer_time / max(measured_batches, 1)) * 1000.0,
        "predictions_file": args.predictions_out.as_posix(),
    }
    with args.timing_out.open("w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
