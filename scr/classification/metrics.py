import argparse
import json
from pathlib import Path
from typing import Dict, List

from metrics_utils import compute_topk_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Top-1/Top-5 from predictions and ground truth")
    parser.add_argument("--predictions", type=Path, default=REPO_ROOT / "experiments/classification/predictions.jsonl")
    parser.add_argument("--ground-truth", type=Path, default=REPO_ROOT / "data/imagenet/val_map.txt")
    parser.add_argument("--output-json", type=Path, default=REPO_ROOT / "experiments/classification/metrics.json")
    parser.add_argument("--ground-truth-offset", type=int, default=-1)
    return parser.parse_args()


def load_ground_truth(path: Path, offset: int) -> Dict[str, int]:
    data: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, cls = line.split()
            data[name] = int(cls) + offset
    return data


def load_predictions(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    metrics = compute_topk_metrics(load_predictions(args.predictions), load_ground_truth(args.ground_truth, args.ground_truth_offset))
    result = {
        "predictions_file": args.predictions.as_posix(),
        "ground_truth_file": args.ground_truth.as_posix(),
        "ground_truth_offset": args.ground_truth_offset,
        **metrics,
    }
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
