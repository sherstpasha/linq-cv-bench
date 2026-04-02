import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, TextIO, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run parallel direct TPU accuracy for ResNet-50 by sharding ImageNet across multiple TPU devices"
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--program-path",
        type=Path,
        default=REPO_ROOT / "artifacts/classification_HPD1M_O5/resnet50_b1.tpu",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "data/evaluation/imagenet",
    )
    parser.add_argument("--val-map", type=Path, default=None)
    parser.add_argument(
        "--build-summary",
        type=Path,
        default=REPO_ROOT / "artifacts/classification_HPD1M_O5/build_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiments/classification_HPD1M_O5_parallel_direct",
    )
    parser.add_argument("--devices", nargs="*", default=None, help="Explicit TPU devices, e.g. /dev/tpu3 /dev/tpu2")
    parser.add_argument("--scales", nargs="*", type=int, default=None, help="Parallel scales, e.g. 1 2 3 4")
    parser.add_argument("--samples", type=int, default=0, help="0 means full val_map.txt")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_val_rows(path: Path) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_name, label = line.split()
            rows.append((image_name, int(label)))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def detect_available_devices(explicit_devices: Sequence[str] | None) -> List[str]:
    if explicit_devices:
        return list(explicit_devices)
    try:
        import pytpu as tpu  # type: ignore
    except Exception as error:
        raise RuntimeError("Missing dependency: pytpu") from error
    devices = list(tpu.Device.list_devices())
    if not devices:
        raise RuntimeError("TPU device not found (Device.list_devices() is empty)")
    return devices


def write_shards(rows: Sequence[Tuple[str, int]], out_dir: Path, shard_count: int) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    shards: List[List[Tuple[str, int]]] = [[] for _ in range(shard_count)]
    for idx, row in enumerate(rows):
        shards[idx % shard_count].append(row)

    paths: List[Path] = []
    for idx, shard_rows in enumerate(shards):
        shard_path = out_dir / f"val_map_shard_{idx}.txt"
        shard_text = "\n".join(f"{image} {label}" for image, label in shard_rows) + "\n"
        shard_path.write_text(shard_text, encoding="utf-8")
        paths.append(shard_path)
    return paths


def run_scale(
    py: str,
    program_path: Path,
    dataset_dir: Path,
    build_summary: Path,
    run_root: Path,
    devices: Sequence[str],
    shard_paths: Sequence[Path],
    repeats: int,
    warmup_batches: int,
) -> Dict:
    iteration_summaries: List[Dict] = []

    for run_index in range(1, repeats + 1):
        iter_dir = run_root / f"iter_{run_index:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        processes: List[Tuple[int, str, subprocess.Popen, Path, Path, TextIO, TextIO]] = []
        start = time.perf_counter()
        for shard_idx, (device, shard_path) in enumerate(zip(devices, shard_paths)):
            summary_out = iter_dir / f"summary_{shard_idx}.json"
            predictions_out = iter_dir / f"predictions_{shard_idx}.jsonl"
            stdout_path = iter_dir / f"stdout_{shard_idx}.txt"
            stderr_path = iter_dir / f"stderr_{shard_idx}.txt"
            stdout_file = stdout_path.open("w", encoding="utf-8")
            stderr_file = stderr_path.open("w", encoding="utf-8")

            cmd = [
                py,
                (THIS_DIR / "run_resnet50_accuracy.py").as_posix(),
                "--program-path",
                program_path.as_posix(),
                "--dataset-dir",
                dataset_dir.as_posix(),
                "--build-summary",
                build_summary.as_posix(),
                "--val-map",
                shard_path.as_posix(),
                "--predictions-out",
                predictions_out.as_posix(),
                "--summary-out",
                summary_out.as_posix(),
                "--device",
                device,
                "--warmup-batches",
                str(warmup_batches),
            ]
            process = subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file, text=True)
            processes.append((shard_idx, device, process, stdout_path, stderr_path, stdout_file, stderr_file))

        failed: List[Dict] = []
        for shard_idx, device, process, stdout_path, stderr_path, stdout_file, stderr_file in processes:
            returncode = process.wait()
            stdout_file.close()
            stderr_file.close()
            if returncode != 0:
                failed.append(
                    {
                        "shard_index": shard_idx,
                        "device": device,
                        "returncode": returncode,
                        "stdout": stdout_path.as_posix(),
                        "stderr": stderr_path.as_posix(),
                    }
                )
        end = time.perf_counter()
        if failed:
            raise RuntimeError(f"Parallel run failed: {failed}")

        shard_summaries = [load_json(iter_dir / f"summary_{idx}.json") for idx in range(len(devices))]
        total = sum(item["total"] for item in shard_summaries)
        good_top1 = sum(item["good_top1"] for item in shard_summaries)
        good_top5 = sum(item["good_top5"] for item in shard_summaries)
        wall_sec = end - start

        aggregate = {
            "n_devices": len(devices),
            "run_index": run_index,
            "wall_sec": wall_sec,
            "effective_samples": total,
            "good_top1": good_top1,
            "good_top5": good_top5,
            "top1_accuracy": 100.0 * good_top1 / total,
            "top5_accuracy": 100.0 * good_top5 / total,
            "effective_fps": total / wall_sec,
            "devices": list(devices),
            "shards": [
                {
                    "device": item["device"],
                    "samples": item["total"],
                    "throughput_img_per_sec": item["throughput_img_per_sec"],
                    "summary_file": f"summary_{idx}.json",
                    "stdout_file": f"stdout_{idx}.txt",
                    "stderr_file": f"stderr_{idx}.txt",
                }
                for idx, item in enumerate(shard_summaries)
            ],
        }
        aggregate_path = iter_dir / "aggregate.json"
        aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        iteration_summaries.append(aggregate)

    mean_summary = {
        "runs": repeats,
        "n_devices": len(devices),
        "effective_samples": iteration_summaries[0]["effective_samples"],
        "top1_accuracy_mean": statistics.mean(item["top1_accuracy"] for item in iteration_summaries),
        "top5_accuracy_mean": statistics.mean(item["top5_accuracy"] for item in iteration_summaries),
        "wall_sec_mean": statistics.mean(item["wall_sec"] for item in iteration_summaries),
        "effective_fps_mean": statistics.mean(item["effective_fps"] for item in iteration_summaries),
        "iterations": iteration_summaries,
    }
    summary_path = run_root / f"aggregate_{repeats}runs.json"
    summary_path.write_text(json.dumps(mean_summary, indent=2), encoding="utf-8")
    return mean_summary


def main() -> None:
    args = parse_args()
    py = args.python.as_posix()

    if not args.program_path.exists():
        raise FileNotFoundError(f"TPU program not found: {args.program_path}")
    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {args.dataset_dir}")
    if not args.build_summary.exists():
        raise FileNotFoundError(f"Build summary not found: {args.build_summary}")
    if args.repeats <= 0:
        raise RuntimeError("--repeats must be > 0")

    val_map = args.val_map or (args.dataset_dir / "val_map.txt")
    if not val_map.exists():
        raise FileNotFoundError(f"val_map.txt not found: {val_map}")

    devices = detect_available_devices(args.devices)
    scales = args.scales or list(range(1, len(devices) + 1))
    if not scales:
        raise RuntimeError("No scales requested")
    if min(scales) <= 0:
        raise RuntimeError("--scales must contain only positive values")
    if max(scales) > len(devices):
        raise RuntimeError(f"Requested scale {max(scales)} exceeds available device count {len(devices)}")

    rows = load_val_rows(val_map)
    if args.samples > 0:
        rows = rows[: min(args.samples, len(rows))]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict] = {}
    for scale in scales:
        run_root = args.output_dir / f"n{scale}"
        shard_paths = write_shards(rows, run_root / "valmaps", scale)
        summary = run_scale(
            py=py,
            program_path=args.program_path,
            dataset_dir=args.dataset_dir,
            build_summary=args.build_summary,
            run_root=run_root,
            devices=devices[:scale],
            shard_paths=shard_paths,
            repeats=args.repeats,
            warmup_batches=args.warmup_batches,
        )
        results[f"n{scale}"] = summary

    baseline_key = f"n{min(scales)}"
    baseline_fps = results[baseline_key]["effective_fps_mean"]
    baseline_devices = results[baseline_key]["n_devices"]
    for key, summary in results.items():
        summary["speedup_vs_baseline"] = summary["effective_fps_mean"] / baseline_fps
        summary["scaling_efficiency_vs_baseline"] = summary["speedup_vs_baseline"] / (
            summary["n_devices"] / baseline_devices
        )

    output_json = args.output_json or (args.output_dir / "results_summary.json")
    final_summary = {
        "pipeline": "resnet50_parallel_direct",
        "program_path": args.program_path.as_posix(),
        "dataset_dir": args.dataset_dir.as_posix(),
        "val_map": val_map.as_posix(),
        "build_summary": args.build_summary.as_posix(),
        "requested_samples": args.samples,
        "effective_samples": len(rows),
        "available_devices": devices,
        "scales": scales,
        "repeats": args.repeats,
        "warmup_batches": args.warmup_batches,
        "results": results,
    }
    output_json.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(f"Saved parallel direct summary: {output_json}")


if __name__ == "__main__":
    main()
