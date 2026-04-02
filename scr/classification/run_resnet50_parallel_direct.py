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
    parser.add_argument("--model-path", type=Path, default=None, help="Optional ONNX path for auto-build")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=REPO_ROOT / "data/calibration/imagenet",
        help="Calibration image directory for auto-build",
    )
    parser.add_argument("--model-name", type=str, default=None, help="Artifact name prefix for auto-build")
    parser.add_argument(
        "--compile-preset",
        type=str,
        default=None,
        choices=["O1", "O5", "DEFAULT"],
        help="Compile preset for auto-build; inferred from paths if omitted",
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


def infer_model_name(program_path: Path, explicit_model_name: str | None) -> str:
    if explicit_model_name:
        return explicit_model_name
    stem = program_path.stem
    if "_b" in stem:
        return stem.split("_b", 1)[0]
    return stem


def infer_compile_preset(
    explicit_compile_preset: str | None,
    program_path: Path,
    build_summary: Path,
    model_path: Path | None,
) -> str:
    if explicit_compile_preset:
        return explicit_compile_preset
    haystacks = [
        program_path.as_posix().upper(),
        build_summary.as_posix().upper(),
        model_path.as_posix().upper() if model_path is not None else "",
    ]
    return "O5" if any("_O5" in item or "/O5" in item for item in haystacks) else "O1"


def infer_model_path(explicit_model_path: Path | None, build_summary: Path, program_path: Path) -> Path:
    if explicit_model_path is not None:
        return explicit_model_path

    artifacts_dir_name = program_path.parent.name
    model_name = infer_model_name(program_path, None)
    candidates = [
        REPO_ROOT / "models/classification" / f"{artifacts_dir_name.replace('classification_', model_name + '_')}.onnx",
        REPO_ROOT / "models/classification" / f"{model_name}_{artifacts_dir_name.split('classification_', 1)[-1]}.onnx",
        REPO_ROOT / "models/classification" / f"{model_name}.onnx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def ensure_build_artifacts(
    py: str,
    program_path: Path,
    build_summary: Path,
    model_path: Path,
    calibration_dir: Path,
    model_name: str,
    compile_preset: str,
) -> List[str] | None:
    if program_path.exists() and build_summary.exists():
        return None
    if not model_path.exists():
        raise FileNotFoundError(
            f"TPU program/build summary missing, and model for auto-build not found: {model_path}"
        )
    if not calibration_dir.exists():
        raise FileNotFoundError(
            f"TPU program/build summary missing, and calibration dir for auto-build not found: {calibration_dir}"
        )

    artifacts_dir = program_path.parent
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        py,
        (THIS_DIR / "build_resnet50_program.py").as_posix(),
        "--model-path",
        model_path.as_posix(),
        "--calibration-dir",
        calibration_dir.as_posix(),
        "--artifacts-dir",
        artifacts_dir.as_posix(),
        "--model-name",
        model_name,
        "--compile-preset",
        compile_preset,
        "--batch-sizes",
        "1",
        "--metadata-out",
        build_summary.as_posix(),
    ]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if not program_path.exists():
        raise FileNotFoundError(f"Auto-build completed without producing TPU program: {program_path}")
    if not build_summary.exists():
        raise FileNotFoundError(f"Auto-build completed without producing build summary: {build_summary}")
    return cmd


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

    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {args.dataset_dir}")
    if args.repeats <= 0:
        raise RuntimeError("--repeats must be > 0")

    val_map = args.val_map or (args.dataset_dir / "val_map.txt")
    if not val_map.exists():
        raise FileNotFoundError(f"val_map.txt not found: {val_map}")

    model_name = infer_model_name(args.program_path, args.model_name)
    model_path = infer_model_path(args.model_path, args.build_summary, args.program_path)
    compile_preset = infer_compile_preset(args.compile_preset, args.program_path, args.build_summary, model_path)
    auto_build_cmd = ensure_build_artifacts(
        py=py,
        program_path=args.program_path,
        build_summary=args.build_summary,
        model_path=model_path,
        calibration_dir=args.calibration_dir,
        model_name=model_name,
        compile_preset=compile_preset,
    )

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
        "model_path": model_path.as_posix(),
        "model_name": model_name,
        "compile_preset": compile_preset,
        "calibration_dir": args.calibration_dir.as_posix(),
        "dataset_dir": args.dataset_dir.as_posix(),
        "val_map": val_map.as_posix(),
        "build_summary": args.build_summary.as_posix(),
        "auto_build": {
            "executed": auto_build_cmd is not None,
            "command": auto_build_cmd,
        },
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
