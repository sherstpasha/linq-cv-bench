import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, TextIO, Tuple

from run_resnet50_performance import parse_mlperf_summary, resolve_qps


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run parallel MLPerf performance for ResNet-50 across multiple TPU devices"
    )
    parser.add_argument("--mlperf-binary", type=str, default="mlperf")
    parser.add_argument(
        "--program-path",
        type=Path,
        default=REPO_ROOT / "artifacts/classification_HPD1M_O5/resnet50_b1.tpu",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--qps", type=int, default=0)
    parser.add_argument("--scales", nargs="*", type=int, default=None, help="Parallel process counts, e.g. 1 2 3 4")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiments/classification_HPD1M_O5_parallel_mlperf",
    )
    parser.add_argument("--devices", nargs="*", default=None, help="Optional expected TPU devices, for reporting only")
    parser.add_argument("--poll-interval-sec", type=float, default=0.2)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def detect_available_devices(explicit_devices: Sequence[str] | None) -> List[str]:
    if explicit_devices:
        return list(explicit_devices)
    try:
        import pytpu as tpu  # type: ignore
    except Exception:
        return []
    try:
        devices = list(tpu.Device.list_devices())
    except Exception:
        return []
    return devices


def snapshot_open_tpu_devices(pid: int) -> List[str]:
    fd_root = Path(f"/proc/{pid}/fd")
    if not fd_root.exists():
        return []
    seen: Set[str] = set()
    try:
        iterator = list(fd_root.iterdir())
    except PermissionError:
        return []
    except OSError:
        return []
    for fd_path in iterator:
        try:
            target = fd_path.resolve(strict=True)
        except Exception:
            continue
        target_str = target.as_posix()
        if target_str.startswith("/dev/tpu"):
            seen.add(target_str)
    return sorted(seen)


def run_scale(
    mlperf_binary: str,
    program_path: Path,
    batch_size: int,
    qps: int,
    run_root: Path,
    scale: int,
    repeats: int,
    poll_interval_sec: float,
    devices: Sequence[str] | None,
) -> Dict:
    iteration_summaries: List[Dict] = []
    effective_qps = resolve_qps(batch_size, qps)

    for run_index in range(1, repeats + 1):
        iter_dir = run_root / f"iter_{run_index:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        processes: List[Tuple[int, subprocess.Popen, Path, Path, Path, TextIO, TextIO, Optional[str]]] = []
        observed_devices: Dict[int, Set[str]] = {}
        start = time.perf_counter()

        for proc_idx in range(scale):
            proc_dir = iter_dir / f"proc_{proc_idx}"
            proc_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = proc_dir / "mlperf_stdout.txt"
            stderr_path = proc_dir / "mlperf_stderr.txt"
            stdout_file = stdout_path.open("w", encoding="utf-8")
            stderr_file = stderr_path.open("w", encoding="utf-8")
            cmd = [
                mlperf_binary,
                "-s",
                "offline",
                "-m",
                "performance",
                "-p",
                program_path.as_posix(),
                "-q",
                str(effective_qps),
            ]
            assigned_device = None
            if devices:
                assigned_device = devices[proc_idx]
                cmd.extend(["--dev", assigned_device])
            process = subprocess.Popen(cmd, cwd=proc_dir, stdout=stdout_file, stderr=stderr_file, text=True)
            processes.append((proc_idx, process, proc_dir, stdout_path, stderr_path, stdout_file, stderr_file, assigned_device))
            observed_devices[proc_idx] = set()

        alive = True
        while alive:
            alive = False
            for proc_idx, process, _, _, _, _, _, _ in processes:
                if process.poll() is None:
                    alive = True
                for device in snapshot_open_tpu_devices(process.pid):
                    observed_devices[proc_idx].add(device)
            if alive:
                time.sleep(poll_interval_sec)

        end = time.perf_counter()

        process_summaries: List[Dict] = []
        for proc_idx, process, proc_dir, stdout_path, stderr_path, stdout_file, stderr_file, assigned_device in processes:
            stdout_file.close()
            stderr_file.close()
            summary_file = proc_dir / "mlperf_log_summary.txt"
            result, samples_per_second = parse_mlperf_summary(summary_file)
            process_summaries.append(
                {
                    "proc_index": proc_idx,
                    "pid": process.pid,
                    "returncode": process.returncode,
                    "summary_file": summary_file.as_posix(),
                    "stdout_file": stdout_path.as_posix(),
                    "stderr_file": stderr_path.as_posix(),
                    "result": result,
                    "samples_per_second": samples_per_second,
                    "assigned_device": assigned_device if devices else None,
                    "observed_devices": sorted(observed_devices[proc_idx]),
                    "is_valid": result == "VALID" and samples_per_second is not None,
                }
            )

        valid_samples = [item["samples_per_second"] for item in process_summaries if item["is_valid"]]
        all_observed_devices = sorted({device for item in process_summaries for device in item["observed_devices"]})
        aggregate = {
            "run_index": run_index,
            "n_processes": scale,
            "batch_size": batch_size,
            "requested_qps_per_process": qps,
            "effective_qps_per_process": effective_qps,
            "wall_sec": end - start,
            "all_valid": len(valid_samples) == scale,
            "valid_process_count": len(valid_samples),
            "total_samples_per_second": sum(valid_samples) if len(valid_samples) == scale else None,
            "distinct_device_count": len(all_observed_devices),
            "observed_devices": all_observed_devices,
            "processes": process_summaries,
        }
        (iter_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        iteration_summaries.append(aggregate)

    valid_iterations = [item for item in iteration_summaries if item["all_valid"]]
    summary = {
        "runs": repeats,
        "n_processes": scale,
        "batch_size": batch_size,
        "requested_qps_per_process": qps,
        "effective_qps_per_process": effective_qps,
        "all_iterations_valid": len(valid_iterations) == repeats,
        "valid_iteration_count": len(valid_iterations),
        "mean_total_samples_per_second": (
            statistics.mean(item["total_samples_per_second"] for item in valid_iterations)
            if valid_iterations
            else None
        ),
        "mean_wall_sec": statistics.mean(item["wall_sec"] for item in iteration_summaries),
        "mean_distinct_device_count": statistics.mean(item["distinct_device_count"] for item in iteration_summaries),
        "iterations": iteration_summaries,
    }
    (run_root / f"aggregate_{repeats}runs.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if not args.program_path.exists():
        raise FileNotFoundError(f"TPU program not found: {args.program_path}")
    if args.batch_size <= 0:
        raise RuntimeError("--batch-size must be > 0")
    if args.repeats <= 0:
        raise RuntimeError("--repeats must be > 0")
    if args.poll_interval_sec <= 0:
        raise RuntimeError("--poll-interval-sec must be > 0")

    available_devices = detect_available_devices(args.devices)
    max_scale = len(available_devices) if available_devices else 4
    scales = args.scales or list(range(1, max_scale + 1))
    if not scales:
        raise RuntimeError("No scales requested")
    if min(scales) <= 0:
        raise RuntimeError("--scales must contain only positive values")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict] = {}
    for scale in scales:
        run_root = args.output_dir / f"n{scale}"
        summary = run_scale(
            mlperf_binary=args.mlperf_binary,
            program_path=args.program_path,
            batch_size=args.batch_size,
            qps=args.qps,
            run_root=run_root,
            scale=scale,
            repeats=args.repeats,
            poll_interval_sec=args.poll_interval_sec,
            devices=available_devices[:scale] if available_devices else None,
        )
        results[f"n{scale}"] = summary

    baseline_key = f"n{min(scales)}"
    baseline = results[baseline_key]["mean_total_samples_per_second"]
    if baseline:
        baseline_scale = results[baseline_key]["n_processes"]
        for summary in results.values():
            total = summary["mean_total_samples_per_second"]
            if total is None:
                summary["speedup_vs_baseline"] = None
                summary["scaling_efficiency_vs_baseline"] = None
            else:
                summary["speedup_vs_baseline"] = total / baseline
                summary["scaling_efficiency_vs_baseline"] = summary["speedup_vs_baseline"] / (
                    summary["n_processes"] / baseline_scale
                )
    else:
        for summary in results.values():
            summary["speedup_vs_baseline"] = None
            summary["scaling_efficiency_vs_baseline"] = None

    output_json = args.output_json or (args.output_dir / "results_summary.json")
    final_summary = {
        "pipeline": "resnet50_parallel_mlperf",
        "program_path": args.program_path.as_posix(),
        "batch_size": args.batch_size,
        "requested_qps_per_process": args.qps,
        "repeats": args.repeats,
        "poll_interval_sec": args.poll_interval_sec,
        "available_devices": available_devices,
        "scales": scales,
        "results": results,
        "note": (
            "Each mlperf process is started with explicit --dev <device> when devices are provided. Observed device assignment is additionally inferred from /proc/<pid>/fd while processes are running."
        ),
    }
    output_json.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(f"Saved parallel MLPerf exploratory summary: {output_json}")


if __name__ == "__main__":
    main()
