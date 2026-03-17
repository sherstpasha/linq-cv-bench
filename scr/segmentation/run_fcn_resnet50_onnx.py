import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
BATCH_SUFFIX_RE = re.compile(r"_b\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FCN-ResNet50 ONNX accuracy and performance on CPU or CUDA"
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "experiments/segmentation/fcn_resnet50.onnx")
    parser.add_argument("--voc-root", type=Path, default=REPO_ROOT / "data/evaluation/VOCdevkit/VOC2012")
    parser.add_argument("--provider", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--accuracy-batch-size", type=int, default=1)
    parser.add_argument("--accuracy-limit", type=int, default=0)
    parser.add_argument("--accuracy-warmup-batches", type=int, default=3)
    parser.add_argument("--performance-samples-b1", type=int, default=500)
    parser.add_argument("--performance-samples-b8", type=int, default=1000)
    parser.add_argument("--performance-warmup-batches", type=int, default=3)
    parser.add_argument("--height", type=int, default=520)
    parser.add_argument("--width", type=int, default=520)
    parser.add_argument("--num-classes", type=int, default=21)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--export-opset", type=int, default=18)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--reexport-model", action="store_true")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=REPO_ROOT / "experiments/segmentation_onnx",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def run(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {"error": f"missing file: {path.as_posix()}"}
    return json.loads(path.read_text(encoding="utf-8"))


def batch_variant_path(model_path: Path, batch_size: int) -> Path:
    stem = model_path.stem
    if BATCH_SUFFIX_RE.search(stem):
        stem = BATCH_SUFFIX_RE.sub(f"_b{batch_size}", stem)
    else:
        stem = f"{stem}_b{batch_size}"
    return model_path.with_name(stem + model_path.suffix)


def main() -> None:
    args = parse_args()
    py = args.python.as_posix()
    args.experiments_dir.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or (args.experiments_dir / "results_summary.json")

    accuracy_summary = args.experiments_dir / "accuracy" / "summary.json"
    performance_b1_summary = args.experiments_dir / "performance" / "b1" / "summary.json"
    performance_b8_summary = args.experiments_dir / "performance" / "b8" / "summary.json"

    model_b1 = batch_variant_path(args.model_path, 1)
    model_b8 = batch_variant_path(args.model_path, 8)
    export_cmds: List[List[str]] = []
    for batch_size, model_variant in ((1, model_b1), (8, model_b8)):
        if args.reexport_model or not model_variant.exists():
            cmd = [
                py,
                (THIS_DIR / "export_fcn_resnet50_to_onnx.py").as_posix(),
                "--output",
                model_variant.as_posix(),
                "--opset",
                str(args.export_opset),
                "--height",
                str(args.height),
                "--width",
                str(args.width),
                "--batch-size",
                str(batch_size),
            ]
            if args.no_pretrained:
                cmd.append("--no-pretrained")
            run(cmd)
            export_cmds.append(cmd)

    accuracy_cmd: Optional[List[str]] = [
        py,
        (THIS_DIR / "run_fcn_resnet50_onnx_accuracy.py").as_posix(),
        "--model-path",
        model_b1.as_posix(),
        "--voc-root",
        args.voc_root.as_posix(),
        "--provider",
        args.provider,
        "--batch-size",
        str(args.accuracy_batch_size),
        "--warmup-batches",
        str(args.accuracy_warmup_batches),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num-classes",
        str(args.num_classes),
        "--ignore-index",
        str(args.ignore_index),
        "--predictions-dir",
        (args.experiments_dir / "accuracy" / "predictions").as_posix(),
        "--summary-out",
        accuracy_summary.as_posix(),
    ]
    if args.accuracy_limit > 0:
        accuracy_cmd += ["--limit", str(args.accuracy_limit)]
    run(accuracy_cmd)

    performance_cmds: List[List[str]] = []
    for batch_size, sample_count, out_path in (
        (1, args.performance_samples_b1, performance_b1_summary),
        (8, args.performance_samples_b8, performance_b8_summary),
    ):
        cmd = [
            py,
            (THIS_DIR / "run_fcn_resnet50_onnx_performance.py").as_posix(),
            "--model-path",
            (model_b1 if batch_size == 1 else model_b8).as_posix(),
            "--voc-root",
            args.voc_root.as_posix(),
            "--provider",
            args.provider,
            "--batch-size",
            str(batch_size),
            "--samples",
            str(sample_count),
            "--warmup-batches",
            str(args.performance_warmup_batches),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--summary-out",
            out_path.as_posix(),
        ]
        run(cmd)
        performance_cmds.append(cmd)

    summary = {
        "pipeline": "fcn_resnet50_onnx",
        "model_path_base": args.model_path.as_posix(),
        "model_path_b1": model_b1.as_posix(),
        "model_path_b8": model_b8.as_posix(),
        "voc_root": args.voc_root.as_posix(),
        "provider": args.provider,
        "experiments_dir": args.experiments_dir.as_posix(),
        "model_export": {
            "commands": export_cmds,
            "executed": bool(export_cmds),
        },
        "accuracy": {
            "command": accuracy_cmd,
            "summary": load_json(accuracy_summary),
        },
        "performance": {
            "commands": performance_cmds,
            "b1": load_json(performance_b1_summary),
            "b8": load_json(performance_b8_summary),
        },
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved ONNX segmentation summary: {output_json}")


if __name__ == "__main__":
    main()
