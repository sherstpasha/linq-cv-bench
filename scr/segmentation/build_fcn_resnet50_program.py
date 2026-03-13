import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize and compile FCN-ResNet50 ONNX into TPU program(s)"
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "experiments/segmentation/fcn_resnet50.onnx",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=REPO_ROOT / "data/calibration/VOCdevkit/VOC2012/JPEGImages",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/segmentation",
    )
    parser.add_argument("--model-name", type=str, default="fcn_resnet50")
    parser.add_argument("--height", type=int, default=520)
    parser.add_argument("--width", type=int, default=520)
    parser.add_argument("--percentile", type=float, default=100.0)
    parser.add_argument("--batch-axis", type=int, default=0)
    parser.add_argument("--num-calibration-images", type=int, default=0)
    parser.add_argument("--calibration-chunk-size", type=int, default=64)
    parser.add_argument(
        "--compile-preset",
        type=str,
        default="O1",
        choices=["O1", "O5", "DEFAULT"],
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--input-tensor-name", type=str, default=None)
    parser.add_argument("--output-tensor-name", type=str, default=None)
    parser.add_argument("--save-quantized-graph-pb", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def run(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    py = args.python.as_posix()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    qm_path = args.artifacts_dir / f"{args.model_name}.qm"
    quantized_graph_pb = args.artifacts_dir / f"{args.model_name}_quantized.pb"
    output_json = args.output_json or (args.artifacts_dir / "build_summary.json")

    quantize_cmd = [
        py,
        (THIS_DIR / "quantize_fcn_resnet50_h1.py").as_posix(),
        "--model-path",
        args.model_path.as_posix(),
        "--calibration-dir",
        args.calibration_dir.as_posix(),
        "--output-qm",
        qm_path.as_posix(),
        "--num-calibration-images",
        str(args.num_calibration_images),
        "--calibration-chunk-size",
        str(args.calibration_chunk_size),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--percentile",
        str(args.percentile),
        "--batch-axis",
        str(args.batch_axis),
    ]
    if args.input_tensor_name:
        quantize_cmd += ["--input-tensor-name", args.input_tensor_name]
    if args.output_tensor_name:
        quantize_cmd += ["--output-tensor-name", args.output_tensor_name]
    if args.save_quantized_graph_pb:
        quantize_cmd += ["--save-quantized-graph-pb", quantized_graph_pb.as_posix()]
    run(quantize_cmd)

    compiled_programs: Dict[str, str] = {}
    compile_cmds: List[List[str]] = []
    for batch_size in args.batch_sizes:
        output_tpu = args.artifacts_dir / f"{args.model_name}_b{batch_size}.tpu"
        compile_cmd = [
            py,
            (THIS_DIR / "compile_fcn_resnet50_h1.py").as_posix(),
            "--input-qm",
            qm_path.as_posix(),
            "--output-tpu",
            output_tpu.as_posix(),
            "--batch-size",
            str(batch_size),
            "--preset",
            args.compile_preset,
        ]
        run(compile_cmd)
        compile_cmds.append(compile_cmd)
        compiled_programs[str(batch_size)] = output_tpu.as_posix()

    summary = {
        "model_name": args.model_name,
        "model_path": args.model_path.as_posix(),
        "calibration_dir": args.calibration_dir.as_posix(),
        "num_calibration_images": args.num_calibration_images,
        "calibration_chunk_size": args.calibration_chunk_size,
        "height": args.height,
        "width": args.width,
        "percentile": args.percentile,
        "batch_axis": args.batch_axis,
        "compile_preset": args.compile_preset,
        "batch_sizes": args.batch_sizes,
        "input_tensor_name": args.input_tensor_name,
        "output_tensor_name": args.output_tensor_name,
        "qm_path": qm_path.as_posix(),
        "compiled_programs": compiled_programs,
        "quantized_graph_pb": quantized_graph_pb.as_posix() if args.save_quantized_graph_pb else None,
        "commands": {
            "quantize": quantize_cmd,
            "compile": compile_cmds,
        },
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved build summary: {output_json}")


if __name__ == "__main__":
    main()
