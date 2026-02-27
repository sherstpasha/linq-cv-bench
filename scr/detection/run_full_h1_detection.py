import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def resolve_model_path(user_model_path: Path) -> Path:
    candidates = [
        user_model_path,
        REPO_ROOT / "experiments_cuda/detection/yolov5su.onnx",
        REPO_ROOT / "experiments_cpu/detection/yolov5su.onnx",
        REPO_ROOT / "experiments/detection/yolov5su.onnx",
    ]
    for p in candidates:
        if p.exists():
            return p
    return user_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full H1 detection pipeline: quantize -> compile -> TPU infer -> metrics")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))

    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "experiments/detection/yolov5su.onnx")
    parser.add_argument("--weights", type=Path, default=REPO_ROOT / "yolov5su.pt")
    parser.add_argument("--calibration-dir", type=Path, default=REPO_ROOT / "data/calibration/MSCOCO2017/val2017")
    parser.add_argument("--img-dir", type=Path, default=REPO_ROOT / "data/evaluation/MSCOCO2017/val2017")
    parser.add_argument("--ann-file", type=Path, default=REPO_ROOT / "data/evaluation/MSCOCO2017/annotations/instances_val2017.json")

    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "experiments/detection")
    parser.add_argument("--experiment-suffix", type=str, default="h1tpu")
    parser.add_argument("--no-auto-export", action="store_true", help="Do not export ONNX automatically if model is missing")

    parser.add_argument("--input-tensor-name", type=str, default=None)
    parser.add_argument("--output-tensor-name", type=str, default=None)
    parser.add_argument("--num-calibration-images", type=int, default=0)
    parser.add_argument("--calibration-chunk-size", type=int, default=128)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--percentile", type=float, default=100.0)
    parser.add_argument("--batch-axis", type=int, default=0)
    parser.add_argument("--save-quantized-graph-pb", action="store_true")

    parser.add_argument("--compile-preset", choices=["O1", "O5", "DEFAULT"], default="O5")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.65)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup-images", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    py = args.python.as_posix()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.experiment_suffix.strip()
    if not suffix:
        raise RuntimeError("--experiment-suffix must not be empty")
    experiment_tag = f"experiment_{suffix}_b{args.batch_size}"

    qm_path = output_dir / f"yolov5su_{experiment_tag}_quantized.qm"
    tpu_path = output_dir / f"yolov5su_{experiment_tag}.tpu"
    predictions_path = output_dir / f"predictions_{experiment_tag}.json"
    timing_path = output_dir / f"inference_timing_{experiment_tag}.json"
    metrics_path = output_dir / f"metrics_{experiment_tag}.json"
    quantized_pb_path = output_dir / f"yolov5su_{experiment_tag}_quantized.pb"
    run_params_path = output_dir / f"run_params_{experiment_tag}.json"

    model_path = resolve_model_path(args.model_path)
    if not model_path.exists():
        if args.no_auto_export:
            raise FileNotFoundError(
                f"Model not found: {args.model_path}. Also checked experiments_cuda/experiments_cpu/experiments defaults."
            )
        model_path = REPO_ROOT / "experiments/detection/yolov5su.onnx"
        run(
            [
                py,
                (THIS_DIR / "export_yolov5su_to_onnx.py").as_posix(),
                "--weights",
                args.weights.as_posix(),
                "--output",
                model_path.as_posix(),
                "--imgsz",
                str(args.img_size),
            ]
        )
    print(f"Using ONNX model: {model_path}")

    quantize_cmd = [
        py,
        (THIS_DIR / "quantize_yolov5_h1.py").as_posix(),
        "--model-path",
        model_path.as_posix(),
        "--calibration-dir",
        args.calibration_dir.as_posix(),
        "--output-qm",
        qm_path.as_posix(),
        "--num-calibration-images",
        str(args.num_calibration_images),
        "--calibration-chunk-size",
        str(args.calibration_chunk_size),
        "--img-size",
        str(args.img_size),
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
        quantize_cmd += ["--save-quantized-graph-pb", quantized_pb_path.as_posix()]
    run(quantize_cmd)

    compile_cmd = [
        py,
        (THIS_DIR / "compile_yolov5_h1.py").as_posix(),
        "--input-qm",
        qm_path.as_posix(),
        "--output-tpu",
        tpu_path.as_posix(),
        "--batch-size",
        str(args.batch_size),
        "--preset",
        args.compile_preset,
    ]
    run(compile_cmd)

    infer_cmd = [
        py,
        (THIS_DIR / "infer_yolov5_h1_tpu.py").as_posix(),
        "--program-path",
        tpu_path.as_posix(),
        "--img-dir",
        args.img_dir.as_posix(),
        "--ann-file",
        args.ann_file.as_posix(),
        "--predictions-out",
        predictions_path.as_posix(),
        "--timing-out",
        timing_path.as_posix(),
        "--img-size",
        str(args.img_size),
        "--conf-thres",
        str(args.conf_thres),
        "--iou-thres",
        str(args.iou_thres),
        "--max-det",
        str(args.max_det),
        "--warmup-images",
        str(args.warmup_images),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.input_tensor_name:
        infer_cmd += ["--input-tensor-name", args.input_tensor_name]
    if args.output_tensor_name:
        infer_cmd += ["--output-tensor-name", args.output_tensor_name]
    if args.limit > 0:
        infer_cmd += ["--limit", str(args.limit)]
    if args.device:
        infer_cmd += ["--device", args.device]
    run(infer_cmd)

    metrics_cmd = [
        py,
        (THIS_DIR / "metrics.py").as_posix(),
        "--ann-file",
        args.ann_file.as_posix(),
        "--predictions",
        predictions_path.as_posix(),
        "--output-json",
        metrics_path.as_posix(),
    ]
    if args.limit > 0:
        metrics_cmd += ["--limit", str(args.limit)]
    run(metrics_cmd)

    run_params = {
        "pipeline": "detection_h1_full",
        "python": py,
        "experiment_suffix": suffix,
        "experiment_tag": experiment_tag,
        "params": {
            "model_path": args.model_path.as_posix(),
            "resolved_model_path": model_path.as_posix(),
            "weights": args.weights.as_posix(),
            "calibration_dir": args.calibration_dir.as_posix(),
            "img_dir": args.img_dir.as_posix(),
            "ann_file": args.ann_file.as_posix(),
            "output_dir": args.output_dir.as_posix(),
            "input_tensor_name": args.input_tensor_name,
            "output_tensor_name": args.output_tensor_name,
            "num_calibration_images": args.num_calibration_images,
            "calibration_chunk_size": args.calibration_chunk_size,
            "img_size": args.img_size,
            "percentile": args.percentile,
            "batch_axis": args.batch_axis,
            "save_quantized_graph_pb": args.save_quantized_graph_pb,
            "compile_preset": args.compile_preset,
            "batch_size": args.batch_size,
            "device": args.device,
            "conf_thres": args.conf_thres,
            "iou_thres": args.iou_thres,
            "max_det": args.max_det,
            "limit": args.limit,
            "warmup_images": args.warmup_images,
        },
        "artifacts": {
            "qm": qm_path.as_posix(),
            "tpu_program": tpu_path.as_posix(),
            "predictions": predictions_path.as_posix(),
            "timing": timing_path.as_posix(),
            "metrics": metrics_path.as_posix(),
            "quantized_graph_pb": quantized_pb_path.as_posix() if args.save_quantized_graph_pb else None,
        },
        "commands": {
            "quantize": quantize_cmd,
            "compile": compile_cmd,
            "infer": infer_cmd,
            "metrics": metrics_cmd,
        },
    }
    with run_params_path.open("w", encoding="utf-8") as f:
        json.dump(run_params, f, indent=2)

    print("Done.")
    print(f"QM: {qm_path}")
    print(f"TPU program: {tpu_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Timing: {timing_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Run params: {run_params_path}")


if __name__ == "__main__":
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    main()
