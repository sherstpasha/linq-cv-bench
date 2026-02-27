import argparse
import json
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
        REPO_ROOT / "experiments_cuda/detection_model/retinanet_resnet50_fpn.onnx",
        REPO_ROOT / "experiments_cpu/detection_model/retinanet_resnet50_fpn.onnx",
        REPO_ROOT / "experiments/detection_model/retinanet_resnet50_fpn.onnx",
    ]
    for p in candidates:
        if p.exists():
            return p
    return user_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full H1 RetinaNet pipeline: export/quantize/compile/infer/metrics")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))

    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "experiments/detection_model/retinanet_resnet50_fpn.onnx")
    parser.add_argument("--calibration-dir", type=Path, default=REPO_ROOT / "data/calibration/MSCOCO2017/val2017")
    parser.add_argument("--img-dir", type=Path, default=REPO_ROOT / "data/evaluation/MSCOCO2017/val2017")
    parser.add_argument("--ann-file", type=Path, default=REPO_ROOT / "data/evaluation/MSCOCO2017/annotations/instances_val2017.json")

    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "experiments/detection_model")
    parser.add_argument("--experiment-suffix", type=str, default="h1tpu")
    parser.add_argument("--no-auto-export", action="store_true")

    parser.add_argument("--input-tensor-name", type=str, default=None)
    parser.add_argument("--output-tensor-name", type=str, default=None)
    parser.add_argument("--num-calibration-images", type=int, default=0)
    parser.add_argument("--calibration-chunk-size", type=int, default=64)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--percentile", type=float, default=100.0)
    parser.add_argument("--batch-axis", type=int, default=0)
    parser.add_argument("--save-quantized-graph-pb", action="store_true")

    parser.add_argument("--compile-preset", choices=["O1", "O5", "DEFAULT"], default="O1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup-images", type=int, default=10)
    parser.add_argument("--max-det", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    py = args.python.as_posix()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.experiment_suffix.strip()
    if not suffix:
        raise RuntimeError("--experiment-suffix must not be empty")
    experiment_tag = f"experiment_{suffix}_b{args.batch_size}"

    model_path = resolve_model_path(args.model_path)

    onnx_path = args.output_dir / f"retinanet_resnet50_fpn_{experiment_tag}.onnx"
    qm_path = args.output_dir / f"retinanet_resnet50_fpn_{experiment_tag}_quantized.qm"
    tpu_path = args.output_dir / f"retinanet_resnet50_fpn_{experiment_tag}.tpu"
    preds_onnx = args.output_dir / f"predictions_{experiment_tag}_onnx.json"
    timing_onnx = args.output_dir / f"inference_timing_{experiment_tag}_onnx.json"
    metrics_onnx = args.output_dir / f"metrics_{experiment_tag}_onnx.json"
    preds_h1 = args.output_dir / f"predictions_{experiment_tag}_h1tpu.json"
    timing_h1 = args.output_dir / f"inference_timing_{experiment_tag}_h1tpu.json"
    metrics_h1 = args.output_dir / f"metrics_{experiment_tag}_h1tpu.json"
    quant_pb = args.output_dir / f"retinanet_resnet50_fpn_{experiment_tag}_quantized.pb"
    run_params = args.output_dir / f"run_params_{experiment_tag}.json"

    if not model_path.exists():
        if args.no_auto_export:
            raise FileNotFoundError(f"Model not found: {args.model_path}")
        run(
            [
                py,
                (THIS_DIR / "export_retinanet_to_onnx.py").as_posix(),
                "--output",
                onnx_path.as_posix(),
                "--height",
                str(args.height),
                "--width",
                str(args.width),
                "--batch-size",
                "1",
                "--max-det",
                str(args.max_det),
            ]
        )
        model_path = onnx_path
    print(f"Using ONNX model: {model_path}")

    onnx_infer_cmd = [
        py,
        (THIS_DIR / "infer_retinanet_onnx.py").as_posix(),
        "--model-path",
        model_path.as_posix(),
        "--img-dir",
        args.img_dir.as_posix(),
        "--ann-file",
        args.ann_file.as_posix(),
        "--predictions-out",
        preds_onnx.as_posix(),
        "--timing-out",
        timing_onnx.as_posix(),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--batch-size",
        str(args.batch_size),
        "--conf-thres",
        str(args.conf_thres),
        "--warmup-images",
        str(args.warmup_images),
    ]
    if args.limit > 0:
        onnx_infer_cmd += ["--limit", str(args.limit)]
    run(onnx_infer_cmd)

    onnx_metrics_cmd = [
        py,
        (REPO_ROOT / "scr/detection/metrics.py").as_posix(),
        "--ann-file",
        args.ann_file.as_posix(),
        "--predictions",
        preds_onnx.as_posix(),
        "--output-json",
        metrics_onnx.as_posix(),
    ]
    if args.limit > 0:
        onnx_metrics_cmd += ["--limit", str(args.limit)]
    run(onnx_metrics_cmd)

    quant_cmd = [
        py,
        (THIS_DIR / "quantize_retinanet_h1.py").as_posix(),
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
        quant_cmd += ["--input-tensor-name", args.input_tensor_name]
    if args.output_tensor_name:
        quant_cmd += ["--output-tensor-name", args.output_tensor_name]
    if args.save_quantized_graph_pb:
        quant_cmd += ["--save-quantized-graph-pb", quant_pb.as_posix()]
    run(quant_cmd)

    compile_cmd = [
        py,
        (THIS_DIR / "compile_retinanet_h1.py").as_posix(),
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

    h1_infer_cmd = [
        py,
        (THIS_DIR / "infer_retinanet_h1_tpu.py").as_posix(),
        "--program-path",
        tpu_path.as_posix(),
        "--img-dir",
        args.img_dir.as_posix(),
        "--ann-file",
        args.ann_file.as_posix(),
        "--predictions-out",
        preds_h1.as_posix(),
        "--timing-out",
        timing_h1.as_posix(),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--batch-size",
        str(args.batch_size),
        "--conf-thres",
        str(args.conf_thres),
        "--warmup-images",
        str(args.warmup_images),
    ]
    if args.input_tensor_name:
        h1_infer_cmd += ["--input-tensor-name", args.input_tensor_name]
    if args.output_tensor_name:
        h1_infer_cmd += ["--output-tensor-name", args.output_tensor_name]
    if args.limit > 0:
        h1_infer_cmd += ["--limit", str(args.limit)]
    if args.device:
        h1_infer_cmd += ["--device", args.device]
    run(h1_infer_cmd)

    h1_metrics_cmd = [
        py,
        (REPO_ROOT / "scr/detection/metrics.py").as_posix(),
        "--ann-file",
        args.ann_file.as_posix(),
        "--predictions",
        preds_h1.as_posix(),
        "--output-json",
        metrics_h1.as_posix(),
    ]
    if args.limit > 0:
        h1_metrics_cmd += ["--limit", str(args.limit)]
    run(h1_metrics_cmd)

    report = {
        "pipeline": "detection_model_retinanet_h1_full",
        "experiment_tag": experiment_tag,
        "params": {
            "model_path": args.model_path.as_posix(),
            "resolved_model_path": model_path.as_posix(),
            "calibration_dir": args.calibration_dir.as_posix(),
            "img_dir": args.img_dir.as_posix(),
            "ann_file": args.ann_file.as_posix(),
            "batch_size": args.batch_size,
            "height": args.height,
            "width": args.width,
            "compile_preset": args.compile_preset,
            "num_calibration_images": args.num_calibration_images,
            "calibration_chunk_size": args.calibration_chunk_size,
            "percentile": args.percentile,
            "conf_thres": args.conf_thres,
            "limit": args.limit,
            "warmup_images": args.warmup_images,
            "max_det": args.max_det,
            "device": args.device,
        },
        "artifacts": {
            "onnx_model": model_path.as_posix(),
            "onnx_predictions": preds_onnx.as_posix(),
            "onnx_timing": timing_onnx.as_posix(),
            "onnx_metrics": metrics_onnx.as_posix(),
            "qm": qm_path.as_posix(),
            "tpu_program": tpu_path.as_posix(),
            "h1_predictions": preds_h1.as_posix(),
            "h1_timing": timing_h1.as_posix(),
            "h1_metrics": metrics_h1.as_posix(),
            "quantized_graph_pb": quant_pb.as_posix() if args.save_quantized_graph_pb else None,
        },
        "commands": {
            "onnx_infer": onnx_infer_cmd,
            "onnx_metrics": onnx_metrics_cmd,
            "quantize": quant_cmd,
            "compile": compile_cmd,
            "h1_infer": h1_infer_cmd,
            "h1_metrics": h1_metrics_cmd,
        },
    }

    with run_params.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Done.")
    print(f"ONNX metrics: {metrics_onnx}")
    print(f"H1 metrics: {metrics_h1}")
    print(f"Run params: {run_params}")


if __name__ == "__main__":
    main()
