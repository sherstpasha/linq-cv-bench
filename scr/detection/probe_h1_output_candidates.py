import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent

DEFAULT_CANDIDATES = [
    "model_24_Concat_3_3/concat:0",
    "model_24_Concat_3_3/transpose:0",
    "model_24_Concat_2_3/concat:0",
    "model_24_Mul_2_2/Mul:0",
]


def run(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    env = os.environ.copy()
    env["YOLO_AUTOINSTALL"] = "False"
    subprocess.run(cmd, check=True, env=env)


def safe_name(tensor_name: str) -> str:
    base = tensor_name.replace(":0", "").replace("/", "_").replace(".", "_")
    short = hashlib.md5(tensor_name.encode("utf-8")).hexdigest()[:8]
    return f"{base[:40]}_{short}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe H1 YOLO output tensor candidates end-to-end")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "experiments_cuda/detection/yolov5su.onnx")
    parser.add_argument("--calibration-dir", type=Path, default=REPO_ROOT / "data/calibration/MSCOCO2017/val2017")
    parser.add_argument("--img-dir", type=Path, default=REPO_ROOT / "data/evaluation/MSCOCO2017/val2017")
    parser.add_argument("--ann-file", type=Path, default=REPO_ROOT / "data/evaluation/MSCOCO2017/annotations/instances_val2017.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "experiments/detection/output_probe")
    parser.add_argument("--candidates", type=str, default=",".join(DEFAULT_CANDIDATES), help="Comma-separated output tensor names")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--compile-preset", choices=["O1", "O5", "DEFAULT"], default="O1")
    parser.add_argument("--percentile", type=float, default=100.0)
    parser.add_argument("--num-calibration-images", type=int, default=256)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--conf-thres", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    py = args.python.as_posix()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = [x.strip() for x in args.candidates.split(",") if x.strip()]
    if not candidates:
        raise RuntimeError("No candidates provided")

    results: List[Dict] = []
    for idx, candidate in enumerate(candidates, start=1):
        tag = f"cand{idx}_{safe_name(candidate)}"
        workdir = args.output_dir / tag
        workdir.mkdir(parents=True, exist_ok=True)

        qm_path = workdir / "model.qm"
        tpu_path = workdir / "model.tpu"
        pred_path = workdir / "predictions.json"
        timing_path = workdir / "timing.json"

        row: Dict = {"candidate": candidate, "tag": tag, "status": "ok"}
        try:
            run(
                [
                    py,
                    (THIS_DIR / "quantize_yolov5_h1.py").as_posix(),
                    "--model-path",
                    args.model_path.as_posix(),
                    "--calibration-dir",
                    args.calibration_dir.as_posix(),
                    "--output-qm",
                    qm_path.as_posix(),
                    "--output-tensor-name",
                    candidate,
                    "--num-calibration-images",
                    str(args.num_calibration_images),
                    "--percentile",
                    str(args.percentile),
                ]
            )
            run(
                [
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
            )
            run(
                [
                    py,
                    (THIS_DIR / "infer_yolov5_h1_tpu.py").as_posix(),
                    "--program-path",
                    tpu_path.as_posix(),
                    "--img-dir",
                    args.img_dir.as_posix(),
                    "--ann-file",
                    args.ann_file.as_posix(),
                    "--predictions-out",
                    pred_path.as_posix(),
                    "--timing-out",
                    timing_path.as_posix(),
                    "--batch-size",
                    str(args.batch_size),
                    "--limit",
                    str(args.limit),
                    "--warmup-images",
                    "0",
                    "--conf-thres",
                    str(args.conf_thres),
                    "--output-tensor-name",
                    candidate,
                ]
            )
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
            row["detections"] = int(timing.get("detections", 0))
            row["throughput_img_per_sec"] = timing.get("throughput_img_per_sec")
            row["timing_file"] = timing_path.as_posix()
            row["predictions_file"] = pred_path.as_posix()
        except Exception as e:
            row["status"] = "failed"
            row["error"] = str(e)

        results.append(row)
        summary = {"results": results}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({"results": results}, indent=2))
    print(f"Saved: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
