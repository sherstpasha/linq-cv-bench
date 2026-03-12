import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from coco_utils import letterbox


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build quantized and compiled IVA H1 artifacts for YOLOv8s")
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "models/detection/yolov8s.onnx")
    parser.add_argument("--calibration-dir", type=Path, default=REPO_ROOT / "data/calibration/MSCOCO2017/val2017")
    parser.add_argument("--artifacts-dir", type=Path, default=REPO_ROOT / "artifacts/detection/yolov8s")
    parser.add_argument("--model-name", type=str, default="yolov8s")
    parser.add_argument("--input-tensor-name", type=str, default=None)
    parser.add_argument("--output-tensor-name", type=str, default=None)
    parser.add_argument("--num-calibration-images", type=int, default=500)
    parser.add_argument("--calibration-chunk-size", type=int, default=128)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--percentile", type=float, default=100.0)
    parser.add_argument("--batch-axis", type=int, default=0)
    parser.add_argument("--save-quantized-graph-pb", action="store_true")
    parser.add_argument("--compile-preset", type=str, default="O1", choices=["O1", "O5", "DEFAULT"])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--metadata-out", type=Path, default=None)
    return parser.parse_args()


def list_images(root: Path) -> List[Path]:
    images: List[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return images


def preprocess_yolo(path: Path, img_size: int) -> np.ndarray:
    img = cv2.imread(path.as_posix())
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img_lb, _, _, _ = letterbox(img, img_size)
    x = img_lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return x.astype(np.float32)


def build_calibration_tensor_memmap(
    calibration_dir: Path,
    num_images: int,
    chunk_size: int,
    img_size: int,
    tmp_dir: Path,
) -> Tuple[np.memmap, Path, int]:
    images = list_images(calibration_dir)
    if not images:
        raise RuntimeError(f"No images found in calibration dir: {calibration_dir}")
    if num_images > 0:
        images = images[: min(num_images, len(images))]
    if chunk_size <= 0:
        raise RuntimeError("--calibration-chunk-size must be > 0")

    sample_count = len(images)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="yolov8_calibration_",
        suffix=".dat",
        dir=tmp_dir.as_posix(),
        delete=False,
    ) as file:
        memmap_path = Path(file.name)

    batch = np.memmap(
        memmap_path.as_posix(),
        dtype=np.float32,
        mode="w+",
        shape=(sample_count, 3, img_size, img_size),
    )

    total_chunks = math.ceil(sample_count / chunk_size)
    for chunk_idx in range(total_chunks):
        start = chunk_idx * chunk_size
        end = min(sample_count, (chunk_idx + 1) * chunk_size)
        tensors = [preprocess_yolo(img_path, img_size=img_size) for img_path in images[start:end]]
        batch[start:end] = np.stack(tensors, axis=0).astype(np.float32)
        batch.flush()

    return batch, memmap_path, sample_count


def force_static_onnx_input_shape(onnx_model: Any, input_name: str, img_size: int) -> None:
    for value_info in onnx_model.graph.input:
        if value_info.name != input_name:
            continue
        dims = value_info.type.tensor_type.shape.dim
        if len(dims) != 4:
            raise RuntimeError(f"Expected 4D input tensor, got {len(dims)} dims for '{input_name}'")
        dims[0].dim_value = 1
        dims[1].dim_value = 3
        dims[2].dim_value = int(img_size)
        dims[3].dim_value = int(img_size)
        return
    raise KeyError(f"Input tensor '{input_name}' not found in ONNX graph")


def load_converted_graph(onnx_model: Any) -> Tuple[Any, Dict[str, str]]:
    try:
        from tpu_framework import onnx_to_tf  # type: ignore
    except Exception:
        from onnx_direct import onnx_to_tf  # type: ignore

    converted = None
    errors: List[str] = []
    for kwargs in (
        {"onnx_model": onnx_model, "try_simplify": True},
        {"onnx_model": onnx_model, "try_simplify": False},
        {"onnx_model": onnx_model},
    ):
        try:
            converted = onnx_to_tf(**kwargs)
            break
        except Exception as error:
            errors.append(f"{kwargs}: {type(error).__name__}: {error}")
    if converted is None:
        raise RuntimeError("Failed to convert ONNX to TF graph: " + " | ".join(errors))

    if isinstance(converted, tuple):
        tf_graph = converted[0]
        mapping = converted[1] if len(converted) > 1 and isinstance(converted[1], dict) else {}
        return tf_graph, mapping
    return converted, {}


def to_graph_def(graph_like: Any) -> Any:
    import tensorflow as tf  # type: ignore

    if isinstance(graph_like, tf.Graph):
        return graph_like.as_graph_def()
    if isinstance(graph_like, tf.compat.v1.GraphDef):
        return graph_like
    raise TypeError(f"Unsupported graph type: {type(graph_like)}")


def map_tensor_name(name: str, mapping: Dict[str, str]) -> str:
    return mapping.get(name, name)


def tensor_name_to_node_name(name: str) -> str:
    return name.split(":", 1)[0]


def unique(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def resolve_preset(name: str) -> Any:
    name = name.upper()
    if name == "DEFAULT":
        from tpu_framework import DEFAULT  # type: ignore

        return DEFAULT
    from tpu_compiler.compiler import O1, O5  # type: ignore

    if name == "O1":
        return O1
    if name == "O5":
        return O5
    raise RuntimeError(f"Unsupported preset: {name}")


def load_export_metadata(model_path: Path) -> Dict[str, Any]:
    metadata_path = model_path.with_suffix(".json")
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not args.calibration_dir.exists():
        raise FileNotFoundError(f"Calibration dir not found: {args.calibration_dir}")

    try:
        import onnx
        from tpu_framework import (  # type: ignore
            Network,
            QuantizedModel,
            RegularModel,
            TPU_128x128_PARAMS,
            TpuProgram,
            compiler,
        )
    except Exception as error:
        raise RuntimeError("Missing dependencies: onnx, tpu_framework, tpu_compiler") from error

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    metadata_out = args.metadata_out or (args.artifacts_dir / "build_summary.json")
    qm_path = args.artifacts_dir / f"{args.model_name}.qm"
    quant_pb = args.artifacts_dir / f"{args.model_name}_quantized.pb"
    export_metadata = load_export_metadata(args.model_path)

    onnx_model = onnx.load(args.model_path.as_posix())
    if not onnx_model.graph.input or not onnx_model.graph.output:
        raise RuntimeError("ONNX model must define at least one input and one output")

    onnx_input_default = onnx_model.graph.input[0].name
    onnx_output_default = onnx_model.graph.output[0].name
    onnx_input_name = args.input_tensor_name or onnx_input_default
    onnx_output_name = args.output_tensor_name or onnx_output_default
    force_static_onnx_input_shape(onnx_model, input_name=onnx_input_name, img_size=args.img_size)

    converted_graph, mapping = load_converted_graph(onnx_model)
    graph_def = to_graph_def(converted_graph)

    mapped_input = map_tensor_name(onnx_input_name, mapping)
    mapped_output = map_tensor_name(onnx_output_name, mapping)

    calib_tensor, memmap_path, calibration_sample_count = build_calibration_tensor_memmap(
        calibration_dir=args.calibration_dir,
        num_images=args.num_calibration_images,
        chunk_size=args.calibration_chunk_size,
        img_size=args.img_size,
        tmp_dir=args.artifacts_dir,
    )
    calibration_dict = {mapped_input: calib_tensor}

    selected_output: Optional[str] = None
    compiled_programs: Dict[str, str] = {}
    try:
        input_shapes = {mapped_input: (1, 3, args.img_size, args.img_size)}
        output_candidates = unique(
            [
                mapped_output,
                tensor_name_to_node_name(mapped_output),
                onnx_output_name,
                f"{onnx_output_name}:0",
            ]
        )

        regular_model = None
        errors: Dict[str, str] = {}
        for output_candidate in output_candidates:
            model_kwargs = {
                "original_graph_def": graph_def,
                "input_shapes": input_shapes,
                "output_nodes": [output_candidate],
            }
            if mapping:
                model_kwargs["anchors_mapping"] = mapping
            try:
                regular_model = RegularModel(**model_kwargs)
                selected_output = output_candidate
                break
            except Exception as error:
                errors[output_candidate] = str(error)

        if regular_model is None:
            raise RuntimeError(f"Could not initialize RegularModel with outputs {output_candidates}. Errors: {errors}")

        try:
            thresholds = regular_model.calibrate(calibration_data=calibration_dict, percentile=args.percentile)
        except TypeError:
            thresholds = regular_model.calibrate(calibration_data=calibration_dict)

        quantized_model = regular_model.quantize(thresholds)
        quantized_model.save(file_dir=args.artifacts_dir.as_posix(), file_name=qm_path.name)

        if args.save_quantized_graph_pb:
            quant_graph = quantized_model.as_graph(batch_size=1, batch_axis=args.batch_axis)
            quant_graph_def = to_graph_def(quant_graph)
            quant_pb.write_bytes(quant_graph_def.SerializeToString())

        preset = resolve_preset(args.compile_preset)
        for batch_size in args.batch_sizes:
            if batch_size <= 0:
                raise RuntimeError(f"Batch size must be > 0, got {batch_size}")
            network, _ = Network.from_quantized_model(quantized_model)
            network.set_batch(batch_size)
            executable, tensor_descriptions = compiler.compile_(
                hardware_parameters=TPU_128x128_PARAMS,
                network=network,
                parameters=preset,
            )
            tpu_program = TpuProgram.from_executable(executable, tensor_descriptions)
            output_path = args.artifacts_dir / f"{args.model_name}_b{batch_size}.tpu"
            tpu_program.to_file(output_path.as_posix())
            compiled_programs[str(batch_size)] = output_path.as_posix()

        summary = {
            "model_name": args.model_name,
            "model_path": args.model_path.as_posix(),
            "calibration_dir": args.calibration_dir.as_posix(),
            "calibration_samples": calibration_sample_count,
            "compile_preset": args.compile_preset,
            "batch_sizes": args.batch_sizes,
            "onnx_input_name": onnx_input_name,
            "onnx_output_name": onnx_output_name,
            "mapped_input_name": mapped_input,
            "mapped_output_name": mapped_output,
            "input_layout": "nchw",
            "runtime_input_value_range": "unit_float",
            "calibration_preprocess": "yolo_letterbox_unit_float",
            "input_shape": [1, 3, args.img_size, args.img_size],
            "selected_output_node": selected_output,
            "model_export_metadata": export_metadata,
            "qm_path": qm_path.as_posix(),
            "compiled_programs": compiled_programs,
            "quantized_graph_pb": quant_pb.as_posix() if args.save_quantized_graph_pb else None,
        }
        metadata_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved build summary: {metadata_out}")
    finally:
        try:
            del calib_tensor
        except Exception:
            pass
        try:
            memmap_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
