import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build quantized and compiled IVA H1 artifacts for ResNet-50"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "models/classification/resnet50.onnx",
        help="Path to source ONNX model",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=REPO_ROOT / "data/calibration/imagenet",
        help="Calibration image directory",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/classification",
        help="Directory for .qm, .tpu and metadata outputs",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet50",
        help="Artifact name prefix",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8],
        help="Batch sizes to compile",
    )
    parser.add_argument("--input-tensor-name", type=str, default=None)
    parser.add_argument("--output-tensor-name", type=str, default=None)
    parser.add_argument(
        "--input-layout",
        type=str,
        default="auto",
        choices=["auto", "nchw", "nhwc"],
        help="Input tensor layout for calibration and compile; auto uses converted TF graph shape",
    )
    parser.add_argument(
        "--input-value-range",
        type=str,
        default="auto",
        choices=["auto", "normalized", "unit_float", "uint8"],
        help="Calibration value range; auto uses export metadata next to the ONNX model if available",
    )
    parser.add_argument("--num-calibration-images", type=int, default=0)
    parser.add_argument("--calibration-chunk-size", type=int, default=256)
    parser.add_argument("--percentile", type=float, default=100.0)
    parser.add_argument("--batch-axis", type=int, default=0)
    parser.add_argument(
        "--compile-preset",
        type=str,
        default="O1",
        choices=["O1", "O5", "DEFAULT"],
    )
    parser.add_argument("--save-quantized-graph-pb", action="store_true")
    parser.add_argument("--metadata-out", type=Path, default=None)
    return parser.parse_args()


def list_images(root: Path) -> List[Path]:
    images: List[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return images


def load_model_export_metadata(model_path: Path) -> Dict[str, Any]:
    metadata_path = model_path.with_suffix(".json")
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def preprocess_resnet50(image: Image.Image, input_layout: str, input_value_range: str) -> np.ndarray:
    image = image.convert("RGB")
    width, height = image.size
    scale = 256.0 / min(width, height)
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    image = image.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))

    arr = np.asarray(image, dtype=np.float32)
    if input_value_range == "normalized":
        arr = arr / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    elif input_value_range == "unit_float":
        arr = arr / 255.0
    elif input_value_range != "uint8":
        raise RuntimeError(f"Unsupported input value range: {input_value_range}")
    if input_layout == "nchw":
        arr = np.transpose(arr, (2, 0, 1))
    elif input_layout != "nhwc":
        raise RuntimeError(f"Unsupported input layout: {input_layout}")
    return arr.astype(np.float32)


def build_calibration_tensor_memmap(
    calibration_dir: Path,
    num_images: int,
    chunk_size: int,
    tmp_dir: Path,
    input_layout: str,
    input_value_range: str,
    input_height: int,
    input_width: int,
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
        prefix="calibration_",
        suffix=".dat",
        dir=tmp_dir.as_posix(),
        delete=False,
    ) as temp_file:
        memmap_path = Path(temp_file.name)

    calibration_batch = np.memmap(
        memmap_path.as_posix(),
        dtype=np.float32,
        mode="w+",
        shape=(sample_count, 3, input_height, input_width)
        if input_layout == "nchw"
        else (sample_count, input_height, input_width, 3),
    )

    total_chunks = math.ceil(sample_count / chunk_size)
    for chunk_idx in range(total_chunks):
        start = chunk_idx * chunk_size
        end = min(sample_count, (chunk_idx + 1) * chunk_size)
        tensors: List[np.ndarray] = []
        for image_path in images[start:end]:
            with Image.open(image_path) as image:
                tensors.append(
                    preprocess_resnet50(
                        image,
                        input_layout=input_layout,
                        input_value_range=input_value_range,
                    )
                )
        calibration_batch[start:end] = np.stack(tensors, axis=0).astype(np.float32)
        calibration_batch.flush()

    return calibration_batch, memmap_path, sample_count


def load_converted_graph(onnx_model: Any) -> Tuple[Any, Dict[str, str]]:
    try:
        from tpu_framework import onnx_to_tf  # type: ignore
    except Exception:
        from onnx_direct import onnx_to_tf  # type: ignore

    try:
        converted = onnx_to_tf(onnx_model=onnx_model, try_simplify=True)
    except TypeError:
        converted = onnx_to_tf(onnx_model)

    if isinstance(converted, tuple):
        mapping = converted[1] if len(converted) > 1 and isinstance(converted[1], dict) else {}
        return converted[0], mapping
    return converted, {}


def to_graph_def(graph_like: Any) -> Any:
    import tensorflow as tf  # type: ignore

    if isinstance(graph_like, tf.Graph):
        return graph_like.as_graph_def()
    if isinstance(graph_like, tf.compat.v1.GraphDef):
        return graph_like
    raise TypeError(f"Unsupported graph type: {type(graph_like)}")


def infer_input_shape(graph_like: Any, tensor_name: str) -> Optional[Tuple[int, int, int, int]]:
    import tensorflow as tf  # type: ignore

    if isinstance(graph_like, tf.Graph):
        try:
            tensor = graph_like.get_tensor_by_name(tensor_name)
        except KeyError:
            return None
        shape = tensor.shape.as_list()
        if len(shape) != 4:
            return None
        resolved = [1 if dim is None else int(dim) for dim in shape]
        return tuple(resolved)  # type: ignore[return-value]
    return None


def resolve_input_layout(
    requested_layout: str,
    inferred_shape: Optional[Tuple[int, int, int, int]],
) -> Tuple[str, Tuple[int, int, int, int]]:
    if inferred_shape is None:
        if requested_layout == "auto":
            return "nchw", (1, 3, 224, 224)
        if requested_layout == "nchw":
            return "nchw", (1, 3, 224, 224)
        return "nhwc", (1, 224, 224, 3)

    if requested_layout == "auto":
        if inferred_shape[1] == 3:
            return "nchw", inferred_shape
        if inferred_shape[3] == 3:
            return "nhwc", inferred_shape
        raise RuntimeError(f"Could not infer input layout from shape {inferred_shape}")

    if requested_layout == "nchw" and inferred_shape[1] != 3:
        raise RuntimeError(f"Requested NCHW, but converted graph shape is {inferred_shape}")
    if requested_layout == "nhwc" and inferred_shape[3] != 3:
        raise RuntimeError(f"Requested NHWC, but converted graph shape is {inferred_shape}")
    return requested_layout, inferred_shape


def resolve_input_value_range(requested: str, export_metadata: Dict[str, Any]) -> str:
    if requested != "auto":
        return requested
    if export_metadata.get("input_value_range") in {"normalized", "unit_float", "uint8"}:
        return str(export_metadata["input_value_range"])
    return "normalized"


def map_tensor_name(name: str, mapping: Dict[str, str]) -> str:
    return mapping.get(name, name)


def tensor_name_to_node_name(name: str) -> str:
    return name.split(":", 1)[0]


def unique(items: List[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


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


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not args.calibration_dir.exists():
        raise FileNotFoundError(f"Calibration dir not found: {args.calibration_dir}")
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise RuntimeError("--batch-sizes must contain only positive values")

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
    export_metadata = load_model_export_metadata(args.model_path)
    qm_path = args.artifacts_dir / f"{args.model_name}.qm"
    quantized_graph_path = args.artifacts_dir / f"{args.model_name}_quantized.pb"

    print(f"Loading ONNX model: {args.model_path}")
    onnx_model = onnx.load(args.model_path.as_posix())
    if not onnx_model.graph.input or not onnx_model.graph.output:
        raise RuntimeError("ONNX model must define at least one input and one output")

    onnx_input_name = args.input_tensor_name or onnx_model.graph.input[0].name
    onnx_output_name = args.output_tensor_name or onnx_model.graph.output[0].name

    converted_graph, mapping = load_converted_graph(onnx_model)
    graph_def = to_graph_def(converted_graph)

    mapped_input_name = map_tensor_name(onnx_input_name, mapping)
    mapped_output_name = map_tensor_name(onnx_output_name, mapping)
    inferred_input_shape = infer_input_shape(converted_graph, mapped_input_name)
    input_layout, input_shape = resolve_input_layout(args.input_layout, inferred_input_shape)
    input_value_range = resolve_input_value_range(args.input_value_range, export_metadata)
    if input_layout == "nchw":
        _, _, input_height, input_width = input_shape
    else:
        _, input_height, input_width, _ = input_shape
    output_candidates = unique(
        [
            mapped_output_name,
            tensor_name_to_node_name(mapped_output_name),
            onnx_output_name,
            f"{onnx_output_name}:0",
        ]
    )

    calibration_tensor, memmap_path, calibration_sample_count = build_calibration_tensor_memmap(
        calibration_dir=args.calibration_dir,
        num_images=args.num_calibration_images,
        chunk_size=args.calibration_chunk_size,
        tmp_dir=args.artifacts_dir,
        input_layout=input_layout,
        input_value_range=input_value_range,
        input_height=input_height,
        input_width=input_width,
    )

    try:
        regular_model = None
        selected_output: Optional[str] = None
        last_error: Optional[Exception] = None

        for output_candidate in output_candidates:
            model_kwargs = {
                "original_graph_def": graph_def,
                "input_shapes": {mapped_input_name: input_shape},
                "output_nodes": [output_candidate],
            }
            if mapping:
                model_kwargs["anchors_mapping"] = mapping
            try:
                regular_model = RegularModel(**model_kwargs)
                selected_output = output_candidate
                break
            except Exception as error:
                last_error = error

        if regular_model is None:
            raise RuntimeError(
                f"Could not initialize RegularModel with output candidates: {output_candidates}"
            ) from last_error

        calibration_data = {mapped_input_name: calibration_tensor}
        try:
            thresholds = regular_model.calibrate(
                calibration_data=calibration_data,
                percentile=args.percentile,
            )
        except TypeError:
            thresholds = regular_model.calibrate(calibration_data=calibration_data)

        quantized_model = regular_model.quantize(thresholds)
        quantized_model.save(file_dir=args.artifacts_dir.as_posix(), file_name=qm_path.name)

        if args.save_quantized_graph_pb:
            quant_graph = quantized_model.as_graph(batch_size=1, batch_axis=args.batch_axis)
            quant_graph_def = to_graph_def(quant_graph)
            with quantized_graph_path.open("wb") as file:
                file.write(quant_graph_def.SerializeToString())

        compiled_programs: Dict[str, str] = {}
        preset = resolve_preset(args.compile_preset)
        loaded_quantized_model = QuantizedModel.load(qm_path.as_posix())
        for batch_size in sorted(set(args.batch_sizes)):
            network, _ = Network.from_quantized_model(loaded_quantized_model)
            network.set_batch(batch_size)
            executable, tensor_descriptions = compiler.compile_(
                hardware_parameters=TPU_128x128_PARAMS,
                network=network,
                parameters=preset,
            )
            tpu_program = TpuProgram.from_executable(executable, tensor_descriptions)
            program_path = args.artifacts_dir / f"{args.model_name}_b{batch_size}.tpu"
            tpu_program.to_file(program_path.as_posix())
            compiled_programs[str(batch_size)] = program_path.as_posix()

        summary = {
            "model_name": args.model_name,
            "model_path": args.model_path.as_posix(),
            "calibration_dir": args.calibration_dir.as_posix(),
            "calibration_samples": calibration_sample_count,
            "compile_preset": args.compile_preset,
            "batch_sizes": sorted(set(args.batch_sizes)),
            "onnx_input_name": onnx_input_name,
            "onnx_output_name": onnx_output_name,
            "mapped_input_name": mapped_input_name,
            "input_layout": input_layout,
            "input_value_range": input_value_range,
            "input_shape": list(input_shape),
            "selected_output_node": selected_output,
            "model_export_metadata": export_metadata or None,
            "qm_path": qm_path.as_posix(),
            "compiled_programs": compiled_programs,
            "quantized_graph_pb": quantized_graph_path.as_posix() if args.save_quantized_graph_pb else None,
        }
        metadata_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved build summary: {metadata_out}")
    finally:
        try:
            memmap_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
