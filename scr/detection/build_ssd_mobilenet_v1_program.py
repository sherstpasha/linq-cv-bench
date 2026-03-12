import argparse
import copy
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_INPUT_NAME = "image_tensor:0"
DEFAULT_OUTPUT_NAMES = [
    "detection_boxes:0",
    "detection_classes:0",
    "detection_scores:0",
    "num_detections:0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build quantized and compiled IVA H1 artifacts for SSD-MobileNetV1"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "models/detection/ssd_mobilenet_v1_10.onnx",
        help="Path to source ONNX model",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=REPO_ROOT / "data/calibration/MSCOCO2017/val2017",
        help="Calibration image directory",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/detection/ssd_mobilenet_v1",
        help="Directory for .qm, .tpu and metadata outputs",
    )
    parser.add_argument("--model-name", type=str, default="ssd_mobilenet_v1")
    parser.add_argument("--image-size", type=int, default=300)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--num-calibration-images", type=int, default=500)
    parser.add_argument("--calibration-chunk-size", type=int, default=128)
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


def load_model_download_metadata(model_path: Path) -> Dict[str, Any]:
    metadata_path = model_path.with_suffix(".json")
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def onnx_elem_type_to_dtype(onnx_module: Any, elem_type: int) -> np.dtype:
    if elem_type == onnx_module.TensorProto.UINT8:
        return np.dtype(np.uint8)
    if elem_type == onnx_module.TensorProto.FLOAT:
        return np.dtype(np.float32)
    raise RuntimeError(f"Unsupported ONNX input element type: {elem_type}")


def ensure_static_input_model(
    onnx_module: Any,
    source_path: Path,
    static_path: Path,
    image_size: int,
) -> Tuple[Any, str, List[str], np.dtype, Tuple[int, int, int, int], bool]:
    model = onnx_module.load(source_path.as_posix())
    if not model.graph.input or not model.graph.output:
        raise RuntimeError("ONNX model must define at least one input and one output")

    input_value_info = model.graph.input[0]
    input_name = input_value_info.name
    output_names = [output.name for output in model.graph.output]

    elem_type = input_value_info.type.tensor_type.elem_type
    input_dtype = onnx_elem_type_to_dtype(onnx_module, elem_type)

    desired_shape = [1, image_size, image_size, 3]
    dims = input_value_info.type.tensor_type.shape.dim
    current_shape: List[Optional[int]] = []
    for dim in dims:
        if dim.HasField("dim_value"):
            current_shape.append(int(dim.dim_value))
        else:
            current_shape.append(None)

    needs_static_copy = current_shape != desired_shape
    static_model = copy.deepcopy(model)
    static_dims = static_model.graph.input[0].type.tensor_type.shape.dim
    for dim, value in zip(static_dims, desired_shape):
        dim.ClearField("dim_param")
        dim.dim_value = value

    static_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_module.save(static_model, static_path.as_posix())

    return (
        static_model,
        input_name,
        output_names,
        input_dtype,
        tuple(desired_shape),  # type: ignore[return-value]
        needs_static_copy,
    )


def preprocess_for_calibration(
    image: Image.Image,
    image_size: int,
    input_layout: str,
    input_dtype: np.dtype,
) -> np.ndarray:
    image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.uint8)
    if input_dtype == np.dtype(np.float32):
        arr = arr.astype(np.float32)
    elif input_dtype != np.dtype(np.uint8):
        raise RuntimeError(f"Unsupported calibration dtype: {input_dtype}")

    if input_layout == "nchw":
        arr = np.transpose(arr, (2, 0, 1))
    elif input_layout != "nhwc":
        raise RuntimeError(f"Unsupported input layout: {input_layout}")
    return arr


def build_calibration_tensor_memmap(
    calibration_dir: Path,
    num_images: int,
    chunk_size: int,
    tmp_dir: Path,
    input_layout: str,
    input_dtype: np.dtype,
    input_shape: Tuple[int, int, int, int],
    image_size: int,
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
        prefix="ssd_calibration_",
        suffix=".dat",
        dir=tmp_dir.as_posix(),
        delete=False,
    ) as temp_file:
        memmap_path = Path(temp_file.name)

    if input_layout == "nchw":
        shape = (sample_count, 3, input_shape[2], input_shape[3])
    else:
        shape = (sample_count, input_shape[1], input_shape[2], input_shape[3])

    calibration_batch = np.memmap(
        memmap_path.as_posix(),
        dtype=input_dtype,
        mode="w+",
        shape=shape,
    )

    total_chunks = math.ceil(sample_count / chunk_size)
    for chunk_index in range(total_chunks):
        start = chunk_index * chunk_size
        end = min(sample_count, (chunk_index + 1) * chunk_size)
        tensors: List[np.ndarray] = []
        for image_path in images[start:end]:
            with Image.open(image_path) as image:
                tensors.append(
                    preprocess_for_calibration(
                        image=image,
                        image_size=image_size,
                        input_layout=input_layout,
                        input_dtype=input_dtype,
                    )
                )
        calibration_batch[start:end] = np.stack(tensors, axis=0).astype(input_dtype)
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


def resolve_input_layout(inferred_shape: Optional[Tuple[int, int, int, int]]) -> Tuple[str, Tuple[int, int, int, int]]:
    if inferred_shape is None:
        return "nhwc", (1, 300, 300, 3)
    if inferred_shape[3] == 3:
        return "nhwc", inferred_shape
    if inferred_shape[1] == 3:
        return "nchw", inferred_shape
    raise RuntimeError(f"Could not infer input layout from shape {inferred_shape}")


def map_tensor_name(name: str, mapping: Dict[str, str]) -> str:
    return mapping.get(name, name)


def tensor_name_to_node_name(name: str) -> str:
    return name.split(":", 1)[0]


def unique_output_groups(groups: Sequence[Sequence[str]]) -> List[List[str]]:
    result: List[List[str]] = []
    seen = set()
    for group in groups:
        normalized = tuple(item for item in group if item)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(list(normalized))
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
    if args.image_size <= 0:
        raise RuntimeError("--image-size must be > 0")
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
    download_metadata = load_model_download_metadata(args.model_path)
    static_model_path = args.artifacts_dir / f"{args.model_name}_static.onnx"
    qm_path = args.artifacts_dir / f"{args.model_name}.qm"
    quantized_graph_path = args.artifacts_dir / f"{args.model_name}_quantized.pb"

    (
        static_onnx_model,
        onnx_input_name,
        onnx_output_names,
        input_dtype,
        static_input_shape,
        used_static_copy,
    ) = ensure_static_input_model(
        onnx_module=onnx,
        source_path=args.model_path,
        static_path=static_model_path,
        image_size=args.image_size,
    )

    converted_graph, mapping = load_converted_graph(static_onnx_model)
    graph_def = to_graph_def(converted_graph)

    mapped_input_name = map_tensor_name(onnx_input_name, mapping)
    inferred_input_shape = infer_input_shape(converted_graph, mapped_input_name)
    input_layout, input_shape = resolve_input_layout(inferred_input_shape or static_input_shape)
    mapped_output_names = [map_tensor_name(name, mapping) for name in onnx_output_names]

    calibration_tensor, memmap_path, calibration_sample_count = build_calibration_tensor_memmap(
        calibration_dir=args.calibration_dir,
        num_images=args.num_calibration_images,
        chunk_size=args.calibration_chunk_size,
        tmp_dir=args.artifacts_dir,
        input_layout=input_layout,
        input_dtype=input_dtype,
        input_shape=input_shape,
        image_size=args.image_size,
    )

    try:
        regular_model = None
        selected_output_nodes: Optional[List[str]] = None
        last_error: Optional[Exception] = None

        output_groups = unique_output_groups(
            [
                mapped_output_names,
                [tensor_name_to_node_name(name) for name in mapped_output_names],
                [name if name.endswith(":0") else f"{name}:0" for name in mapped_output_names],
                [tensor_name_to_node_name(name if name.endswith(":0") else f"{name}:0") for name in mapped_output_names],
            ]
        )

        for output_group in output_groups:
            model_kwargs = {
                "original_graph_def": graph_def,
                "input_shapes": {mapped_input_name: input_shape},
                "output_nodes": output_group,
            }
            if mapping:
                model_kwargs["anchors_mapping"] = mapping
            try:
                regular_model = RegularModel(**model_kwargs)
                selected_output_nodes = output_group
                break
            except Exception as error:
                last_error = error

        if regular_model is None:
            raise RuntimeError(
                f"Could not initialize RegularModel with output groups: {output_groups}"
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
            "source_model_path": args.model_path.as_posix(),
            "static_model_path": static_model_path.as_posix(),
            "used_static_copy": used_static_copy,
            "model_download_metadata": download_metadata or None,
            "calibration_dir": args.calibration_dir.as_posix(),
            "calibration_samples": calibration_sample_count,
            "image_size": args.image_size,
            "compile_preset": args.compile_preset,
            "batch_sizes": sorted(set(args.batch_sizes)),
            "onnx_input_name": onnx_input_name,
            "onnx_output_names": onnx_output_names,
            "mapped_input_name": mapped_input_name,
            "mapped_output_names": mapped_output_names,
            "input_layout": input_layout,
            "input_dtype": str(input_dtype),
            "input_shape": list(input_shape),
            "selected_output_nodes": selected_output_nodes,
            "calibration_preprocess": "resize_uint8",
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
