import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show TensorFlow graph details after ONNX -> onnx_direct conversion")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "experiments/classification/resnet50.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument("--max-ops", type=int, default=200, help="Max operations to print")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save graph summary JSON")
    return parser.parse_args()


def as_graph_def(converted_model: Any, tf_module: Any) -> Any:
    if isinstance(converted_model, tuple):
        converted_model = converted_model[0]
    if isinstance(converted_model, tf_module.Graph):
        return converted_model.as_graph_def()
    if isinstance(converted_model, tf_module.compat.v1.GraphDef):
        return converted_model
    raise TypeError(f"Unsupported converted model type: {type(converted_model)}")


def tensor_shape(t: Any) -> List[Any]:
    try:
        shape = t.shape.as_list()
        return [int(x) if isinstance(x, int) else x for x in shape]
    except Exception:
        return []


def build_summary(graph: Any) -> Dict[str, Any]:
    placeholders = []
    for op in graph.get_operations():
        if op.type == "Placeholder":
            for out in op.outputs:
                placeholders.append({"name": out.name, "dtype": out.dtype.name, "shape": tensor_shape(out)})

    leaves = []
    for op in graph.get_operations():
        for out in op.outputs:
            if len(out.consumers()) == 0 and out.dtype.name.startswith("float"):
                leaves.append({"name": out.name, "dtype": out.dtype.name, "shape": tensor_shape(out), "op_type": op.type})

    op_rows = []
    for op in graph.get_operations():
        op_rows.append(
            {
                "name": op.name,
                "type": op.type,
                "inputs": [x.name for x in op.inputs],
                "outputs": [x.name for x in op.outputs],
            }
        )

    return {
        "num_operations": len(op_rows),
        "placeholders": placeholders,
        "leaf_float_tensors": leaves,
        "operations": op_rows,
    }


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    try:
        import onnx
        import tensorflow as tf
        from onnx_direct import onnx_to_tf
    except Exception as e:
        raise RuntimeError("Missing dependencies: onnx, tensorflow, onnx_direct") from e

    model = onnx.load(args.model_path.as_posix())
    converted = onnx_to_tf(model)
    graph_def = as_graph_def(converted, tf)

    graph = tf.Graph()
    with graph.as_default():
        tf.graph_util.import_graph_def(graph_def, name="")

    summary = build_summary(graph)
    ops = summary["operations"]

    print(f"Model: {args.model_path}")
    print(f"Operations: {summary['num_operations']}")
    print("\nInputs (Placeholders):")
    if summary["placeholders"]:
        for x in summary["placeholders"]:
            print(f"  - {x['name']} | {x['dtype']} | shape={x['shape']}")
    else:
        print("  (none)")

    print("\nLikely outputs (leaf float tensors):")
    if summary["leaf_float_tensors"]:
        for x in summary["leaf_float_tensors"]:
            print(f"  - {x['name']} | {x['dtype']} | shape={x['shape']} | op={x['op_type']}")
    else:
        print("  (none)")

    print(f"\nOperations preview (first {min(args.max_ops, len(ops))}):")
    for row in ops[: args.max_ops]:
        print(f"  - {row['name']} [{row['type']}]")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSaved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
