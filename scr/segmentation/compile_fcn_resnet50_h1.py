import argparse
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile H1 quantized FCN-ResNet50 (.qm) to TPU program (.tpu)")
    parser.add_argument(
        "--input-qm",
        type=Path,
        default=REPO_ROOT / "experiments/segmentation/fcn_resnet50_h1_quantized.qm",
        help="Path to quantized model (.qm)",
    )
    parser.add_argument(
        "--output-tpu",
        type=Path,
        default=REPO_ROOT / "experiments/segmentation/fcn_resnet50_h1.tpu",
        help="Path to output TPU program (.tpu)",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for compiled network")
    parser.add_argument(
        "--preset",
        type=str,
        default="O5",
        choices=["O1", "O5", "DEFAULT"],
        help="Compiler preset",
    )
    return parser.parse_args()


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
    if not args.input_qm.exists():
        raise FileNotFoundError(f"Input .qm not found: {args.input_qm}")

    try:
        from tpu_framework import Network, QuantizedModel, TPU_128x128_PARAMS, TpuProgram, compiler  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependencies from tpu_framework") from e

    print(f"Loading quantized model: {args.input_qm}")
    quantized_model = QuantizedModel.load(args.input_qm.as_posix())
    network, _ = Network.from_quantized_model(quantized_model)
    network.set_batch(args.batch_size)

    preset = resolve_preset(args.preset)
    print(f"Compiling with preset={args.preset}, batch={args.batch_size}")
    tlm_program = compiler.compile_(
        hardware_parameters=TPU_128x128_PARAMS,
        network=network,
        parameters=preset,
    )

    executable, tensor_descriptions = tlm_program
    tpu_program = TpuProgram.from_executable(executable, tensor_descriptions)
    args.output_tpu.parent.mkdir(parents=True, exist_ok=True)
    tpu_program.to_file(args.output_tpu.as_posix())
    print(f"Saved TPU program: {args.output_tpu}")


if __name__ == "__main__":
    main()
