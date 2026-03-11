import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MLPerf accuracy output for a standard ResNet-50 ImageNet classifier"
    )
    parser.add_argument("--mlperf-accuracy-file", type=Path, required=True)
    parser.add_argument("--imagenet-val-file", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "int8", "int32", "int64"],
        help="Force prediction decoding dtype; auto handles scalar class indices and common logit layouts",
    )
    parser.add_argument(
        "--label-shift",
        default="auto",
        help="Subtract this integer from labels in val_map.txt before comparison; auto detects 0/1-based labels",
    )
    return parser.parse_args()


DTYPE_MAP = {
    "float32": np.float32,
    "int8": np.int8,
    "int32": np.int32,
    "int64": np.int64,
}


def load_imagenet_rows(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_name, label = line.split()
            rows.append((image_name, int(label)))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def resolve_label_shift(labels: list[int], raw_value: str) -> int:
    if raw_value != "auto":
        return int(raw_value)
    minimum = min(labels)
    maximum = max(labels)
    if minimum == 1 and maximum == 1000:
        return 1
    if minimum == 0 and maximum == 999:
        return 0
    return 0


def decode_with_dtype(raw: bytes, dtype_name: str) -> tuple[int, str]:
    array = np.frombuffer(raw, dtype=DTYPE_MAP[dtype_name])
    if array.size == 0:
        raise RuntimeError("Empty prediction payload")
    if array.size == 1:
        return int(array[0]), f"{dtype_name}_scalar"
    return int(array.argmax()), f"{dtype_name}_vector"


def decode_prediction(raw_hex: str, dtype_name: str) -> tuple[int, str]:
    raw = bytes.fromhex(raw_hex)
    if dtype_name != "auto":
        return decode_with_dtype(raw, dtype_name)

    if len(raw) == 8:
        return int(np.frombuffer(raw, dtype=np.int64)[0]), "int64_scalar"
    if len(raw) == 4:
        return int(np.frombuffer(raw, dtype=np.int32)[0]), "int32_scalar"
    if len(raw) == 1000:
        return int(np.frombuffer(raw, dtype=np.int8).argmax()), "int8_logits"
    if len(raw) == 4000:
        logits = np.frombuffer(raw, dtype=np.float32)
        if np.isfinite(logits).all():
            return int(logits.argmax()), "float32_logits"
        return int(np.frombuffer(raw, dtype=np.int32).argmax()), "int32_logits"
    if len(raw) % 8 == 0:
        values = np.frombuffer(raw, dtype=np.int64)
        if values.size == 1:
            return int(values[0]), "int64_scalar"
        return int(values.argmax()), "int64_vector"
    if len(raw) % 4 == 0:
        values = np.frombuffer(raw, dtype=np.int32)
        if values.size == 1:
            return int(values[0]), "int32_scalar"
        return int(values.argmax()), "int32_vector"

    values = np.frombuffer(raw, dtype=np.int8)
    if values.size == 1:
        return int(values[0]), "int8_scalar"
    return int(values.argmax()), "int8_vector"


def main() -> None:
    args = parse_args()
    imagenet_rows = load_imagenet_rows(args.imagenet_val_file)
    labels = [label for _, label in imagenet_rows]
    label_shift = resolve_label_shift(labels, args.label_shift)

    with args.mlperf_accuracy_file.open("r", encoding="utf-8") as file:
        results = json.load(file)

    seen: set[int] = set()
    good = 0
    detected_format = None
    for item in results:
        idx = item["qsl_idx"]
        if idx in seen:
            continue
        seen.add(idx)

        image_name, raw_label = imagenet_rows[idx]
        expected_label = raw_label - label_shift
        predicted_label, detected_format = decode_prediction(item["data"], args.dtype)

        if predicted_label == expected_label:
            good += 1
        elif args.verbose:
            print(
                f"{image_name}, expected: {expected_label} "
                f"(raw {raw_label}), found {predicted_label}"
            )

    print(f"prediction_format={detected_format}")
    print(f"label_shift={label_shift}")
    print(f"accuracy={100.0 * good / len(seen):.3f}%, good={good}, total={len(seen)}")


if __name__ == "__main__":
    main()
