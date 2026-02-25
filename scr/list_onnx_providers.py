import argparse
import json

import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List available ONNX Runtime execution providers")
    parser.add_argument("--json", action="store_true", help="Print output as JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = {
        "onnxruntime_version": ort.__version__,
        "available_providers": ort.get_available_providers(),
    }

    if args.json:
        print(json.dumps(info, indent=2))
        return

    print(f"onnxruntime: {info['onnxruntime_version']}")
    print("available providers:")
    for provider in info["available_providers"]:
        print(f"- {provider}")


if __name__ == "__main__":
    main()
