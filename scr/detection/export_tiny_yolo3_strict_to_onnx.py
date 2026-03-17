import argparse
import configparser
import io
import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
DEFAULT_WEIGHTS_URL = "https://data.pjreddie.com/files/yolov3-tiny.weights"
DEFAULT_ANCHORS = "10,14 23,27 37,58 81,82 135,169 344,319"
DEFAULT_MASKS = "3,4,5|0,1,2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export strict tiny_yolo3 ONNX from pjreddie cfg/weights")
    parser.add_argument("--python", type=Path, default=Path(__import__("sys").executable))
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "models/detection/tiny_yolo3_strict.onnx")
    parser.add_argument("--cfg-path", type=Path, default=REPO_ROOT / "models/detection/yolov3-tiny.cfg")
    parser.add_argument("--weights-path", type=Path, default=REPO_ROOT / "models/detection/yolov3-tiny.weights")
    parser.add_argument("--cfg-url", type=str, default=DEFAULT_CFG_URL)
    parser.add_argument("--weights-url", type=str, default=DEFAULT_WEIGHTS_URL)
    parser.add_argument("--height", type=int, default=416)
    parser.add_argument("--width", type=int, default=416)
    parser.add_argument("--batch-size", type=int, default=0, help="0 exports dynamic batch dimension")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--anchors", type=str, default=DEFAULT_ANCHORS)
    parser.add_argument("--masks", type=str, default=DEFAULT_MASKS)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def run(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def unique_config_sections(config_file: Path) -> io.StringIO:
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    with config_file.open("r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("["):
                section = line.strip().strip("[]")
                renamed = f"{section}_{section_counters[section]}"
                section_counters[section] += 1
                line = line.replace(section, renamed)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream


def ensure_download(url: str, output_path: Path, force: bool) -> None:
    if output_path.exists() and not force:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from urllib.request import Request, urlopen

    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request) as response, output_path.open("wb") as file:
        shutil.copyfileobj(response, file)


def build_patched_converter_script(output_path: Path) -> None:
    script = r'''
import configparser
import io
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf


def unique_config_sections(config_file: Path) -> io.StringIO:
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    with config_file.open("r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("["):
                section = line.strip().strip("[]")
                renamed = f"{section}_{section_counters[section]}"
                section_counters[section] += 1
                line = line.replace(section, renamed)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream


def convert(cfg_path: Path, weights_path: Path, output_h5: Path, input_height: int, input_width: int) -> None:
    weights_file = weights_path.open("rb")
    major, minor, revision = np.ndarray(shape=(3,), dtype="int32", buffer=weights_file.read(12))
    if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
        _ = np.ndarray(shape=(1,), dtype="int64", buffer=weights_file.read(8))
    else:
        _ = np.ndarray(shape=(1,), dtype="int32", buffer=weights_file.read(4))

    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_sections(cfg_path))

    input_layer = tf.keras.layers.Input(shape=(input_height, input_width, 3), name="input_1")
    prev_layer = input_layer
    all_layers = []
    out_index = []
    weight_decay = float(cfg_parser["net_0"].get("decay", 5e-4)) if "net_0" in cfg_parser.sections() else 5e-4

    conv_index = 0
    bn_index = 0
    for section in cfg_parser.sections():
        if section.startswith("convolutional"):
            filters = int(cfg_parser[section]["filters"])
            size = int(cfg_parser[section]["size"])
            stride = int(cfg_parser[section]["stride"])
            pad = int(cfg_parser[section]["pad"])
            activation = cfg_parser[section]["activation"]
            batch_normalize = "batch_normalize" in cfg_parser[section]
            padding = "same" if pad == 1 and stride == 1 else "valid"

            prev_shape = tf.keras.backend.int_shape(prev_layer)
            weights_shape = (size, size, prev_shape[-1], filters)
            darknet_shape = (filters, weights_shape[2], size, size)
            weights_size = int(np.product(weights_shape))

            conv_bias = np.ndarray(shape=(filters,), dtype="float32", buffer=weights_file.read(filters * 4))
            if batch_normalize:
                bn_weights = np.ndarray(shape=(3, filters), dtype="float32", buffer=weights_file.read(filters * 12))
                bn_weight_list = [bn_weights[0], conv_bias, bn_weights[1], bn_weights[2]]

            conv_weights = np.ndarray(shape=darknet_shape, dtype="float32", buffer=weights_file.read(weights_size * 4))
            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])

            if stride > 1:
                prev_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(prev_layer)
            conv_layer = tf.keras.layers.Conv2D(
                filters,
                (size, size),
                strides=(stride, stride),
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                use_bias=not batch_normalize,
                activation=None,
                padding=padding,
                name=f"conv2d_{conv_index}",
            )
            x = conv_layer(prev_layer)
            conv_layer.set_weights([conv_weights] if batch_normalize else [conv_weights, conv_bias])
            prev_layer = x

            if batch_normalize:
                bn_layer = tf.keras.layers.BatchNormalization(name=f"batch_normalization_{bn_index}")
                x = bn_layer(prev_layer)
                bn_layer.set_weights(bn_weight_list)
                prev_layer = x
                bn_index += 1

            if activation == "leaky":
                prev_layer = tf.keras.layers.LeakyReLU(alpha=0.1)(prev_layer)
            elif activation != "linear":
                raise RuntimeError(f"Unsupported activation: {activation}")

            all_layers.append(prev_layer)
            conv_index += 1
        elif section.startswith("route"):
            ids = [int(item) for item in cfg_parser[section]["layers"].split(",")]
            layers = [all_layers[item] for item in ids]
            if len(layers) > 1:
                prev_layer = tf.keras.layers.Concatenate()(layers)
            else:
                prev_layer = layers[0]
            all_layers.append(prev_layer)
        elif section.startswith("maxpool"):
            size = int(cfg_parser[section]["size"])
            stride = int(cfg_parser[section]["stride"])
            prev_layer = tf.keras.layers.MaxPooling2D(
                pool_size=(size, size), strides=(stride, stride), padding="same"
            )(prev_layer)
            all_layers.append(prev_layer)
        elif section.startswith("shortcut"):
            index = int(cfg_parser[section]["from"])
            prev_layer = tf.keras.layers.Add()([all_layers[index], prev_layer])
            all_layers.append(prev_layer)
        elif section.startswith("upsample"):
            stride = int(cfg_parser[section]["stride"])
            prev_layer = tf.keras.layers.UpSampling2D(stride)(prev_layer)
            all_layers.append(prev_layer)
        elif section.startswith("yolo"):
            out_index.append(len(all_layers) - 1)
            all_layers.append(None)
            prev_layer = all_layers[-1]
        elif section.startswith("net"):
            continue
        else:
            raise RuntimeError(f"Unsupported section: {section}")

    if not out_index:
        out_index.append(len(all_layers) - 1)

    model = tf.keras.Model(inputs=input_layer, outputs=[all_layers[i] for i in out_index], name="tiny_yolo3_strict")
    model.save(output_h5.as_posix())
    weights_file.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=Path, required=True)
    parser.add_argument("--weights-path", type=Path, required=True)
    parser.add_argument("--output-h5", type=Path, required=True)
    parser.add_argument("--height", type=int, default=416)
    parser.add_argument("--width", type=int, default=416)
    args = parser.parse_args()
    convert(args.cfg_path, args.weights_path, args.output_h5, args.height, args.width)
'''
    output_path.write_text(script, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cfg_path.parent.mkdir(parents=True, exist_ok=True)
    args.weights_path.parent.mkdir(parents=True, exist_ok=True)

    ensure_download(args.cfg_url, args.cfg_path, args.force_download)
    ensure_download(args.weights_url, args.weights_path, args.force_download)

    with tempfile.TemporaryDirectory(prefix="tinyyolo3_strict_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        converter_path = tmpdir / "convert_tiny_yolo3.py"
        h5_path = tmpdir / "tiny_yolo3_strict.h5"
        build_patched_converter_script(converter_path)

        run(
            [
                args.python.as_posix(),
                converter_path.as_posix(),
                "--cfg-path",
                args.cfg_path.as_posix(),
                "--weights-path",
                args.weights_path.as_posix(),
                "--output-h5",
                h5_path.as_posix(),
                "--height",
                str(args.height),
                "--width",
                str(args.width),
            ]
        )

        export_script = tmpdir / "export_to_onnx.py"
        export_script.write_text(
            """
from pathlib import Path
import argparse
import tensorflow as tf
import tf2onnx

parser = argparse.ArgumentParser()
parser.add_argument('--h5-path', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--height', type=int, default=416)
parser.add_argument('--width', type=int, default=416)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--opset', type=int, default=13)
args = parser.parse_args()
model = tf.keras.models.load_model(args.h5_path.as_posix(), compile=False)
batch_dim = None if args.batch_size <= 0 else args.batch_size
spec = (tf.TensorSpec((batch_dim, args.height, args.width, 3), tf.float32, name='input_1'),)
tf2onnx.convert.from_keras(model, input_signature=spec, opset=args.opset, output_path=args.output.as_posix())
""",
            encoding="utf-8",
        )
        run(
            [
                args.python.as_posix(),
                export_script.as_posix(),
                "--h5-path",
                h5_path.as_posix(),
                "--output",
                args.output.as_posix(),
                "--height",
                str(args.height),
                "--width",
                str(args.width),
                "--batch-size",
                str(args.batch_size),
                "--opset",
                str(args.opset),
            ]
        )

    metadata = {
        "model_name": "tiny_yolo3_strict",
        "model_variant": "yolo_heads",
        "cfg_path": args.cfg_path.as_posix(),
        "weights_path": args.weights_path.as_posix(),
        "cfg_url": args.cfg_url,
        "weights_url": args.weights_url,
        "input_layout": "nhwc",
        "input_color_order": "bgr",
        "input_value_range": "unit_float",
        "image_size": args.height,
        "static_batch_size": args.batch_size if args.batch_size > 0 else 0,
        "anchors": args.anchors,
        "masks": args.masks,
        "num_classes": 80,
        "source": "pjreddie_darknet_cfg_weights_keras_compatible_export",
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Exported strict tiny_yolo3 ONNX model to: {args.output}")
    print(f"Saved strict tiny_yolo3 metadata: {metadata_path}")


if __name__ == "__main__":
    main()
