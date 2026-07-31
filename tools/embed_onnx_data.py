"""Embed ONNX external tensor data so OpenCV DNN can load the model."""

import argparse
import onnx

parser = argparse.ArgumentParser()
parser.add_argument("model", help="ONNX model with adjacent .data files")
args = parser.parse_args()
model = onnx.load(args.model, load_external_data=True)
onnx.save_model(model, args.model, save_as_external_data=False)
print(f"embedded tensor data in {args.model}")
