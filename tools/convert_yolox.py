#!/usr/bin/env python3
"""Convert the official YOLOX-S checkpoint (Apache-2.0) to a candle-ready safetensors file.

YOLOX is Apache-2.0 (https://github.com/Megvii-BaseDetection/YOLOX). The official
pretrained `yolox_s.pth` is downloaded from the YOLOX GitHub release if it is not
already present, and written as `models/yolox_s.safetensors` with tensor names that
match the candle port in `app/src/yolox.rs` 1:1.

Usage:
    python3 tools/convert_yolox.py

Requires: torch, safetensors (pip install torch safetensors)
"""

import argparse
import os
import sys
from pathlib import Path

import torch

try:
    from safetensors.torch import save_file
except ImportError:
    sys.exit("missing dependency: pip install safetensors")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = REPO_ROOT / "models" / "yolox_s.pth"
DEFAULT_DST = REPO_ROOT / "models" / "yolox_s.safetensors"
URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth"


def download(url: str, dst: Path) -> None:
    import urllib.request

    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dst)
    print(f"downloaded {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="path to yolox_s.pth")
    ap.add_argument(
        "--dst", type=Path, default=DEFAULT_DST, help="output safetensors path"
    )
    args = ap.parse_args()

    if not args.src.exists():
        download(URL, args.src)

    ckpt = torch.load(args.src, map_location="cpu")
    sd = ckpt["model"]
    print(f"loaded {args.src} with {len(sd)} tensors")

    out = {}
    for key, value in sd.items():
        if key.endswith("num_batches_tracked"):
            continue
        out[key] = value.contiguous()

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, args.dst)
    print(f"wrote {args.dst} ({len(out)} tensors)")


if __name__ == "__main__":
    main()
