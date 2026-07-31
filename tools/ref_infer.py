#!/usr/bin/env python
import argparse
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, "/tmp")
sys.path.insert(0, "/tmp/stub")
from yolox_ref.yolox import YOLOX
from yolox_ref.yolo_pafpn import YOLOPAFPN
from yolox_ref.yolo_head import YOLOXHead

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("ckpt")
    ap.add_argument("out_dir")
    args = ap.parse_args()

    model = YOLOX(
        backbone=YOLOPAFPN(depth=0.33, width=0.50),
        head=YOLOXHead(80, width=0.50),
    )
    model.eval()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)

    img = cv2.imread(args.image)  # BGR
    x, r = preproc(img, (640, 640), MEAN, STD)
    xs = torch.from_numpy(x).unsqueeze(0)

    with torch.no_grad():
        outs = model.head(model.backbone(xs))  # decoded outputs [1, N, 85]
    outs = outs[0].numpy()  # [N, 85]
    print("decoded outputs shape:", outs.shape)
    print("box range:", outs[:, :4].min(), outs[:, :4].max())

    reg_raw, obj_raw, cls_raw = [], [], []
    for k, stride in enumerate([8, 16, 32]):
        feat = model.backbone(xs)
        h = model.backbone(xs)
        with torch.no_grad():
            fpn = model.backbone(xs)
            xin = fpn
            _head = model.head
            x = _head.stems[k](xin[k])
            cls_feat = _head.cls_convs[k](x)
            cls_output = _head.cls_preds[k](cls_feat)
            reg_feat = _head.reg_convs[k](x)
            reg_output = _head.reg_preds[k](reg_feat)
            obj_output = _head.obj_preds[k](reg_feat)
        reg_raw.append(reg_output[0].numpy())
        obj_raw.append(obj_output[0].numpy())
        cls_raw.append(cls_output[0].numpy())
        print(
            f"level {k} stride {stride}: reg {reg_output.shape} obj {obj_output.shape} cls {cls_output.shape}"
        )

    for i, (r_, o_, c_) in enumerate(zip(reg_raw, obj_raw, cls_raw)):
        np.save(f"{args.out_dir}/reg_{i}.npy", r_)
        np.save(f"{args.out_dir}/obj_{i}.npy", o_)
        np.save(f"{args.out_dir}/cls_{i}.npy", c_)
    np.save(f"{args.out_dir}/decoded.npy", outs)

    # quick person-detection sanity: class 0, conf threshold 0.3
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    obj_s = sig(obj_raw[0]).reshape(-1)
    cls0_s = sig(cls_raw[0][0]).reshape(-1)
    print("level0: obj max", obj_s.max(), "person cls0 max", cls0_s.max())


if __name__ == "__main__":
    main()
