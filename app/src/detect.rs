//! YOLO ONNX inference through OpenCV's DNN module.

use anyhow::{Context, Result};
use image::RgbImage;
use opencv::{core, dnn, imgproc, prelude::*};

#[derive(Debug, Clone)]
pub struct Detection { pub x1: f32, pub y1: f32, pub x2: f32, pub y2: f32, pub confidence: f32 }

/// Runs a YOLOv5/YOLOv8-style ONNX model and keeps COCO class 0 (person).
pub fn detect(net: &mut dnn::Net, img: &RgbImage, conf: f32, nms: f32) -> Result<Vec<Detection>> {
    let rgb = core::Mat::from_slice(img.as_raw())?;
    let rgb = rgb.reshape(3, img.height() as i32)?;
    let mut bgr = core::Mat::default();
    imgproc::cvt_color(&rgb, &mut bgr, imgproc::COLOR_RGB2BGR, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    let blob = dnn::blob_from_image(&bgr, 1.0 / 255.0, core::Size::new(640, 640),
        core::Scalar::default(), true, false, core::CV_32F)?;
    net.set_input(&blob, "", 1.0, core::Scalar::default())?;
    let mut out = core::Mat::default();
    net.forward_layer_def(&mut out)?;
    let data = out.data_typed::<f32>().context("YOLO output is not f32")?;
    let cols = if out.dims() == 3 { out.mat_size().get(2)? as usize } else { out.cols() as usize };
    let rows = data.len() / cols;
    let (sx, sy) = (img.width() as f32 / 640.0, img.height() as f32 / 640.0);
    let mut boxes = core::Vector::<core::Rect>::new();
    let mut scores = core::Vector::<f32>::new();
    for row in data.chunks(cols).take(rows) {
        if cols < 6 { continue; }
        let (cx, cy, bw, bh, score, class_id) = if cols == 6 {
            (row[0], row[1], row[2], row[3], row[4], row[5] as i32)
        } else {
            let start = if cols > 85 { 5 } else { 4 };
            let (class_id, class_score) = row[start..].iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
            (row[0], row[1], row[2], row[3], row[4] * class_score, class_id as i32)
        };
        if class_id != 0 || score < conf { continue; }
        boxes.push(core::Rect::new(((cx - bw / 2.0) * sx) as i32, ((cy - bh / 2.0) * sy) as i32,
            (bw * sx) as i32, (bh * sy) as i32));
        scores.push(score);
    }
    let mut keep = core::Vector::<i32>::new();
    dnn::nms_boxes(&boxes, &scores, conf, nms, &mut keep, 1.0, 0)?;
    Ok(keep.iter().map(|i| { let r = boxes.get(i as usize).unwrap(); Detection {
        x1: r.x as f32, y1: r.y as f32, x2: (r.x + r.width) as f32, y2: (r.y + r.height) as f32,
        confidence: scores.get(i as usize).unwrap(),
    }}).collect())
}
