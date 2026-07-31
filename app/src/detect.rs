//! YOLO ONNX inference through OpenCV's DNN module.

use anyhow::{Context, Result};
use image::RgbImage;
use opencv::{core, dnn, imgproc, prelude::*};

#[derive(Debug, Clone)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

/// Runs a YOLOv5/YOLOv8-style ONNX model and keeps COCO class 0 (person).
pub fn detect(net: &mut dnn::Net, img: &RgbImage, conf: f32, nms: f32) -> Result<Vec<Detection>> {
    let t0 = std::time::Instant::now();
    eprintln!("[detect] start {}x{}", img.width(), img.height());
    let rgb = core::Mat::from_slice(img.as_raw())?;
    let rgb = rgb.reshape(3, img.height() as i32)?;
    let mut bgr = core::Mat::default();
    imgproc::cvt_color(
        &rgb,
        &mut bgr,
        imgproc::COLOR_RGB2BGR,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    // YOLOv5 is trained with letterboxed images; stretching the frame to 640x640
    // produces badly scaled boxes and commonly results in no usable detections.
    let h = bgr.rows() as f32;
    let w = bgr.cols() as f32;
    let scale = (640.0 / w).min(640.0 / h);
    let nw = (w * scale).round() as i32;
    let nh = (h * scale).round() as i32;
    let mut resized = core::Mat::default();
    imgproc::resize(
        &bgr,
        &mut resized,
        core::Size::new(nw, nh),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;
    let mut padded =
        core::Mat::new_rows_cols_with_default(640, 640, bgr.typ(), core::Scalar::all(114.0))?;
    let roi = core::Rect::new(0, 0, nw, nh);
    let mut dst = core::Mat::roi_mut(&mut padded, roi)?;
    resized.copy_to(&mut dst)?;
    let blob = dnn::blob_from_image(
        &padded,
        1.0 / 255.0,
        core::Size::new(640, 640),
        core::Scalar::default(),
        true,
        false,
        core::CV_32F,
    )?;
    eprintln!("[detect] blob {:.2?}", t0.elapsed());
    net.set_input(&blob, "", 1.0, core::Scalar::default())?;
    eprintln!("[detect] forward begin");
    let mut out = core::Mat::default();
    net.forward_layer_def(&mut out)?;
    eprintln!(
        "[detect] forward end dims={} total={} {:.2?}",
        out.dims(),
        out.total(),
        t0.elapsed()
    );
    let data = out.data_typed::<f32>().context("YOLO output is not f32")?;
    let cols = if out.dims() == 3 {
        out.mat_size().get(2)? as usize
    } else {
        out.cols() as usize
    };
    let rows = data.len() / cols;
    let inv_scale = 1.0 / scale;
    let mut boxes = core::Vector::<core::Rect>::new();
    let mut scores = core::Vector::<f32>::new();
    eprintln!("[detect] decode rows={} cols={}", rows, cols);
    for row in data.chunks(cols).take(rows) {
        if cols < 6 {
            continue;
        }
        let (cx, cy, bw, bh, score, class_id) = if cols == 6 {
            (row[0], row[1], row[2], row[3], row[4], row[5] as i32)
        } else {
            // YOLOv5: [cx,cy,w,h,obj,80 class scores] (85 columns).
            // YOLOv8: [cx,cy,w,h,80 class scores] (84 columns).
            let start = if cols == 85 { 5 } else { 4 };
            let (class_id, class_score) = row[start..]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            (
                row[0],
                row[1],
                row[2],
                row[3],
                row[4] * class_score,
                class_id as i32,
            )
        };
        if class_id != 0
            || !score.is_finite()
            || score < conf
            || !cx.is_finite()
            || !cy.is_finite()
            || !bw.is_finite()
            || !bh.is_finite()
            || bw <= 1.0
            || bh <= 1.0
        {
            continue;
        }
        let x = ((cx - bw / 2.0) * inv_scale).clamp(0.0, img.width() as f32 - 1.0);
        let y = ((cy - bh / 2.0) * inv_scale).clamp(0.0, img.height() as f32 - 1.0);
        let x2 = ((cx + bw / 2.0) * inv_scale).clamp(x + 1.0, img.width() as f32);
        let y2 = ((cy + bh / 2.0) * inv_scale).clamp(y + 1.0, img.height() as f32);
        boxes.push(core::Rect::new(
            x as i32,
            y as i32,
            (x2 - x).max(1.0) as i32,
            (y2 - y).max(1.0) as i32,
        ));
        scores.push(score);
        // Keep pathological exports from feeding tens of thousands of boxes
        // into NMS and blocking the UI thread.
        if scores.len() >= 4096 {
            break;
        }
    }
    let mut keep = core::Vector::<i32>::new();
    eprintln!("[detect] nms candidates={}", scores.len());
    dnn::nms_boxes(&boxes, &scores, conf, nms, &mut keep, 1.0, 0)?;
    eprintln!("[detect] done keep={} {:.2?}", keep.len(), t0.elapsed());
    Ok(keep
        .iter()
        .map(|i| {
            let r = boxes.get(i as usize).unwrap();
            Detection {
                x1: r.x as f32,
                y1: r.y as f32,
                x2: (r.x + r.width) as f32,
                y2: (r.y + r.height) as f32,
                // confidence: scores.get(i as usize).unwrap(),
            }
        })
        .collect())
}
