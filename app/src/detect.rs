//! RF-DETR ONNX inference through OpenCV's DNN module.

use anyhow::{Context, Result};
use image::RgbImage;
use opencv::{core, dnn, imgproc, prelude::*};

const INPUT_SIZE: i32 = 576;
// RF-DETR's exported COCO head includes the background slot at index 0.
const PERSON_CLASS: usize = 1;
const NUM_CLASSES: usize = 91;

#[derive(Debug, Clone)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

/// Runs the RF-DETR Medium export and keeps COCO's person class.
///
/// The exported graph returns normalized `cx, cy, width, height` boxes in
/// `dets[1, 300, 4]` and class logits in `labels[1, 300, 91]`. RF-DETR is a
/// set-prediction model, so duplicate suppression is not applied here; the
/// 300 query slots are already unique detections.
pub fn detect(net: &mut dnn::Net, img: &RgbImage, conf: f32, _nms: f32) -> Result<Vec<Detection>> {
    let t0 = std::time::Instant::now();
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

    // RF-DETR's Python preprocessing resizes (rather than letterboxes), then
    // applies ImageNet RGB normalization.
    let mut resized = core::Mat::default();
    imgproc::resize(
        &bgr,
        &mut resized,
        core::Size::new(INPUT_SIZE, INPUT_SIZE),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;
    let pixels = resized.data_bytes()?;
    let plane = (INPUT_SIZE * INPUT_SIZE) as usize;
    let means = [0.485_f32, 0.456, 0.406];
    let stds = [0.229_f32, 0.224, 0.225];
    let mut input = vec![0.0_f32; 3 * plane];
    for y in 0..INPUT_SIZE as usize {
        for x in 0..INPUT_SIZE as usize {
            let pixel = (y * INPUT_SIZE as usize + x) * 3;
            for channel in 0..3 {
                // resized is BGR; the model expects RGB in NCHW order.
                let value = pixels[pixel + (2 - channel)] as f32 / 255.0;
                input[channel * plane + y * INPUT_SIZE as usize + x] =
                    (value - means[channel]) / stds[channel];
            }
        }
    }
    let input_storage = core::Mat::from_slice(&input)?;
    let input = input_storage.reshape_nd(1, &[1, 3, INPUT_SIZE, INPUT_SIZE])?;
    net.set_input(&input, "", 1.0, core::Scalar::default())?;

    let names = net.get_unconnected_out_layers_names()?;
    let mut outputs = core::Vector::<core::Mat>::new();
    net.forward(&mut outputs, &names)?;
    if outputs.len() < 2 {
        anyhow::bail!(
            "RF-DETR ONNX returned {} outputs, expected dets and labels",
            outputs.len()
        );
    }
    let dets = outputs.get(0)?;
    let labels = outputs.get(1)?;
    let boxes = dets
        .data_typed::<f32>()
        .context("RF-DETR boxes are not f32")?;
    let logits = labels
        .data_typed::<f32>()
        .context("RF-DETR class logits are not f32")?;
    if boxes.len() < 300 * 4 || logits.len() < 300 * NUM_CLASSES {
        anyhow::bail!(
            "unexpected RF-DETR output sizes: boxes={}, labels={}",
            boxes.len(),
            logits.len()
        );
    }

    let mut result = Vec::new();
    for query in 0..300 {
        let score = sigmoid(logits[query * NUM_CLASSES + PERSON_CLASS]);
        if !score.is_finite() || score < conf {
            continue;
        }
        let cx = boxes[query * 4];
        let cy = boxes[query * 4 + 1];
        let width = boxes[query * 4 + 2];
        let height = boxes[query * 4 + 3];
        if ![cx, cy, width, height].iter().all(|v| v.is_finite()) || width <= 0.0 || height <= 0.0 {
            continue;
        }
        let x1 = ((cx - width / 2.0) * img.width() as f32).clamp(0.0, img.width() as f32 - 1.0);
        let y1 = ((cy - height / 2.0) * img.height() as f32).clamp(0.0, img.height() as f32 - 1.0);
        let x2 = ((cx + width / 2.0) * img.width() as f32).clamp(x1 + 1.0, img.width() as f32);
        let y2 = ((cy + height / 2.0) * img.height() as f32).clamp(y1 + 1.0, img.height() as f32);
        result.push(Detection { x1, y1, x2, y2 });
    }
    eprintln!(
        "[detect] RF-DETR queries={} keep={} {:.2?}",
        boxes.len() / 4,
        result.len(),
        t0.elapsed()
    );
    Ok(result)
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}
