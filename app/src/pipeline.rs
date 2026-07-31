//! Asynchronous frame-processing boundary used by the GUI.
//!
//! The GUI only submits frame paths and receives completed results. New
//! detector/tracker implementations can be placed behind `FramePipeline`
//! without coupling them to egui.
use crate::{detect, detect::Detection};
use anyhow::Result;
use image::RgbImage;
use opencv::dnn;
use std::{
    path::PathBuf,
    sync::mpsc::{self, Receiver, Sender},
    thread,
};

pub struct FrameBatch {
    pub cameras: Vec<PathBuf>,
}
pub struct FrameResult {
    pub images: Vec<RgbImage>,
    pub detections: Vec<Vec<Detection>>,
}

pub trait FramePipeline: Send {
    fn submit(&self, frames: FrameBatch, confidence: f32) -> Result<()>;
    fn try_result(&self) -> Option<Result<FrameResult>>;
}

pub struct OpenCvOnnxPipeline {
    tx: Sender<(FrameBatch, f32)>,
    rx: Receiver<Option<Result<FrameResult>>>,
}

impl OpenCvOnnxPipeline {
    pub fn start(model_path: PathBuf) -> Result<Self> {
        let (tx, requests) = mpsc::channel::<(FrameBatch, f32)>();
        let (results, rx) = mpsc::channel();
        thread::Builder::new()
            .name("mc-mot-pipeline".into())
            .spawn(move || {
                let mut net = match dnn::read_net_from_onnx_def(model_path) {
                    Ok(n) => n,
                    Err(e) => {
                        let _ = results.send(Some(Err(e.into())));
                        return;
                    }
                };
                while let Ok((pair, confidence)) = requests.recv() {
                    let result = (|| -> Result<FrameResult> {
                        let mut images = Vec::with_capacity(pair.cameras.len());
                        let mut detections = Vec::with_capacity(pair.cameras.len());
                        for path in pair.cameras {
                            let image = image::open(path)?.to_rgb8();
                            let camera_detections =
                                detect::detect(&mut net, &image, confidence, 0.45)?;
                            images.push(image);
                            detections.push(camera_detections);
                        }
                        Ok(FrameResult { images, detections })
                    })();
                    if results.send(Some(result)).is_err() {
                        break;
                    }
                }
            })?;
        Ok(Self { tx, rx })
    }
}

impl FramePipeline for OpenCvOnnxPipeline {
    fn submit(&self, frames: FrameBatch, confidence: f32) -> Result<()> {
        self.tx.send((frames, confidence)).map_err(Into::into)
    }
    fn try_result(&self) -> Option<Result<FrameResult>> {
        self.rx.try_recv().ok().flatten()
    }
}
