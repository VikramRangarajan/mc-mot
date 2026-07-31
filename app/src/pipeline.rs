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

pub struct FramePair {
    pub camera1: PathBuf,
    pub camera2: PathBuf,
}
pub struct FrameResult {
    pub image1: RgbImage,
    pub image2: RgbImage,
    pub detections1: Vec<Detection>,
    pub detections2: Vec<Detection>,
}

pub trait FramePipeline: Send {
    fn submit(&self, frames: FramePair, confidence: f32) -> Result<()>;
    fn try_result(&self) -> Option<Result<FrameResult>>;
}

pub struct OpenCvOnnxPipeline {
    tx: Sender<(FramePair, f32)>,
    rx: Receiver<Option<Result<FrameResult>>>,
}

impl OpenCvOnnxPipeline {
    pub fn start(model_path: PathBuf) -> Result<Self> {
        let (tx, requests) = mpsc::channel::<(FramePair, f32)>();
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
                        let image1 = image::open(&pair.camera1)?.to_rgb8();
                        let image2 = image::open(&pair.camera2)?.to_rgb8();
                        let detections1 = detect::detect(&mut net, &image1, confidence, 0.45)?;
                        let detections2 = detect::detect(&mut net, &image2, confidence, 0.45)?;
                        Ok(FrameResult {
                            image1,
                            image2,
                            detections1,
                            detections2,
                        })
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
    fn submit(&self, frames: FramePair, confidence: f32) -> Result<()> {
        self.tx.send((frames, confidence)).map_err(Into::into)
    }
    fn try_result(&self) -> Option<Result<FrameResult>> {
        self.rx.try_recv().ok().flatten()
    }
}
