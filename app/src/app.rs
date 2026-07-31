use ab_glyph::FontArc;
use eframe::egui;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::calibration;
use crate::detect::Detection;
use crate::draw;
use crate::multicam::MultiCameraTracker;
use crate::pipeline::{FrameBatch, FramePipeline, OpenCvOnnxPipeline};
use crate::sort::{Sort, Track, shared_id};
use crate::video::{self, ExtractedVideo};

/// A single camera source: a list of frame images (either picked directly or
/// extracted from a picked video).
#[derive(Default)]
pub struct CameraSource {
    pub name: String,
    pub frames: Vec<PathBuf>,
    /// Holds the temp dir alive while extracted video frames are in use.
    video: Option<ExtractedVideo>,
}

pub struct McMotApp {
    homographies: Vec<[[f64; 3]; 3]>,
    font: FontArc,
    status: String,

    /// Frame sources for all cameras. Camera 0 is the reference plane.
    cameras: Vec<CameraSource>,

    /// Per-camera SORT trackers.
    trackers: Vec<Sort>,

    global_tracker: Option<MultiCameraTracker>,

    /// Number of frames processed so far.
    frame: usize,
    running: bool,

    /// Last displayed results.
    vis: Vec<Option<egui::ColorImage>>,
    tex: Vec<Option<egui::TextureHandle>>,
    tracks: Vec<Vec<Track>>,
    global_ids: Vec<HashMap<i64, i64>>,

    /// Centroid history per camera (for trail drawing).
    histories: Vec<draw::History>,

    /// Detection / tracking parameters (tunable in the toolbar).
    conf: f32,
    min_hits: i64,
    max_age: i64,
    iou_thres: f64,
    fps: f64,
    pipeline: Option<OpenCvOnnxPipeline>,
    pending_pipeline: bool,
    playback_speed: f32,
    cache: Vec<Option<CachedFrame>>,
}

#[derive(Clone)]
struct CachedFrame {
    tracks: Vec<Vec<Track>>,
    global_ids: Vec<HashMap<i64, i64>>,
}

impl McMotApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let _ = cc;

        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("models/rfdetr-medium.onnx");
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let homographies = vec![identity];
        let status = "add camera sources to begin".to_string();

        // Load a system font for drawing labels.
        let font = load_font();

        let mut app = Self {
            homographies,
            font,
            status,
            cameras: Vec::new(),
            trackers: Vec::new(),
            global_tracker: None,
            frame: 0,
            running: false,
            vis: Vec::new(),
            tex: Vec::new(),
            tracks: Vec::new(),
            global_ids: Vec::new(),
            histories: Vec::new(),
            conf: 0.30,
            min_hits: 1,
            max_age: 30,
            iou_thres: 0.10,
            fps: 0.0,
            pipeline: None,
            pending_pipeline: false,
            playback_speed: 1.0,
            cache: Vec::new(),
        };

        match OpenCvOnnxPipeline::start(model_path) {
            Ok(pipeline) => app.pipeline = Some(pipeline),
            Err(e) => app.status = format!("failed to start pipeline worker: {e}"),
        }
        app
    }

    fn reset_pipeline(&mut self) {
        let count = shared_id();
        self.trackers = (0..self.cameras.len())
            .map(|_| Sort::new(self.max_age, self.min_hits, self.iou_thres, &count))
            .collect();
        self.global_tracker = None;
        self.frame = 0;
        self.vis = vec![None; self.cameras.len()];
        self.tex = (0..self.cameras.len()).map(|_| None).collect();
        self.tracks = vec![Vec::new(); self.cameras.len()];
        self.histories = (0..self.cameras.len()).map(|_| HashMap::new()).collect();
        self.global_ids = vec![HashMap::new(); self.cameras.len()];
        self.pending_pipeline = false;
    }

    /// Re-estimates every camera's mapping into camera 0's reference plane.
    fn recalibrate_homography(&mut self) {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let Some(reference) = self.cameras.first().and_then(|c| c.frames.first()) else {
            self.homographies = Vec::new();
            self.status = "add camera sources to begin".into();
            return;
        };
        let Ok(reference) = image::open(reference).map(|image| image.to_rgb8()) else {
            self.homographies = vec![identity; self.cameras.len()];
            self.status = "could not load reference calibration frame".into();
            return;
        };
        let mut homographies = vec![identity];
        self.status = "calibrating homography online...".into();
        for camera in self.cameras.iter().skip(1) {
            let Some(path) = camera.frames.first() else {
                homographies.push(identity);
                continue;
            };
            match image::open(path).map(|image| image.to_rgb8()) {
                Ok(image) => match calibration::estimate(&reference, &image) {
                    Ok(h) => homographies.push(calibration::invert(&h)),
                    Err(e) => {
                        homographies.push(identity);
                        self.status = format!("homography failed for {}: {e}", camera.name);
                    }
                },
                Err(e) => {
                    homographies.push(identity);
                    self.status = format!("could not load {} calibration frame: {e}", camera.name);
                }
            }
        }
        self.homographies = homographies;
        if self.status == "calibrating homography online..." {
            self.status = if self.cameras.len() == 1 {
                "reference camera loaded; add another camera to calibrate".into()
            } else {
                format!(
                    "online SIFT homography calibrated for {} cameras",
                    self.cameras.len()
                )
            };
        }
    }

    fn source_changed(&mut self) {
        self.reset_pipeline();
        self.cache.clear();
        self.recalibrate_homography();
    }

    /// Runs one synchronized frame from every camera through detect -> SORT -> global tracker.
    fn step(&mut self) {
        eprintln!("[app] step frame={}", self.frame);
        if self.cameras.is_empty() {
            self.status = "add camera sources first".to_string();
            return;
        }
        if self.homographies.len() != self.cameras.len() {
            self.status = "homographies are not ready".to_string();
            return;
        }
        let i = self.frame;
        let frame_count = self
            .cameras
            .iter()
            .map(|camera| camera.frames.len())
            .min()
            .unwrap_or(0);
        if i >= frame_count {
            self.status = "all frames processed".to_string();
            self.running = false;
            return;
        }

        if let Some(Some(cached)) = self.cache.get(i).cloned()
            && cached.tracks.len() == self.cameras.len()
            && cached.global_ids.len() == self.cameras.len()
        {
            for (camera_index, camera) in self.cameras.iter().enumerate() {
                let Ok(image) = image::open(&camera.frames[i]).map(|image| image.to_rgb8()) else {
                    continue;
                };
                let rendered = draw::draw_tracks(
                    &image,
                    &cached.tracks[camera_index],
                    &cached.global_ids[camera_index],
                    &mut self.histories[camera_index],
                    &self.font,
                );
                self.vis[camera_index] = Some(egui::ColorImage::from_rgb(
                    [rendered.width() as usize, rendered.height() as usize],
                    rendered.as_raw(),
                ));
            }
            self.tracks = cached.tracks;
            self.global_ids = cached.global_ids;
            self.frame += 1;
            self.status = format!("cached frame {}", i);
            return;
        }

        let result = if self.pending_pipeline {
            let Some(pipeline) = &self.pipeline else {
                self.status = "pipeline worker unavailable".into();
                return;
            };
            match pipeline.try_result() {
                Some(Ok(result)) => {
                    self.pending_pipeline = false;
                    result
                }
                Some(Err(e)) => {
                    self.pending_pipeline = false;
                    self.status = format!("pipeline failed: {e}");
                    return;
                }
                None => return,
            }
        } else {
            let Some(pipeline) = &self.pipeline else {
                self.status = "pipeline worker unavailable".into();
                return;
            };
            let cameras = self
                .cameras
                .iter()
                .map(|camera| camera.frames[i].clone())
                .collect();
            if let Err(e) = pipeline.submit(FrameBatch { cameras }, self.conf) {
                self.status = format!("pipeline submit failed: {e}");
                return;
            }
            self.pending_pipeline = true;
            return;
        };
        if result.images.len() != self.cameras.len()
            || result.detections.len() != self.cameras.len()
        {
            self.status = "pipeline returned the wrong number of cameras".into();
            return;
        }
        let t0 = std::time::Instant::now();

        // SORT expects integer bboxes.
        let to_xyxy = |d: &Detection| {
            [
                d.x1 as i64 as f64,
                d.y1 as i64 as f64,
                d.x2 as i64 as f64,
                d.y2 as i64 as f64,
            ]
        };
        let tracks: Vec<Vec<Track>> = result
            .detections
            .iter()
            .enumerate()
            .map(|(camera_index, detections)| {
                let boxes: Vec<[f64; 4]> = detections.iter().map(to_xyxy).collect();
                let labels = vec![0i64; boxes.len()];
                self.trackers[camera_index].update(&boxes, &labels)
            })
            .collect();

        let global_tracker = self.global_tracker.get_or_insert_with(|| {
            MultiCameraTracker::new(self.homographies.clone(), self.iou_thres)
        });
        let global_ids = global_tracker.update(&tracks);

        let to_color = |im: &image::RgbImage| {
            egui::ColorImage::from_rgb([im.width() as usize, im.height() as usize], im.as_raw())
        };
        for camera_index in 0..self.cameras.len() {
            let rendered = draw::draw_tracks(
                &result.images[camera_index],
                &tracks[camera_index],
                &global_ids[camera_index],
                &mut self.histories[camera_index],
                &self.font,
            );
            self.vis[camera_index] = Some(to_color(&rendered));
        }
        self.tracks = tracks;
        self.global_ids = global_ids;
        if self.cache.len() <= i {
            self.cache.resize_with(i + 1, || None);
        }
        self.cache[i] = Some(CachedFrame {
            tracks: self.tracks.clone(),
            global_ids: self.global_ids.clone(),
        });
        self.frame += 1;
        let elapsed = t0.elapsed();
        self.fps = 1.0 / elapsed.as_secs_f64();
        self.status = format!(
            "frame {}: {} / {} tracks ({:?})",
            i,
            self.tracks.iter().map(Vec::len).sum::<usize>(),
            self.cameras.len(),
            elapsed
        );
        eprintln!("[app] frame complete {}", i);
    }
}

fn pick_source(cam: &mut CameraSource) -> String {
    let files = rfd::FileDialog::new()
        .add_filter(
            "Media (images / videos)",
            &[
                "png", "jpg", "jpeg", "bmp", "gif", "webp", "tiff", "ico", "pnm", "mp4", "mov",
                "avi", "mkv", "webm", "m4v", "ts", "mpeg", "mpg", "wmv",
            ],
        )
        .add_filter(
            "Images",
            &[
                "png", "jpg", "jpeg", "bmp", "gif", "webp", "tiff", "ico", "pnm",
            ],
        )
        .add_filter(
            "Videos",
            &[
                "mp4", "mov", "avi", "mkv", "webm", "m4v", "ts", "mpeg", "mpg", "wmv",
            ],
        )
        .pick_files();

    let Some(files) = files else {
        return format!("{}: selection cancelled", cam.name);
    };

    let single_video = files.len() == 1 && video::is_video(&files[0]);
    if single_video {
        match video::extract_frames(&files[0]) {
            Ok(extracted) => {
                let n = extracted.frames.len();
                let frames = extracted.frames.clone();
                cam.video = Some(extracted);
                cam.frames = frames;
                format!("{}: video -> {} frames extracted", cam.name, n)
            }
            Err(e) => format!("{}: failed to extract video: {e}", cam.name),
        }
    } else {
        cam.video = None;
        cam.frames = files;
        format!("{}: {} frame(s) selected", cam.name, cam.frames.len())
    }
}

fn load_font() -> FontArc {
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ] {
        if let Ok(data) = std::fs::read(path)
            && let Ok(font) = FontArc::try_from_vec(data)
        {
            return font;
        }
    }
    FontArc::try_from_slice(include_bytes!(
        "/System/Library/Fonts/Supplemental/Arial.ttf"
    ))
    .unwrap_or_else(|_| panic!("no font available"))
}

/// Shows one camera preview scaled to fit the given max size, preserving
/// aspect ratio.
fn show_image(
    ui: &mut egui::Ui,
    label: &str,
    img: &Option<egui::ColorImage>,
    tex: &mut Option<egui::TextureHandle>,
    max: egui::Vec2,
) {
    if let Some(img) = img {
        ui.label(label);
        if let Some(texture) = tex {
            texture.set(img.clone(), egui::TextureOptions::LINEAR);
        } else {
            *tex = Some(
                ui.ctx()
                    .load_texture(label, img.clone(), egui::TextureOptions::LINEAR),
            );
        }
        let texture = tex.as_ref().unwrap();
        let img_size = egui::vec2(img.size[0] as f32, img.size[1] as f32);
        let scale = (max.x / img_size.x).min(max.y / img_size.y).min(1.0);
        let shown = img_size * scale;
        ui.add(
            egui::Image::new(egui::load::SizedTexture::new(texture, shown))
                .max_size(shown)
                .maintain_aspect_ratio(true),
        );
    }
}

impl eframe::App for McMotApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("toolbar").show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui.button("Add Camera...").clicked() {
                    let index = self.cameras.len();
                    let mut camera = CameraSource {
                        name: format!("Camera {}", index + 1),
                        ..Default::default()
                    };
                    let message = pick_source(&mut camera);
                    if !camera.frames.is_empty() {
                        self.cameras.push(camera);
                        self.source_changed();
                    } else {
                        self.status = message;
                    }
                }
                ui.separator();
                if ui.button("Clear").clicked() {
                    self.cameras.clear();
                    self.reset_pipeline();
                    self.cache.clear();
                    self.homographies.clear();
                    self.status = "add camera sources to begin".into();
                }
                ui.separator();
                ui.label(format!(
                    "{} camera(s) | frame {}",
                    self.cameras.len(),
                    self.frame
                ));
            });
            if !self.cameras.is_empty() {
                ui.horizontal_wrapped(|ui| {
                    for camera in &self.cameras {
                        ui.label(format!("{}: {} frame(s)", camera.name, camera.frames.len()));
                    }
                });
            }
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("conf");
                ui.add(
                    egui::Slider::new(&mut self.conf, 0.005..=0.5)
                        .logarithmic(true)
                        .show_value(true),
                );
                ui.label("min hits");
                ui.add(egui::Slider::new(&mut self.min_hits, 1..=10).show_value(true));
                ui.label("max age");
                ui.add(egui::Slider::new(&mut self.max_age, 1..=120).show_value(true));
                ui.label("SORT IOU");
                ui.add(
                    egui::Slider::new(&mut self.iou_thres, 0.0..=1.0)
                        .step_by(0.01)
                        .show_value(true),
                );
                if self.fps > 0.0 {
                    ui.separator();
                    ui.label(format!("{:.1} fps", self.fps));
                }
            });
            ui.add_space(4.0);
        });

        egui::Panel::bottom("transport").show(ui, |ui| {
            let frame_count = self
                .cameras
                .iter()
                .map(|camera| camera.frames.len())
                .min()
                .unwrap_or(0);
            let mut selected = self.frame.min(frame_count.saturating_sub(1));
            if frame_count > 0
                && ui
                    .add(egui::Slider::new(&mut selected, 0..=frame_count - 1).text("Frame"))
                    .changed()
                && selected != self.frame
            {
                // Scrubbing always takes control away from playback.
                self.running = false;
                self.reset_pipeline();
                self.frame = selected;
                // Show the selected source images immediately. Tracking overlays
                // are applied when cached metadata is available or when the
                // selected frame is processed with Step/Play.
                for (camera_index, camera) in self.cameras.iter().enumerate() {
                    if let Ok(image) = image::open(&camera.frames[selected]) {
                        let image = image.to_rgb8();
                        self.vis[camera_index] = Some(egui::ColorImage::from_rgb(
                            [image.width() as usize, image.height() as usize],
                            image.as_raw(),
                        ));
                    }
                }
                self.status = format!("seeked to frame {selected}");
            }
            ui.horizontal(|ui| {
                if ui
                    .button(if self.running { "Pause" } else { "Play" })
                    .clicked()
                {
                    self.running = !self.running;
                    if self.running {
                        ui.ctx().request_repaint();
                    }
                }
                if ui.button("Step").clicked() && !self.running {
                    self.step();
                    ui.ctx().request_repaint();
                }
                ui.separator();
                ui.label("Speed");
                egui::ComboBox::from_id_salt("playback_speed")
                    .selected_text(format!("{}x", self.playback_speed))
                    .show_ui(ui, |ui| {
                        for speed in [0.5_f32, 1.0, 1.5, 2.0] {
                            ui.selectable_value(
                                &mut self.playback_speed,
                                speed,
                                format!("{}x", speed),
                            );
                        }
                    });
                ui.label(format!(
                    "frame {}/{}",
                    self.frame,
                    frame_count.saturating_sub(1)
                ));
            });
        });

        egui::CentralPanel::default().show(ui, |ui| {
            if self.pending_pipeline {
                ui.ctx()
                    .request_repaint_after(std::time::Duration::from_millis(16));
            }
            if self.running {
                self.step();
                // keep stepping until both sources are exhausted
                if self.frame
                    >= self
                        .cameras
                        .iter()
                        .map(|camera| camera.frames.len())
                        .min()
                        .unwrap_or(0)
                {
                    self.running = false;
                }
                // The source videos are 30 FPS. Inference remains asynchronous;
                // this controls the pacing between completed frame requests.
                let interval = 1.0 / (30.0 * self.playback_speed as f64);
                ui.ctx()
                    .request_repaint_after(std::time::Duration::from_secs_f64(interval));
            }

            if !self.status.is_empty() {
                ui.label(&self.status);
                ui.separator();
            }

            if self.vis.iter().all(Option::is_none) {
                ui.centered_and_justified(|ui| {
                    ui.label(
                        "Add two or more camera sources, then press Play.\n\
                         The pipeline is: RF-DETR ONNX person detection -> per-camera SORT ->\n\
                         homography-based global track fusion.",
                    );
                });
            } else {
                egui::ScrollArea::both().show(ui, |ui| {
                    let avail = ui.available_size();
                    let spacing = 12.0;
                    let columns = (self.cameras.len() as f32).sqrt().ceil().max(1.0) as usize;
                    let max = egui::vec2(
                        (avail.x - spacing * (columns.saturating_sub(1) as f32)) / columns as f32,
                        (avail.y - spacing * (columns.saturating_sub(1) as f32)) / columns as f32,
                    );
                    egui::Grid::new("camera_grid")
                        .num_columns(columns)
                        .spacing(egui::vec2(spacing, spacing))
                        .show(ui, |ui| {
                            for (camera_index, camera) in self.cameras.iter().enumerate() {
                                show_image(
                                    ui,
                                    &camera.name,
                                    &self.vis[camera_index],
                                    &mut self.tex[camera_index],
                                    max,
                                );
                                if (camera_index + 1) % columns == 0 {
                                    ui.end_row();
                                }
                            }
                        });
                });
            }
        });
    }
}
