use ab_glyph::FontArc;
use eframe::egui;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::detect::{self, Detection};
use crate::draw;
use crate::multicam::MultiCameraTracker;
use crate::npy;
use crate::sort::{Sort, Track, shared_id};
use crate::video::{self, ExtractedVideo};
use opencv::dnn;

/// A single camera source: a list of frame images (either picked directly or
/// extracted from a picked video).
#[derive(Default)]
pub struct CameraSource {
    pub name: String,
    pub frames: Vec<PathBuf>,
    /// Holds the temp dir alive while extracted video frames are in use.
    video: Option<ExtractedVideo>,
}

impl CameraSource {
    fn clear(&mut self) {
        self.frames.clear();
        self.video = None;
    }
}

pub struct McMotApp {
    model: Option<dnn::Net>,
    homographies: Vec<[[f64; 3]; 3]>,
    font: FontArc,
    status: String,

    /// Frame sources for camera 1 and camera 2.
    cam1: CameraSource,
    cam2: CameraSource,

    /// Per-camera SORT trackers.
    tracker1: Sort,
    tracker2: Sort,

    global_tracker: Option<MultiCameraTracker>,

    /// Number of frames processed so far.
    frame: usize,
    running: bool,

    /// Last displayed results.
    vis1: Option<egui::ColorImage>,
    vis2: Option<egui::ColorImage>,
    tex1: Option<egui::TextureHandle>,
    tex2: Option<egui::TextureHandle>,
    tracks1: Vec<Track>,
    tracks2: Vec<Track>,
    global_ids: Vec<HashMap<i64, i64>>,

    /// Centroid history per camera (for trail drawing).
    hist1: draw::History,
    hist2: draw::History,

    /// Detection / tracking parameters (tunable in the toolbar).
    conf: f32,
    min_hits: i64,
    max_age: i64,
    iou_thres: f64,
    fps: f64,
}

fn homography_to_array(data: &[f64]) -> [[f64; 3]; 3] {
    let mut h = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            h[r][c] = data[r * 3 + c];
        }
    }
    h
}

fn invert_homography(h: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
        - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
        + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);
    let d = det;
    let mut r = [[0.0; 3]; 3];
    r[0][0] = (h[1][1] * h[2][2] - h[1][2] * h[2][1]) / d;
    r[0][1] = (h[0][2] * h[2][1] - h[0][1] * h[2][2]) / d;
    r[0][2] = (h[0][1] * h[1][2] - h[0][2] * h[1][1]) / d;
    r[1][0] = (h[1][2] * h[2][0] - h[1][0] * h[2][2]) / d;
    r[1][1] = (h[0][0] * h[2][2] - h[0][2] * h[2][0]) / d;
    r[1][2] = (h[0][2] * h[1][0] - h[0][0] * h[1][2]) / d;
    r[2][0] = (h[1][0] * h[2][1] - h[1][1] * h[2][0]) / d;
    r[2][1] = (h[0][1] * h[2][0] - h[0][0] * h[2][1]) / d;
    r[2][2] = (h[0][0] * h[1][1] - h[0][1] * h[1][0]) / d;
    r
}

impl McMotApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let _ = cc;

        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("models/yolov5m.onnx");
        let homography_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("data/cam4_H_cam1.npy");

        let model = match dnn::read_net_from_onnx_def(model_path.to_str().unwrap()) {
            Ok(mut net) => {
                if let Err(e) = opencv::prelude::NetTrait::set_preferable_backend(&mut net, dnn::DNN_BACKEND_OPENCV) {
                    eprintln!("warning: could not select OpenCV DNN backend: {e}");
                }
                Some(net)
            }
            Err(e) => {
                eprintln!("warning: failed to load ONNX model from {model_path:?}: {e}");
                None
            }
        };

        let (homographies, status) = match npy::load_f64_2d(homography_path.to_str().unwrap()) {
            Ok((data, rows, cols)) if rows == 3 && cols == 3 => {
                let h_c4_from_c1 = homography_to_array(&data);
                let h_c1_from_c4 = invert_homography(&h_c4_from_c1);
                let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
                (
                    vec![identity, h_c1_from_c4],
                    format!("homography loaded from {homography_path:?}"),
                )
            }
            Ok(_) => (Vec::new(), "homography has wrong shape".to_string()),
            Err(e) => (Vec::new(), format!("failed to load homography: {e}")),
        };

        // Load a system font for drawing labels.
        let font = load_font();

        let count = shared_id();
        let tracker1 = Sort::new(30, 1, 0.1, &count);
        let tracker2 = Sort::new(30, 1, 0.1, &count);

        let mut app = Self {
            model,
            homographies,
            font,
            status,
            cam1: CameraSource {
                name: "Camera 1".into(),
                ..Default::default()
            },
            cam2: CameraSource {
                name: "Camera 2".into(),
                ..Default::default()
            },
            tracker1,
            tracker2,
            global_tracker: None,
            frame: 0,
            running: false,
            vis1: None,
            vis2: None,
            tex1: None,
            tex2: None,
            tracks1: Vec::new(),
            tracks2: Vec::new(),
            global_ids: Vec::new(),
            hist1: HashMap::new(),
            hist2: HashMap::new(),
            conf: 0.30,
            min_hits: 1,
            max_age: 30,
            iou_thres: 0.10,
            fps: 0.0,
        };

        // Debug defaults: load the repository videos automatically when present.
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
        for (path, cam) in [(root.join("data/cam1.mp4"), &mut app.cam1), (root.join("data/cam4.mp4"), &mut app.cam2)] {
            if path.exists() {
                match video::extract_frames(&path) {
                    Ok(extracted) => { cam.frames = extracted.frames.clone(); cam.video = Some(extracted); }
                    Err(e) => app.status = format!("failed to load {}: {e}", path.display()),
                }
            }
        }
        app
    }

    fn reset_pipeline(&mut self) {
        let count = shared_id();
        self.tracker1 = Sort::new(self.max_age, self.min_hits, self.iou_thres, &count);
        self.tracker2 = Sort::new(self.max_age, self.min_hits, self.iou_thres, &count);
        self.global_tracker = None;
        self.frame = 0;
        self.vis1 = None;
        self.vis2 = None;
        self.tex1 = None;
        self.tex2 = None;
        self.tracks1.clear();
        self.tracks2.clear();
        self.hist1.clear();
        self.hist2.clear();
    }

    /// Runs one frame couple through detect -> SORT -> global tracker.
    fn step(&mut self) {
        eprintln!("[app] step frame={}", self.frame);
        let Some(model) = self.model.as_mut() else {
            self.status = "no model loaded".to_string();
            return;
        };
        if self.homographies.len() < 2 {
            self.status = "homography not loaded".to_string();
            return;
        }
        if self.cam1.frames.is_empty() || self.cam2.frames.is_empty() {
            self.status = "add frames for both cameras first".to_string();
            return;
        }

        let i = self.frame;
        if i >= self.cam1.frames.len() || i >= self.cam2.frames.len() {
            self.status = "all frames processed".to_string();
            self.running = false;
            return;
        }

        let img1 = match image::open(&self.cam1.frames[i]) {
            Ok(im) => im.to_rgb8(),
            Err(e) => {
                self.status = format!("failed to open {}: {e}", self.cam1.frames[i].display());
                return;
            }
        };
        let img2 = match image::open(&self.cam2.frames[i]) {
            Ok(im) => im.to_rgb8(),
            Err(e) => {
                self.status = format!("failed to open {}: {e}", self.cam2.frames[i].display());
                return;
            }
        };
        eprintln!("[app] images loaded frame={}", i);

        let t0 = std::time::Instant::now();
        let conf = self.conf;
        let dets1 = match detect::detect(model, &img1, conf, 0.45) {
            Ok(d) => d,
            Err(e) => {
                self.status = format!("camera 1 detection failed: {e}");
                return;
            }
        };
        let dets2 = match detect::detect(model, &img2, conf, 0.45) {
            Ok(d) => d,
            Err(e) => {
                self.status = format!("camera 2 detection failed: {e}");
                return;
            }
        };
        eprintln!("[app] detections {} {}", dets1.len(), dets2.len());

        // SORT expects integer bboxes.
        let to_xyxy = |d: &Detection| {
            [
                d.x1 as i64 as f64,
                d.y1 as i64 as f64,
                d.x2 as i64 as f64,
                d.y2 as i64 as f64,
            ]
        };
        let dets1: Vec<[f64; 4]> = dets1.iter().map(to_xyxy).collect();
        let dets2: Vec<[f64; 4]> = dets2.iter().map(to_xyxy).collect();

        let labels1 = vec![0i64; dets1.len()];
        let labels2 = vec![0i64; dets2.len()];

        let tracks1 = self.tracker1.update(&dets1, &labels1);
        let tracks2 = self.tracker2.update(&dets2, &labels2);
        eprintln!("[app] sort tracks {} {}", tracks1.len(), tracks2.len());

        let global_tracker = self
            .global_tracker
            .get_or_insert_with(|| MultiCameraTracker::new(self.homographies.clone(), self.iou_thres));
        let global_ids = global_tracker.update(&[tracks1.clone(), tracks2.clone()]);
        eprintln!("[app] global tracker done");

        let vis1 = draw::draw_tracks(&img1, &tracks1, &global_ids[0], &mut self.hist1, &self.font);
        let vis2 = draw::draw_tracks(&img2, &tracks2, &global_ids[1], &mut self.hist2, &self.font);

        let to_color = |im: &image::RgbImage| {
            egui::ColorImage::from_rgb([im.width() as usize, im.height() as usize], im.as_raw())
        };

        self.vis1 = Some(to_color(&vis1));
        self.vis2 = Some(to_color(&vis2));
        self.tracks1 = tracks1;
        self.tracks2 = tracks2;
        self.global_ids = global_ids;
        self.frame += 1;
        let elapsed = t0.elapsed();
        self.fps = 1.0 / elapsed.as_secs_f64();
        self.status = format!(
            "frame {}: {} / {} tracks ({:?})",
            i,
            self.tracks1.len(),
            self.tracks2.len(),
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
                if ui.button("Add Camera 1...").clicked() {
                    let cam = &mut self.cam1;
                    let msg = pick_source(cam);
                    self.status = msg;
                }
                if ui.button("Add Camera 2...").clicked() {
                    let cam = &mut self.cam2;
                    let msg = pick_source(cam);
                    self.status = msg;
                }
                ui.separator();
                if ui.button("Run").clicked() {
                    self.running = true;
                    self.reset_pipeline();
                    self.step();
                }
                if ui.button(if self.running { "Pause" } else { "Play" }).clicked() {
                    self.running = !self.running;
                    if self.running { ui.ctx().request_repaint(); }
                }
                if ui.button("Step").clicked() && !self.running {
                        if self.frame == 0 {
                            self.reset_pipeline();
                        }
                        self.step();
                }
                if ui.button("Clear").clicked() {
                    self.cam1.clear();
                    self.cam2.clear();
                    self.reset_pipeline();
                    self.status.clear();
                }
                ui.separator();
                ui.label(format!(
                    "camera 1: {} frame(s) | camera 2: {} frame(s) | frame {}",
                    self.cam1.frames.len(),
                    self.cam2.frames.len(),
                    self.frame
                ));
            });
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
                ui.add(egui::Slider::new(&mut self.iou_thres, 0.0..=1.0).step_by(0.01).show_value(true));
                if self.fps > 0.0 {
                    ui.separator();
                    ui.label(format!("{:.1} fps", self.fps));
                }
            });
            ui.add_space(4.0);
        });

        egui::CentralPanel::default().show(ui, |ui| {
            if self.running {
                self.step();
                // keep stepping until both sources are exhausted
                if self.frame >= self.cam1.frames.len() || self.frame >= self.cam2.frames.len() {
                    self.running = false;
                }
                ui.ctx().request_repaint();
            }

            if !self.status.is_empty() {
                ui.label(&self.status);
                ui.separator();
            }

            if self.vis1.is_none() && self.vis2.is_none() {
                ui.centered_and_justified(|ui| {
                    ui.label(
                        "Add images or videos for camera 1 and camera 2, then press Run.\n\
                         The pipeline is: YOLO ONNX person detection -> per-camera SORT ->\n\
                         homography-based global track fusion.",
                    );
                });
            } else {
                egui::ScrollArea::both().show(ui, |ui| {
                    let avail = ui.available_size();
                    let spacing = 16.0;
                    let max = egui::vec2((avail.x - spacing) / 2.0, avail.y);
                    ui.horizontal(|ui| {
                        show_image(ui, &self.cam1.name, &self.vis1, &mut self.tex1, max);
                        show_image(ui, &self.cam2.name, &self.vis2, &mut self.tex2, max);
                    });
                });
            }
        });
    }
}
