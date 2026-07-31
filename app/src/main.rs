#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod calibration;
mod detect;
mod draw;
mod multicam;
mod pipeline;
mod sort;
mod video;

use eframe::egui;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 800.0]),
        ..Default::default()
    };
    eframe::run_native(
        "MC-MOT",
        options,
        Box::new(|cc| Ok(Box::new(app::McMotApp::new(cc)))),
    )
}
