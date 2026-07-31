//! Lightweight drawing helpers for RGB frames (no image-processing dependency).
use crate::sort::Track;
use image::{Rgb, RgbImage};
use std::collections::HashMap;

pub fn color_from_id(id: i64) -> [u8; 3] {
    let mut s = id as u64;
    let mut next = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        (s >> 24) as u8
    };
    [next(), next(), next()]
}
pub type History = HashMap<i64, Vec<(i32, i32)>>;

fn pixel(img: &mut RgbImage, x: i32, y: i32, c: Rgb<u8>) {
    if x >= 0 && y >= 0 && x < img.width() as i32 && y < img.height() as i32 {
        img.put_pixel(x as u32, y as u32, c);
    }
}
fn line(img: &mut RgbImage, mut x0: i32, mut y0: i32, x1: i32, y1: i32, c: Rgb<u8>) {
    let dx = (i64::from(x1) - i64::from(x0)).unsigned_abs() as i64;
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -((i64::from(y1) - i64::from(y0)).unsigned_abs() as i64);
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        pixel(img, x0, y0, c);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e = 2 * err;
        if e >= dy {
            err += dy;
            x0 += sx;
        }
        if e <= dx {
            err += dx;
            y0 += sy;
        }
    }
}
fn rect(img: &mut RgbImage, x1: i32, y1: i32, x2: i32, y2: i32, c: Rgb<u8>) {
    // Draw a 3-pixel outline for visibility at the GUI's downscaled preview.
    for d in -1..=1 {
        line(img, x1 + d, y1 + d, x2 - d, y1 + d, c);
        line(img, x2 - d, y1 + d, x2 - d, y2 - d, c);
        line(img, x2 - d, y2 - d, x1 + d, y2 - d, c);
        line(img, x1 + d, y2 - d, x1 + d, y1 + d, c);
    }
}
pub fn draw_tracks(
    img: &RgbImage,
    tracks: &[Track],
    ids: &HashMap<i64, i64>,
    history: &mut History,
    _font: &ab_glyph::FontArc,
) -> RgbImage {
    let mut out = img.clone();
    let max_x = out.width().saturating_sub(1) as f64;
    let max_y = out.height().saturating_sub(1) as f64;
    for t in tracks {
        let gid = ids.get(&t.id).copied().unwrap_or(t.id);
        let p = ((t.x1 + t.x2) as i32 / 2, (t.y1 + t.y2) as i32 / 2);
        let trail = history.entry(gid).or_default();
        trail.push(p);
        if trail.len() > 120 {
            trail.drain(..trail.len() - 120);
        }
    }
    for t in tracks {
        let gid = ids.get(&t.id).copied().unwrap_or(t.id);
        let c = Rgb(color_from_id(gid));
        let x1 = t.x1.clamp(0.0, max_x) as i32;
        let y1 = t.y1.clamp(0.0, max_y) as i32;
        let x2 = t.x2.clamp(0.0, max_x) as i32;
        let y2 = t.y2.clamp(0.0, max_y) as i32;
        rect(&mut out, x1, y1, x2, y2, c);
        if let Some(points) = history.get(&gid) {
            for pair in points.windows(2) {
                line(&mut out, pair[0].0.clamp(0, max_x as i32), pair[0].1.clamp(0, max_y as i32), pair[1].0.clamp(0, max_x as i32), pair[1].1.clamp(0, max_y as i32), c);
            }
        }
    }
    out
}
