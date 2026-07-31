//! Port of `homography_tracker.py`: fuses per-camera SORT tracks into global
//! identities by projecting bboxes into a common reference plane and matching
//! with pairwise IOU.

use std::collections::HashMap;

use crate::sort::{Track, associate_detections_to_trackers};

/// Applies a homography to a bbox in [x1, y1, x2, y2] form, truncating the
/// projected corners to integers (mirrors `modify_bbox_source`). Keeps the
/// id/label columns unchanged.
fn modify_bbox_source(tracks: &[Track], h: &[[f64; 3]; 3]) -> Vec<Track> {
    tracks
        .iter()
        .map(|t| {
            let p0 = hom(h, t.x1, t.y1);
            let p1 = hom(h, t.x2, t.y2);
            Track {
                x1: p0.0 as i64 as f64,
                y1: p0.1 as i64 as f64,
                x2: p1.0 as i64 as f64,
                y2: p1.1 as i64 as f64,
                id: t.id,
                label: t.label,
            }
        })
        .collect()
}

/// h @ [x, y, 1] normalised back to Euclidean coordinates.
fn hom(h: &[[f64; 3]; 3], x: f64, y: f64) -> (f64, f64) {
    let px = h[0][0] * x + h[0][1] * y + h[0][2];
    let py = h[1][0] * x + h[1][1] * y + h[1][2];
    let s = h[2][0] * x + h[2][1] * y + h[2][2];
    if !s.is_finite() || s.abs() < 1e-9 {
        return (f64::NAN, f64::NAN);
    }
    let x = px / s;
    let y = py / s;
    if x.is_finite() && y.is_finite() {
        (x, y)
    } else {
        (f64::NAN, f64::NAN)
    }
}

/// Global tracker over multiple camera sources. `homographies[i]` maps camera i
/// into the common reference plane.
pub struct MultiCameraTracker {
    num_sources: usize,
    homographies: Vec<[[f64; 3]; 3]>,
    iou_thres: f64,
    next_id: i64,
    ids: Vec<HashMap<i64, i64>>,
    age: Vec<HashMap<i64, i64>>,
}

impl MultiCameraTracker {
    pub fn new(homographies: Vec<[[f64; 3]; 3]>, iou_thres: f64) -> Self {
        let num_sources = homographies.len();
        Self {
            num_sources,
            homographies,
            iou_thres,
            next_id: 1,
            ids: vec![HashMap::new(); num_sources],
            age: vec![HashMap::new(); num_sources],
        }
    }

    /// Consumes one frame of per-camera tracks (each `Track.id` is the local
    /// SORT id) and returns, per camera, a mapping from local id to global id.
    pub fn update(&mut self, tracks: &[Vec<Track>]) -> Vec<HashMap<i64, i64>> {
        if tracks.len() != self.num_sources {
            return vec![HashMap::new(); self.num_sources];
        }
        // Project tracks to the common reference plane
        let proj: Vec<Vec<Track>> = tracks
            .iter()
            .enumerate()
            .map(|(i, trks)| modify_bbox_source(trks, &self.homographies[i]))
            .collect();

        // Give every visible local track an ID before cross-camera association.
        // This makes one-camera and N-camera operation well-defined.
        for (source, source_tracks) in proj.iter().enumerate() {
            for track in source_tracks {
                self.ids[source].entry(track.id).or_insert_with(|| {
                    let id = self.next_id;
                    self.next_id += 1;
                    id
                });
                *self.age[source].entry(track.id).or_insert(0) += 1;
            }
        }

        // Associate every camera pair in the common reference plane. Matching
        // is used to merge identities, so a third or fourth camera can join an
        // identity already established by an earlier pair.
        for i in 0..self.num_sources {
            for j in (i + 1)..self.num_sources {
                let dets: Vec<[f64; 4]> = proj[i].iter().map(box4).collect();
                let trks: Vec<[f64; 4]> = proj[j].iter().map(box4).collect();
                let (matches, _, _) =
                    associate_detections_to_trackers(&dets, &trks, self.iou_thres);
                for (idx_i, idx_j) in matches {
                    let id_i = proj[i][idx_i].id;
                    let id_j = proj[j][idx_j].id;
                    let global_i = self.ids[i][&id_i];
                    let global_j = self.ids[j][&id_j];
                    if global_i != global_j {
                        let age_i = self.age[i][&id_i];
                        let age_j = self.age[j][&id_j];
                        let keep = if age_i >= age_j { global_i } else { global_j };
                        let replace = if keep == global_i { global_j } else { global_i };
                        for source_ids in &mut self.ids {
                            for id in source_ids.values_mut() {
                                if *id == replace {
                                    *id = keep;
                                }
                            }
                        }
                    }
                }
            }
        }

        self.ids.clone()
    }
}

fn box4(t: &Track) -> [f64; 4] {
    [t.x1, t.y1, t.x2, t.y2]
}
