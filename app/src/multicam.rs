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
    if !s.is_finite() || s.abs() < 1e-9 { return (f64::NAN, f64::NAN); }
    let x = px / s;
    let y = py / s;
    if x.is_finite() && y.is_finite() { (x, y) } else { (f64::NAN, f64::NAN) }
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
        // Project tracks to the common reference plane
        let proj: Vec<Vec<Track>> = tracks
            .iter()
            .enumerate()
            .map(|(i, trks)| modify_bbox_source(trks, &self.homographies[i]))
            .collect();

        // For each pair of sources
        for i in 0..self.num_sources {
            for j in (i + 1)..self.num_sources {
                // Match tracks with IOU
                let mut matched: HashMap<i64, bool> = HashMap::new();
                let dets: Vec<[f64; 4]> = proj[i].iter().map(box4).collect();
                let trks: Vec<[f64; 4]> = proj[j].iter().map(box4).collect();
                let (matches, unmatches_i, unmatches_j) =
                    associate_detections_to_trackers(&dets, &trks, self.iou_thres);

                // Set global ids for the matched tracks
                for (idx_i, idx_j) in matches {
                    let id_i = proj[i][idx_i].id;
                    let id_j = proj[j][idx_j].id;
                    let match_i = self.ids[i].get(&id_i).copied();
                    let match_j = self.ids[j].get(&id_j).copied();

                    // If track i has a global id and is at least as old as track j
                    let assigned = if let Some(mi) = match_i {
                        let age_i = self.age[i].get(&id_i).copied().unwrap_or(0);
                        let age_j = self.age[j].get(&id_j).copied().unwrap_or(0);
                        if age_i >= age_j && !matched.contains_key(&mi) {
                            self.ids[j].insert(id_j, mi);
                            matched.insert(mi, true);
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    // Else if track j has a global id
                    let assigned = if assigned {
                        true
                    } else if let Some(mj) = match_j {
                        if let std::collections::hash_map::Entry::Vacant(e) = matched.entry(mj) {
                            self.ids[i].insert(id_i, mj);
                            e.insert(true);
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    // None of them has a global id
                    if !assigned {
                        self.ids[i].insert(id_i, self.next_id);
                        self.ids[j].insert(id_j, self.next_id);
                        matched.insert(self.next_id, true);
                        self.next_id += 1;
                    }

                    // Increment track age
                    let a = self.age[i].get(&id_i).copied().unwrap_or(0) + 1;
                    self.age[i].insert(id_i, a);
                    let a = self.age[j].get(&id_j).copied().unwrap_or(0) + 1;
                    self.age[j].insert(id_j, a);
                }

                // Set global ids for unmatched tracks of source i
                for &idx_i in &unmatches_i {
                    let id_i = proj[i][idx_i].id;
                    let match_i = self.ids[i].get(&id_i).copied();
                    if match_i.is_none() || matched.contains_key(&match_i.unwrap()) {
                        self.ids[i].insert(id_i, self.next_id);
                        matched.insert(self.next_id, true);
                        self.next_id += 1;
                    }
                    let a = self.age[i].get(&id_i).copied().unwrap_or(0) + 1;
                    self.age[i].insert(id_i, a);
                }

                // Set global ids for unmatched tracks of source j
                for &idx_j in &unmatches_j {
                    let id_j = proj[j][idx_j].id;
                    let match_j = self.ids[j].get(&id_j).copied();
                    if match_j.is_none() || matched.contains_key(&match_j.unwrap()) {
                        self.ids[j].insert(id_j, self.next_id);
                        matched.insert(self.next_id, true);
                        self.next_id += 1;
                    }
                    let a = self.age[j].get(&id_j).copied().unwrap_or(0) + 1;
                    self.age[j].insert(id_j, a);
                }
            }
        }

        self.ids.clone()
    }
}

fn box4(t: &Track) -> [f64; 4] {
    [t.x1, t.y1, t.x2, t.y2]
}
