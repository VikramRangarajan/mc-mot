//! SORT: Simple Online and Realtime Tracking, ported from
//! https://github.com/abewley/sort (used by the original mc-mot `sort.py`).

use nalgebra::{SMatrix, SVector, Vector4};
use std::cell::Cell;
use std::rc::Rc;

type KalmanX = SVector<f64, 7>;

/// A single SORT output track: integer-valued box + local tracker id (1-based)
/// + detection label (class index).
#[derive(Debug, Clone, Copy)]
pub struct Track {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
    pub id: i64,
    pub label: i64,
}

/// Hungarian algorithm (O(n^3), e-maxx variant) for minimum-cost assignment on a
/// rectangular cost matrix. Returns for each row the assigned column, or
/// `usize::MAX` if that row is unmatched.
fn linear_assignment(cost: &[Vec<f64>]) -> Vec<usize> {
    let n = cost.len();
    if n == 0 {
        return Vec::new();
    }
    let real_m = cost[0].len();
    let m = real_m.max(n);
    let size = m + 1;
    let mut padded = cost.to_vec();
    for row in &mut padded { row.resize(m, 0.0); }
    let mut u = vec![0f64; size];
    let mut v = vec![0f64; size];
    let mut p = vec![0usize; size];
    let mut way = vec![0usize; size];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; size];
        let mut used = vec![false; size];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;
            for j in 1..=m {
                if !used[j] {
                let cur = padded[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut ans = vec![usize::MAX; n];
    for j in 1..=m {
        if p[j] != 0 {
            ans[p[j] - 1] = if j <= real_m { j - 1 } else { usize::MAX };
        }
    }
    ans
}

fn iou(a: [f64; 4], b: [f64; 4]) -> f64 {
    if a.iter().any(|v| !v.is_finite()) || b.iter().any(|v| !v.is_finite()) {
        return 0.0;
    }
    let xx1 = a[0].max(b[0]);
    let yy1 = a[1].max(b[1]);
    let xx2 = a[2].min(b[2]);
    let yy2 = a[3].min(b[3]);
    let w = (xx2 - xx1).max(0.0);
    let h = (yy2 - yy1).max(0.0);
    let wh = w * h;
    let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - wh;
    if union <= 0.0 { 0.0 } else { wh / union }
}

/// Bbox in [x1, y1, x2, y2] form -> measurement [x, y, s, r], where (x, y) is
/// the centre, s the area and r the aspect ratio.
fn convert_bbox_to_z(bbox: [f64; 4]) -> Vector4<f64> {
    let w = bbox[2] - bbox[0];
    let h = bbox[3] - bbox[1];
    let x = bbox[0] + w / 2.0;
    let y = bbox[1] + h / 2.0;
    let s = w * h;
    let r = w / h;
    Vector4::new(x, y, s, r)
}

/// State [x, y, s, r, ...] -> bbox in [x1, y1, x2, y2] form.
fn convert_x_to_bbox(x: &KalmanX) -> [f64; 4] {
    let w = (x[2] * x[3]).sqrt();
    let h = x[2] / w;
    [
        x[0] - w / 2.0,
        x[1] - h / 2.0,
        x[0] + w / 2.0,
        x[1] + h / 2.0,
    ]
}

/// Mirrors the class-level `KalmanBoxTracker.count` used by the reference, so
/// that tracker ids are unique across all Sort instances in the process.
pub type SharedId = Rc<Cell<i64>>;

pub fn shared_id() -> SharedId {
    Rc::new(Cell::new(0))
}

struct KalmanBoxTracker {
    kf: KalmanFilter,
    time_since_update: i64,
    id: i64,
    hits: i64,
    hit_streak: i64,
    age: i64,
    label: i64,
}

impl KalmanBoxTracker {
    fn new(bbox: [f64; 4], label: i64, count: &SharedId) -> Self {
        let mut kf = KalmanFilter::new();
        let z = convert_bbox_to_z(bbox);
        kf.x[0] = z[0];
        kf.x[1] = z[1];
        kf.x[2] = z[2];
        kf.x[3] = z[3];
        let id = count.get();
        count.set(id + 1);
        Self {
            kf,
            time_since_update: 0,
            id,
            hits: 0,
            hit_streak: 0,
            age: 0,
            label,
        }
    }

    fn update(&mut self, bbox: [f64; 4]) {
        self.time_since_update = 0;
        self.hits += 1;
        self.hit_streak += 1;
        self.kf.update(convert_bbox_to_z(bbox));
    }

    fn predict(&mut self) -> [f64; 4] {
        if self.kf.x[6] + self.kf.x[2] <= 0.0 {
            self.kf.x[6] *= 0.0;
        }
        self.kf.predict();
        self.age += 1;
        if self.time_since_update > 0 {
            self.hit_streak = 0;
        }
        self.time_since_update += 1;
        convert_x_to_bbox(&self.kf.x)
    }

    fn get_state(&self) -> [f64; 4] {
        convert_x_to_bbox(&self.kf.x)
    }
}

/// Constant-velocity 7-state / 4-measurement Kalman filter, matching the
/// filterpy `KalmanFilter` configuration used by `sort.py`.
struct KalmanFilter {
    f: SMatrix<f64, 7, 7>,
    h: SMatrix<f64, 4, 7>,
    x: KalmanX,
    p: SMatrix<f64, 7, 7>,
    q: SMatrix<f64, 7, 7>,
    r: SMatrix<f64, 4, 4>,
}

impl KalmanFilter {
    fn new() -> Self {
        let f = SMatrix::<f64, 7, 7>::from_fn(|r, c| {
            if r == c || (r == 0 && c == 4) || (r == 1 && c == 5) || (r == 2 && c == 6) {
                1.0
            } else {
                0.0
            }
        });

        let mut h = SMatrix::<f64, 4, 7>::zeros();
        for i in 0..4 {
            h[(i, i)] = 1.0;
        }

        // filterpy defaults are R=I, Q=I, P=I; then sort.py adjusts:
        //   R[2:, 2:] *= 10 ; P[4:, 4:] *= 1000 ; P *= 10
        //   Q[-1, -1] *= 0.01 ; Q[4:, 4:] *= 0.01
        let mut r = SMatrix::<f64, 4, 4>::identity();
        r[(2, 2)] *= 10.0;
        r[(3, 3)] *= 10.0;

        let mut p = SMatrix::<f64, 7, 7>::identity();
        for i in 4..7 {
            p[(i, i)] *= 1000.0;
        }
        p *= 10.0;

        let mut q = SMatrix::<f64, 7, 7>::identity();
        q[(6, 6)] *= 0.01;
        for i in 4..7 {
            q[(i, i)] *= 0.01;
        }

        Self {
            f,
            h,
            x: KalmanX::zeros(),
            p,
            q,
            r,
        }
    }

    fn predict(&mut self) {
        self.x = self.f * self.x;
        self.p = self.f * self.p * self.f.transpose() + self.q;
    }

    fn update(&mut self, z: Vector4<f64>) {
        let y = z - self.h * self.x;
        let s = self.h * self.p * self.h.transpose() + self.r;
        let Some(s_inv) = s.try_inverse() else {
            return;
        };
        let k = self.p * self.h.transpose() * s_inv;
        self.x += k * y;
        self.p = self.p - k * self.h * self.p;
    }
}

/// Pairwise IOU matrix between detections and tracker predictions.
fn iou_matrix(dets: &[[f64; 4]], trks: &[[f64; 4]]) -> Vec<Vec<f64>> {
    dets.iter()
        .map(|d| trks.iter().map(|t| iou(*d, *t)).collect())
        .collect()
}

/// Assigns detections to tracked objects. Mirrors
/// `sort.py::associate_detections_to_trackers`. Both detections and tracker
/// predictions are given as [x1, y1, x2, y2] boxes.
pub fn associate_detections_to_trackers(
    dets: &[[f64; 4]],
    trks: &[[f64; 4]],
    iou_threshold: f64,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    if trks.is_empty() {
        return (Vec::new(), (0..dets.len()).collect(), Vec::new());
    }

    let iou_m = iou_matrix(dets, trks);

    let matched_indices: Vec<(usize, usize)> = if !iou_m.is_empty() && !iou_m[0].is_empty() {
        let rows = iou_m.len();
        let cols = iou_m[0].len();
        let a_sum_rows = iou_m
            .iter()
            .map(|r| r.iter().filter(|&&v| v > iou_threshold).count())
            .max()
            .unwrap();
        let a_sum_cols = (0..cols)
            .map(|j| (0..rows).filter(|&i| iou_m[i][j] > iou_threshold).count())
            .max()
            .unwrap();
        if a_sum_rows == 1 && a_sum_cols == 1 {
            let mut m = Vec::new();
            for (i, row) in iou_m.iter().enumerate().take(rows) {
                for (j, value) in row.iter().enumerate().take(cols) {
                    if *value > iou_threshold {
                        m.push((i, j));
                    }
                }
            }
            m
        } else {
            let cost: Vec<Vec<f64>> = iou_m
                .iter()
                .map(|row| row.iter().map(|v| -*v).collect())
                .collect();
            let assign = linear_assignment(&cost);
            assign
                .iter()
                .enumerate()
                .filter_map(|(i, &j)| if j != usize::MAX { Some((i, j)) } else { None })
                .collect()
        }
    } else {
        Vec::new()
    };

    let mut unmatched_dets: Vec<usize> = Vec::new();
    for d in 0..dets.len() {
        if !matched_indices.iter().any(|(i, _)| *i == d) {
            unmatched_dets.push(d);
        }
    }
    let mut unmatched_trks: Vec<usize> = Vec::new();
    for t in 0..trks.len() {
        if !matched_indices.iter().any(|(_, j)| *j == t) {
            unmatched_trks.push(t);
        }
    }

    // filter out matches with low IOU
    let mut matches: Vec<(usize, usize)> = Vec::new();
    for (i, j) in matched_indices {
        if iou_m[i][j] < iou_threshold {
            unmatched_dets.push(i);
            unmatched_trks.push(j);
        } else {
            matches.push((i, j));
        }
    }

    (matches, unmatched_dets, unmatched_trks)
}

/// SORT tracker: runs one Kalman tracker per object and performs IOU-based
/// association each frame.
pub struct Sort {
    max_age: i64,
    min_hits: i64,
    iou_threshold: f64,
    trackers: Vec<KalmanBoxTracker>,
    frame_count: i64,
    count: SharedId,
}

impl Sort {
    pub fn new(max_age: i64, min_hits: i64, iou_threshold: f64, count: &SharedId) -> Self {
        Self {
            max_age,
            min_hits,
            iou_threshold,
            trackers: Vec::new(),
            frame_count: 0,
            count: Rc::clone(count),
        }
    }

    /// Advances the tracker one frame with the given detections
    /// ([x1, y1, x2, y2] rows) and labels (class index per detection). Returns
    /// the confirmed tracks as [x1, y1, x2, y2, id, label] with integer coords.
    pub fn update(&mut self, dets: &[[f64; 4]], labels: &[i64]) -> Vec<Track> {
        self.frame_count += 1;

        // predict existing trackers
        let mut trks: Vec<[f64; 4]> = Vec::with_capacity(self.trackers.len());
        let mut to_del: Vec<usize> = Vec::new();
        for (t, trk) in self.trackers.iter_mut().enumerate() {
            let pos = trk.predict();
            if pos.iter().any(|v| v.is_nan()) {
                to_del.push(t);
                continue;
            }
            trks.push(pos);
        }
        for &t in to_del.iter().rev() {
            self.trackers.remove(t);
        }

        let (matched, unmatched_dets, _unmatched_trks) =
            associate_detections_to_trackers(dets, &trks, self.iou_threshold);

        // update matched trackers with assigned detections
        for (m_det, m_trk) in matched {
            self.trackers[m_trk].update([
                dets[m_det][0],
                dets[m_det][1],
                dets[m_det][2],
                dets[m_det][3],
            ]);
        }

        // create and initialise new trackers for unmatched detections
        for &i in &unmatched_dets {
            let bbox = [dets[i][0], dets[i][1], dets[i][2], dets[i][3]];
            let label = labels.get(i).copied().unwrap_or(0);
            self.trackers
                .push(KalmanBoxTracker::new(bbox, label, &self.count));
        }

        // produce output tracks and remove dead tracklets
        let mut ret: Vec<Track> = Vec::new();
        let mut i = self.trackers.len();
        let mut to_remove: Vec<usize> = Vec::new();
        for trk in self.trackers.iter().rev() {
            i -= 1;
            let d = trk.get_state();
            let confirmed = trk.time_since_update < 1
                && (trk.hit_streak >= self.min_hits || self.frame_count <= self.min_hits);
            if confirmed {
                ret.push(Track {
                    x1: d[0] as i64 as f64,
                    y1: d[1] as i64 as f64,
                    x2: d[2] as i64 as f64,
                    y2: d[3] as i64 as f64,
                    id: trk.id + 1, // +1 as MOT benchmark requires positive ids
                    label: trk.label,
                });
            }
            if trk.time_since_update > self.max_age {
                to_remove.push(i);
            }
        }
        for i in to_remove {
            self.trackers.remove(i);
        }
        ret
    }
}
