#[path = "../multicam.rs"]
mod multicam;
#[path = "../sort.rs"]
mod sort;

fn main() -> anyhow::Result<()> {
    let count = sort::shared_id();
    let mut tracker1 = sort::Sort::new(30, 3, 0.3, &count);
    let mut tracker2 = sort::Sort::new(30, 3, 0.3, &count);

    // Use precomputed detection boxes: open wildtrack frames and run the detector is
    // overkill here; instead simulate a couple of moving boxes per camera.
    // Frame 0: two people per camera.
    let dets1 = [
        [100.0, 200.0, 150.0, 400.0, 0.9],
        [300.0, 220.0, 360.0, 430.0, 0.8],
    ];
    let dets2 = [
        [80.0, 180.0, 130.0, 380.0, 0.9],
        [280.0, 200.0, 340.0, 410.0, 0.8],
    ];
    let to4 = |d: &[f64; 5]| [d[0], d[1], d[2], d[3]];
    let mut t1 = tracker1.update(
        &dets1.iter().map(to4).collect::<Vec<_>>(),
        &vec![0; dets1.len()],
    );
    let mut t2 = tracker2.update(
        &dets2.iter().map(to4).collect::<Vec<_>>(),
        &vec![0; dets2.len()],
    );
    println!("frame 0 tracks1={t1:?} tracks2={t2:?}");

    // Build tracks as [x1,y1,x2,y2,id,label] columns for the global tracker.
    let as_rows = |ts: &[sort::Track]| -> Vec<[f64; 6]> {
        ts.iter()
            .map(|t| [t.x1, t.y1, t.x2, t.y2, t.id as f64, t.label as f64])
            .collect()
    };

    // Synthetic tracker test uses identity homographies. Runtime calibration
    // is performed by the SIFT/RANSAC pipeline in the main application.
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut global = multicam::MultiCameraTracker::new(vec![identity, identity], 0.20);

    let rows1: Vec<sort::Track> = std::mem::take(&mut t1);
    let rows2: Vec<sort::Track> = std::mem::take(&mut t2);
    let ids = global.update(&[rows1, rows2]);
    println!("global ids frame0 = {ids:?}");
    let _ = as_rows;

    // Frame 1: the boxes move slightly; confirm ids are consistent.
    let dets1 = [
        [105.0, 205.0, 155.0, 405.0, 0.9],
        [305.0, 225.0, 365.0, 435.0, 0.8],
    ];
    let dets2 = [
        [85.0, 185.0, 135.0, 385.0, 0.9],
        [285.0, 205.0, 345.0, 415.0, 0.8],
    ];
    let t1 = tracker1.update(&dets1.iter().map(to4).collect::<Vec<_>>(), &[0; 2]);
    let t2 = tracker2.update(&dets2.iter().map(to4).collect::<Vec<_>>(), &[0; 2]);
    let ids = global.update(&[t1, t2]);
    println!("global ids frame1 = {ids:?}");
    Ok(())
}
