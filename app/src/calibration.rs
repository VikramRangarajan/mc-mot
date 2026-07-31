//! Online SIFT/BFMatcher/RANSAC homography calibration.
use anyhow::{Result, bail};
use image::RgbImage;
use opencv::{core, features, geometry, imgproc, prelude::*};

pub fn estimate(first: &RgbImage, second: &RgbImage) -> Result<[[f64; 3]; 3]> {
    let to_gray = |image: &RgbImage| -> Result<core::Mat> {
        let raw = core::Mat::from_slice(image.as_raw())?;
        let rgb = raw.reshape(3, image.height() as i32)?;
        let mut gray = core::Mat::default();
        imgproc::cvt_color(
            &rgb,
            &mut gray,
            imgproc::COLOR_RGB2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        Ok(gray)
    };
    let gray1 = to_gray(first)?;
    let gray2 = to_gray(second)?;
    let mut sift = features::SIFT::create(1000, 3, 0.04, 10.0, 1.6, false)?;
    let mut keypoints1 = core::Vector::<core::KeyPoint>::new();
    let mut keypoints2 = core::Vector::<core::KeyPoint>::new();
    let mut descriptors1 = core::Mat::default();
    let mut descriptors2 = core::Mat::default();
    sift.detect_and_compute_def(
        &gray1,
        &core::no_array(),
        &mut keypoints1,
        &mut descriptors1,
    )?;
    sift.detect_and_compute_def(
        &gray2,
        &core::no_array(),
        &mut keypoints2,
        &mut descriptors2,
    )?;
    if keypoints1.len() < 4 || keypoints2.len() < 4 {
        bail!("not enough SIFT keypoints");
    }

    let mut matcher = features::BFMatcher::create(core::NORM_L2, false)?;
    let mut knn = core::Vector::<core::Vector<core::DMatch>>::new();
    let mut train = core::Vector::<core::Mat>::new();
    train.push(descriptors2);
    matcher.add(&train)?;
    matcher.knn_match_def(&descriptors1, &mut knn, 2)?;
    let mut src = core::Vector::<core::Point2f>::new();
    let mut dst = core::Vector::<core::Point2f>::new();
    for pair in knn.iter() {
        if pair.len() < 2 {
            continue;
        }
        let a = pair.get(0)?;
        let b = pair.get(1)?;
        if a.distance < 0.75 * b.distance {
            src.push(keypoints1.get(a.query_idx as usize)?.pt());
            dst.push(keypoints2.get(a.train_idx as usize)?.pt());
        }
    }
    if src.len() < 4 {
        bail!("not enough good SIFT matches: {}", src.len());
    }
    let mut mask = core::Mat::default();
    let h = geometry::find_homography_1(&src, &dst, &mut mask, geometry::RANSAC, 5.0)?;
    if h.empty() {
        bail!("RANSAC homography estimation failed");
    }
    let mut out = [[0.0; 3]; 3];
    for (r, row) in out.iter_mut().enumerate() {
        for (c, value) in row.iter_mut().enumerate() {
            *value = *h.at_2d::<f64>(r as i32, c as i32)?;
        }
    }
    Ok(out)
}

pub fn invert(h: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let d = h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
        - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
        + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);
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
