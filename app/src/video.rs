//! Video support: extracts all frames of a video to a temporary directory as
//! PNGs using the system `ffmpeg` binary, then hands them off to the existing
//! image pipeline. Keeps the crate dependency list MIT/Apache-2.0.

use std::path::{Path, PathBuf};
use std::process::Command;

const VIDEO_EXTS: &[&str] = &[
    "mp4", "mov", "avi", "mkv", "webm", "m4v", "ts", "mpeg", "mpg", "wmv",
];

pub fn is_video(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| VIDEO_EXTS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Simple self-deleting temp dir (avoids a tempfile dependency).
pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn create(prefix: &str) -> std::io::Result<TempDir> {
        let base = std::env::temp_dir();
        for _ in 0..100 {
            let unique = format!(
                "{prefix}-{}-{:x}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.subsec_nanos())
                    .unwrap_or(0)
            );
            let path = base.join(&unique);
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(TempDir { path }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not create temp dir",
        ))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// All frames of the video as extracted PNG paths.
pub struct ExtractedVideo {
    pub frames: Vec<PathBuf>,
    pub _temp: TempDir,
}

/// Extracts every frame of `video` into a fresh temp dir as
/// `frame_00000.png`, `frame_00001.png`, ... using the `ffmpeg` CLI.
pub fn extract_frames(video: &Path) -> anyhow::Result<ExtractedVideo> {
    let temp = TempDir::create("mc-mot-video")?;
    let pattern = temp.path().join("frame_%05d.png");

    let out = Command::new("ffmpeg")
        .args(["-y", "-i"])
        .arg(video)
        .args(["-vsync", "0", "-start_number", "0", "-qscale:v", "2"])
        .arg(&pattern)
        .output()?;
    if !out.status.success() {
        anyhow::bail!(
            "ffmpeg failed: {}\n{}",
            String::from_utf8_lossy(&out.stderr),
            String::from_utf8_lossy(&out.stdout)
        );
    }

    let mut frames: Vec<PathBuf> = std::fs::read_dir(temp.path())?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "png").unwrap_or(false))
        .collect();
    frames.sort_by_key(|p| {
        p.file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.rsplit('_').next())
            .and_then(|n| n.parse::<u64>().ok())
            .unwrap_or(0)
    });

    if frames.is_empty() {
        anyhow::bail!("no frames extracted from {}", video.display());
    }

    Ok(ExtractedVideo {
        frames,
        _temp: temp,
    })
}
