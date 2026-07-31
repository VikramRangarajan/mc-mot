#[path = "../video.rs"]
mod video;
fn main() -> anyhow::Result<()> {
    let input = std::path::Path::new("/Users/vikramrangarajan/Documents/3d/mc-mot/data/cam1.mp4");
    assert!(
        video::is_video(input),
        "test input must have a supported video extension"
    );
    let v = video::extract_frames(input)?;
    println!("frames: {}", v.frames.len());
    println!("first: {:?}", v.frames.first());
    println!("last: {:?}", v.frames.last());
    Ok(())
}
