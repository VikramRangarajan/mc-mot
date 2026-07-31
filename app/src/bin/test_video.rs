#[path = "../video.rs"]
mod video;
fn main() -> anyhow::Result<()> {
    let v = video::extract_frames(&std::path::Path::new("/Users/vikramrangarajan/Documents/3d/mc-mot/data/cam1.mp4"))?;
    println!("frames: {}", v.frames.len());
    println!("first: {:?}", v.frames.first());
    println!("last: {:?}", v.frames.last());
    Ok(())
}
