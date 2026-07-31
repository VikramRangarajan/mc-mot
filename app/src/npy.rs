//! Minimal .npy reader for float64 2-D arrays (little-endian), enough to load
//! the `cam4_H_cam1.npy` homography.

use std::fs::File;
use std::io::Read;

/// Reads a 2-D float64 .npy file into a row-major Vec plus its (rows, cols).
pub fn load_f64_2d(path: &str) -> anyhow::Result<(Vec<f64>, usize, usize)> {
    let mut buf = Vec::new();
    File::open(path)?.read_to_end(&mut buf)?;

    let magic = b"\x93NUMPY";
    if &buf[..6] != magic {
        anyhow::bail!("not a numpy file: {path}");
    }
    let major = buf[6];
    let (header_len_bytes, header_len) = match major {
        1 => {
            let mut b = [0u8; 2];
            b.copy_from_slice(&buf[8..10]);
            (2, u16::from_le_bytes(b) as usize)
        }
        2 | 3 => {
            let mut b = [0u8; 4];
            b.copy_from_slice(&buf[8..12]);
            (4, u32::from_le_bytes(b) as usize)
        }
        other => anyhow::bail!("unsupported npy version {other}"),
    };
    let header_start = 8 + header_len_bytes;
    let data_offset = header_start + header_len;
    let header = String::from_utf8_lossy(&buf[header_start..data_offset]);

    // descr: '<f8'
    let descr = extract(&header, "descr").unwrap_or_default();
    if !descr.contains("f8") && !descr.contains("f4") {
        anyhow::bail!("expected float dtype, got {descr}");
    }
    // shape: (3, 3)
    let shape = extract(&header, "shape").unwrap_or_default();
    let dims: Vec<usize> = shape
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();
    if dims.len() != 2 {
        anyhow::bail!("expected 2-D array, got shape {shape}");
    }
    let (rows, cols) = (dims[0], dims[1]);

    let mut data = Vec::with_capacity(rows * cols);
    for chunk in buf[data_offset..].chunks_exact(8) {
        data.push(f64::from_le_bytes(chunk.try_into().unwrap()));
    }
    if data.len() != rows * cols {
        anyhow::bail!("array length mismatch");
    }
    Ok((data, rows, cols))
}

fn extract(header: &str, key: &str) -> Option<String> {
    let needle = format!("'{key}':");
    let start = header.find(&needle)? + needle.len();
    let rest = &header[start..];
    let rest = rest.trim_start();
    // Value is either a quoted string, a tuple, or a bare keyword.
    let end = if rest.starts_with('\'') {
        rest[1..].find('\'')? + 2
    } else if rest.starts_with('(') {
        let mut depth = 0i32;
        for (i, ch) in rest.char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(rest[..=i].to_string());
                    }
                }
                _ => {}
            }
            let _ = i;
        }
        rest.len()
    } else {
        rest.find([',', '}']).unwrap_or(rest.len())
    };
    Some(rest[..end].trim().to_string())
}
