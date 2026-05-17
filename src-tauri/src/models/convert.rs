use std::path::{Path, PathBuf};
use std::process::Command;

use super::ModelType;
use crate::{Result, UtaiError};

pub fn convert_pth_to_onnx(
    pth_path: &PathBuf,
    models_dir: &PathBuf,
    model_type: &ModelType,
) -> Result<PathBuf> {
    let subdir = match model_type {
        ModelType::Rvc => "rvc",
        ModelType::SoVits => "sovits",
        ModelType::S2H => "s2h",
        ModelType::F0 => "f0",
        ModelType::NsfHifigan => "nsf_hifigan",
    };

    let output_dir = models_dir.join(subdir);
    std::fs::create_dir_all(&output_dir)?;

    let stem = pth_path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let onnx_path = output_dir.join(format!("{}.onnx", stem));

    let python = find_converter_python();
    let script = PathBuf::from("converter/convert.py");

    let output = Command::new(&python)
        .arg(&script)
        .arg("--input")
        .arg(pth_path)
        .arg("--output")
        .arg(&onnx_path)
        .arg("--type")
        .arg(subdir)
        .output()
        .map_err(|e| {
            UtaiError::Model(format!(
                "Failed to run converter (python={}): {}",
                python.display(),
                e
            ))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(UtaiError::Model(format!(
            "Conversion failed: {}",
            stderr.trim()
        )));
    }

    if !onnx_path.exists() {
        return Err(UtaiError::Model(
            "Conversion completed but output file not found".to_string(),
        ));
    }

    tracing::info!(
        "Converted {} -> {}",
        pth_path.display(),
        onnx_path.display()
    );
    Ok(onnx_path)
}

fn find_converter_python() -> PathBuf {
    // Converter can use embedded Python (has PyTorch CPU)
    let embedded = PathBuf::from("./python/python.exe");
    if embedded.exists() {
        return embedded;
    }

    // Fallback: system Python with torch installed
    PathBuf::from("python")
}
