use std::path::PathBuf;
use std::process::Command;

use super::ModelType;
use crate::{Result, UtaiError};

pub fn convert_pth_to_onnx(
    pth_path: &PathBuf,
    models_dir: &PathBuf,
    model_type: &ModelType,
    app_dir: &PathBuf,
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

    let python = crate::util::find_python(&app_dir.join("converter"), app_dir);
    let script = app_dir.join("converter").join("convert.py");

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

pub fn convert_index_to_npy(
    index_path: &PathBuf,
    output_path: &PathBuf,
    app_dir: &PathBuf,
) -> Result<PathBuf> {
    let python = crate::util::find_python(&app_dir.join("converter"), app_dir);
    let script = app_dir.join("converter").join("extract_index.py");

    if !script.exists() {
        return Err(UtaiError::Model(format!(
            "Index extractor not found: {}",
            script.display()
        )));
    }

    let output = Command::new(&python)
        .arg(&script)
        .arg("--input")
        .arg(index_path)
        .arg("--output")
        .arg(output_path)
        .output()
        .map_err(|e| {
            UtaiError::Model(format!(
                "Failed to run index extractor (python={}): {}",
                python.display(),
                e
            ))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(UtaiError::Model(format!(
            "Index extraction failed: {}",
            stderr.trim()
        )));
    }

    if !output_path.exists() {
        return Err(UtaiError::Model(
            "Index extraction completed but .npy file not found".to_string(),
        ));
    }

    tracing::info!(
        "Extracted index {} -> {}",
        index_path.display(),
        output_path.display()
    );
    Ok(output_path.clone())
}

// find_converter_python moved to crate::util::find_python (shared with commands/msst_models.rs).
