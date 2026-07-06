use std::path::Path;

use super::ModelType;
use crate::{Result, UtaiError};

/// Prefer stderr for the error detail, but fall back to stdout — convert.py prints some
/// failures (e.g. unsupported checkpoint layouts) to stdout before exiting non-zero.
fn spawn_error_detail(output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr);
    if stderr.trim().is_empty() {
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    } else {
        stderr.trim().to_string()
    }
}

/// Convert a `.pth` checkpoint to ONNX at EXACTLY `onnx_path` — the caller owns the naming
/// (registry keys every artifact on the user-chosen model name, NOT the source file stem).
/// The converter writes the sidecar `.json` next to the output.
pub fn convert_pth_to_onnx(
    pth_path: &Path,
    onnx_path: &Path,
    model_type: &ModelType,
    app_dir: &Path,
) -> Result<()> {
    let python = crate::util::find_python(&app_dir.join("converter"), app_dir);
    let script = app_dir.join("converter").join("convert.py");

    let output = crate::util::python_command(&python)
        .arg(&script)
        .arg("--input")
        .arg(pth_path)
        .arg("--output")
        .arg(onnx_path)
        .arg("--type")
        .arg(super::type_subdir(model_type))
        .output()
        .map_err(|e| {
            UtaiError::Model(format!(
                "Failed to run converter (python={}): {}",
                python.display(),
                e
            ))
        })?;

    if !output.status.success() {
        return Err(UtaiError::Model(format!(
            "Conversion failed: {}",
            spawn_error_detail(&output)
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
    Ok(())
}

pub fn convert_index_to_npy(
    index_path: &Path,
    output_path: &Path,
    app_dir: &Path,
) -> Result<()> {
    let python = crate::util::find_python(&app_dir.join("converter"), app_dir);
    let script = app_dir.join("converter").join("extract_index.py");

    if !script.exists() {
        return Err(UtaiError::Model(format!(
            "Index extractor not found: {}",
            script.display()
        )));
    }

    let output = crate::util::python_command(&python)
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
        return Err(UtaiError::Model(format!(
            "Index extraction failed: {}",
            spawn_error_detail(&output)
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
    Ok(())
}

/// so-vits-svc companion assets: cluster kmeans .pt / feature-retrieval .pkl →
/// per-speaker .npy files in `outdir` (converter/export_cluster.py naming:
/// `<speaker_name>.centers.npy` / `<speaker_id>.index_vectors.npy` — exactly what the
/// SoVITS inference pipeline probes for).
pub fn convert_cluster_assets(input: &Path, outdir: &Path, app_dir: &Path) -> Result<()> {
    let python = crate::util::find_python(&app_dir.join("converter"), app_dir);
    let script = app_dir.join("converter").join("export_cluster.py");

    if !script.exists() {
        return Err(UtaiError::Model(format!(
            "Cluster converter not found: {}",
            script.display()
        )));
    }
    std::fs::create_dir_all(outdir)?;

    let output = crate::util::python_command(&python)
        .arg(&script)
        .arg("--input")
        .arg(input)
        .arg("--outdir")
        .arg(outdir)
        .output()
        .map_err(|e| {
            UtaiError::Model(format!(
                "Failed to run cluster converter (python={}): {}",
                python.display(),
                e
            ))
        })?;

    if !output.status.success() {
        return Err(UtaiError::Model(format!(
            "Cluster conversion failed: {}",
            spawn_error_detail(&output)
        )));
    }
    Ok(())
}

/// SoVITS shallow-diffusion attachment: the separate diffusion `.pt` (+ its `config.yaml`)
/// → `encoder.onnx` + `denoiser.onnx` + `diffusion.json` inside `outdir`
/// (converter/export_diffusion.py). `config` = the user-picked yaml; None lets the script
/// auto-resolve it next to the .pt (same stem → unique .yaml in dir → config.yaml), erroring
/// in Chinese when ambiguous.
pub fn convert_diffusion_assets(
    input: &Path,
    config: Option<&Path>,
    outdir: &Path,
    app_dir: &Path,
) -> Result<()> {
    let python = crate::util::find_python(&app_dir.join("converter"), app_dir);
    let script = app_dir.join("converter").join("export_diffusion.py");

    if !script.exists() {
        return Err(UtaiError::Model(format!(
            "Diffusion converter not found: {}",
            script.display()
        )));
    }
    std::fs::create_dir_all(outdir)?;

    let mut cmd = crate::util::python_command(&python);
    cmd.arg(&script).arg("--input").arg(input);
    if let Some(cfg) = config {
        cmd.arg("--config").arg(cfg);
    }
    let output = cmd.arg("--outdir").arg(outdir).output().map_err(|e| {
        UtaiError::Model(format!(
            "Failed to run diffusion converter (python={}): {}",
            python.display(),
            e
        ))
    })?;

    if !output.status.success() {
        return Err(UtaiError::Model(format!(
            "Diffusion conversion failed: {}",
            spawn_error_detail(&output)
        )));
    }
    if !outdir.join("diffusion.json").exists() {
        return Err(UtaiError::Model(
            "Diffusion conversion completed but diffusion.json not found".to_string(),
        ));
    }
    Ok(())
}

/// NSF-HiFiGAN vocoder resource (S40): a torch checkpoint ({'generator':sd}
/// deploy format, a SingingVocoders lightning training ckpt, or our training
/// weights/ snapshot) → `<stem>.onnx` + `<stem>.json` + `<stem>_mel.npy`
/// inside `outdir` (converter/export_nsf_hifigan.py --stem). `config` = the
/// user-picked config.json; None lets the script auto-resolve it next to the
/// checkpoint (original load_config semantics), erroring in Chinese when
/// absent. The script self-checks torch-vs-ORT numerics before returning
/// (mini_nsf / PC-NSF configs are rejected loudly — 一期 classic-only).
pub fn convert_vocoder_to_onnx(
    input: &Path,
    config: Option<&Path>,
    outdir: &Path,
    stem: &str,
    app_dir: &Path,
) -> Result<()> {
    let python = crate::util::find_python(&app_dir.join("converter"), app_dir);
    let script = app_dir.join("converter").join("export_nsf_hifigan.py");

    if !script.exists() {
        return Err(UtaiError::Model(format!(
            "Vocoder converter not found: {}",
            script.display()
        )));
    }
    std::fs::create_dir_all(outdir)?;

    let mut cmd = crate::util::python_command(&python);
    cmd.arg(&script).arg("--model").arg(input);
    if let Some(cfg) = config {
        cmd.arg("--config").arg(cfg);
    }
    let output = cmd
        .arg("--outdir")
        .arg(outdir)
        .arg("--stem")
        .arg(stem)
        .output()
        .map_err(|e| {
            UtaiError::Model(format!(
                "Failed to run vocoder converter (python={}): {}",
                python.display(),
                e
            ))
        })?;

    if !output.status.success() {
        return Err(UtaiError::Model(format!(
            "Vocoder conversion failed: {}",
            spawn_error_detail(&output)
        )));
    }
    for suffix in [".onnx", ".json", "_mel.npy"] {
        let p = outdir.join(format!("{}{}", stem, suffix));
        if !p.exists() {
            return Err(UtaiError::Model(format!(
                "Vocoder conversion completed but {} not found",
                p.display()
            )));
        }
    }
    Ok(())
}

// find_converter_python moved to crate::util::find_python (shared with commands/msst_models.rs).
