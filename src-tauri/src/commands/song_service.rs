use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use tauri::State;

use crate::AppState;

/// 模型工作流根目录 — 用户本地安装路径
const MODEL_WORKFLOW_ROOT: &str = "F:\\002 AIgequ";

/// 模型目录配置：(显示名, 子路径, 工作目录, 端口, 需要启动的 py 文件)
struct ModelSpec {
    display_name: &'static str,
    // 相对于 MODEL_WORKFLOW_ROOT 的完整子路径
    rel_path: &'static str,
    // 启动 app.py 时需要先 cd 进去的目录
    // （YuE-2 是 yuE-2/yuE-2，HeartMuLa/ACE-Step 是它们的子目录）
    app_py: &'static str,
    python_rel: &'static str, // 相对于 app_dir
    gradio_port: u16,
    health_path: &'static str, // 例如 /api/health 或根路径
}

const YUE2_SPEC: ModelSpec = ModelSpec {
    display_name: "YuE-2",
    rel_path: "yuE-2\\yuE-2",
    app_py: "app.py",
    python_rel: "py312\\python.exe",
    gradio_port: 7863,
    health_path: "/api/health",
};

const HEARTMULA_SPEC: ModelSpec = ModelSpec {
    display_name: "HeartMuLa",
    rel_path: "HeartMuLa-HL\\HeartMuLa",
    app_py: "app.py",
    python_rel: "py312\\python.exe",
    gradio_port: 7861,
    health_path: "/", // Gradio 根路径即可
};

const ACESTEP_SPEC: ModelSpec = ModelSpec {
    display_name: "ACE-Step 1.5",
    rel_path: "ACE-Step-1.5-XL-Turbo-comfy\\ACE-Step-1.5-XL-Turbo-comfy",
    app_py: "app.py",
    python_rel: "py312\\python.exe",
    gradio_port: 7861,
    health_path: "/", // 和 HeartMuLa 同端口，互斥启动
};

fn resolve_spec_for_model(model: &str) -> Option<ModelSpec> {
    let m = model.to_lowercase();
    if m.contains("yue") || m.contains("yue2") {
        Some(YUE2_SPEC)
    } else if m.contains("heart") || m.contains("heartmula") {
        Some(HEARTMULA_SPEC)
    } else if m.contains("ace") || m.contains("step") {
        Some(ACESTEP_SPEC)
    } else {
        None
    }
}

fn model_dir(spec: &ModelSpec) -> PathBuf {
    PathBuf::from(MODEL_WORKFLOW_ROOT).join(spec.rel_path)
}

async fn check_service_health(url: &str) -> bool {
    let client = match crate::download::client() {
        Ok(c) => c,
        Err(_) => return false,
    };

    match client
        .get(url)
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await
    {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    }
}

/// 根据 .bat 文件复刻环境变量 + 启动 app.py
async fn launch_model_service(spec: &ModelSpec) -> Result<String, String> {
    let app_dir = model_dir(spec);
    let python_path = app_dir.join(spec.python_rel);
    let app_py = app_dir.join(spec.app_py);

    tracing::info!("====== starting {} service ======", spec.display_name);
    tracing::info!("  app_dir: {}", app_dir.display());
    tracing::info!("  python:  {} (exists={})", python_path.display(), python_path.exists());
    tracing::info!("  app.py:  {} (exists={})", app_py.display(), app_py.exists());

    if !app_dir.exists() {
        return Err(format!(
            "{} 模型工作流目录不存在: {}\n请确认模型已安装在 {}",
            spec.display_name,
            app_dir.display(),
            MODEL_WORKFLOW_ROOT
        ));
    }
    if !python_path.exists() {
        return Err(format!("{} Python 不存在: {}", spec.display_name, python_path.display()));
    }
    if !app_py.exists() {
        return Err(format!("{} app.py 不存在: {}", spec.display_name, app_py.display()));
    }

    // 检查是否已经在运行
    let url = format!("http://127.0.0.1:{}{}", spec.gradio_port, spec.health_path);
    if check_service_health(&url).await {
        tracing::info!("  service already running ✓ ({})", url);
        return Ok(format!("http://127.0.0.1:{}", spec.gradio_port));
    }

    // 复刻 .bat 里的环境变量
    let python_dir = python_path.parent().unwrap();
    let ffmpeg_path = python_dir.join("ffmpeg").join("bin");
    let sox_path = python_dir.join("sox-14-4-2");
    let torch_home = app_dir.join("cache");
    let hf_home = app_dir.join("hf_download");

    tracing::info!("  launching {} (port={})...", spec.display_name, spec.gradio_port);

    let mut cmd = tokio::process::Command::new(&python_path);
    cmd.arg("-s")
        .arg(&app_py)
        .current_dir(&app_dir)
        .env("PYTHONHOME", "")
        .env("PYTHONPATH", "")
        .env("PYTHONEXECUTABLE", &python_path)
        .env("PYTHON_EXECUTABLE", &python_path)
        .env("PYTHON_BIN_PATH", &python_path)
        .env("PYTHON_LIB_PATH", python_dir.join("Lib").join("site-packages"))
        .env("TORCH_HOME", &torch_home)
        .env("HF_HOME", &hf_home)
        .env("HF_ENDPOINT", "https://hf-mirror.com")
        .env("GRADIO_TEMP_DIR", app_dir.join("tmp"))
        .env("CUDA_HOME", python_dir.join("Library"))
        .env(
            "PATH",
            format!(
                "{};{};{};{};{}",
                python_dir.display(),
                python_dir.join("Scripts").display(),
                ffmpeg_path.display(),
                sox_path.display(),
                std::env::var("PATH").unwrap_or_default()
            ),
        )
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    // HeartMuLa/ACE-Step 还需要 MODELSCOPE_CACHE
    if spec.display_name != "YuE-2" {
        cmd.env("MODELSCOPE_CACHE", app_dir.join("ms_download"));
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("启动 {} 失败: {}", spec.display_name, e))?;

    let pid = child.id();
    tracing::info!("  process started pid={:?}, waiting for health check...", pid);

    // 等待端口就绪（YuE-2 冷启动较慢，等待最多 120 秒）
    let max_wait = 120u64;
    for i in 0..max_wait {
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        if check_service_health(&url).await {
            tracing::info!("  {} started ✓ (port={}, pid={:?}, took={}s)",
                spec.display_name, spec.gradio_port, pid, i + 1);
            return Ok(format!("http://127.0.0.1:{}", spec.gradio_port));
        }
        if i % 10 == 0 && i > 0 {
            tracing::info!("  ... still waiting for {} to become ready ({}/{})", spec.display_name, i, max_wait);
        }
    }

    // 超时 — 杀掉进程并报错
    if let Some(p) = child.id() {
        let _ = child.kill().await;
        tracing::warn!("  {} startup timed out, killed pid={}", spec.display_name, p);
    }
    Err(format!(
        "{} 服务启动超时（{}秒内端口 {} 未就绪）。请手动运行 `开始.bat` 检查是否有错误输出。",
        spec.display_name, max_wait, spec.gradio_port
    ))
}

#[tauri::command]
pub async fn auto_start_service(
    _state: State<'_, Arc<AppState>>,
    model: String,
) -> Result<String, String> {
    tracing::info!("auto_start_service request: model={}", model);

    let spec = resolve_spec_for_model(&model)
        .ok_or_else(|| format!("未知模型: {}。支持: YuE-2, HeartMuLa, ACE-Step", model))?;

    launch_model_service(&spec).await
}
