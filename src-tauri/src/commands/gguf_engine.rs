//! GGUF 快速推理引擎管理 —— audiocpp_server.exe 生命周期 + HTTP 客户端
//!
//! 设计：
//! - 常驻进程：启动时自动拉起 audiocpp_server.exe，加载 yue2 GGUF 模型到 CUDA
//! - HTTP 客户端：通过 reqwest 调用 /v1/tasks/run，解析 SSE 流式进度
//! - 健康检查：启动后探测 /health 端点，确保服务可用
//! - 自动回退：GGUF 引擎不可用时，上层逻辑回退到 Python sidecar

use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::time::sleep;

use crate::AppState;

/// GGUF 引擎守护进程句柄
struct GgufDaemon {
    child: Child,
    pid: Option<u32>,
    port: u16,
    base_url: String,
}

/// 全局守护进程单槽（OnceLock + tokio::sync::Mutex 模式，对齐 song.rs）
fn gguf_daemon() -> &'static tokio::sync::Mutex<Option<GgufDaemon>> {
    static CELL: std::sync::OnceLock<tokio::sync::Mutex<Option<GgufDaemon>>> =
        std::sync::OnceLock::new();
    CELL.get_or_init(|| tokio::sync::Mutex::new(None))
}

/// audiocpp_server.exe 可执行文件路径解析
fn resolve_gguf_server_exe(state: &AppState) -> PathBuf {
    // 优先查找 models/song/yue2/ 目录下的 audiocpp_server.exe
    let yue2_dir = state.song_models_dir.join("yue2");
    let exe_in_yue2 = yue2_dir.join("audiocpp_server.exe");
    if exe_in_yue2.exists() {
        tracing::info!("Found audiocpp_server.exe in yue2 dir: {}", exe_in_yue2.display());
        return exe_in_yue2;
    }

    // 兜底：app_dir/audiocpp_server.exe（开发环境）
    let exe_in_app = state.app_dir.join("audiocpp_server.exe");
    tracing::warn!("GGUF server not found in yue2 dir, using fallback: {}", exe_in_app.display());
    exe_in_app
}

/// server.json 配置文件路径
fn resolve_gguf_config(state: &AppState) -> PathBuf {
    state.song_models_dir.join("yue2").join("server.json")
}

/// 健康检查：探测 /health 端点
async fn health_check(base_url: &str, max_attempts: u32) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {e}"))?;

    for attempt in 1..=max_attempts {
        tracing::debug!("Health check attempt {}/{}", attempt, max_attempts);
        match client.get(format!("{}/health", base_url)).send().await {
            Ok(resp) if resp.status().is_success() => {
                let body = resp.text().await.unwrap_or_default();
                tracing::info!("GGUF engine healthy: {}", body);
                return Ok(());
            }
            Ok(resp) => {
                tracing::warn!("Health check failed: HTTP {}", resp.status());
            }
            Err(e) => {
                tracing::warn!("Health check error: {}", e);
            }
        }
        if attempt < max_attempts {
            sleep(Duration::from_secs(2)).await;
        }
    }
    Err("Health check timeout after all attempts".to_string())
}

/// 启动 GGUF 守护进程
async fn start_daemon(state: &AppState, port: u16) -> Result<GgufDaemon, String> {
    let exe = resolve_gguf_server_exe(state);
    if !exe.exists() {
        return Err(format!("GGUF_SERVER_NOT_FOUND: {}", exe.display()));
    }

    let config = resolve_gguf_config(state);
    if !config.exists() {
        return Err(format!("GGUF_CONFIG_NOT_FOUND: {}", config.display()));
    }

    tracing::info!("Starting GGUF daemon: exe={}, config={}, port={}", 
                   exe.display(), config.display(), port);

    let mut cmd = Command::new(&exe);
    cmd.arg("--config").arg(&config)
       .arg("--port").arg(port.to_string())
       .arg("--log")
       .stdout(std::process::Stdio::piped())
       .stderr(std::process::Stdio::piped())
       .kill_on_drop(true);

    let mut child = cmd.spawn()
        .map_err(|e| format!("Failed to spawn audiocpp_server: {e}"))?;

    let pid = child.id();
    tracing::info!("GGUF daemon spawned: pid={:?}", pid);

    // 启动后台任务监听 stdout/stderr
    if let Some(stdout) = child.stdout.take() {
        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::debug!("[gguf-stdout] {}", line);
            }
        });
    }
    if let Some(stderr) = child.stderr.take() {
        tokio::spawn(async move {
            let reader = BufReader::new(stderr);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::info!("[gguf-stderr] {}", line);
            }
        });
    }

    let base_url = format!("http://127.0.0.1:{}", port);
    
    // 健康检查：最多等待 15 秒（模型加载需要时间）
    health_check(&base_url, 10).await?;

    Ok(GgufDaemon {
        child,
        pid,
        port,
        base_url,
    })
}

/// 确保守护进程运行（公开接口）
pub async fn ensure_gguf_daemon(state: &AppState, port: u16) -> Result<String, String> {
    let mut slot = gguf_daemon().lock().await;
    
    if let Some(daemon) = slot.as_mut() {
        // 检查现有进程是否存活
        match daemon.child.try_wait() {
            Ok(None) => {
                // 进程仍在运行，返回 base_url
                return Ok(daemon.base_url.clone());
            }
            Ok(Some(status)) => {
                tracing::warn!("GGUF daemon exited unexpectedly: {:?}", status);
            }
            Err(e) => {
                tracing::warn!("GGUF daemon check failed: {}", e);
            }
        }
        // 进程已死，清理槽位
        *slot = None;
    }

    // 启动新守护进程
    let daemon = start_daemon(state, port).await?;
    let base_url = daemon.base_url.clone();
    *slot = Some(daemon);
    
    Ok(base_url)
}

/// 关闭守护进程
pub async fn shutdown_gguf_daemon() -> Result<(), String> {
    let mut slot = gguf_daemon().lock().await;
    if let Some(mut daemon) = slot.take() {
        tracing::info!("Shutting down GGUF daemon: pid={:?}", daemon.pid);
        let _ = daemon.child.kill().await;
        let _ = daemon.child.wait().await;
    }
    Ok(())
}

/// YuE2 生成请求参数
#[derive(Debug, Serialize)]
pub struct Yue2GenerateRequest {
    pub model: String,
    pub request: Yue2RequestBody,
}

#[derive(Debug, Serialize)]
pub struct Yue2RequestBody {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    pub options: Yue2Options,
}

#[derive(Debug, Serialize)]
pub struct Yue2Options {
    pub style: String,
    pub lyrics: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cot: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_inference_steps: Option<u32>,
}

/// SSE 事件解析
#[derive(Debug, Deserialize)]
struct SseEvent {
    #[serde(default)]
    event: String,
    #[serde(default)]
    data: String,
}

/// 生成进度事件载荷（对齐 song.rs 的 SongProgress）
#[derive(Debug, Clone, Serialize)]
pub struct GgufProgress {
    pub stage: String,
    pub current: u64,
    pub total: u64,
    pub stem_label: String,
}

/// 通过 GGUF 引擎生成音乐（HTTP + SSE 流式进度）
pub async fn generate_via_gguf(
    state: &AppState,
    _app: &tauri::AppHandle,
    request: Yue2GenerateRequest,
) -> Result<Vec<u8>, String> {
    // 确保守护进程运行
    let base_url = ensure_gguf_daemon(state, 8080).await?;
    
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))  // 5 分钟超时（生成可能很长）
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {e}"))?;

    let url = format!("{}/v1/tasks/run", base_url);
    tracing::info!("Sending GGUF generation request to: {}", url);

    let response = client
        .post(&url)
        .json(&request)
        .send()
        .await
        .map_err(|e| format!("HTTP request failed: {e}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("GGUF generation failed: HTTP {} - {}", status, body));
    }

    // 解析 JSON 响应（audiocpp_server 返回嵌套结构）
    #[derive(Deserialize)]
    struct GgufResponse {
        data: GgufData,
    }
    
    #[derive(Deserialize)]
    struct GgufData {
        audio: String,  // base64 编码的 WAV
    }

    let resp_json = response.json::<GgufResponse>().await
        .map_err(|e| format!("Failed to parse response: {e}"))?;

    // Base64 解码
    let audio_bytes = base64_decode(&resp_json.data.audio)
        .map_err(|e| format!("Failed to decode audio: {e}"))?;

    tracing::info!("GGUF generation completed: {} bytes", audio_bytes.len());
    Ok(audio_bytes)
}

fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    use base64::{engine::general_purpose, Engine as _};
    general_purpose::STANDARD.decode(s)
        .map_err(|e| format!("Base64 decode error: {e}"))
}
