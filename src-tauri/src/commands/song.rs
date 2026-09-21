//! 歌曲制作（Song Studio）后端命令 —— 音乐模型下载/管理 + 生成历史持久化 + 外部推理服务对接。
//!
//! 设计对齐（详见 `.trae/documents/song-production-plan.md`）：
//! - 音乐模型落盘 `<data>/models/song/`，与 `msst`/`amt` 并列，下载统一走 `download.rs`
//!   （.part 续传 + 镜像轮换 + stall 看门狗 + sha256 校验后改名）。
//! - 生成历史落 `<data>/song_history.json`，直接照抄 `storage.rs` 的 `preset_file` /
//!   `load_workflow_presets` / `save_workflow_preset` 范式。
//! - 阶段 A：外部推理服务（YuE2 / ACE-Step 均为 Linux/Python 栈）默认不自动拉起，
//!   仅探测可用性（`song_service_probe`）与转发生成请求（`song_generate`）。
//! - 下载进度事件名 `song-download-progress`（独立于 MSST/AMT，避免卡片状态串台）。

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use tauri::{Emitter, State};

use crate::AppState;

/// 规划 13.3：当前运行中的 song sidecar 进程 pid（用于 song_cancel 强制终止）。
/// 同一时刻至多一个 song_generate 在跑（前端任务队列串行化），单槽即可。
static SONG_SIDECAR_PID: Mutex<Option<u32>> = Mutex::new(None);

/// 常驻守护进程句柄：进程 + stdin 写端 + stdout 行读取器。
/// 仅当用户在设置里打开「歌曲生成常驻模式」时存活；空闲超过阈值后 Python 侧会把
/// 权重搬回 CPU 释放显存（进程仍在，省掉 Python 启动 + torch import + 权重反序列化）。
struct DaemonHandle {
    child: tokio::process::Child,
    stdin: tokio::process::ChildStdin,
    stdout: tokio::io::Lines<tokio::io::BufReader<tokio::process::ChildStdout>>,
    pid: Option<u32>,
    idle_timeout: u32,
}

/// 守护进程单槽。必须用 tokio 的异步锁：持锁期间要 await（写 stdin / 读 stdout），
/// std::sync::Mutex 的 guard 不能跨 await 点。外面套 OnceLock 是因为
/// `tokio::sync::Mutex::new` 不是 const fn，没法直接写在 static 初始化式里。
fn song_daemon() -> &'static tokio::sync::Mutex<Option<DaemonHandle>> {
    static CELL: std::sync::OnceLock<tokio::sync::Mutex<Option<DaemonHandle>>> =
        std::sync::OnceLock::new();
    CELL.get_or_init(|| tokio::sync::Mutex::new(None))
}

/// 常驻状态事件载荷（对应 Python 的 `@@KEEPALIVE@@`）。
/// state: ready | vram_released | vram_restored | shutdown
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SongResidentState {
    state: String,
    detail: String,
}

/// 常驻进程对外状态，供设置页展示「当前是否有常驻进程」。
#[derive(Debug, Clone, Serialize)]
pub struct SongResidentStatus {
    pub running: bool,
    pub pid: Option<u32>,
    pub idle_timeout: u32,
}

/// 一次常驻生成的结果。必须区分两种失败：
/// - `Done(Err)`：守护进程正常跑完但业务失败（模型报错等）→ 直接返回，不要重跑；
/// - `Unavailable`：守护进程拉不起来或中途死了 → 回退一次性 spawn。
/// 合成一个 Result 的话，业务失败会被误当成"常驻不可用"而白跑一遍。
enum ResidentAttempt {
    Done(Result<serde_json::Value, String>),
    Unavailable(String),
}

async fn ensure_daemon(
    state: &AppState,
    idle_timeout: u32,
) -> Result<(), String> {
    let mut slot = song_daemon().lock().await;
    if let Some(handle) = slot.as_mut() {
        if handle.idle_timeout == idle_timeout {
            match handle.child.try_wait() {
                Ok(None) => return Ok(()),
                Ok(Some(status)) => {
                    tracing::warn!("Daemon exited unexpectedly: {:?}", status);
                }
                Err(e) => {
                    tracing::warn!("Daemon check failed: {}", e);
                }
            }
        } else {
            tracing::info!("Idle timeout changed, restarting daemon");
        }
        let _ = shutdown_daemon_locked(&mut *slot).await;
    }

    let sidecar_dir = resolve_song_sidecar_dir(state);
    let sidecar_script = sidecar_dir.join("song_sidecar.py");
    let python = resolve_song_python(&state.app_dir);

    if !sidecar_script.exists() {
        return Err(format!(
            "SONG_SIDECAR_NOT_FOUND: {}",
            sidecar_script.display()
        ));
    }

    tracing::info!("Starting resident daemon (idle_timeout={}s)...", idle_timeout);
    // ⚠️ 同 one-shot：缓存/字节码写入必须避开 src-tauri（tauri dev 监视器会杀应用）
    let ace_work_dir = state
        .song_models_dir
        .parent()
        .and_then(std::path::Path::parent)
        .map(|d| d.join("cache").join("acestep_work"))
        .unwrap_or_else(|| std::env::temp_dir().join("muno_acestep_work"));
    let _ = std::fs::create_dir_all(&ace_work_dir);
    let mut child = tokio::process::Command::new(&python)
        .arg("-u")
        .arg(&sidecar_script)
        .arg("--daemon")
        .arg("--idle-timeout")
        .arg(idle_timeout.to_string())
        .current_dir(&sidecar_dir)
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("ACESTEP_PROJECT_ROOT", ace_work_dir.to_string_lossy().as_ref())
        .env("SONG_MODELS_DIR", state.song_models_dir.to_string_lossy().as_ref())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("SONG_DAEMON_SPAWN_FAILED: {e}"))?;

    let pid = child.id();
    let stdin = child.stdin.take().ok_or("cannot take daemon stdin")?;
    let stdout = child.stdout.take().ok_or("cannot take daemon stdout")?;
    use tokio::io::{AsyncBufReadExt, BufReader};
    let stdout = BufReader::new(stdout).lines();

    tracing::info!("Daemon spawned, pid={:?}", pid);
    *slot = Some(DaemonHandle {
        child,
        stdin,
        stdout,
        pid,
        idle_timeout,
    });

    Ok(())
}

async fn shutdown_daemon_locked(slot: &mut Option<DaemonHandle>) -> Result<(), String> {
    if let Some(mut handle) = slot.take() {
        tracing::info!("Shutting down daemon pid={:?}", handle.pid);
        use tokio::io::AsyncWriteExt;
        let shutdown_cmd = r#"{"cmd":"shutdown"}"#.to_string() + "\n";
        let _ = handle.stdin.write_all(shutdown_cmd.as_bytes()).await;
        let _ = handle.stdin.flush().await;
        drop(handle.stdin);
        let timeout = tokio::time::Duration::from_secs(5);
        let _ = tokio::time::timeout(timeout, handle.child.wait()).await;
        let _ = handle.child.kill().await;
    }
    Ok(())
}

/// 守护进程失败时，把攒下来的非协议 stdout 行拼成诊断串。
/// 只取尾部若干行：日志可能很长，全塞进错误信息里对排查没帮助。
fn tail_lines(lines: &[String]) -> String {
    const KEEP: usize = 20;
    let start = lines.len().saturating_sub(KEEP);
    if lines.is_empty() {
        "<empty>".to_string()
    } else {
        lines[start..].join("\n")
    }
}

async fn dispatch_to_daemon(
    app: &tauri::AppHandle,
    config_path: &std::path::Path,
    timeout_secs: u64,
) -> ResidentAttempt {
    let mut slot = song_daemon().lock().await;
    let handle = match slot.as_mut() {
        Some(h) => h,
        None => return ResidentAttempt::Unavailable("daemon not running".into()),
    };

    use tokio::io::AsyncWriteExt;
    let generate_cmd = serde_json::json!({
        "cmd": "generate",
        "config_path": config_path.to_string_lossy()
    });
    let line = serde_json::to_string(&generate_cmd).unwrap() + "\n";
    if let Err(e) = handle.stdin.write_all(line.as_bytes()).await {
        tracing::warn!("Failed to write generate command: {}", e);
        return ResidentAttempt::Unavailable(format!("stdin write failed: {e}"));
    }
    if let Err(e) = handle.stdin.flush().await {
        tracing::warn!("Failed to flush stdin: {}", e);
        return ResidentAttempt::Unavailable(format!("stdin flush failed: {e}"));
    }

    let start = std::time::Instant::now();
    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(timeout_secs);
    let mut stdout_lines: Vec<String> = Vec::new();

    let result = loop {
        let line_fut = handle.stdout.next_line();
        let line_result = tokio::time::timeout_at(deadline, line_fut).await;
        match line_result {
            Err(_) => {
                return ResidentAttempt::Done(Err(format!(
                    "SONG_DAEMON_TIMEOUT: no result after {}s; daemon stdout tail: {}",
                    timeout_secs,
                    tail_lines(&stdout_lines)
                )));
            }
            Ok(Err(e)) => {
                return ResidentAttempt::Unavailable(format!(
                    "stdout read failed: {e}; tail: {}",
                    tail_lines(&stdout_lines)
                ));
            }
            Ok(Ok(None)) => {
                return ResidentAttempt::Unavailable(format!(
                    "daemon stdout closed; tail: {}",
                    tail_lines(&stdout_lines)
                ));
            }
            Ok(Ok(Some(line))) => {
                if let Some(rest) = line.strip_prefix("@@PROGRESS@@") {
                    if let Ok(payload) = serde_json::from_str::<serde_json::Value>(rest) {
                        let stage = payload.get("stage").and_then(|v| v.as_str()).unwrap_or("");
                        let progress_float = payload.get("progress").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        let progress = (progress_float * 100.0) as u64;
                        let total = payload.get("total").and_then(|v| v.as_u64()).unwrap_or(100);
                        let stem_label = payload.get("stem_label").and_then(|v| v.as_str()).unwrap_or("");
                        let _ = app.emit(
                            "song-generation-progress",
                            SongProgress {
                                stage: stage.to_string(),
                                current: progress,
                                total,
                                stem_label: stem_label.to_string(),
                            },
                        );
                    }
                } else if let Some(rest) = line.strip_prefix("@@RESULT@@") {
                    if let Ok(payload) = serde_json::from_str::<serde_json::Value>(rest.trim_start()) {
                        break payload;
                    }
                } else if let Some(rest) = line.strip_prefix("@@KEEPALIVE@@") {
                    if let Ok(payload) = serde_json::from_str::<SongResidentState>(rest) {
                        tracing::debug!("daemon keepalive: state={}, detail={}", payload.state, payload.detail);
                    }
                } else if !line.is_empty() {
                    tracing::info!("daemon-stdout: {}", line);
                    stdout_lines.push(line);
                }
            }
        }
    };

    let elapsed = start.elapsed().as_secs_f64();
    tracing::info!("Daemon job finished in {:.2}s", elapsed);

    ResidentAttempt::Done(Ok(result))
}

#[tauri::command]
pub async fn song_resident_status(_state: State<'_, Arc<AppState>>) -> Result<SongResidentStatus, String> {
    let slot = song_daemon().lock().await;
    match slot.as_ref() {
        Some(handle) => Ok(SongResidentStatus {
            running: true,
            pid: handle.pid,
            idle_timeout: handle.idle_timeout,
        }),
        None => Ok(SongResidentStatus {
            running: false,
            pid: None,
            idle_timeout: 0,
        }),
    }
}

#[tauri::command]
pub async fn song_resident_shutdown(_state: State<'_, Arc<AppState>>) -> Result<(), String> {
    let mut slot = song_daemon().lock().await;
    shutdown_daemon_locked(&mut *slot).await
}

#[tauri::command]
pub fn song_cancel() -> Result<(), String> {
    let pid = SONG_SIDECAR_PID.lock().map_err(|e| e.to_string())?.take();
    match pid {
        Some(pid) => super::amt::kill_pid(pid).map_err(|e| format!("SONG_KILL_FAILED: {e}")),
        None => Ok(()),
    }
}

/// 与 `storage::data_root` 同语义：`<cache_dir>/..`，兜底 `app_dir/data`。
/// song.rs 独立实现一份，避免跨模块私有依赖。
fn data_root(state: &AppState) -> PathBuf {
    state
        .cache_dir
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| state.app_dir.join("data"))
}

fn song_history_file(state: &AppState) -> PathBuf {
    data_root(state).join("song_history.json")
}

/// 文件名清洗：去掉 Windows/跨平台非法字符、路径分隔与控制字符，空白折叠为 `-`。
/// 用于由用户输入的歌曲名生成目录名，避免下游 `join` 越界或建目录失败。
fn sanitize_filename(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| match c {
            '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*' | '.' | '\0' => '-',
            c if c.is_control() || c.is_whitespace() => '-',
            c => c,
        })
        .collect();
    let trimmed = cleaned.trim_matches('-');
    if trimmed.is_empty() {
        "untitled".to_string()
    } else {
        trimmed.to_string()
    }
}

fn timestamp_slug() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}

/// 生成产物默认输出目录：`<data>/songs/<sanitized(song_name)>/`；song_name 为空时用时间戳。
fn default_song_output_dir(state: &AppState, song_name: &str) -> Result<PathBuf, String> {
    let slug = if song_name.trim().is_empty() {
        format!("untitled-{}", timestamp_slug())
    } else {
        sanitize_filename(song_name)
    };
    let dir = data_root(state).join("songs").join(slug);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// 解析 song_sidecar.py 所在目录。优先查找 data_dir，兜底到 app_dir。
fn resolve_song_sidecar_dir(state: &AppState) -> PathBuf {
    // 按优先级尝试多个可能位置（开发环境 & 打包环境）
    let candidates = [
        // 1. src-tauri/python_sidecar（开发环境：cargo tauri dev 的 src-tauri 子目录）
        state.app_dir.join("src-tauri").join("python_sidecar"),
        // 2. python_sidecar 直接在 app_dir 下（打包环境 / 已复制到根）
        state.app_dir.join("python_sidecar"),
        // 3. data/python_sidecar（类似 AMT 的 data 布局）
        state.app_dir.join("data").join("python_sidecar"),
        // 4. 缓存目录父级下（自定义安装）
        state.cache_dir.parent().unwrap_or(&state.cache_dir).join("python_sidecar"),
    ];
    for dir in &candidates {
        if dir.join("song_sidecar.py").exists() {
            tracing::info!("  found song_sidecar at: {}", dir.display());
            return dir.clone();
        }
    }
    // 找不到也返回最后一个候选，让上层报清晰的错误
    candidates[0].clone()
}

/// 解析用于 Song Sidecar 的 Python 解释器。
/// 优先使用专用的歌曲生成 venv（runtime/python310），其中已安装 torch + 三个模型库。
fn resolve_song_python(app_dir: &PathBuf) -> PathBuf {
    // 开发环境：src-tauri/runtime/python310/Scripts/python.exe
    let dev_python = app_dir.join("src-tauri").join("runtime").join("python310").join("Scripts").join("python.exe");
    if dev_python.exists() {
        tracing::info!("Using song venv (dev): {}", dev_python.display());
        return dev_python;
    }
    
    // 打包环境：runtime/python310/Scripts/python.exe
    let prod_python = app_dir.join("runtime").join("python310").join("Scripts").join("python.exe");
    if prod_python.exists() {
        tracing::info!("Using song venv (prod): {}", prod_python.display());
        return prod_python;
    }
    
    // 兜底：使用 converter_python（但可能缺少 torch 等依赖）
    tracing::warn!("Song venv not found, falling back to converter_python");
    crate::pyenv::converter_python(app_dir)
}

// ---------------------------------------------------------------------------
// 音乐模型目录 / 列表 / 下载 / 删除
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn get_song_models_dir(state: State<'_, Arc<AppState>>) -> Result<String, String> {
    let dir = state.song_models_dir.clone();
    std::fs::create_dir_all(&dir).map_err(|e| format!("Failed to create song models dir: {e}"))?;
    Ok(dir.to_string_lossy().to_string())
}

/// 已下载音乐模型的简单清单：`<filename, size>`。前端与 `SONG_MODEL_CATALOG` 比对得出
/// 安装状态。递归扫描，扩展名白名单只认模型类产物。
#[tauri::command]
pub fn list_song_models(state: State<'_, Arc<AppState>>) -> Result<Vec<SongModelFile>, String> {
    let dir = &state.song_models_dir;
    let mut models = Vec::new();
    if !dir.exists() {
        return Ok(models);
    }
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(current) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    let ext = path
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("")
                        .to_lowercase();
                    if ["safetensors", "ckpt", "pt", "pth", "onnx", "whl", "bin", "yaml", "json", "sf2", "mid", "wav"]
                        .contains(&ext.as_str())
                    {
                        let rel = path.strip_prefix(dir).unwrap_or(&path);
                        let filename = rel.to_string_lossy().to_string().replace('\\', "/");
                        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                        models.push(SongModelFile { filename, size });
                    }
                }
            }
        }
    }
    models.sort_by(|a, b| a.filename.cmp(&b.filename));
    Ok(models)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SongModelFile {
    pub filename: String,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize)]
struct SongDownloadProgress {
    id: String,
    downloaded: u64,
    total: u64,
    stage: String,
}

#[tauri::command]
pub async fn download_song_model(
    app: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    urls: Vec<String>,
    id: String,
    filename: String,
    sha256: Option<String>,
) -> Result<String, String> {
    let dir = state.song_models_dir.clone();
    std::fs::create_dir_all(&dir).map_err(|e| format!("Failed to create models dir: {e}"))?;

    let dest = dir.join(&filename);
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let client = crate::download::client().map_err(|e| e.to_string())?;
    let app_emit = app.clone();
    let model_id = id.clone();
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut last_emit: u64 = 0;

    crate::download::download(
        &client,
        &crate::download::DownloadRequest {
            urls,
            dest: dest.clone(),
            sha256,
            expected_size: None,
        },
        &cancel,
        move |done, total| {
            // Range 续传时 done 会回落（新起点更小），重置 high-water 以免进度条倒退。
            if done < last_emit {
                last_emit = 0;
            }
            if done.saturating_sub(last_emit) > 1_000_000 || Some(done) == total {
                last_emit = done;
                let _ = app_emit.emit(
                    "song-download-progress",
                    SongDownloadProgress {
                        id: model_id.clone(),
                        downloaded: done,
                        total: total.unwrap_or(0),
                        stage: "download".into(),
                    },
                );
            }
        },
    )
    .await
    .map_err(|e| e.to_string())?;

    tracing::info!("Downloaded song model: {}", filename);
    Ok(dest.to_string_lossy().to_string())
}

#[tauri::command]
pub fn delete_song_model(state: State<'_, Arc<AppState>>, filename: String) -> Result<(), String> {
    let dir = &state.song_models_dir;
    // ⛔ 路径穿越防护：拒绝任何解析后越出 song_models_dir 的路径（`..` / 绝对路径等）。
    let path = dir.join(&filename);
    let canon_dir = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.clone());
    let canon_path = match std::fs::canonicalize(&path) {
        Ok(p) => p,
        // 目标不存在（或已被删除）——幂等地视为成功，避免前端反复删除报错。
        Err(_) => return Ok(()),
    };
    if !canon_path.starts_with(&canon_dir) {
        return Err("SONG_DELETE_REJECTED: path escapes models dir".into());
    }
    if path.is_dir() {
        std::fs::remove_dir_all(&path).map_err(|e| format!("SONG_DELETE_FAILED: {e}"))?;
    } else {
        std::fs::remove_file(&path).map_err(|e| format!("SONG_DELETE_FAILED: {e}"))?;
    }
    // 清理空父目录（不越出 song_models_dir）。
    let mut parent = path.parent();
    while let Some(p) = parent {
        if p == dir || !p.starts_with(dir) {
            break;
        }
        if let Ok(entries) = std::fs::read_dir(p) {
            if entries.count() == 0 {
                let _ = std::fs::remove_dir(p);
            }
        }
        parent = p.parent();
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 生成历史持久化（照抄 storage.rs 的 preset_file 范式）
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn load_song_history(state: State<'_, Arc<AppState>>) -> Result<String, String> {
    let path = song_history_file(&state);
    match std::fs::read_to_string(&path) {
        Ok(s) => Ok(s),
        Err(_) => Ok("[]".into()),
    }
}

/// `entries` 为前端已序列化的完整历史数组（`Vec<SongHistoryEntry>`），整体覆盖写入。
#[tauri::command]
pub fn save_song_history(
    state: State<'_, Arc<AppState>>,
    entries: serde_json::Value,
) -> Result<(), String> {
    let root = data_root(&state);
    std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
    let path = song_history_file(&state);
    let out = serde_json::to_string_pretty(&entries).map_err(|e| e.to_string())?;
    // 原子写入：tmp + 同目录 rename，崩溃时不会留下半截 JSON（下一轮 load 解析即失败）。
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, out).map_err(|e| e.to_string())?;
    crate::util::rename_with_retry(&tmp, &path, "SONG_HISTORY_WRITE")
}

// ---------------------------------------------------------------------------
// 外部推理服务对接（阶段 A：只探测 + 转发，不自动拉起 Python 服务）
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct SongServiceProbe {
    pub ok: bool,
    pub status: u16,
    pub message: String,
}

#[tauri::command]
pub async fn song_service_probe(url: String) -> Result<SongServiceProbe, String> {
    let client = crate::download::client().map_err(|e| e.to_string())?;
    let base = url.trim().trim_end_matches('/').to_string();
    if base.is_empty() {
        return Ok(SongServiceProbe {
            ok: false,
            status: 0,
            message: "SONG_SERVICE_URL_EMPTY".into(),
        });
    }
    match client
        .get(format!("{base}/"))
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
    {
        Ok(resp) => Ok(SongServiceProbe {
            ok: resp.status().is_success(),
            status: resp.status().as_u16(),
            message: resp.status().to_string(),
        }),
        Err(e) => Ok(SongServiceProbe {
            ok: false,
            status: 0,
            message: format!("{e}"),
        }),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SongGenRequest {
    // Common parameters
    pub model: String,
    pub song_name: String,
    pub lyrics: String,
    pub prompt: String,
    pub audio_duration: f64,
    pub seed: Option<i64>,
    pub cfg_scale: Option<f64>,
    pub format: String,
    pub output_dir: String,
    
    // YuE-2 specific parameters
    pub cot: Option<String>,
    pub abc: Option<String>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub repetition_penalty: Option<f64>,
    pub penalty_window: Option<u32>,
    pub min_tokens: Option<u32>,
    pub max_tokens: Option<u32>,
    pub ode_steps: Option<u32>,
    pub ode_method: Option<String>,
    
    // ACE-Step specific parameters
    pub vram_mode: Option<String>,
    pub inference_steps: Option<u32>,
    pub guidance_scale: Option<f64>,
    pub audio_format: Option<String>,
    pub mp3_bitrate: Option<String>,
    pub mp3_sample_rate: Option<u32>,
    pub task: Option<String>,
    pub repaint_start: Option<f64>,
    pub repaint_end: Option<f64>,
    pub src_audio_path: Option<String>,
    // ACE-Step v1.5 多轨任务（lego/extract/complete）扩展字段
    pub track_name: Option<String>,
    pub complete_track_classes: Option<Vec<String>>,
    pub audio_cover_strength: Option<f64>,
    pub edit_target_prompt: Option<String>,
    pub edit_target_lyrics: Option<String>,
    pub edit_n_min: Option<u32>,
    pub edit_n_max: Option<u32>,
    pub edit_n_avg: Option<u32>,
    pub batch_size: Option<u32>,
    
    // HeartMuLa specific parameters
    pub topk: Option<u32>,
    
    // Output products (applies to all models)
    pub want_midi: Option<bool>,
    pub want_stems: Option<bool>,
    pub want_lrc: Option<bool>,
}

impl Default for SongGenRequest {
    fn default() -> Self {
        Self {
            model: "yue2".into(),
            song_name: String::new(),
            lyrics: String::new(),
            prompt: String::new(),
            audio_duration: 60.0,
            seed: None,
            cfg_scale: None,
            format: "wav".into(),
            output_dir: String::new(),
            
            // YuE-2 defaults
            cot: None,
            abc: None,
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            penalty_window: None,
            min_tokens: None,
            max_tokens: None,
            ode_steps: None,
            ode_method: None,
            
            // ACE-Step defaults
            vram_mode: None,
            inference_steps: None,
            guidance_scale: None,
            audio_format: None,
            mp3_bitrate: None,
            mp3_sample_rate: None,
            task: None,
            repaint_start: None,
            repaint_end: None,
            src_audio_path: None,
            track_name: None,
            complete_track_classes: None,
            audio_cover_strength: None,
            edit_target_prompt: None,
            edit_target_lyrics: None,
            edit_n_min: None,
            edit_n_max: None,
            edit_n_avg: None,
            batch_size: None,
            
            // HeartMuLa defaults
            topk: None,
            
            // Output products defaults
            want_midi: None,
            want_stems: None,
            want_lrc: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SongOutput {
    pub label: String,
    pub audio_path: Option<String>,
    pub midi_path: Option<String>,
    pub lrc_path: Option<String>,
    pub abc_path: Option<String>,
    pub stems: Option<std::collections::HashMap<String, String>>,
    pub processing_time_secs: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct SongProgress {
    stage: String,
    current: u64,
    total: u64,
    stem_label: String,
}

/// 使用 Python Sidecar 生成歌曲（ACE-Step / YuE-2）。
/// 支持多轨道输出（vocals, accompaniment, drums, bass）、MIDI生成、歌词时间轴。
/// 进度通过 @@PROGRESS@@ 协议实时推送到前端，最终结果通过 @@RESULT@@ 返回。
#[tauri::command]
pub async fn song_generate(
    app: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    mut req: SongGenRequest,
) -> Result<Vec<SongOutput>, String> {
    tracing::info!("======== song_generate start ========");
    tracing::info!("  model={}, song_name={}", req.model, req.song_name);

    let output_dir = if req.output_dir.trim().is_empty() {
        default_song_output_dir(&state, &req.song_name)?
    } else {
        std::fs::create_dir_all(&req.output_dir).map_err(|e| e.to_string())?;
        PathBuf::from(&req.output_dir)
    };
    req.output_dir = output_dir.to_string_lossy().to_string();

    let models_dir = state.song_models_dir.to_string_lossy().to_string();
    let mut config = serde_json::to_value(&req).map_err(|e| format!("Failed to serialize request: {}", e))?;
    if let Some(obj) = config.as_object_mut() {
        obj.insert("models_dir".to_string(), serde_json::json!(models_dir));
    }

    let cache_dir = state.cache_dir.clone();
    std::fs::create_dir_all(&cache_dir).map_err(|e| format!("SONG_CACHE_DIR_ERROR: {e}"))?;
    let config_filename = format!("song_gen_config_{}.json", uuid::Uuid::new_v4());
    let config_path = cache_dir.join(&config_filename);
    let config_text = serde_json::to_string_pretty(&config)
        .map_err(|e| format!("SONG_CONFIG_ERROR: {e}"))?;
    std::fs::write(&config_path, &config_text)
        .map_err(|e| format!("SONG_CONFIG_WRITE_FAILED: {e}"))?;

    tracing::info!("  config written to: {}", config_path.display());

    let result_json = if req.model.starts_with("yue2") {
        tracing::info!("  yue2 model detected, trying GGUF fast path...");
        match try_gguf_generate(&app, &state, &req).await {
            Ok(audio_bytes) => {
                tracing::info!("  GGUF generation succeeded, saving audio...");
                let audio_filename = format!("{}_yue2.wav", sanitize_filename(&req.song_name));
                let audio_path = output_dir.join(&audio_filename);
                std::fs::write(&audio_path, &audio_bytes)
                    .map_err(|e| format!("Failed to write audio: {e}"))?;
                
                serde_json::json!({
                    "status": "success",
                    "data": {
                        "audio_path": audio_path.to_string_lossy().to_string()
                    }
                })
            }
            Err(e) => {
                tracing::warn!("  GGUF generation failed ({}), falling back to Python", e);
                run_generation_fallback(&app, &state, &config_path).await?
            }
        }
    } else {
        run_generation_fallback(&app, &state, &config_path).await?
    };

    let _ = std::fs::remove_file(&config_path);

    let data = result_json.get("data").ok_or_else(|| {
        "SONG_NO_DATA: @@RESULT@@ missing 'data' field".to_string()
    })?;

    let get_opt_str = |key: &str| -> Option<String> {
        data.get(key).and_then(|v| v.as_str()).map(|s| s.to_string())
    };
    let get_opt_map = |key: &str| -> Option<HashMap<String, String>> {
        data.get(key).and_then(|v| v.as_object()).map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        })
    };

    let processing_time = data.get("processing_time_secs").and_then(|v| v.as_f64());
    
    let main_output = SongOutput {
        label: req.song_name.clone(),
        audio_path: get_opt_str("audio_path"),
        midi_path: get_opt_str("midi_path"),
        lrc_path: get_opt_str("lrc_path"),
        abc_path: get_opt_str("abc_path"),
        stems: get_opt_map("stems"),
        processing_time_secs: processing_time,
    };

    Ok(vec![main_output])
}

async fn try_gguf_generate(
    app: &tauri::AppHandle,
    state: &AppState,
    req: &SongGenRequest,
) -> Result<Vec<u8>, String> {
    use crate::commands::gguf_engine::{Yue2GenerateRequest, Yue2RequestBody, Yue2Options};
    
    let style = req.prompt.clone();
    let lyrics = req.lyrics.clone();
    
    let request = Yue2GenerateRequest {
        model: "yue2".to_string(),
        request: Yue2RequestBody {
            seed: req.seed,
            options: Yue2Options {
                style,
                lyrics,
                cot: req.cot.clone(),
                num_inference_steps: req.inference_steps,
            },
        },
    };
    
    crate::commands::gguf_engine::generate_via_gguf(state, app, request).await
}

async fn run_generation_fallback(
    app: &tauri::AppHandle,
    state: &AppState,
    config_path: &std::path::Path,
) -> Result<serde_json::Value, String> {
    // 强制禁用常驻模式，始终使用按需启动，避免9GB内存预加载
    tracing::info!("  using on-demand one-shot spawn (resident mode disabled to prevent preload)...");
    let result_json = run_oneshot(&app, &state, &config_path).await?;
    Ok(result_json)
}

async fn run_oneshot(
    app: &tauri::AppHandle,
    state: &AppState,
    config_path: &std::path::Path,
) -> Result<serde_json::Value, String> {
    let start = std::time::Instant::now();
    let sidecar_dir = resolve_song_sidecar_dir(state);
    let sidecar_script = sidecar_dir.join("song_sidecar.py");
    let python = resolve_song_python(&state.app_dir);

    if !sidecar_script.exists() {
        return Err(format!(
            "SONG_SIDECAR_NOT_FOUND: {}",
            sidecar_script.display()
        ));
    }

    tracing::info!("  spawning one-shot sidecar...");

    // ⚠️ ACE-Step 运行时缓存必须指向 src-tauri 之外！
    // tauri dev 监视 src-tauri/：sidecar 往 python_sidecar/.cache/acestep/progress_estimates.json
    // 写文件会触发监视器杀掉应用重启（生成中"软件自动退出"的根因）。
    // ACESTEP_PROJECT_ROOT 让 ACE-Step 把所有 .cache 写到 data/cache/acestep_work。
    let ace_work_dir = state
        .song_models_dir
        .parent()
        .and_then(std::path::Path::parent)
        .map(|d| d.join("cache").join("acestep_work"))
        .unwrap_or_else(|| std::env::temp_dir().join("muno_acestep_work"));
    let _ = std::fs::create_dir_all(&ace_work_dir);
    tracing::info!("  ACESTEP_PROJECT_ROOT={}", ace_work_dir.display());

    let mut child = tokio::process::Command::new(&python)
        .arg("-u")
        .arg(&sidecar_script)
        .arg("--config")
        .arg(format!("@{}", config_path.to_string_lossy()))
        .current_dir(&sidecar_dir)
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("ACESTEP_PROJECT_ROOT", ace_work_dir.to_string_lossy().as_ref())
        .env("SONG_MODELS_DIR", state.song_models_dir.to_string_lossy().as_ref())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("SONG_SPAWN_FAILED: {e}"))?;

    tracing::info!("  sidecar spawned, pid={:?}", child.id());
    if let Some(pid) = child.id() {
        if let Ok(mut slot) = SONG_SIDECAR_PID.lock() {
            *slot = Some(pid);
        }
    }

    use tokio::io::{AsyncBufReadExt, BufReader};
    let stdout = child.stdout.take().ok_or("cannot take stdout")?;
    let stderr = child.stderr.take().ok_or("cannot take stderr")?;

    let mut result_json: Option<serde_json::Value> = None;
    let mut stdout_lines: Vec<String> = Vec::new();
    let mut stderr_lines: Vec<String> = Vec::new();

    let app_handle = app.clone();
    let stdout_task = tokio::spawn(async move {
        let mut reader = BufReader::new(stdout).lines();
        let mut result: Option<serde_json::Value> = None;
        let mut lines: Vec<String> = Vec::new();
        
        while let Ok(Some(line)) = reader.next_line().await {
            if let Some(rest) = line.strip_prefix("@@PROGRESS@@") {
                if let Ok(payload) = serde_json::from_str::<serde_json::Value>(rest) {
                    let stage = payload.get("stage").and_then(|v| v.as_str()).unwrap_or("");
                    let progress_float = payload.get("progress").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let progress = (progress_float * 100.0) as u64;
                    let total = payload.get("total").and_then(|v| v.as_u64()).unwrap_or(100);
                    let stem_label = payload.get("stem_label").and_then(|v| v.as_str()).unwrap_or("");
                    let _ = app_handle.emit(
                        "song-generation-progress",
                        SongProgress {
                            stage: stage.to_string(),
                            current: progress,
                            total,
                            stem_label: stem_label.to_string(),
                        },
                    );
                }
            } else if let Some(rest) = line.strip_prefix("@@RESULT@@") {
                if let Ok(payload) = serde_json::from_str::<serde_json::Value>(rest.trim_start()) {
                    result = Some(payload);
                }
            } else if !line.is_empty() {
                tracing::info!("sidecar: {}", line);
                lines.push(line);
            }
        }
        (result, lines)
    });

    let stderr_task = tokio::spawn(async move {
        let mut reader = BufReader::new(stderr).lines();
        let mut lines: Vec<String> = Vec::new();
        
        while let Ok(Some(line)) = reader.next_line().await {
            lines.push(line);
        }
        lines
    });

    let (stdout_result, stderr_result) = tokio::join!(stdout_task, stderr_task);
    
    let (result_from_stdout, stdout_lines_result) = stdout_result.map_err(|e| format!("stdout task failed: {e}"))?;
    result_json = result_from_stdout;
    stdout_lines = stdout_lines_result;
    stderr_lines = stderr_result.map_err(|e| format!("stderr task failed: {e}"))?;

    let status = child.wait().await.map_err(|e| format!("wait failed: {e}"))?;
    if let Ok(mut slot) = SONG_SIDECAR_PID.lock() {
        *slot = None;
    }

    let elapsed = start.elapsed().as_secs_f64();
    tracing::info!("One-shot generation finished in {:.2}s", elapsed);

    let stderr_text = stderr_lines.join("\n");

    if !stderr_text.trim().is_empty() {
        tracing::info!("sidecar stderr: {}", stderr_text.trim());
    }

    if !status.success() {
        let error_msg = result_json
            .as_ref()
            .and_then(|r| r.get("error"))
            .and_then(|e| e.as_str())
            .unwrap_or("unknown error");
        return Err(format!(
             "SONG_GENERATION_FAILED: exit {:?}\n{}\nstdout: {}\nstderr: {}",
             status.code(),
             error_msg,
             stdout_lines.join("\n"),
             stderr_text,
         ));
    }

    let mut result_json = result_json.ok_or_else(|| {
        format!(
            "SONG_NO_RESULT: sidecar finished without @@RESULT@@\nstdout: {}\nstderr: {}",
            stdout_lines.join("\n"),
            stderr_text,
        )
    })?;

    let success = result_json.get("success").and_then(|v| v.as_bool()).unwrap_or(false);
    if !success {
        let error_msg = result_json
            .get("error")
            .and_then(|e| e.as_str())
            .unwrap_or("unknown error");
        return Err(format!(
            "SONG_GENERATION_FAILED: success=false\n{}\nstdout: {}\nstderr: {}",
            error_msg,
            stdout_lines.join("\n"),
            stderr_text,
        ));
    }

    if let Some(data) = result_json.get_mut("data").and_then(|v| v.as_object_mut()) {
        data.insert("processing_time_secs".to_string(), serde_json::json!(elapsed));
    }

    Ok(result_json)
}

/// 极简 ABC 符号谱 → MIDI（单旋律轨：音高 A-G + 升降号 + 八度 + 时值）。
/// 供 YuE2 的 `abc` 产物映射为 MIDI 轨道；不支持的语法原样跳过并继续，绝不 panic。
/// 约定：`C`=中央 C(MIDI 60)，`c`=高八度；`'` 升八度、`,` 降八度；`^` 升半音、`_` 降半音。
#[tauri::command]
pub fn abc_to_midi(abc: String, out_path: String) -> Result<String, String> {
    write_midi_from_abc(&abc, &out_path)?;
    Ok(out_path)
}

fn abc_tokens(input: &str) -> Vec<(i64, String)> {
    // 只保留第一个曲谱块（T: 之后的正文），并按 | 分小节；返回 (小节序号, 小节原文)。
    let mut body = String::new();
    let mut in_body = false;
    for line in input.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('%') {
            continue;
        }
        // 头部字段（K:/M:/L:/Q:/X:/T:）
        if t.len() >= 2 && t.as_bytes()[1] == b':' && t.as_bytes()[0].is_ascii_alphabetic() {
            in_body = true;
            continue;
        }
        if in_body {
            body.push_str(t);
            body.push(' ');
        }
    }
    let mut out = Vec::new();
    for (i, bar) in body.split('|').enumerate() {
        let b = bar.trim();
        if !b.is_empty() {
            out.push((i as i64, b.to_string()));
        }
    }
    out
}

/// 默认时值（拍）→ 每音符 tick。L: 未指定时按 1/8。
fn abc_default_len(input: &str) -> f64 {
    for line in input.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("L:") {
            let r = rest.trim();
            if let Some((a, b)) = r.split_once('/') {
                let a: f64 = a.trim().parse().unwrap_or(1.0);
                let b: f64 = b.trim().parse().unwrap_or(8.0);
                if b > 0.0 {
                    return a / b;
                }
            }
        }
    }
    1.0 / 8.0
}

fn abc_pitch(letter: char, acc: i32, oct: i32) -> i32 {
    // C=0, D=2, E=4, F=5, G=7, A=9, B=11
    let base: i32 = match letter.to_ascii_uppercase() {
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        'A' => 9,
        'B' => 11,
        _ => return -1,
    };
    // ABC：大写=C4，小写=C5；oct 额外偏移（' 升、, 降）。
    let case_oct = if letter.is_ascii_uppercase() { 4 } else { 5 };
    // MIDI 编号 = (octave + 1) * 12 + pitchClass（中央 C = C4 = 60）。
    let midi = (case_oct + oct + 1) * 12 + base + acc;
    midi.clamp(0, 127)
}

fn write_midi_from_abc(input: &str, out_path: &str) -> Result<(), String> {
    use midly::num::{u15, u24, u28, u4, u7};
    use midly::{Format, Header, MetaMessage, MidiMessage, Smf, Timing, Track, TrackEvent, TrackEventKind};

    let default_len = abc_default_len(input);
    let ticks_per_beat = 480u16;
    let ticks_per_note = (default_len * ticks_per_beat as f64).round().max(1.0) as u32;

    let mut smf = Smf::new(Header::new(Format::SingleTrack, Timing::Metrical(u15::new(ticks_per_beat))));
    // conductor：Tempo 120 默认。
    let mut track: Track = vec![TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::new(500_000))),
    }];

    for (_bar, text) in abc_tokens(input) {
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0usize;
        while i < chars.len() {
            let c = chars[i];
            // 小节内分隔符 / 空格 / 连音符
            if c.is_whitespace() || c == '|' || c == '(' || c == ')' || c == '[' || c == ']' {
                i += 1;
                continue;
            }
            if c == 'z' || c == 'Z' || c == 'x' || c == 'X' {
                // 休止符
                let mut dur = ticks_per_note as i64;
                i += 1;
                if i < chars.len() && chars[i].is_ascii_digit() {
                    let mut s = String::new();
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        s.push(chars[i]);
                        i += 1;
                    }
                    dur = (s.parse::<f64>().unwrap_or(1.0) * ticks_per_note as f64) as i64;
                }
                let _ = dur; // 休止：仅推进时间，这里 delta 累积由最终音符绝对 tick 决定，跳过即可
                continue;
            }
            if c.is_ascii_alphabetic() {
                // 意外音 / 和弦装饰 [CEG] 内字母也在此，逐个作为旋律音处理
                let mut acc = 0i32;
                // 前置变音记号
                let mut k = i;
                while k < chars.len() && (chars[k] == '^' || chars[k] == '_' || chars[k] == '=') {
                    if chars[k] == '^' {
                        acc += 1;
                    } else if chars[k] == '_' {
                        acc -= 1;
                    }
                    k += 1;
                }
                if k >= chars.len() || !chars[k].is_ascii_alphabetic() {
                    i = k;
                    continue;
                }
                let letter = chars[k];
                k += 1;
                let mut oct = 0i32;
                while k < chars.len() && (chars[k] == '\'' || chars[k] == ',') {
                    if chars[k] == '\'' {
                        oct += 1;
                    } else {
                        oct -= 1;
                    }
                    k += 1;
                }
                let mut mult = 1.0f64;
                // 时值乘数：数字、或 / // 的除号
                if k < chars.len() && chars[k].is_ascii_digit() {
                    let mut s = String::new();
                    while k < chars.len() && chars[k].is_ascii_digit() {
                        s.push(chars[k]);
                        k += 1;
                    }
                    mult = s.parse::<f64>().unwrap_or(1.0);
                } else if k < chars.len() && chars[k] == '/' {
                    let mut div = 1.0f64;
                    while k < chars.len() && chars[k] == '/' {
                        div *= 2.0;
                        k += 1;
                    }
                    if k < chars.len() && chars[k].is_ascii_digit() {
                        let mut s = String::new();
                        while k < chars.len() && chars[k].is_ascii_digit() {
                            s.push(chars[k]);
                            k += 1;
                        }
                        div = s.parse::<f64>().unwrap_or(div);
                    }
                    mult = 1.0 / div;
                }
                let pitch = abc_pitch(letter, acc, oct);
                let dur = (mult * ticks_per_note as f64).round().max(1.0) as u32;
                if pitch >= 0 {
                    let vel = u7::new(100);
                    let key = u7::new(pitch as u8);
                    track.push(TrackEvent {
                        delta: u28::new(0),
                        kind: TrackEventKind::Midi {
                            channel: u4::new(0),
                            message: MidiMessage::NoteOn { key, vel },
                        },
                    });
                    track.push(TrackEvent {
                        delta: u28::new(dur.min(0x0FFF_FFFF)),
                        kind: TrackEventKind::Midi {
                            channel: u4::new(0),
                            message: MidiMessage::NoteOff { key, vel: u7::new(0) },
                        },
                    });
                }
                i = k;
                continue;
            }
            i += 1;
        }
    }

    track.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });
    smf.tracks.push(track);

    let mut buf: Vec<u8> = Vec::new();
    smf.write(&mut buf).map_err(|e| format!("SONG_ABC_WRITE_FAIL: {e}"))?;
    std::fs::write(out_path, buf).map_err(|e| format!("SONG_ABC_WRITE_FAIL: {e}"))
}

/// 发行包 ZIP 内的条目：磁盘路径 → zip 内显示名（可含 "01_母带/" 目录前缀）。
#[derive(Deserialize)]
pub struct ZipEntry {
    pub path: String,
    pub name: String,
}

/// 规划 12.6：通用 ZIP 打包（发行包/全量下载）。缺失的源文件跳过；
/// 一个有效条目都没有时按 EXPORT_NO_FILES 失败（复用 amt_export_zip 的既有 CODE）。
#[tauri::command]
pub fn zip_files(entries: Vec<ZipEntry>, out_path: String) -> Result<(), String> {
    use std::io::Write;

    let save_pb = PathBuf::from(&out_path);
    if let Some(parent) = save_pb.parent() {
        if !parent.exists() {
            return Err(format!("EXPORT_DIR_NOT_FOUND: {}", parent.display()));
        }
    }
    let file = std::fs::File::create(&out_path)
        .map_err(|e| format!("CREATE_ZIP_FAILED: {e}"))?;
    let mut zip = zip::ZipWriter::new(file);
    let opts = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    let mut added = 0usize;
    for entry in entries {
        let pb = PathBuf::from(&entry.path);
        if !pb.is_file() {
            continue;
        }
        zip.start_file(entry.name.clone(), opts)
            .map_err(|e| format!("ZIP_ENTRY_FAILED: {e} ({})", entry.name))?;
        let bytes = std::fs::read(&pb)
            .map_err(|e| format!("READ_MIDI_FAILED: {e} ({})", pb.display()))?;
        zip.write_all(&bytes).map_err(|e| format!("WRITE_ZIP_FAILED: {e}"))?;
        added += 1;
    }
    if added == 0 {
        return Err("EXPORT_NO_FILES: no valid files to bundle".to_string());
    }
    zip.finish().map_err(|e| format!("FINALIZE_ZIP_FAILED: {e}"))?;
    Ok(())
}

