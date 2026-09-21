#!/usr/bin/env python3
"""
Song generation sidecar for MunoAI.
Embedded inference with YuE-2, ACE-Step, and HeartMuLa.
"""

import sys
import json
import re
import argparse
import os
from pathlib import Path
from typing import Dict, Any, Optional
import time
import threading


def emit_progress(stage: str, progress: float, message: str = ""):
    """Emit progress update following the @@PROGRESS@@ protocol."""
    data = {
        "stage": stage,
        "progress": progress,
        "message": message
    }
    print(f"@@PROGRESS@@{json.dumps(data)}", flush=True)


def emit_result(success: bool, data: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
    """Emit final result following the @@RESULT@@ protocol."""
    result = {
        "success": success,
        "data": data or {},
        "error": error
    }
    print(f"@@RESULT@@{json.dumps(result)}", flush=True)


def emit_keepalive(state: str, detail: str = ""):
    """Emit daemon lifecycle event following the @@KEEPALIVE@@ protocol.

    state: ready | vram_released | vram_restored | shutdown
    Rust 侧只需识别 ready（可以投递任务）和 shutdown（进程即将退出），
    其余状态供日志/诊断使用。
    """
    print(f"@@KEEPALIVE@@{json.dumps({'state': state, 'detail': detail})}", flush=True)


# ── 常驻（daemon）模式 ────────────────────────────────────────────────
# 默认关闭，只有 Rust 以 --daemon 启动时才启用。常驻省掉的是 Python 启动、
# torch import 和权重反序列化（约 20~30 秒），**不是**长期占着显存：空闲超过
# --idle-timeout 后所有权重会被搬到 CPU 并 empty_cache()，显存归还系统；
# 下次任务到达时再搬回 GPU（约 5~8 秒，不必重新读盘）。
_MODEL_CACHE: Dict[str, Any] = {}
_RESIDENT_LOCK = threading.RLock()
_STATE: Dict[str, Any] = {
    "daemon": False,
    "idle_timeout": 300,
    "last_activity": time.time(),
    "vram_released": True,
    "busy": False,
}


def _iter_torch_modules(obj, _depth=0, _seen=None):
    """产出 obj 持有的所有 torch.nn.Module（用于整体搬迁显存）。

    只读 __dict__（不走 dir()/getattr，避免触发 property 副作用），按类型识别
    而非按字段名，这样上游 ACE-Step 调整内部结构也不会失效。
    """
    try:
        import torch
    except Exception:
        return
    if obj is None or _depth > 3:
        return
    if _seen is None:
        _seen = set()
    if id(obj) in _seen:
        return
    _seen.add(id(obj))
    if isinstance(obj, torch.nn.Module):
        yield obj
        return
    d = getattr(obj, "__dict__", None)
    if not isinstance(d, dict):
        return
    for val in list(d.values()):
        if val is None or isinstance(val, (str, bytes, int, float, bool)):
            continue
        yield from _iter_torch_modules(val, _depth + 1, _seen)


def _move_cached_models(device: str) -> int:
    """把缓存里的所有权重搬到 device（"cpu" 释放显存 / "cuda" 恢复推理）。"""
    moved = 0
    try:
        import torch
    except Exception:
        return 0
    import gc
    for entry in list(_MODEL_CACHE.values()):
        for holder in (entry.get("dit"), entry.get("llm")):
            for mod in _iter_torch_modules(holder):
                try:
                    mod.to(device)
                    moved += 1
                except Exception as e:
                    print(f"[warn] move module to {device} failed: {e}",
                          file=sys.stderr, flush=True)
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return moved


def release_vram(reason: str = "idle") -> None:
    """释放显存：权重搬到 CPU，进程与已导入的 torch 保持存活。"""
    with _RESIDENT_LOCK:
        if _STATE["vram_released"]:
            return
        moved = _move_cached_models("cpu")
        _STATE["vram_released"] = True
    emit_keepalive("vram_released", f"{reason}: {moved} modules moved to CPU")


def restore_vram(device: str) -> None:
    """任务到达时把权重搬回 GPU（无需重新读盘）。"""
    with _RESIDENT_LOCK:
        if not _STATE["vram_released"]:
            return
        if device == "cuda" and _MODEL_CACHE:
            moved = _move_cached_models("cuda")
            emit_keepalive("vram_restored", f"{moved} modules moved to GPU")
        _STATE["vram_released"] = False


def _idle_watchdog() -> None:
    """守护线程：空闲超时后释放显存（进程不退出）。"""
    while True:
        time.sleep(5)
        try:
            if _STATE["busy"] or _STATE["vram_released"]:
                continue
            if time.time() - _STATE["last_activity"] >= _STATE["idle_timeout"]:
                release_vram("idle timeout")
        except Exception as e:
            print(f"[warn] idle watchdog error: {e}", file=sys.stderr, flush=True)


# YuE-2 的 VAE 固定 48kHz（yue2/modeling_vae.py 的 sample_rate 默认值，
# pipeline 内部也硬编码 48000）。
YUE2_SAMPLE_RATE = 48000


def _sanitize_yue2_abc(abc_score: str) -> str:
    """清洗 YuE-2 输出的 ABC。

    YuE-2 的 ABC 常有两类畸形，music21 会直接抛异常：
    1. 拍号字段粘上音符，如 "M:Fmf4e2c3A3F,6|Z|"（解析时 split('/') 崩溃）；
    2. voice 行缺冒号且音符粘连，如 "V Vocal cleABcB2z6..."。
    另外曲中多次换拍（M:5/8）会让 music21 写 MIDI 时插入重复 TimeSignature
    对象触发 StreamException，因此只保留首个有效拍号。
    """
    out_lines, seen_meter = [], False
    for line in abc_score.splitlines():
        s = line.strip()
        if s.startswith("M:"):
            if not seen_meter and re.match(r"^M:\s*\d+/\d+\s*$", s):
                out_lines.append(s)
                seen_meter = True
            continue
        m = re.match(r"^V\s+\S+\s+(\S*[|Z].*)$", s)
        if m:  # voice 头缺冒号且粘了音符：拆成 voice 声明 + 单独的音符行
            out_lines.append("V:" + s[1:].split()[0])
            out_lines.append(m.group(1))
            continue
        out_lines.append(s)
    return "\n".join(out_lines)


def convert_abc_to_midi(abc_score: str, out_path: Path) -> Optional[Path]:
    """把 YuE-2 产出的 ABC 乐谱转成 MIDI。失败不影响音频主产物。"""
    try:
        from music21 import converter, meter
    except ImportError:
        print("[warn] music21 未安装，跳过 MIDI 导出", file=sys.stderr, flush=True)
        return None
    try:
        score = converter.parse(_sanitize_yue2_abc(abc_score), format="abc")
        # music21 复用同一 TimeSignature 对象，写 MIDI 时 conductorStream
        # 二次插入同一实例会抛 "already found in this Stream"，替换为独立副本
        for ts in list(score.recurse().getElementsByClass(meter.TimeSignature)):
            site = ts.activeSite
            if site is not None:
                site.replace(ts, meter.TimeSignature(ts.ratioString))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        score.write("midi", fp=str(out_path))
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
        print("[warn] MIDI 写出为空", file=sys.stderr, flush=True)
        return None
    except Exception as exc:
        print(f"[warn] ABC 转 MIDI 失败: {exc}", file=sys.stderr, flush=True)
        return None


def separate_stems(audio_path: Path, out_dir: Path) -> Optional[Dict[str, str]]:
    """用 demucs 把成品音频分成 4 轨。demucs 缺失或失败时返回 None。"""
    try:
        import demucs.separate  # noqa: F401
    except ImportError:
        print("[warn] demucs 未安装，跳过多轨分离", file=sys.stderr, flush=True)
        return None
    import subprocess
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "demucs.separate",
             "-n", "htdemucs", "--out", str(out_dir), str(audio_path)],
            capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode != 0:
            print(f"[warn] demucs 失败: {proc.stderr[-600:]}", file=sys.stderr, flush=True)
            return None
    except Exception as exc:
        print(f"[warn] demucs 执行异常: {exc}", file=sys.stderr, flush=True)
        return None
    # demucs 输出到 <out_dir>/<model>/<音频名>/{vocals,drums,bass,other}.wav
    stems: Dict[str, str] = {}
    for wav in out_dir.rglob("*.wav"):
        stems[wav.stem] = str(wav)
    return stems or None


def write_plain_lrc(lyrics: str, out_path: Path) -> Optional[Path]:
    """写出无时间轴的 LRC。模型不返回逐行时间戳，这里只做歌词落盘，不伪造时间。"""
    if not lyrics or not lyrics.strip():
        return None
    try:
        lines = [ln.strip() for ln in lyrics.splitlines() if ln.strip()]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path
    except Exception as exc:
        print(f"[warn] LRC 写出失败: {exc}", file=sys.stderr, flush=True)
        return None


def validate_model_files(models_dir: Path, model_id: str) -> bool:
    """Validate that required model files exist locally."""
    model_path = models_dir / model_id
    if not model_path.exists():
        return False
    
    if model_id == "yue2-3b":
        # YuE-2: check model root and vae subdirectory
        required = ["config.json", "model.safetensors", "qwen.tiktoken"]
        vae_required = ["config.json", "model.safetensors"]
        if not all((model_path / f).exists() for f in required):
            return False
        if not all((model_path / "vae" / f).exists() for f in vae_required):
            return False
    elif model_id == "acestep-v1-3.5b":
        # ACE-Step v1: check 4 required subdirectories
        required_dirs = ["music_dcae_f8c8", "music_vocoder", "ace_step_transformer", "umt5-base"]
        if not all((model_path / d).exists() for d in required_dirs):
            return False
    elif model_id == "acestep-v1.5":
        # ACE-Step v1.5: 官方 checkpoints 布局（与 HF ACE-Step/Ace-Step1.5 一致）。
        # DiT 二选一：acestep-v15-xl-turbo（XL 变体，质量更好）或
        # acestep-v15-turbo（标准 turbo，体积更小）。
        ckpt = model_path / "checkpoints"
        base_required = [
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "Qwen3-Embedding-0.6B/model.safetensors",
            "Qwen3-Embedding-0.6B/config.json",
            "acestep-5Hz-lm-1.7B/model.safetensors",
            "acestep-5Hz-lm-1.7B/config.json",
        ]
        xl_required = [
            "acestep-v15-xl-turbo/model.safetensors",
            "acestep-v15-xl-turbo/config.json",
            "acestep-v15-xl-turbo/silence_latent.pt",
        ]
        turbo_required = [
            "acestep-v15-turbo/model.safetensors",
            "acestep-v15-turbo/config.json",
            "acestep-v15-turbo/silence_latent.pt",
        ]
        if not all((ckpt / f).exists() for f in base_required):
            return False
        has_xl = all((ckpt / f).exists() for f in xl_required)
        has_turbo = all((ckpt / f).exists() for f in turbo_required)
        if not (has_xl or has_turbo):
            return False
    elif model_id == "heartmula-3b":
        # HeartMuLa: check directory structure
        required_dirs = ["HeartMuLa-oss-3B", "HeartCodec-oss"]
        required_files = ["tokenizer.json", "gen_config.json"]
        if not all((model_path / d).exists() for d in required_dirs):
            return False
        if not all((model_path / f).exists() for f in required_files):
            return False
    else:
        return False
    
    return True


def generate_yue2_embedded(
    models_dir: Path,
    output_dir: Path,
    song_name: str,
    prompt: str,
    lyrics: str,
    audio_duration: float,
    seed: Optional[int] = None,
    cfg_scale: Optional[float] = None,
    cot: str = "full",
    abc: Optional[str] = None,
    # Sampling overrides: None 表示沿用 checkpoint 自带默认值。
    # 不要在这里写死 max_tokens 之类的小值，否则歌曲会被提前截断。
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    penalty_window: Optional[int] = None,
    min_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    # Generation config
    ode_steps: int = 32,
    ode_method: str = "midpoint",
    # 产物开关
    want_midi: bool = False,
    want_stems: bool = False,
    want_lrc: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Generate song using local YuE-2 model with full parameter support."""
    try:
        from yue2.pipeline import YuE2Pipeline
        from yue2.protocol import SongRequest, Sampling, GenerationConfig, resolve_sampling
        
        emit_progress("initialize", 0.0, "Loading YuE-2 model...")
        
        model_dir = models_dir / "yue2-3b"
        vae_dir = model_dir / "vae"
        
        # 只覆盖调用方显式给出的采样参数，其余沿用 Sampling() 自带默认值
        # （默认 max_tokens=9000；写死小值会让歌曲被提前截断）。
        sampling_overrides = {}
        for key, val in [("temperature", temperature), ("top_p", top_p), ("top_k", top_k),
                         ("repetition_penalty", repetition_penalty), ("penalty_window", penalty_window),
                         ("min_tokens", min_tokens), ("max_tokens", max_tokens)]:
            if val is not None:
                sampling_overrides[key] = val
        sampling = resolve_sampling(sampling_overrides, Sampling()) if sampling_overrides else Sampling()
        
        # ode_steps / ode_method 只能通过构造函数的 generation_config 生效，
        # 因为 synthesize() 读的是 self.generation_config。
        gen_config = GenerationConfig(
            abc=sampling, semantic=sampling,
            ode_steps=ode_steps, ode_method=ode_method
        )
        
        # backend 必须是 "torch-eager"：默认的 "torch" 会启用 GraphAR(CUDA graph)，
        # 而 GraphAR 的 attention_backend="auto" 在 Windows 上会误判为 flash
        # （cuda_graph.py 只检查 op 是否存在 + schema 含 seqused_k，Windows 版 torch 两项都满足，
        # 但内核没编进去），运行时直接抛 "USE_FLASH_ATTENTION was not enabled for build"。
        # pipeline 没有任何参数能把 attention_backend 透传给 GraphAR，只能整体走 eager。
        pipe = YuE2Pipeline(
            model_dir=str(model_dir),
            vae_dir=str(vae_dir),
            device="auto",
            memory_budget_gib=24,
            generation_config=gen_config,
            backend="torch-eager",
            progress=False
        )
        
        emit_progress("plan", 0.2, "Planning song structure...")
        
        # Build request
        # 如果前端没有提供 seed，使用随机种子而不是硬编码的值
        import random
        actual_seed = seed if seed is not None else random.randint(1, 2147483647)
        request = SongRequest(
            style=prompt,
            lyrics=lyrics,
            cot=cot,
            seed=actual_seed,
            abc=abc,
            cfg_scale=cfg_scale
        )
        
        # Execute generation stages with progress tracking
        plan = pipe.plan(request=request)
        emit_progress("semantic", 0.4, "Generating semantic tokens...")
        
        semantic = pipe.generate_semantic(plan)
        emit_progress("synthesize", 0.6, "Synthesizing audio latents...")
        
        latents = pipe.synthesize(semantic)
        emit_progress("decode", 0.8, "Decoding to audio...")
        
        audio_array = pipe.decode(latents)
        
        # Save output with model identifier to prevent overwriting
        output_path = output_dir / f"{song_name}_yue2.wav"
        emit_progress("save", 0.95, "Saving audio file...")
        import soundfile as sf
        sf.write(str(output_path), audio_array, YUE2_SAMPLE_RATE)
        
        # 真实时长按解码出的样本数算。YuE-2 由 max_tokens 决定长度，
        # 不受 audio_duration 控制，直接回显请求值会给出假数据。
        actual_duration = len(audio_array) / float(YUE2_SAMPLE_RATE)
        
        result = {
            "audio_path": str(output_path),
            "duration": actual_duration,
            "model": "yue2-3b",
        }
        
        # YuE-2 的 plan 里带 ABC 乐谱，这是三个模型中唯一的原生符号层产物，
        # 可以无损转成 MIDI（不是从音频反推音符）。
        abc_score = getattr(plan, "abc", None)
        if abc_score:
            abc_path = output_dir / f"{song_name}.abc"
            abc_path.write_text(abc_score, encoding="utf-8")
            result["abc_path"] = str(abc_path)
            if want_midi:
                emit_progress("midi", 0.97, "Converting score to MIDI...")
                midi_path = convert_abc_to_midi(abc_score, output_dir / f"{song_name}.mid")
                if midi_path:
                    result["midi_path"] = str(midi_path)
        
        if want_stems:
            emit_progress("stems", 0.98, "Separating stems...")
            stems = separate_stems(output_path, output_dir / f"{song_name}_stems")
            if stems:
                result["stems"] = stems
        
        if want_lrc:
            lrc_path = write_plain_lrc(lyrics, output_dir / f"{song_name}.lrc")
            if lrc_path:
                result["lrc_path"] = str(lrc_path)
        
        emit_progress("complete", 1.0, "Generation completed")
        return result
        
    except Exception as e:
        raise RuntimeError(f"YuE-2 generation failed: {str(e)}")


def _import_acestep_v15():
    """导入 ACE-Step v1.5 包。

    venv 里可编辑安装的 `acestep`（ace_step 0.2.0）指向 v1 仓库（repos/ACE-Step），
    与 v1.5 仓库（repos/ACE-Step-1.5）包名相同。必须在 import 前把 v1.5 仓库
    插到 sys.path 最前面，否则会加载到 v1 的实现。
    """
    repo_path = None
    # 从 venv 的 python.exe 向上逐级查找 repos/ACE-Step-1.5
    # （实际布局：<app>/src-tauri/runtime/repos/ACE-Step-1.5，
    #   python.exe 在 <app>/src-tauri/runtime/python310/Scripts/ 下）
    probe = Path(sys.executable).resolve().parent
    for _ in range(4):
        candidate = probe / "repos" / "ACE-Step-1.5"
        if (candidate / "acestep" / "inference.py").exists():
            repo_path = candidate
            break
        probe = probe.parent
    if repo_path is None:
        raise RuntimeError(
            "ACE-Step v1.5 repo not found (searched upwards from "
            f"{Path(sys.executable).parent} for repos/ACE-Step-1.5)"
        )
    sys.path.insert(0, str(repo_path))
    import acestep
    if not (Path(acestep.__file__).parent / "inference.py").exists():
        raise RuntimeError(
            f"Loaded acestep package from {acestep.__file__} is not v1.5 "
            "(package name conflict with v1 editable install)"
        )
    return repo_path


def _check_vram_availability() -> tuple[bool, float]:
    try:
        import torch
        if not torch.cuda.is_available():
            return False, 0.0
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        free_gb = free_mem / (1024 ** 3)
        return True, free_gb
    except Exception:
        return False, 0.0


def generate_acestep_embedded(
    models_dir: Path,
    output_dir: Path,
    song_name: str,
    prompt: str,
    lyrics: str,
    audio_duration: float,
    model_id: str = "acestep-v1.5",
    seed: Optional[int] = None,
    inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    audio_format: str = "flac",
    mp3_bitrate: str = "192k",
    mp3_sample_rate: int = 48000,
    vram_mode: str = "auto",
    thinking: bool = True,
    force_thinking: bool = False,
    batch_size: Optional[int] = None,
    task_type: str = "text2music",
    src_audio: Optional[str] = None,
    track_name: Optional[str] = None,
    complete_track_classes: Optional[list] = None,
    instruction: Optional[str] = None,
    repainting_start: Optional[float] = None,
    repainting_end: Optional[float] = None,
    audio_cover_strength: Optional[float] = None,
    task: Optional[str] = None,
    src_audio_path: Optional[str] = None,
    repaint_start: Optional[float] = None,
    repaint_end: Optional[float] = None,
    **kwargs
) -> Dict[str, Any]:
    """Generate song using ACE-Step v1.5 (xl-turbo DiT + 5Hz LM + VAE).

    参数对齐 Rust SongGenRequest：Option 字段会以 None 传入，这里统一做
    None -> 默认值处理。turbo 模型默认 8 步推理。

    task_type 支持 text2music / cover / repaint / lego / extract / complete。
    lego（叠加音轨/干声）、extract（分轨分离）、complete（补全音轨）仅
    acestep-v15-base 模型支持，会自动切换 DiT 变体。
    """
    import shutil
    import tempfile

    try:
        import torch

        # ══════════════════════════════════════════════════════════════════
        #  VRAM 感知策略：根据可用显存动态决定所有加载/推理参数
        # ══════════════════════════════════════════════════════════════════
        has_cuda, free_vram_gb = _check_vram_availability()
        if not has_cuda:
            free_vram_gb = 0.0
            print("[sidecar] CUDA not available, using CPU (will be very slow)",
                  file=sys.stderr, flush=True)

        detected_tier = "quality_16gb" if free_vram_gb >= 14.0 else ("balanced_12gb" if free_vram_gb >= 10.0 else "fast_8gb")
        selected_mode = vram_mode if vram_mode in ("fast_8gb", "balanced_12gb", "quality_16gb") else detected_tier
        if selected_mode != detected_tier:
            print(f"[sidecar] VRAM mode override: requested={selected_mode}, detected={detected_tier}", file=sys.stderr, flush=True)

        # 显存档位控制加载策略。手动档位只改变运行策略，不伪造显卡检测结果。
        if selected_mode == "quality_16gb":
            strategy_tier = "ultra"
        elif selected_mode == "balanced_12gb":
            strategy_tier = "high"
        else:
            strategy_tier = "mid" if has_cuda and free_vram_gb >= 8.0 else "low"

        if strategy_tier == "ultra":

            tier = "ultra"
            dit_dtype = torch.bfloat16
            dit_offload = False
            lm_offload_to_cpu = True
            enable_thinking = False  # ultra 默认也关，除非用户显式传 True
            max_duration = 240.0
        elif strategy_tier == "high":
            tier = "high"
            dit_dtype = torch.bfloat16
            dit_offload = False
            lm_offload_to_cpu = True
            enable_thinking = False
            max_duration = 180.0
        elif strategy_tier == "mid":
            tier = "mid"
            dit_dtype = torch.bfloat16
            dit_offload = False
            lm_offload_to_cpu = True
            enable_thinking = False
            max_duration = 120.0
        elif free_vram_gb >= 6.0:
            tier = "low"
            dit_dtype = torch.float16
            dit_offload = False
            lm_offload_to_cpu = True
            enable_thinking = False
            max_duration = 90.0
        else:
            tier = "min"
            dit_dtype = torch.float16
            dit_offload = True
            lm_offload_to_cpu = True
            enable_thinking = False
            max_duration = 60.0

        # CPU fallback
        if not has_cuda:
            tier = "cpu"
            dit_dtype = None
            dit_offload = True
            lm_offload_to_cpu = True
            enable_thinking = False
            max_duration = 60.0

        # ⚠️ thinking 默认强制关闭（快 20 倍！LM Phase 1+2 ≈ 200s）
        # 只有用户显式在 config 里传 "force_thinking": true 才开启 CoT
        original_thinking = thinking
        thinking = False  # 默认关！
        if force_thinking and tier in ("ultra", "high"):
            thinking = True
            print(f"[sidecar] thinking FORCE-enabled (tier={tier}, ~200s overhead)",
                  file=sys.stderr, flush=True)
        elif original_thinking and tier in ("ultra", "high"):
            print(f"[sidecar] thinking ignored (config had True, but default=off). "
                  f"Use force_thinking=true to enable.", file=sys.stderr, flush=True)

        print(f"[sidecar] VRAM tier={tier} ({free_vram_gb:.1f} GB free) | "
              f"dit_dtype={dit_dtype} | dit_offload={dit_offload} | "
              f"lm_offload={lm_offload_to_cpu} | thinking={thinking} | "
              f"max_duration={max_duration}s",
              file=sys.stderr, flush=True)

        torch.cuda.empty_cache() if has_cuda else None
        
        repo_path = _import_acestep_v15()
        # 关闭遥测类环境噪音；checkpoints 走 SONG_MODELS_DIR 下的官方布局
        ckpt_dir = models_dir / "acestep-v1.5" / "checkpoints"
        os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(ckpt_dir)

        # 确保 repo 下的 checkpoints 是一个指向我们已下载模型的 junction/link，
        # 这样 AceStepHandler 不会每次都重新从 HuggingFace 下载（这是之前卡住的根因）。
        repo_ckpt_link = repo_path / "checkpoints"
        if ckpt_dir.exists():
            if not repo_ckpt_link.exists():
                try:
                    if os.name == "nt":
                        import subprocess
                        subprocess.run(
                            ["cmd", "/c", "mklink", "/J", str(repo_ckpt_link), str(ckpt_dir)],
                            check=True, capture_output=True
                        )
                        print(f"[sidecar] Created junction: {repo_ckpt_link} -> {ckpt_dir}",
                              file=sys.stderr, flush=True)
                    else:
                        os.symlink(str(ckpt_dir), str(repo_ckpt_link))
                except Exception as e:
                    print(f"[warn] Failed to create checkpoints junction: {e}",
                          file=sys.stderr, flush=True)
            elif not repo_ckpt_link.is_dir():
                # 旧的损坏 junction，删除重建
                try:
                    repo_ckpt_link.unlink()
                    import subprocess
                    subprocess.run(
                        ["cmd", "/c", "mklink", "/J", str(repo_ckpt_link), str(ckpt_dir)],
                        check=True, capture_output=True
                    )
                    print(f"[sidecar] Recreated junction: {repo_ckpt_link} -> {ckpt_dir}",
                          file=sys.stderr, flush=True)
                except Exception as e:
                    print(f"[warn] Failed to recreate junction: {e}",
                          file=sys.stderr, flush=True)

        # 防止 ACE-Step 自动下载整个主仓库：本地文件已校验齐全（validate_model_files），
        # 把下载预检替换为恒真。必须 patch `init_service_downloads` 模块内的绑定名
        # （它是 from-import 进来的，patch model_downloader 原模块无效）。
        from acestep.core.generation.handler import init_service_downloads as _isd
        _isd.check_main_model_exists = lambda *a, **k: True

        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler
        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        # Rust 旧字段名 -> 官方字段名（task/src_audio_path/repaint_*）
        task_type = (task_type or task or "text2music").strip()
        src_audio = src_audio or src_audio_path
        if repainting_start is None:
            repainting_start = repaint_start
        if repainting_end is None:
            repainting_end = repaint_end
        # 需要 base 模型的任务（官方 constants.py: TASK_TYPES_TURBO 不含这三个）
        base_only_tasks = ("lego", "extract", "complete")
        needs_base = task_type in base_only_tasks

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # DiT 变体选择：多轨任务强制 base；其余优先 XL（质量更好），其次 turbo
        def _variant_ready(name: str) -> bool:
            return (ckpt_dir / name / "model.safetensors").exists()

        if needs_base:
            if not _variant_ready("acestep-v15-base"):
                raise RuntimeError(
                    "多轨任务（叠加音轨/分轨分离/补全）需要 acestep-v15-base 模型，"
                    "请先在资源管理中下载（约 4.8 GB）"
                )
            dit_variant = "acestep-v15-base"
        elif _variant_ready("acestep-v15-xl-turbo"):
            dit_variant = "acestep-v15-xl-turbo"
        elif _variant_ready("acestep-v15-turbo"):
            dit_variant = "acestep-v15-turbo"
        else:
            raise RuntimeError(
                f"No DiT checkpoint found in {ckpt_dir} "
                "(need acestep-v15-xl-turbo/ or acestep-v15-turbo/)"
            )

        # base 模型官方推荐 50 步推理（turbo 8 步）
        is_base = dit_variant == "acestep-v15-base"
        is_turbo = "turbo" in dit_variant
        if is_base:
            infer_steps = max(1, min(int(inference_steps), 100)) if inference_steps is not None else 50
        elif is_turbo:
            requested_steps = int(inference_steps) if inference_steps is not None else 8
            infer_steps = max(4, min(requested_steps, 8))
            if requested_steps != infer_steps:
                print(f"[sidecar] TURBO steps clamped from {requested_steps} to {infer_steps}", file=sys.stderr, flush=True)
        else:
            infer_steps = max(1, min(int(inference_steps), 100)) if inference_steps is not None else 50

        # 常驻模式下按 (变体, 设备) 复用已加载的权重，省掉约 20~30 秒的反序列化。
        # 非常驻模式 cache_key 查不到，行为与改造前完全一致。
        cache_key = f"acestep15::{dit_variant}::{device}"
        cached = _MODEL_CACHE.get(cache_key) if _STATE["daemon"] else None

        if cached is not None:
            dit = cached["dit"]
            llm = cached["llm"]
            lm_ok = cached["lm_ok"]
            # 权重可能因空闲超时被搬到 CPU（或上轮 thinking 后 LM 被卸载），
            # 这里统一搬回 GPU；已在 GPU 上的 .to() 是空操作。
            if device == "cuda":
                emit_progress("initialize", 0.1, "Restoring resident models to GPU...")
                _move_cached_models(device)
            with _RESIDENT_LOCK:
                _STATE["vram_released"] = False
            emit_progress("initialize", 0.3, f"Resident ACE-Step models ready ({dit_variant})")
            if not lm_ok:
                thinking = False
        else:
            dit = AceStepHandler()
            dtype_label = {torch.bfloat16: "BF16", torch.float16: "FP16", None: "default"}.get(dit_dtype, str(dit_dtype))
            emit_progress("initialize", 0.05, f"Loading ACE-Step v1.5 DiT ({dit_variant}, {dtype_label}, tier={tier})...")
            init_status, enable_generate = dit.initialize_service(
                project_root=str(repo_path),
                config_path=dit_variant,
                device=device,
                use_flash_attention=False,
                compile_model=False,
                offload_to_cpu=dit_offload,
                offload_dit_to_cpu=dit_offload,
                quantization=None,
                prefer_source=None,
            )
            if not enable_generate:
                raise RuntimeError(f"DiT init failed: {init_status}")

            emit_progress("initialize", 0.3, f"Loading 5Hz language model (tier={tier})...")
            llm = LLMHandler()
            lm_dtype = dit_dtype if dit_dtype is not None else None
            lm_status, lm_ok = llm.initialize(
                checkpoint_dir=str(ckpt_dir),
                lm_model_path="acestep-5Hz-lm-1.7B",
                backend="pt",
                device=device,
                offload_to_cpu=lm_offload_to_cpu,
                dtype=lm_dtype,
            )
            if not lm_ok:
                # LM 初始化失败时降级为纯 DiT 模式（跳过 CoT），不直接失败
                print(f"[warn] LM init failed, fallback to thinking=False: {lm_status}",
                      file=sys.stderr, flush=True)
                thinking = False

            if _STATE["daemon"]:
                # 同一时刻只常驻一套变体：切换变体时先丢掉旧的，否则两套 DiT
                # 会同时占着内存（各约 4.8GB）。
                for old_key in [k for k in _MODEL_CACHE if k != cache_key]:
                    _MODEL_CACHE.pop(old_key, None)
                _MODEL_CACHE[cache_key] = {"dit": dit, "llm": llm, "lm_ok": lm_ok}
                with _RESIDENT_LOCK:
                    _STATE["vram_released"] = False

        # LM 用完自动卸载到 CPU：thinking 完成后（generate_with_stop_condition
        # 返回后）把 ~3.4GB 的 LM 权重移出显存，给扩散阶段腾出空间。
        # 16GB 显存上实测峰值会溢出到共享内存 ~3GB，此优化可基本消除溢出。
        if lm_ok and thinking and device == "cuda":
            import gc as _gc
            # 常驻模式会复用同一个 llm 实例，wrapper 必须只包一层：否则每次生成
            # 都再套一层，调用栈逐轮加深、卸载逻辑重复执行。用实例上的标记位保证
            # 幂等，每轮只把 per-run 的 done 状态复位。
            _offload_state = getattr(llm, "_utai_offload_state", None)
            if _offload_state is None:
                _offload_state = {"done": False}
                _orig_generate = llm.generate_with_stop_condition

                def _generate_then_offload(*a, **k):
                    try:
                        return _orig_generate(*a, **k)
                    finally:
                        if not _offload_state["done"]:
                            _offload_state["done"] = True
                            try:
                                m = getattr(llm, "llm", None)
                                if m is not None:
                                    print("[sidecar] LM thinking complete, offloading to CPU to free 3.4GB VRAM...", 
                                          file=sys.stderr, flush=True)
                                    m.to("cpu")
                                _gc.collect()
                                torch.cuda.empty_cache()
                                print("[sidecar] LM offloaded to CPU after thinking",
                                      file=sys.stderr, flush=True)
                            except Exception as _e:
                                print(f"[warn] LM offload failed: {_e}",
                                      file=sys.stderr, flush=True)

                llm._utai_offload_state = _offload_state
                llm.generate_with_stop_condition = _generate_then_offload
            else:
                _offload_state["done"] = False

        emit_progress("generate", 0.35, "Generating music...")

        # 源音频类任务必须有 src_audio
        needs_src = task_type in ("cover", "cover-nofsq", "repaint", "lego", "extract", "complete")
        if needs_src and not src_audio:
            raise RuntimeError(f"task_type={task_type} 需要提供源音频 (src_audio)")
        if src_audio and not Path(src_audio).exists():
            raise RuntimeError(f"源音频不存在: {src_audio}")

        # instruction 未提供时按官方模板自动生成（lego/extract 用轨道名填充）
        if not instruction:
            try:
                instruction = dit.generate_instruction(
                    task_type,
                    track_name=track_name,
                    complete_track_classes=complete_track_classes,
                )
            except Exception:
                instruction = "Fill the audio semantic mask based on the given conditions:"

        # 多轨任务（lego/extract/complete）是 direct-conditioning：LM 跳过，
        # thinking 无意义，显式关闭避免误导
        if needs_base:
            thinking = False

        # ── 参数最终修正（VRAM tier 驱动）─────────────────────────────────
        # duration 硬限制：config 可能传 180s（BASE 默认），但低显存 tier 会爆
        raw_duration = float(audio_duration) if audio_duration and audio_duration > 0 else -1.0
        if raw_duration > 0 and raw_duration > max_duration:
            print(f"[sidecar] Duration {raw_duration}s exceeds tier max {max_duration}s, clamping",
                  file=sys.stderr, flush=True)
            raw_duration = max_duration

        # guidance_scale 按模型变体修正
        if is_base:
            final_guidance = float(guidance_scale) if guidance_scale else 7.0
        else:
            # TURBO/XL-TURBO 模型：默认 4.0，超过 10 强制降到 4.0
            gs = float(guidance_scale) if guidance_scale else 4.0
            final_guidance = 4.0 if gs > 10 else gs

        # inference_steps：TURBO 模型强制限制（已在前面处理过，但兜底）
        if is_turbo:
            infer_steps = max(4, min(int(infer_steps), 8))

        print(f"[sidecar] Final params: duration={raw_duration}s | steps={infer_steps} | "
              f"guidance={final_guidance} | thinking={thinking} | tier={tier}",
              file=sys.stderr, flush=True)

        # ⚠️ 关键：use_cot_* 默认全是 True！必须和 thinking 一起关掉
        # 否则 need_lm_for_cot = True 会强制 LM 跑，白浪费 200+ 秒
        _cot = bool(thinking)  # thinking 关则所有 CoT 关
        params = GenerationParams(
            caption=prompt or "",
            lyrics=lyrics if lyrics and lyrics.strip() else "[Instrumental]",
            duration=raw_duration,
            seed=int(seed) if seed is not None else -1,
            inference_steps=infer_steps,
            guidance_scale=final_guidance,
            thinking=bool(thinking),
            use_cot_metas=_cot,
            use_cot_caption=_cot,
            use_cot_language=_cot,
            vocal_language="unknown",
            task_type=task_type,
            instruction=instruction,
        )
        if thinking:
            print(f"[sidecar] CoT enabled (thinking + cot_metas/caption/language)",
                  file=sys.stderr, flush=True)
        else:
            print(f"[sidecar] CoT disabled → LM skipped → ~200s saved → direct DiT inference",
                  file=sys.stderr, flush=True)
        if src_audio:
            params.src_audio = str(src_audio)
        if audio_cover_strength is not None:
            params.audio_cover_strength = float(audio_cover_strength)
        if task_type == "repaint":
            if repainting_start is not None:
                params.repainting_start = float(repainting_start)
            if repainting_end is not None:
                params.repainting_end = float(repainting_end)

        allowed_formats = {"flac", "wav", "wav32", "mp3", "aac", "opus"}
        output_format = str(audio_format or "flac").strip().lower()
        if output_format not in allowed_formats:
            output_format = "flac"
        config = GenerationConfig(
            batch_size=int(batch_size) if batch_size else 1,
            audio_format=output_format,
            mp3_bitrate=str(mp3_bitrate or "192k"),
            mp3_sample_rate=int(mp3_sample_rate or 48000),
            use_random_seed=seed is None,
            seeds=[int(seed)] if seed is not None else None,
        )

        def _progress_cb(ratio, desc=""):
            try:
                r = max(0.0, min(1.0, float(ratio)))
            except (TypeError, ValueError):
                r = 0.0
            emit_progress("generate", 0.35 + 0.6 * r, str(desc))

        tmp_dir = Path(tempfile.mkdtemp(prefix="acestep15_out_"))
        
        try:
            result = generate_music(
                dit, llm, params, config, save_dir=str(tmp_dir), progress=_progress_cb
            )
            if not result.success or not result.audios:
                raise RuntimeError(result.error or "generate_music returned no audio")
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg:
                print(f"[error] CUDA OOM detected: {e}", file=sys.stderr, flush=True)
                if device == "cuda":
                    try:
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                        print("[sidecar] Cleared CUDA cache after OOM", file=sys.stderr, flush=True)
                    except Exception:
                        pass
                raise RuntimeError(
                    f"显存不足(CUDA OOM)。请尝试: 1)关闭其他占用显存的程序 2)减少生成时长 3)重启软件。原始错误: {e}"
                )
            raise
        except Exception as e:
            print(f"[error] ACE-Step generation exception: {type(e).__name__}: {e}", 
                  file=sys.stderr, flush=True)
            raise

        src_path = Path(result.audios[0]["path"])
        if not src_path.exists():
            raise RuntimeError(f"Generated audio missing: {src_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        extension = "wav" if output_format == "wav32" else output_format
        output_path = output_dir / f"{song_name}_acestep.{extension}"
        shutil.copyfile(src_path, output_path)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        actual_duration = audio_duration
        try:
            import soundfile as sf
            info = sf.info(str(output_path))
            actual_duration = info.duration
        except Exception:
            pass

        emit_progress("complete", 1.0, "Generation completed")
        return {
            "audio_path": str(output_path),
            "duration": actual_duration,
            "model": "acestep-v1.5",
            "seed": seed,
        }

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[error] ACE-Step fatal error:\n{error_trace}", file=sys.stderr, flush=True)
        
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
                import gc
                gc.collect()
        except Exception:
            pass
        
        error_msg = str(e)
        if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
            raise RuntimeError(
                f"ACE-Step 生成失败(显存不足)。建议:\n"
                f"1. 关闭其他占用显存的程序\n"
                f"2. 减少生成时长(当前: {audio_duration}秒)\n"
                f"3. 重启软件清理内存\n"
                f"详细错误: {error_msg}"
            )
        
        raise RuntimeError(f"ACE-Step generation failed: {error_msg}\n{error_trace}")


def generate_heartmula_embedded(
    models_dir: Path,
    output_dir: Path,
    song_name: str,
    prompt: str,
    lyrics: str,
    audio_duration: float,
    seed: Optional[int] = None,
    cfg_scale: float = 1.5,
    temperature: float = 1.0,
    topk: int = 50,
    want_midi: bool = False,
    want_stems: bool = False,
    want_lrc: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Generate song using local HeartMuLa model."""
    try:
        import torch
        from heartlib.pipelines.music_generation import HeartMuLaGenPipeline
        
        emit_progress("initialize", 0.0, "Loading HeartMuLa model...")
        
        pretrained_path = str(models_dir / "heartmula-3b")
        
        # device/dtype 必须是 torch.device / torch.dtype 对象，不能是字符串：
        # mula_dtype 会被拿去 torch.zeros(dtype=...)，字符串会直接抛 TypeError。
        #
        # 显存布局（16GB 显卡上的血泪教训）：
        # - mula 3B bf16 ≈ 7GB，放 GPU；
        # - HeartCodec 6.3GB 权重若以 fp32 放 GPU ≈ 12.6GB，两者相加远超 16GB，
        #   生成中驱动级崩溃会直接整机重启（已实际发生）。
        # - postprocess 会 frames.to(codec_device) 且 _forward 结束就 _unload()，
        #   所以 codec 放 CPU 完全可行：解码时显存已腾空，只多花十几秒 CPU 时间。
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mula_dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
        
        pipe = HeartMuLaGenPipeline.from_pretrained(
            pretrained_path=pretrained_path,
            device={"mula": dev, "codec": torch.device("cpu")},
            dtype={"mula": mula_dtype, "codec": torch.float32},
            version="3B",
            lazy_load=False
        )
        
        emit_progress("generate", 0.2, "Generating audio...")
        
        # Prepare output path
        output_path = output_dir / f"{song_name}.wav"
        
        # Generate
        max_audio_length_ms = int(audio_duration * 1000)
        
        pipe(
            inputs={"lyrics": lyrics, "tags": prompt},
            max_audio_length_ms=max_audio_length_ms,
            save_path=str(output_path),
            topk=topk,
            temperature=temperature,
            cfg_scale=cfg_scale
        )
        
        # HeartMuLa 遇到 EOS 会提前停，实际时长按落盘音频算，不回显请求值。
        actual_duration = audio_duration
        if output_path.exists():
            import soundfile as sf
            info = sf.info(str(output_path))
            actual_duration = info.duration
        
        result = {
            "audio_path": str(output_path),
            "duration": actual_duration,
            "model": "heartmula-3b",
        }
        
        # HeartMuLa 无原生符号层产物（ABC/乐谱），不做 MIDI 转换，
        # 避免用音频反推音符生成不可靠的"假 MIDI"。stems/lrc 可用。
        if want_stems:
            emit_progress("stems", 0.95, "Separating stems...")
            stems = separate_stems(output_path, output_dir / f"{song_name}_stems")
            if stems:
                result["stems"] = stems
        
        if want_lrc:
            lrc_path = write_plain_lrc(lyrics, output_dir / f"{song_name}.lrc")
            if lrc_path:
                result["lrc_path"] = str(lrc_path)
        
        emit_progress("complete", 1.0, "Generation completed")
        return result
        
    except Exception as e:
        raise RuntimeError(f"HeartMuLa generation failed: {str(e)}")


def run_job(config: Dict[str, Any]) -> int:
    """执行单个生成任务。一次性模式和常驻模式共用同一条代码路径，
    唯一区别是常驻模式复用 _MODEL_CACHE 里的权重、且返回后进程不退出。
    """
    # Extract config fields
    model = config.get("model", "").lower()
    output_dir = Path(config.get("output_dir", "."))
    song_name = config.get("song_name", "output")
    prompt = config.get("prompt", "")
    lyrics = config.get("lyrics", "")
    audio_duration = float(config.get("audio_duration", 60.0))
    
    # Get models directory from config or environment
    models_dir_str = config.get("models_dir") or os.environ.get("SONG_MODELS_DIR")
    if not models_dir_str:
        emit_result(success=False, error="models_dir not provided in config or SONG_MODELS_DIR environment")
        return 1
    
    models_dir = Path(models_dir_str)
    if not models_dir.exists():
        emit_result(success=False, error=f"Models directory does not exist: {models_dir}")
        return 1
    
    # Map model names
    model_id_map = {
        "yue2": "yue2-3b",
        "yue-2": "yue2-3b",
        "acestep": "acestep-v1.5",
        "ace-step": "acestep-v1.5",
        "acestep-v1": "acestep-v1-3.5b",
        "acestep-v1.5": "acestep-v1.5",
        "heartmula": "heartmula-3b",
        "heart-mula": "heartmula-3b"
    }
    
    model_id = model_id_map.get(model, model)
    
    # ── stems-only 轻量任务（规划 14.2#1）：对任意源音频直接跑 demucs，
    # 不加载任何生成模型、不做模型文件校验。Rust song_generate 原样透传
    # task="stems_only" + src_audio_path，无需新命令。
    # 注意：必须放在 validate_model_files 之前，否则未下载生成模型时会被拦截。
    if config.get("task") == "stems_only":
        src = config.get("src_audio_path") or ""
        if not src or not Path(src).exists():
            emit_result(success=False, error=f"源音频不存在: {src}")
            return 1
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            emit_progress("initialize", 0.1, "Loading demucs model...")
            stems = separate_stems(Path(src), output_dir / f"{song_name}_stems")
            if not stems:
                emit_result(success=False, error="demucs 分轨失败（首次使用需联网自动安装依赖，请检查网络后重试）")
                return 1
            duration = audio_duration
            try:
                import soundfile as _sf
                duration = _sf.info(src).duration
            except Exception:
                pass
            emit_result(success=True, data={"stems": stems, "duration": duration, "model": "demucs"})
            return 0
        except Exception as e:
            import traceback
            emit_result(success=False, error=f"stems_only failed: {str(e)}\n{traceback.format_exc()}")
            return 1
    
    # Validate model files
    if not validate_model_files(models_dir, model_id):
        emit_result(
            success=False,
            error=f"Model '{model_id}' files not found or incomplete. Please download the model first."
        )
        return 1
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 上面已显式取出的字段必须从 config 里剔除，否则 **config 会与显式关键字重名，
    # 直接抛 TypeError: got multiple values for keyword argument（三个模型都走这条路）。
    explicit_keys = {
        "models_dir",
        "output_dir",
        "song_name",
        "prompt",
        "lyrics",
        "audio_duration",
        "model",
        "model_id",
        "format",
        # 输出产物开关（Rust 侧可能传 null，必须显式 bool 化，避免 None 透传）
        "want_midi",
        "want_stems",
        "want_lrc",
    }
    extra_params = {k: v for k, v in config.items() if k not in explicit_keys}
    want_midi = bool(config.get("want_midi") or False)
    want_stems = bool(config.get("want_stems") or False)
    want_lrc = bool(config.get("want_lrc") or False)

    try:
        # Route to appropriate generator with all config parameters
        if model_id == "yue2-3b":
            result_data = generate_yue2_embedded(
                models_dir=models_dir,
                output_dir=output_dir,
                song_name=song_name,
                prompt=prompt,
                lyrics=lyrics,
                audio_duration=audio_duration,
                want_midi=want_midi,
                want_stems=want_stems,
                want_lrc=want_lrc,
                **extra_params
            )
        elif model_id in ["acestep-v1-3.5b", "acestep-v1.5"]:
            result_data = generate_acestep_embedded(
                models_dir=models_dir,
                output_dir=output_dir,
                song_name=song_name,
                prompt=prompt,
                lyrics=lyrics,
                audio_duration=audio_duration,
                model_id=model_id,
                **extra_params
            )
        elif model_id == "heartmula-3b":
            result_data = generate_heartmula_embedded(
                models_dir=models_dir,
                output_dir=output_dir,
                song_name=song_name,
                prompt=prompt,
                lyrics=lyrics,
                audio_duration=audio_duration,
                want_midi=want_midi,
                want_stems=want_stems,
                want_lrc=want_lrc,
                **extra_params
            )
        else:
            emit_result(success=False, error=f"Unsupported model: {model_id}")
            return 1
        
        emit_result(success=True, data=result_data)
        return 0
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        emit_result(success=False, error=error_msg)
        return 1


def _daemon_loop() -> int:
    """常驻模式主循环：从 stdin 逐行读取 JSON 指令，直到收到 shutdown 或 EOF。

    指令格式（每行一个 JSON 对象）：
      {"cmd": "generate", "config": {...}}   执行一次生成，结束后发 @@RESULT@@
      {"cmd": "generate", "config_path": "..."}  同上，但从文件读 config
      {"cmd": "release_vram"}                立即把权重搬到 CPU 释放显存
      {"cmd": "set_idle_timeout", "seconds": N}  运行时改空闲超时（不丢弃权重）
      {"cmd": "ping"}                        心跳，回一个 ready
      {"cmd": "shutdown"}                    卸载并退出进程

    stdout 上除 @@PROGRESS@@ / @@RESULT@@ / @@KEEPALIVE@@ 外不产生其他协议输出，
    Rust 侧的解析逻辑无需区分一次性/常驻两种模式。
    """
    watchdog = threading.Thread(target=_idle_watchdog, name="utai-idle-watchdog", daemon=True)
    watchdog.start()
    emit_keepalive("ready", f"daemon started, idle_timeout={_STATE['idle_timeout']}s")

    for line in sys.stdin:
        # 去掉 BOM：某些 shell/宿主在首行写入 UTF-8 BOM，会让 json.loads 直接失败
        line = line.strip().lstrip("\ufeff")
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception as e:
            emit_result(success=False, error=f"Malformed daemon command: {e}")
            continue

        cmd = str(msg.get("cmd") or "").lower()

        if cmd == "shutdown":
            release_vram("shutdown")
            _MODEL_CACHE.clear()
            emit_keepalive("shutdown", "daemon exiting")
            return 0

        if cmd == "ping":
            emit_keepalive("ready", "vram_released" if _STATE["vram_released"] else "vram_active")
            continue

        if cmd == "release_vram":
            release_vram("explicit request")
            continue

        if cmd == "set_idle_timeout":
            try:
                _STATE["idle_timeout"] = max(60, int(msg.get("seconds") or 300))
            except (TypeError, ValueError):
                emit_result(success=False, error="set_idle_timeout requires an integer 'seconds'")
                continue
            emit_keepalive("ready", f"idle_timeout={_STATE['idle_timeout']}s")
            continue

        if cmd == "generate":
            config = msg.get("config")
            if config is None:
                cfg_path = msg.get("config_path")
                if not cfg_path:
                    emit_result(success=False, error="generate command requires config or config_path")
                    continue
                try:
                    with open(str(cfg_path).lstrip("@"), "r", encoding="utf-8") as f:
                        config = json.load(f)
                except Exception as e:
                    emit_result(success=False, error=f"Failed to load config: {e}")
                    continue
            if not isinstance(config, dict):
                emit_result(success=False, error="config must be a JSON object")
                continue

            _STATE["busy"] = True
            try:
                # run_job 内部已经用 emit_result 报告成功/失败，异常不会外泄；
                # 这里再兜一层，保证任何意外都不会打断守护循环。
                run_job(config)
            except BaseException as e:
                import traceback
                emit_result(success=False, error=f"{e}\n{traceback.format_exc()}")
            finally:
                _STATE["busy"] = False
                _STATE["last_activity"] = time.time()
            emit_keepalive("ready", "idle")
            continue

        emit_result(success=False, error=f"Unknown daemon command: {cmd}")

    # stdin 被关闭（Rust 侧退出或显式 close）：正常收尾
    release_vram("stdin closed")
    _MODEL_CACHE.clear()
    emit_keepalive("shutdown", "stdin closed")
    return 0


def main():
    """Main entry point. 默认一次性执行；--daemon 进入常驻模式。"""
    parser = argparse.ArgumentParser(description="MunoAI Song Generation Sidecar (Embedded)")
    parser.add_argument("--config", help="Config JSON file path (prefixed with @)")
    parser.add_argument("--daemon", action="store_true",
                        help="Resident mode: keep process alive and read jobs from stdin")
    parser.add_argument("--idle-timeout", type=int, default=300,
                        help="Seconds of inactivity before unloading weights to CPU (daemon only)")
    args = parser.parse_args()

    # ⚠️ 关键防御：ACE-Step 会往 <project_root>/.cache/ 写运行时缓存
    # （progress_estimates.json 等）。若 project_root 落在 src-tauri/ 内，
    # tauri dev 的文件监视器检测到写入后会杀掉整个应用重启 —— 这就是
    # "生成中软件自动退出"的根因。此处把 project_root 重定向到
    # data/cache/acestep_work（Rust 端 spawn 时也会设置同名环境变量，
    # setdefault 保证不覆盖）。手动命令行运行同样受此保护。
    _ace_root = os.environ.get("ACESTEP_PROJECT_ROOT")
    if not _ace_root:
        _models_dir = os.environ.get("SONG_MODELS_DIR", "")
        if _models_dir:
            _data_dir = os.path.dirname(os.path.dirname(_models_dir.rstrip("\\/")))
            _ace_root = os.path.join(_data_dir, "cache", "acestep_work") if _data_dir else None
        if not _ace_root:
            import tempfile
            _ace_root = os.path.join(tempfile.gettempdir(), "muno_acestep_work")
        os.environ["ACESTEP_PROJECT_ROOT"] = _ace_root
    os.makedirs(_ace_root, exist_ok=True)
    print(f"[sidecar] ACESTEP_PROJECT_ROOT={_ace_root} (cache redirected outside src-tauri)",
          file=sys.stderr, flush=True)

    if args.daemon:
        _STATE["daemon"] = True
        _STATE["idle_timeout"] = max(60, int(args.idle_timeout or 300))
        _STATE["last_activity"] = time.time()
        return _daemon_loop()

    if not args.config:
        emit_result(success=False, error="--config is required unless --daemon is set")
        return 1

    config_path = args.config.lstrip("@")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        emit_result(success=False, error=f"Failed to load config: {str(e)}")
        return 1

    return run_job(config)


if __name__ == "__main__":
    sys.exit(main())
