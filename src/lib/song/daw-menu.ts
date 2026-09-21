/**
 * 规划 10.2/10.3/10.4：DAW 右键 → 歌曲制作带参打开 的共用菜单构造。
 * DAW 不直接执行任务：setPendingSongTask + toggleSongStudio，参数确认仍在歌曲弹窗完成。
 * 文案统一走 i18n（10.6），不硬编码到 TrackList/Arrangement。
 * 
 * 规划 10.2 Ctrl 快捷执行（P2-15.b）：分轨这类无需参数的任务可按住 Ctrl 直接执行默认设置。
 */
import i18n from "../../i18n";
import { useAppStore, type PendingSongTask } from "../../store/app";
import { useProjectStore } from "../../store/project";
import { useAudioStore } from "../../store/audio";
import { enqueueSongTask } from "../../store/song-task-queue";
import { backendErrorMessage } from "../backendError";
import type { SongSource } from "./types";
import type { MenuItem } from "../../components/common/ContextMenu";

// 模块级 Ctrl 键状态追踪（P2-15.b：ContextMenu onClick 不传事件，需另外监听）
//
// 只认「右键按下那一刻」的 Ctrl 状态：菜单弹出后用户必然要松开 Ctrl 才能移动鼠标点菜单项，
// 所以不能用「点击时 Ctrl 是否按住」来判断。改为在 mousedown/contextmenu 阶段快照 e.ctrlKey，
// 这样既符合真实操作序列，也彻底避开 keyup 被弹窗/焦点转移吞掉导致 isCtrlPressed 残留 true
// 而意外触发直接分轨（会凭空建出一堆轨道）的风险。
let ctrlAtMenuOpen = false;
if (typeof window !== "undefined") {
  // capture 阶段：早于 React 的 onContextMenu，保证菜单构造时快照已就位
  window.addEventListener("mousedown", (e) => { if (e.button === 2) ctrlAtMenuOpen = e.ctrlKey; }, true);
  window.addEventListener("contextmenu", (e) => { ctrlAtMenuOpen = e.ctrlKey; }, true);
}

export interface DawSongMenuOptions {
  source: SongSource;
  sourceLabel?: string;
  /** 已解析出的音频路径（可用性判断 + 快速校验） */
  audioPath?: string | null;
  /** 回流目标轨道/片段 */
  trackId?: string;
  segmentId?: string;
  /** repaint 选区（源音频秒） */
  range?: [number, number];
}

function openSongStudioWith(p: PendingSongTask) {
  useAppStore.getState().setPendingSongTask(p);
  const app = useAppStore.getState();
  if (!app.songStudioOpen) app.toggleSongStudio();
}

/** 构造"歌曲模型 ▸"二级子菜单项集合（与歌曲弹窗 8 模式对齐，源音频任务仅在有音频时可用） */
export function songTaskSubmenuItems(opts: DawSongMenuOptions): MenuItem[] {
  const t = (k: string) => i18n.t(k);
  const hasAudio = !!opts.audioPath;
  const base = {
    source: opts.source,
    sourceLabel: opts.sourceLabel,
    returnTarget:
      opts.trackId
        ? { kind: "track" as const, trackId: opts.trackId, segmentId: opts.segmentId, align: true }
        : undefined,
  };
  const item = (task: PendingSongTask["task"], key: string, icon: string, needAudio = true, extra?: Partial<PendingSongTask>): MenuItem => ({
    label: t(key),
    icon,
    disabled: needAudio && !hasAudio,
    title: needAudio && !hasAudio ? t("songStudio.noMidiHint") : undefined,
    onClick: () => openSongStudioWith({ ...base, task, ...extra }),
  });

  // 规划 10.2：extract（分轨）支持 Ctrl 快捷执行——按住 Ctrl 直接以 demucs 默认参数执行并回流
  const extractItem = (): MenuItem => {
    return {
      label: t("songMenu.remixExtract"),
      icon: "✂",
      disabled: !hasAudio,
      title: !hasAudio ? t("songStudio.noMidiHint") : undefined,
      onClick: () => {
        if (ctrlAtMenuOpen && hasAudio && opts.audioPath) {
          // Ctrl 快捷路径：直接跑 demucs 分轨（stems task = 快速四分轨，无需 trackClasses 参数）
          const audioPath = opts.audioPath;
          const durationMs = useAudioStore.getState().audioFiles[audioPath]?.durationMs ?? 0;
          const app = useAppStore.getState();
          const label = opts.sourceLabel || t("songTask.stems");
          // 该路径不打开歌曲弹窗，弹窗内的进度条看不到 —— 必须用全局 toast 交代开始/结束/失败，
          // 否则用户按下 Ctrl 点完之后要盲等几十秒，完全没有反馈。
          app.showToast(i18n.t("songMenu.quickStemsStart", { name: label }), "info");
          enqueueSongTask({
            task: "stems",
            label,
            payload: {
              songName: opts.sourceLabel || "quick_stems",
              model: "demucs",
              srcAudioPath: audioPath,
              srcDurationSec: durationMs / 1000,
            },
            onProgress: () => {},
          }).then(async (result) => {
            // 回流：result.outputs[0].stems 是 { vocals, bass, drums, other } 对象，分别建轨
            const stems = result?.outputs?.[0]?.stems;
            if (!stems) {
              app.showToast(i18n.t("songMenu.quickStemsEmpty"), "warning");
              return;
            }
            const project = useProjectStore.getState();
            const startTick = 0;
            // 固定 ticksPerBeat = 480（全局标准），按 tempo 计算 ticks
            const ticksPerBeat = 480;
            const durationTicks = Math.round((durationMs / 1000) * (project.tempo / 60) * ticksPerBeat);

            let added = 0;
            for (const [className, stemPath] of Object.entries(stems)) {
              if (!stemPath) continue;
              const trackLabel = `${label} - ${className}`;
              
              // 新建轨道（匹配 Track 类型）
              project.addTrack({
                id: `track-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
                name: trackLabel,
                trackType: "audio",
                segments: [
                  {
                    id: `seg-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
                    startTick,
                    durationTicks,
                    content: {
                      type: "audioClip",
                      sourcePath: stemPath,
                      offsetMs: 0,
                      totalDurationMs: durationMs,
                    },
                  },
                ],
                laneControls: {},
                expanded: true,
                volumeDb: 0,
                pan: 0,
                solo: false,
                muted: false,
              });
              // 注册音频元数据供波形/播放器
              useAudioStore.getState().loadAudioFile(stemPath);
              added++;
            }
            if (added === 0) app.showToast(i18n.t("songMenu.quickStemsEmpty"), "warning");
            else app.showToast(i18n.t("songMenu.quickStemsDone", { count: added }), "success");
          }).catch((err) => {
            // 失败必须让用户看见：模型未下载、格式不支持、被取消都会走到这里。
            // 只 console.error 的话用户只会觉得「点了没反应」。
            app.showToast(
              i18n.t("songMenu.quickStemsFailed", { error: backendErrorMessage(err) ?? String(err) }),
              "error",
            );
          });
        } else {
          // 普通路径：打开歌曲弹窗让用户选参数
          openSongStudioWith({ ...base, task: "extract" });
        }
      },
    };
  };

  return [
    item("cover", "songMenu.remixCover", "🎤"),
    item("repaint", "songMenu.remixRepaint", "🖌", true, opts.range ? { range: opts.range } : undefined),
    item("complete", "songMenu.remixComplete", "🧩"),
    item("lego", "songMenu.remixLego", "🧱"),
    extractItem(),
    item("sheet", "songMenu.remixSheet", "🎹"),
  ];
}
