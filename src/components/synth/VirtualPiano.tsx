import { useCallback, useEffect, useRef, useState } from "react";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useHistoryStore } from "../../store/history";
import { playSingleNote } from "../../lib/audio/previewNote";
import { resolveMelodyTrackId } from "../../lib/arrangement/autoArrange";
import { ArrangeDialog } from "./ArrangeDialog";
import { MixerConsole } from "./MixerConsole";
import "./VirtualPiano.css";

/** MIDI note number for C3 (中央 C 下八度的 C) */
const C3 = 48;
const OCTAVES = 2;
const WHITE_KEYS_PER_OCTAVE = 7;
const TOTAL_WHITE = OCTAVES * WHITE_KEYS_PER_OCTAVE; // 14 keys

/** 键盘 → MIDI 映射 */
const KEYBOARD_MAP_LOWER: Record<string, number> = {
  z: C3, s: C3 + 1, x: C3 + 2, d: C3 + 3, c: C3 + 4, v: C3 + 5, g: C3 + 6,
  b: C3 + 7, h: C3 + 8, n: C3 + 9, j: C3 + 10, m: C3 + 11,
};
const KEYBOARD_MAP_UPPER: Record<string, number> = {
  q: C3 + 12, "2": C3 + 13, w: C3 + 14, "3": C3 + 15, e: C3 + 16,
  r: C3 + 17, "5": C3 + 18, t: C3 + 19, "6": C3 + 20, y: C3 + 21,
  "7": C3 + 22, u: C3 + 23,
};
const KEYBOARD_MAP = { ...KEYBOARD_MAP_LOWER, ...KEYBOARD_MAP_UPPER };

/** 白键序号 → note number */
function whiteIndexToNote(idx: number): number {
  const octave = Math.floor(idx / WHITE_KEYS_PER_OCTAVE);
  const step = idx % WHITE_KEYS_PER_OCTAVE;
  const whiteSteps = [0, 2, 4, 5, 7, 9, 11];
  return C3 + octave * 12 + (whiteSteps[step] as number);
}

export function VirtualPiano() {
  const [open, setOpen] = useState(false);
  const [activeNotes, setActiveNotes] = useState<Set<number>>(new Set());
  const [arrangeTarget, setArrangeTarget] = useState<string | null>(null);
  const [consoleOpen, setConsoleOpen] = useState(false);
  const pressedKeys = useRef<Set<string>>(new Set());
  const inspectorTrackId = useAppStore((s) => s.inspectorTrackId);

  // —— 底部工作流快捷按钮 ——
  const toast = (msg: string, kind: "info" | "error" = "info") =>
    useAppStore.getState().showToast(msg, kind);

  const openSongStudio = () => useAppStore.getState().toggleSongStudio();

  const openSuperWizard = () => window.dispatchEvent(new CustomEvent("utai:open-super-wizard"));

  const openSmartArrange = () => {
    const tid = resolveMelodyTrackId();
    if (!tid) { toast("请先选中或创建一条有音符的旋律轨", "error"); return; }
    setArrangeTarget(tid);
  };

  // 注: 转MIDI / 和弦MIDI / 识别和弦 / 识别鼓点 已集成到轨道右键菜单 (TrackList)

  // 键盘事件 — 全局监听
  useEffect(() => {
    if (!open) return;
    const onDown = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA")) return;
      // 修饰键组合 (Ctrl/Cmd/Alt) 属于应用快捷键或系统手势 —— 别把 Ctrl+Z/S/C/V… 误触发成琴键音。
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const k = e.key.toLowerCase();
      if (pressedKeys.current.has(k)) return;
      const note = KEYBOARD_MAP[k];
      if (note !== undefined) {
        pressedKeys.current.add(k);
        setActiveNotes((s) => new Set([...s, note]));
        playSingleNote(note);
      }
    };
    const onUp = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      pressedKeys.current.delete(k);
      const note = KEYBOARD_MAP[k];
      if (note !== undefined) {
        setActiveNotes((s) => {
          const next = new Set(s);
          next.delete(note);
          return next;
        });
      }
    };
    window.addEventListener("keydown", onDown);
    window.addEventListener("keyup", onUp);
    return () => {
      window.removeEventListener("keydown", onDown);
      window.removeEventListener("keyup", onUp);
      // 收起/卸载时清掉挂起的按键与高亮 —— 否则按住键时合上钢琴，keyup 监听已移除，
      // pressedKeys 里的键会「卡死」（重开后按它不再发声），activeNotes 残留的高亮也不会消失。
      pressedKeys.current.clear();
      setActiveNotes(new Set());
    };
  }, [open]);

  const writeNoteToTrack = useCallback((note: number) => {
    const trackId = inspectorTrackId;
    if (!trackId || !inspectorTrackId) return;
    const track = useProjectStore.getState().tracks.find((t) => t.id === trackId)!;
    if (!track || (track.trackType !== "instrument" && track.trackType !== "vocal")) return;
    const { playheadTick, updateTrack } = useProjectStore.getState();
    const seg: any = {
      id: "vp-" + Date.now() + "-" + Math.random().toString(36).slice(2, 6),
      startTick: playheadTick,
      lengthTicks: 120,
      durationTicks: 120,
      content: { type: "notes" as const, notes: [{ pitch: note, startTick: 0, lengthTicks: 120, velocity: 80 }] },
    };
    useHistoryStore.getState().beginTransaction();
    updateTrack(trackId, { segments: [...track.segments, seg] });
    useHistoryStore.getState().commitTransaction();
  }, [inspectorTrackId]);

  const onMouseDown = useCallback((note: number) => {
    setActiveNotes((s) => new Set([...s, note]));
    playSingleNote(note);
    writeNoteToTrack(note);
  }, [writeNoteToTrack]);

  const onMouseUp = useCallback((note: number) => {
    setActiveNotes((s) => {
      const next = new Set(s);
      next.delete(note);
      return next;
    });
  }, []);

  // 构建黑键列表 (白键之间, 跳过 E-F 和 B-C)
  const blackKeys: Array<{ note: number; leftPct: number }> = [];
  const whiteSteps = [0, 2, 4, 5, 7, 9, 11];
  for (let o = 0; o < OCTAVES; o++) {
    for (let w = 0; w < 7; w++) {
      if (w === 2 || w === 6) continue; // E/B 后无黑键
      const whiteNote = C3 + o * 12 + (whiteSteps[w] as number);
      const blackNote = whiteNote + 1;
      const leftPct = ((o * 7 + w + 1) / TOTAL_WHITE) * 100;
      blackKeys.push({ note: blackNote, leftPct });
    }
  }

  const whiteNoteNames = ["C", "D", "E", "F", "G", "A", "B"];

  return (
    <div className={`vp-container ${open ? "vp-open" : ""}`}>
      <div className="vp-bar">
        <button className="vp-toggle" onClick={() => setOpen((o) => !o)} title="虚拟钢琴 (2 八度 C3-C5)">
          🎹 {open ? "收起钢琴" : "虚拟钢琴"}
        </button>
        <span className="vp-sep" />
        <button className="vp-quick" onClick={openSongStudio} title="打开歌曲制作工作台">🎧 歌曲制作</button>
        <button className="vp-quick" onClick={openSuperWizard} title="选歌 + 模板 → 一键生成原创歌曲">🪄 超级原创</button>
        <button className="vp-quick" onClick={openSmartArrange} title="AI 自动编曲 (鼓/贝斯/钢琴/铺底)">✨ 智能编曲</button>
        <span className="vp-sep" />
        <button className="vp-quick" onClick={() => setConsoleOpen(true)} title="控制台: 所有轨道音量/平衡/静音独奏 + 总输出">🎛 控制台</button>
      </div>
      {open && (
        <div className="vp-keyboard">
          {/* 白键 */}
          <div className="vp-white-row">
            {Array.from({ length: TOTAL_WHITE }, (_, i) => {
              const note = whiteIndexToNote(i);
              const isActive = activeNotes.has(note);
              const pc = note % 12;
              const name = whiteNoteNames[pc === 0 ? 0 : pc === 2 ? 1 : pc === 4 ? 2 : pc === 5 ? 3 : pc === 7 ? 4 : pc === 9 ? 5 : 6];
              return (
                <button
                  key={i}
                  className={`vp-white ${isActive ? "vp-active" : ""}`}
                  onMouseDown={() => onMouseDown(note)}
                  onMouseUp={() => onMouseUp(note)}
                  onMouseLeave={() => onMouseUp(note)}
                >
                  <span className="vp-white-label">{name}{Math.floor(note / 12) - 1}</span>
                </button>
              );
            })}
          </div>
          {/* 黑键 */}
          <div className="vp-black-row">
            {blackKeys.map(({ note, leftPct }) => {
              const isActive = activeNotes.has(note);
              return (
                <button
                  key={note}
                  className={`vp-black ${isActive ? "vp-active-black" : ""}`}
                  style={{ left: `${leftPct}%` }}
                  onMouseDown={() => onMouseDown(note)}
                  onMouseUp={() => onMouseUp(note)}
                  onMouseLeave={() => onMouseUp(note)}
                />
              );
            })}
          </div>
          <div className="vp-hint">
            💡 键盘: <code>Z S X D C V G B H N J M</code> (低八度) · <code>Q 2 W 3 E R 5 T 6 Y 7 U</code> (高八度) · 点琴键 → 写入当前选中轨
          </div>
        </div>
      )}
      {arrangeTarget && <ArrangeDialog trackId={arrangeTarget} onClose={() => setArrangeTarget(null)} />}
      {consoleOpen && <MixerConsole onClose={() => setConsoleOpen(false)} />}
    </div>
  );
}
