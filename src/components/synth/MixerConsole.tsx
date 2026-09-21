import { useState, useEffect } from "react";
import { useProjectStore } from "../../store/project";
import { useHistoryStore } from "../../store/history";
import { VolumeFader, formatDb, formatPan } from "../common/VolumeFader";
import { OutLevelMeter } from "../common/OutLevelMeter";
import { updateTrackVolume, updateTrackPan, updateTrackAudibility, getContext } from "../../lib/audio/playback";
import { getMasterVolumeDb, setMasterVolume } from "../../lib/audio/effectsBus";
import { loadSetting, saveSetting } from "../../lib/settings";
import { FADER_MIN_DB, FADER_MAX_DB } from "../../lib/constants";
import { trackTypeCssVar } from "../../lib/trackColors";
import "./MixerConsole.css";

/** dB 读数 — 点击变成输入框, 可手动输入精确值 (Enter 确认 / Esc 取消 / 失焦提交) */
function DbReadout({ value, min, max, onCommit, title }: {
  value: number; min: number; max: number; onCommit: (v: number) => void; title: string;
}) {
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState("");

  if (!editing) {
    return (
      <span
        className="mixer-strip-db mixer-strip-db-click"
        title={title}
        onClick={() => { setText(String(Math.round(value * 10) / 10)); setEditing(true); }}
      >{formatDb(value, min)}</span>
    );
  }
  const commit = () => {
    const v = parseFloat(text);
    if (!Number.isNaN(v)) {
      onCommit(Math.max(min, Math.min(max, Math.round(v * 10) / 10)));
    }
    setEditing(false);
  };
  return (
    <input
      className="mixer-strip-db-input"
      type="number"
      step={0.1}
      min={min}
      max={max}
      autoFocus
      value={text}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") commit();
        if (e.key === "Escape") setEditing(false);
      }}
      title="输入 dB 值, Enter 确认, Esc 取消"
    />
  );
}

/** 🎛 控制台 — 全轨道混音台 (Studio Pro Mixer 风格).
 *  底部停靠面板 (非弹窗): 打开后可与页面上的轨道/播放实时交互.
 *  左侧: 每条轨一条通道 (竖直推子 + 分轨电平表 + 静音/独奏 + 平衡);
 *  最右侧: 总输出条 (主音量推子 + 实时电平). */
export function MixerConsole({ onClose }: { onClose: () => void }) {
  const tracks = useProjectStore((s) => s.tracks);
  const updateTrack = useProjectStore((s) => s.updateTrack);
  // 主输出音量 (dB) — 持久化, 打开控制台时恢复
  const [masterDb, setMasterDb] = useState(() => getMasterVolumeDb());

  useEffect(() => {
    const db = loadSetting("utai.masterVolumeDb", 0);
    setMasterVolume(getContext(), db);
    setMasterDb(db);
  }, []);

  const setVol = (id: string, v: number) => {
    updateTrack(id, { volumeDb: v });
    updateTrackVolume(id, v);
  };
  const setPan = (id: string, v: number) => {
    updateTrack(id, { pan: v });
    updateTrackPan(id, v);
  };
  const toggleMute = (id: string, muted: boolean) => {
    useHistoryStore.getState().beginTransaction();
    updateTrack(id, { muted: !muted });
    useHistoryStore.getState().commitTransaction();
    updateTrackAudibility(useProjectStore.getState().tracks);
  };
  const toggleSolo = (id: string, solo: boolean) => {
    useHistoryStore.getState().beginTransaction();
    updateTrack(id, { solo: !solo });
    useHistoryStore.getState().commitTransaction();
    updateTrackAudibility(useProjectStore.getState().tracks);
  };
  const setMaster = (v: number) => {
    setMasterVolume(getContext(), v);
    saveSetting("utai.masterVolumeDb", v);
    setMasterDb(v);
  };

  return (
    <div className="mixer-dock" role="region" aria-label="控制台">
      <div className="mixer-header">
        <span className="mixer-title">🎛 控制台</span>
        <span className="mixer-hint">上下拖动推子调音量 · 静音/独奏 · 平衡=左右声道 · 播放时可实时操作</span>
        <button className="mixer-close" onClick={onClose} title="收起控制台">✕</button>
      </div>
      <div className="mixer-strips">
          {tracks.map((t, i) => {
            const color = t.color || trackTypeCssVar(t.trackType);
            return (
              <div key={t.id} className="mixer-strip" style={{ borderTopColor: color }}>
                <div className="mixer-strip-name" style={{ color }} title={t.name}>
                  <span className="mixer-strip-no" style={{ color }}>{i + 1}</span> {t.name}
                </div>
                <div className="mixer-strip-ms">
                  <button
                    className={`mixer-ms-btn ${t.muted ? "active-mute" : ""}`}
                    onClick={() => toggleMute(t.id, t.muted)}
                    title="静音"
                  >静音</button>
                  <button
                    className={`mixer-ms-btn ${t.solo ? "active-solo" : ""}`}
                    onClick={() => toggleSolo(t.id, t.solo)}
                    title="独奏"
                  >独奏</button>
                </div>
                <div className="mixer-strip-fader">
                  <div className="mixer-fader-meter">
                    <VolumeFader
                      value={t.volumeDb}
                      min={FADER_MIN_DB}
                      max={FADER_MAX_DB}
                      orientation="vertical"
                      width={32}
                      height={170}
                      onChange={(v) => setVol(t.id, v)}
                      onGestureStart={() => useHistoryStore.getState().beginTransaction()}
                      onGestureEnd={() => useHistoryStore.getState().commitTransaction()}
                      tip="轨道音量"
                    />
                    <OutLevelMeter trackId={t.id} width={10} height={170} />
                  </div>
                  <span className="mixer-strip-db-row">
                    <DbReadout value={t.volumeDb} min={FADER_MIN_DB} max={FADER_MAX_DB}
                      onCommit={(v) => setVol(t.id, v)} title="点击手动输入音量 dB" />
                  </span>
                </div>
                <div className="mixer-strip-pan">
                  <span className="mixer-mini-label">平衡</span>
                  <VolumeFader
                    value={t.pan}
                    min={-1}
                    max={1}
                    width={84}
                    step={0.1}
                    fillFrom="center"
                    format={formatPan}
                    onChange={(v) => setPan(t.id, v)}
                    onGestureStart={() => useHistoryStore.getState().beginTransaction()}
                    onGestureEnd={() => useHistoryStore.getState().commitTransaction()}
                    tip="左右声道平衡"
                  />
                </div>
              </div>
            );
          })}
          {tracks.length === 0 && (
            <div className="mixer-empty">工程里还没有轨道 — 先创建一条轨道再来</div>
          )}
          <div className="mixer-strip mixer-master">
            <div className="mixer-strip-name" style={{ color: "#c084fc" }} title="总输出">🔊 总输出</div>
            <div className="mixer-master-body">
              <div className="mixer-fader-meter">
                <VolumeFader
                  value={masterDb}
                  min={FADER_MIN_DB}
                  max={FADER_MAX_DB}
                  orientation="vertical"
                  width={32}
                  height={170}
                  onChange={setMaster}
                  onGestureStart={() => useHistoryStore.getState().beginTransaction()}
                  onGestureEnd={() => useHistoryStore.getState().commitTransaction()}
                  tip="总输出音量 (整个软件的监听音量, 不影响导出)"
                />
                <OutLevelMeter width={16} height={170} />
              </div>
              <span className="mixer-strip-db-row">
                <DbReadout value={masterDb} min={FADER_MIN_DB} max={FADER_MAX_DB}
                  onCommit={setMaster} title="点击手动输入总输出音量 dB" />
              </span>
            </div>
          </div>
      </div>
    </div>
  );
}
