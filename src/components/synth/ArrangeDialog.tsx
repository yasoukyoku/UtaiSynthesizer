// 智能自动编曲面板(三段式):① 歌曲身份证(实时 BPM/调/拍号/和弦) ② 风格情绪 + 乐器分组勾选
// ③ 底部[试听·清空·生成 N 轨]。引擎/动作层见 lib/arrangement/*;本组件只做交互与展示。
import { useEffect, useMemo, useRef, useState } from "react";
import {
  ARRANGE_STYLES,
  ARRANGE_MOODS,
  type ArrangeStyle,
  type ArrangeMood,
} from "../../lib/arrangement/styles";
import {
  runAutoArrange,
  previewAutoArrange,
  stopAutoArrangePreview,
  clearAutoArrange,
  analyzeSourceProfile,
  type ArrangePartKind,
} from "../../lib/arrangement/autoArrange";
import { useProjectStore } from "../../store/project";
import "./ArrangeDialog.css";

/** 22 种风格中文名(轨道名语言无关,这里仅面板展示)。 */
const STYLE_LABELS: Record<ArrangeStyle, string> = {
  pop: "流行", rock: "摇滚", jazz: "爵士", edm: "电子舞曲", latin: "拉丁",
  funk: "放克", rnb: "R&B", ballad: "抒情", country: "乡村", lofi: "Lo-Fi",
  trap: "Trap", reggae: "雷鬼", bossanova: "波萨诺瓦", kpop: "K-Pop",
  chinese: "中国风", ambient: "氛围", techno: "科技舞曲",
  folk: "民谣", citypop: "都市流行", hiphop: "嘻哈", house: "浩室", cinematic: "史诗影视",
};

/** 6 种情绪中文名(只影响力度/密度/铺底八度)。 */
const MOOD_LABELS: Record<ArrangeMood, string> = {
  neutral: "中性", happy: "明亮", sad: "忧伤",
  energetic: "有力度", chill: "松弛", epic: "宏大",
};

interface InstDef { id: ArrangePartKind; name: string; desc: string; }
interface InstGroup { title: string; items: InstDef[]; }

/** 11 类乐器分 4 组(节奏 / 和声 / 铺底 / 旋律)。 */
const INST_GROUPS: InstGroup[] = [
  {
    title: "节奏组",
    items: [
      { id: "drums", name: "鼓", desc: "节拍骨架" },
      { id: "bass", name: "贝斯", desc: "低音根基" },
    ],
  },
  {
    title: "和声组",
    items: [
      { id: "piano", name: "钢琴", desc: "柱式/琶音" },
      { id: "guitarArp", name: "吉他分解", desc: "拨弦" },
      { id: "guitarStrum", name: "吉他扫弦", desc: "柱式" },
      { id: "epiano", name: "电钢琴", desc: "柔和" },
    ],
  },
  {
    title: "铺底组",
    items: [
      { id: "strings", name: "弦乐", desc: "长音铺底" },
      { id: "chords", name: "铺底Pad", desc: "和弦垫" },
      { id: "synthPad", name: "合成Pad", desc: "空间感" },
    ],
  },
  {
    title: "旋律组",
    items: [
      { id: "pluck", name: "琶音", desc: "颗粒点缀" },
      { id: "melody", name: "AI旋律", desc: "自动加花" },
    ],
  },
];

/** 默认四大件:鼓 + 贝斯 + 钢琴 + 弦乐(主流流行最稳的组合)。 */
const DEFAULT_INST: ArrangePartKind[] = ["drums", "bass", "piano", "strings"];
const ALL_INST: ArrangePartKind[] = INST_GROUPS.flatMap((g) => g.items.map((i) => i.id));

interface Props {
  trackId: string;
  onClose: () => void;
}

/** localStorage key: 编曲最后一次成功生成的配置 — 新工程/重启自动恢复。 */
const LS_ARRANGE_LAST = "muno.arrange.last";

interface PresetRec { style: ArrangeStyle; mood: ArrangeMood; instruments: ArrangePartKind[]; }

/** 5 个快捷预设 — 最常用的乐器组合,一键套用。 */
const QUICK_PRESETS: Array<{ name: string; style: ArrangeStyle; mood: ArrangeMood; instruments: ArrangePartKind[]; emoji: string }> = [
  { name: "流行四大件",  emoji: "🎶", style: "pop",      mood: "neutral",   instruments: ["drums", "bass", "piano", "strings"] },
  { name: "民谣吉他组",  emoji: "🎸", style: "folk",     mood: "chill",     instruments: ["drums", "bass", "guitarStrum", "strings", "melody"] },
  { name: "Lo-Fi Chill", emoji: "🌙", style: "lofi",     mood: "chill",     instruments: ["drums", "bass", "epiano", "synthPad", "pluck"] },
  { name: "摇滚齐奏",    emoji: "🎸", style: "rock",     mood: "energetic", instruments: ["drums", "bass", "guitarStrum", "guitarArp", "melody"] },
  { name: "Jazz Trio",   emoji: "🎷", style: "jazz",     mood: "happy",     instruments: ["drums", "bass", "piano"] },
];

function loadLastPreset(): PresetRec | null {
  try {
    const raw = localStorage.getItem(LS_ARRANGE_LAST);
    if (!raw) return null;
    const rec = JSON.parse(raw);
    if (rec && rec.style && rec.mood && Array.isArray(rec.instruments)) return rec;
    return null;
  } catch { return null; }
}

function saveLastPreset(p: PresetRec) {
  try { localStorage.setItem(LS_ARRANGE_LAST, JSON.stringify(p)); } catch { /* ignore */ }
}

export function ArrangeDialog({ trackId, onClose }: Props) {
  const last = loadLastPreset();
  const [style, setStyle] = useState<ArrangeStyle>(last?.style ?? "pop");
  const [mood, setMood] = useState<ArrangeMood>(last?.mood ?? "neutral");
  const [instruments, setInstruments] = useState<ArrangePartKind[]>(last?.instruments ?? DEFAULT_INST);
  const [previewing, setPreviewing] = useState(false);
  const [busy, setBusy] = useState(false);

  // 订阅 tracks:源轨音符一旦变化(如刚转谱/编辑),身份证实时重算。
  const tracks = useProjectStore((s) => s.tracks);
  const track = tracks.find((t) => t.id === trackId);
  const profile = useMemo(
    () => (trackId ? analyzeSourceProfile(trackId) : null),
    [trackId, tracks],
  );

  // 卸载即停止试听,避免面板关了还在响。
  useEffect(() => () => stopAutoArrangePreview(), []);

  // 风格/情绪/勾选一旦变化,立即停掉正在播的旧预览,保证「听到的 = 当前选的」(首次挂载跳过)。
  const firstSel = useRef(true);
  useEffect(() => {
    if (firstSel.current) { firstSel.current = false; return; }
    stopAutoArrangePreview();
    setPreviewing(false);
  }, [style, mood, instruments]);

  const toggle = (id: ArrangePartKind) =>
    setInstruments((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id],
    );
  const toggleGroup = (group: InstGroup, on: boolean) =>
    setInstruments((prev) => {
      const ids = group.items.map((i) => i.id);
      const rest = prev.filter((x) => !ids.includes(x));
      return on ? [...rest, ...ids] : rest;
    });

  const handlePreview = async () => {
    if (previewing) {
      stopAutoArrangePreview();
      setPreviewing(false);
      return;
    }
    if (instruments.length === 0) return;
    setBusy(true);
    const ok = await previewAutoArrange(trackId, style, mood, {
      instruments,
      onEnded: () => setPreviewing(false),
    });
    setPreviewing(ok);
    setBusy(false);
  };

  const handleClear = () => {
    stopAutoArrangePreview();
    setPreviewing(false);
    clearAutoArrange(trackId);
  };

  const handleGenerate = async () => {
    if (instruments.length === 0) return;
    stopAutoArrangePreview();
    setBusy(true);
    const ok = await runAutoArrange(trackId, style, mood, { instruments });
    setBusy(false);
    if (ok) {
      saveLastPreset({ style, mood, instruments });
      onClose();
    }
  };

  return (
    <div className="arrange-overlay" onClick={onClose}>
      <div className="arrange-panel arrange-panel-wide" onClick={(e) => e.stopPropagation()}>
        <div className="arrange-head">
          <span className="arrange-title">智能编曲</span>
          <button className="arrange-close" onClick={onClose}>✕</button>
        </div>

        <div className="arrange-body">
          {/* ① 歌曲身份证:软件自动从源轨扒出的硬骨架,只看不用改 */}
          <div className="idcard">
            <div className="idcard-title">歌曲身份证 · 自动识别</div>
            {profile ? (
              <>
                <div className="idcard-grid">
                  <div className="idcell"><span className="idcell-k">速度</span><span className="idcell-v">{profile.bpm}</span><span className="idcell-u">BPM</span></div>
                  <div className="idcell"><span className="idcell-k">调性</span><span className="idcell-v">{profile.keyLabel}</span></div>
                  <div className="idcell"><span className="idcell-k">拍号</span><span className="idcell-v">{profile.beatsPerBar}/{profile.beatUnit}</span></div>
                  <div className="idcell"><span className="idcell-k">小节</span><span className="idcell-v">{profile.bars}</span></div>
                </div>
                <div className="idcard-chords" title={profile.chordLabels.join("  ")}>
                  <span className="idcell-k">和弦</span>
                  <span className="idcard-chordseq">{profile.chordLabels.join(" · ") || "—"}</span>
                </div>
              </>
            ) : (
              <div className="idcard-empty">这条轨还没有音符 —— 若是音频请先「AI 转谱」再编曲</div>
            )}
          </div>

          {/* ②-quick: 一键预设行 (5 个常用组合,点一下 style+mood+乐器全填好) */}
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", margin: "0 0 8px 0" }}>
            {QUICK_PRESETS.map((p) => (
              <button
                key={p.name}
                type="button"
                style={{
                  fontSize: 11, padding: "3px 8px", borderRadius: 14,
                  border: "1px solid var(--border-default)", background: (style === p.style && mood === p.mood && instruments.length === p.instruments.length) ? "var(--accent-primary)" : "var(--bg-surface)",
                  color: "var(--text-primary)", cursor: "pointer", fontFamily: "inherit",
                }}
                onClick={() => { setStyle(p.style); setMood(p.mood); setInstruments(p.instruments); }}
                title={`${STYLE_LABELS[p.style]} · ${MOOD_LABELS[p.mood]} · ${p.instruments.length} 轨`}
              >
                {p.emoji} {p.name}
              </button>
            ))}
          </div>

          {/* ②-a 风格 / 情绪 */}
          <div className="arrange-row">
            <label className="arrange-field">
              <span>风格</span>
              <select value={style} onChange={(e) => setStyle(e.target.value as ArrangeStyle)}>
                {ARRANGE_STYLES.map((s) => (
                  <option key={s} value={s}>{STYLE_LABELS[s]}</option>
                ))}
              </select>
            </label>
            <label className="arrange-field">
              <span>情绪</span>
              <select value={mood} onChange={(e) => setMood(e.target.value as ArrangeMood)}>
                {ARRANGE_MOODS.map((m) => (
                  <option key={m} value={m}>{MOOD_LABELS[m]}</option>
                ))}
              </select>
            </label>
          </div>

          {/* ②-b 乐器分组勾选 */}
          <div className="inst-block">
            <div className="inst-block-head">
              <span>乐器自由组合</span>
              <span className="inst-block-actions">
                <button type="button" onClick={() => setInstruments(ALL_INST)}>全选</button>
                <button type="button" onClick={() => setInstruments([])}>清空</button>
              </span>
            </div>
            {INST_GROUPS.map((g) => {
              const ids = g.items.map((i) => i.id);
              const onCount = ids.filter((id) => instruments.includes(id)).length;
              return (
                <div className="inst-group" key={g.title}>
                  <div className="inst-group-head">
                    <span className="inst-group-title">{g.title}</span>
                    <button
                      type="button"
                      className="inst-group-toggle"
                      onClick={() => toggleGroup(g, onCount !== ids.length)}
                    >
                      {onCount === ids.length ? "取消" : "整组"}
                    </button>
                  </div>
                  <div className="inst-chips">
                    {g.items.map((it) => {
                      const on = instruments.includes(it.id);
                      return (
                        <button
                          type="button"
                          key={it.id}
                          className={`inst-chip${on ? " on" : ""}`}
                          aria-pressed={on}
                          onClick={() => toggle(it.id)}
                          title={it.desc}
                        >
                          <span className="inst-chip-name">{it.name}</span>
                          <span className="inst-chip-desc">{it.desc}</span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* ③ 底部操作 */}
        <div className="arrange-foot arrange-foot-split">
          <button className="arrange-btn" onClick={handleClear}>清空伴奏</button>
          <div className="arrange-foot-right">
            <button
              className={`arrange-btn${previewing ? " active" : ""}`}
              disabled={busy || instruments.length === 0 || !profile}
              onClick={handlePreview}
            >
              {previewing ? "■ 停止" : "▶ 试听 8 小节"}
            </button>
            <button
              className="arrange-btn primary"
              disabled={busy || instruments.length === 0 || !profile}
              onClick={handleGenerate}
            >
              {busy ? "处理中…" : `生成 ${instruments.length} 轨`}
            </button>
          </div>
        </div>
        {track && <div className="arrange-source">编曲源：<strong>{track.name}</strong></div>}
      </div>
    </div>
  );
}
