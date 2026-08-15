import { useEffect, useState } from "react";
import { ParamSlider } from "../workflow/nodes/ParamSlider";
import { t18 } from "../../lib/models/msst-catalog";
import { autoBounds, boundsAreEdited, clampBounds, targetRange } from "../../lib/vocal/rangeBounds";
import { midiName, setVocalRangeBounds, type SpeakerRangeRecord } from "../../lib/vocal/rangeTest";
import type { VoiceType } from "../../store/voice-models";

/** S146e —— 「可用范围 / 目标范围」编辑器,资源管理器与人声侧栏**共用这一份**。
 *
 *  ⭐ S146f 术语:磁盘键名仍是 `usable` / `comfort`(改键名会作废所有现存记录),但 UI 上
 *  叫「可用范围」与「目标范围」。「舒适区」这个旧词是**误导**的 —— 它听起来是描述性的
 *  (模型哪儿舒服),而它已经是**指令性**的:被救的音落在哪里,以用户设的为准。
 *
 *  ⛔ 三条形状是从 S146e 侦察量出来的坑,别改回去:
 *
 *  ⒜ **本地暂存 + 显式 OK**,不是每帧写盘。侧栏那个 `Slider` 组件会 `beginTransaction()`
 *     开项目历史事务,而真正的写入是 `set_model_vocal_range` → 磁盘 sidecar,**根本不进
 *     undo** ⇒ 直接复用它,拖一次滑条 = 几十次磁盘写 + 几十次 `fetchModels()` 全量重扫,
 *     外加 undo 栈里一串什么都撤销不了的空事务。
 *
 *  ⒝ **两个边界一次写完**(`setVocalRangeBounds`)。后端 `validate_range_record` 会拒绝
 *     逃出扫描带的组合(`RANGE_INVALID`),而那条错误今天在 UI 上完全不可见。
 *     ⛔ S146f 起两个旋钮**正交**:`usable` 说「哪些音要救」,`comfort` 说「落点去哪」,
 *     comfort 由**扫描带**约束而不是由 usable 约束,谁也不许动谁 —— 旧夹取把用户的
 *     comfort 从 79 跟着拖到 74,此后落点永远够不着它,那个旋钮就悄悄不做事了。
 *
 *  ⒞ **必须写出歌手名**。侧栏拿的是 `governingSpeakerId`(spk_mix 里权重最大的那位),
 *     用户从没显式选过它 —— 不标注就是无声地改错人。
 *
 *  ⚠ 这两个值是**模型的磁盘 sidecar**,全局于整台机器:不进 `.usp`、不进 undo、换项目
 *     重开 app 都还在,发给别人不跟着走。所以底下那行说明文字是功能的一部分,不是装饰。 */
export function RangeBoundsEditor({
  sp, modelName, backend, speakerId, speakerLabel, lang, onClose,
}: {
  sp: SpeakerRangeRecord;
  modelName: string;
  backend: Exclude<VoiceType, "vocoder">;
  speakerId: number;
  /** 正在被改的那位歌手的显示名。侧栏必须传(见 ⒞);单歌手模型可省。 */
  speakerLabel?: string;
  lang: string;
  onClose?: () => void;
}) {
  const auto = autoBounds(sp);
  const shownTarget = targetRange(sp);
  const [uLo, setULo] = useState(sp.usable[0]);
  const [uHi, setUHi] = useState(sp.usable[1]);
  const [cLo, setCLo] = useState(shownTarget[0]);
  const [cHi, setCHi] = useState(shownTarget[1]);
  const [busy, setBusy] = useState(false);

  // 换歌手 / 换模型 = 换记录 ⇒ 重新播种,否则四个滑条还停在上一位歌手的数字上,
  // 而 OK 会把那些数字写到这一位头上。
  useEffect(() => {
    setULo(sp.usable[0]);
    setUHi(sp.usable[1]);
    const c = targetRange(sp);
    setCLo(c[0]);
    setCHi(c[1]);
  }, [modelName, backend, speakerId, sp]);

  // 显示的永远是**夹取之后**的值:滑条不该展示一个后端不会接受的组合。
  const preview = clampBounds(sp, [uLo, uHi], [cLo, cHi]);

  const commit = async (usable: [number, number], comfort: [number, number]) => {
    setBusy(true);
    try {
      await setVocalRangeBounds(modelName, backend, speakerId, usable, comfort);
    } finally {
      setBusy(false);
      onClose?.();
    }
  };

  return (
    <div className="rbe">
      <ParamSlider
        label={t18({ zh: "可用下限", en: "Usable low", ja: "使用可能域 下限" }, lang)}
        title={t18({
          zh: "模型到底唱不唱得上去的硬边界。收窄它 = 更多音被判成唱不了,于是更多地方触发音域扩展。",
          en: "The hard bound on what the model can voice at all. Narrowing it marks more notes unsingable, so range extension fires in more places.",
          ja: "モデルが発声できるかどうかの硬い境界。狭めるとより多くの音が「歌えない」と判定され、音域拡張の発動箇所が増えます。",
        }, lang)}
        min={auto.usable[0]} max={auto.usable[1]} step={1} value={preview.usable[0]}
        onChange={(v) => setULo(v)} format={midiName}
      />
      <ParamSlider
        label={t18({ zh: "可用上限", en: "Usable high", ja: "使用可能域 上限" }, lang)}
        min={auto.usable[0]} max={auto.usable[1]} step={1} value={preview.usable[1]}
        onChange={(v) => setUHi(v)} format={midiName}
      />
      <ParamSlider
        label={t18({ zh: "目标下限", en: "Target low", ja: "目標範囲 下限" }, lang)}
        title={t18({
          zh: "被救的音落在这个范围里。算法判某个落点没问题而你听着不行时,把上限往下拖,救援就会落得更低——这里写的永远说了算。够不着则退回自动落点,日志里会说。「还原」回到扫描量出来的值。",
          en: "Rescued notes land inside this range — what you set here is always obeyed. When the algorithm calls a landing fine and your ears disagree, drag the ceiling down and the rescues follow. Out of reach ⇒ it falls back to the automatic landing and says so in the log. Reset returns it to the scanned value.",
          ja: "救済された音はこの範囲に着地します——ここで指定した値が常に優先されます。アルゴリズムが問題ないと判定した着地先が耳に合わないときは上限を下げれば救済もそれに従います。届かない場合は自動の着地先に戻り、ログに記録されます。「リセット」で測定値に戻ります。",
        }, lang)}
        min={auto.usable[0]} max={auto.usable[1]} step={1} value={preview.comfort[0]}
        onChange={(v) => setCLo(v)} format={midiName}
      />
      <ParamSlider
        label={t18({ zh: "目标上限", en: "Target high", ja: "目標範囲 上限" }, lang)}
        min={auto.usable[0]} max={auto.usable[1]} step={1} value={preview.comfort[1]}
        onChange={(v) => setCHi(v)} format={midiName}
      />
      <div className="rbe-actions">
        <button
          className="rm-range-btn" disabled={busy}
          onClick={() => void commit(preview.usable, preview.comfort)}
        >
          OK
        </button>
        <button
          className="rm-range-btn" disabled={busy || !boundsAreEdited(sp)}
          title={t18({
            zh: boundsAreEdited(sp) ? "还原为自动检测值" : "当前就是自动检测值",
            en: boundsAreEdited(sp) ? "Reset to the detected values" : "Already at the detected values",
            ja: boundsAreEdited(sp) ? "自動検出値に戻す" : "すでに自動検出値です",
          }, lang)}
          onClick={() => void commit(auto.usable, auto.comfort)}
        >
          {t18({ zh: "还原", en: "Reset", ja: "リセット" }, lang)}
        </button>
        {onClose && (
          <button className="rm-range-btn" disabled={busy} onClick={onClose}>
            {t18({ zh: "取消", en: "Cancel", ja: "キャンセル" }, lang)}
          </button>
        )}
      </div>
      <div className="rbe-note">
        {t18({
          zh: `这两个边界属于模型「${modelName}」${speakerLabel ? `的歌手「${speakerLabel}」` : ""},不属于这条轨道:改动立即对所有用到它的轨道生效,且不随项目保存、撤销不了。`,
          en: `These bounds belong to the model "${modelName}"${speakerLabel ? `, singer "${speakerLabel}"` : ""} — not to this track. Changes apply to every track using it, are not saved with the project, and cannot be undone.`,
          ja: `この2つの境界はモデル「${modelName}」${speakerLabel ? `の話者「${speakerLabel}」` : ""}に属し、このトラックのものではありません。変更は使用中の全トラックに即時反映され、プロジェクトには保存されず、元に戻せません。`,
        }, lang)}
      </div>
    </div>
  );
}
