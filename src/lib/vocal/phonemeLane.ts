import type { EsDialectId, Note } from "../../types/project";
import type { VocalTokens } from "../vocalNotes";
import { buildScoreTriples, type ScoreTriple } from "./vocalRender";

/**
 * EVERY input the read-only phoneme lane's preview depends on, as ONE named object.
 *
 * ⚠ Why this exists rather than an inline `JSON.stringify([...])` in the component: the lane caches its
 * last result under a signature and skips the IPC when the signature is unchanged. If an input reaches
 * `preview_vocal_phonemes` but NOT the signature, the lane keeps painting the previous answer — the
 * switch looks broken, or worse, silently shows the other arm's layout. S88 shipped exactly that shape
 * once (`AutoTuneWatcher`'s skip-sig omitted the two lyric tokens, so changing a token never
 * re-tuned — "the fix was dormant at precisely the moment it mattered"), and the S89 preroll switch is
 * the same hazard class: a mutation test proved the inline signature stayed green with the switch
 * removed from it.
 *
 * The defence is structural: both the signature and the IPC payload are derived from THIS object, and
 * `phonemeLane.test.ts` asserts that perturbing ANY field moves the signature. A future input can
 * therefore only be added by adding a field here — which the property test immediately holds to
 * account.
 */
export interface PhonemeLaneInputs {
  notes: Note[];
  tempo: number;
  /** Both lyric triggers: re-pointing either one re-resolves which notes produce phones at all. */
  tokens: VocalTokens;
  /** Track default language (per-note overrides ride on the notes). */
  langId: number;
  /** S89 「自动咬字时序」: changes WHERE every onset consonant sits. */
  consonantPreroll: boolean;
  /** S91 「音素约定」: changes WHICH phones every English note has at all. */
  phonemeSet?: string;
  /** S167 (§E4): changes WHICH phones every Spanish note has (θ/s · ʎ/ʝ). */
  esDialect?: EsDialectId;
}

/** The per-note phone-timing term of the lane signature — the edit changes the SPLIT the preview
 *  returns, so it must move the cache key (same hazard class as the S89 switch). */
function phoneTimingSig(n: Note): string {
  const pt = n.phoneTiming;
  return pt ? `${pt.phones.join(",")}~${pt.scale.join(",")}~${(pt.gainDb ?? []).join(",")}` : "";
}

/** The lane's cache key. Cheap: reads the notes' fields directly, no triple building. */
export function phonemeLaneSig(i: PhonemeLaneInputs): string {
  return JSON.stringify([
    i.notes.map((n) => [n.tick, n.duration, n.lyric, n.pitch, n.lang ?? "", n.phonemeInput ?? "", phoneTimingSig(n)]),
    i.langId,
    i.tokens.breath,
    i.tokens.rest,
    i.tempo,
    i.consonantPreroll,
    i.phonemeSet ?? "",
    i.esDialect ?? "",
  ]);
}

/** The `preview_vocal_phonemes` payload + the parallel arrays the lane needs to map spans back to notes.
 *  Built from the SAME inputs object as the signature — that is the whole point. */
export function phonemeLaneRequest(i: PhonemeLaneInputs): {
  args: {
    score: ScoreTriple[];
    defaultLang: number;
    consonantPreroll: boolean;
    phonemeSet: string | null;
    esDialect: string | null;
  };
  tripleNoteIds: (string | null)[];
  ticksPerFrame: number;
} {
  const { triples, tripleNoteIds, ticksPerFrame } = buildScoreTriples(i.notes, i.tempo, i.tokens, i.langId);
  return {
    args: {
      score: triples,
      defaultLang: i.langId,
      consonantPreroll: i.consonantPreroll,
      phonemeSet: i.phonemeSet ?? null,
      esDialect: i.esDialect ?? null,
    },
    tripleNoteIds,
    ticksPerFrame,
  };
}

/** S167c —— 提交端 `score2cv.rs::apply_phone_edits` 的 scale 夹持,镜像常量。 */
export const PHONE_SCALE_MIN = 0.1;
export const PHONE_SCALE_MAX = 10;

/**
 * S167c —— `score2cv.rs::redistribute_conserving` 的 **faithful mirror**(逐整数一致)。
 *
 * ⚠ 为什么允许这份「重复」(feedback_no_duplication_drift 的例外,理由写死在这):拖动预览
 * 必须画出「松手后真正会生效」的分配 —— 旧的对内 1 帧下限预览允许提交端会取整/夹掉的帧数,
 * 于是松手「回弹」(用户 2026-08-31)。每次 mousemove 走 IPC 不现实 ⇒ 唯一诚实的路是把
 * 提交端的整数数学镜像过来,并用**与 Rust 端相同的测试向量**钉住
 * (`redistribute_conserving_floors_sums_and_is_deterministic` ↔ 本文件的 test)。
 * 改任何一侧都必须同步另一侧与两份向量。
 *
 * 语义:按权重分 `total` 帧,每份 ≥ 1,Σ == total(floor+1 起步,余数按最大小数位分,
 * 平手按下标升序 —— 与 Rust 的 `then(a.cmp(&b))` 一致,决定性)。
 */
export function redistributeConserving(total: number, w: readonly number[]): number[] {
  const n = w.length;
  const spare = total - n;
  const out = new Array<number>(n).fill(1);
  if (spare <= 0) return out;
  const sum = w.reduce((a, x) => a + Math.max(0, x), 0);
  if (sum <= 0) {
    out[n - 1]! += spare;
    return out;
  }
  const exact = w.map((x) => (spare * Math.max(0, x)) / sum);
  const extra = exact.map((e) => Math.floor(e));
  let left = spare - extra.reduce((a, x) => a + x, 0);
  const order = exact.map((_, i) => i).sort((a, b) => {
    const ra = exact[a]! - extra[a]!;
    const rb = exact[b]! - extra[b]!;
    return rb - ra || a - b;
  });
  for (const i of order) {
    if (left <= 0) break;
    extra[i]! += 1;
    left -= 1;
  }
  return out.map((v, i) => v + extra[i]!);
}

/**
 * S167c —— 「span i 之后的边界能不能拖」的**唯一**谓词:命中测试与把手绘制共用它,
 * 亮色竖线永远不会画在拖不动的地方(用户:「有些白色竖线 hover 却拖不动」= 画的集合与
 * 命中的集合各写了一份的漂移形状)。
 *
 * 规则:左侧 span 真实存在(frames > 0)、它的音符可编辑(有 tripleNoteId;空拍 rest 没有)、
 * 越过中间同音符的零宽 dropped 标记后,右侧还有**同一个音符**的真实音素。
 */
export function boundaryDraggableAfter(
  spans: readonly { evt: number; frames: number }[],
  editableByEvt: { readonly [evt: number]: unknown },
  i: number,
): boolean {
  const s = spans[i];
  if (!s || s.frames <= 0) return false;
  if (!editableByEvt[s.evt]) return false;
  let j = i + 1;
  while (j < spans.length && spans[j]!.frames <= 0 && spans[j]!.evt === s.evt) j++;
  const next = spans[j];
  return !!next && next.evt === s.evt && next.frames > 0;
}
