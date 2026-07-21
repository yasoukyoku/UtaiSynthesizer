// S73b — 自动音高常开 watcher(SynthV Sing 模式同构,用户拍板「未调教段使用自动音高+长开启」)。
// App 常驻挂载(RenderLinkWatcher 先例)。订阅 tracks/tempo,松手级提交后 debounce 扫一遍所有
// vocal notes 段:follow 开着的轨,对可调教音符(资格=applyAutoTune 的 refresh 判定,用户调教
// 一律绕行)静默补 θ(相位保留)。
//
// 收敛与 undo 语义(设计核心,别破坏;S73b 审查修复全在此):
//  - 写入走 applyAutoTune(..., {silent:true}) = history.runSilent:不进撤销栈、不砍 redo。
//  - ★手势事务窗:txnDepth>0 时 sweep 整体让路(入口检查)+ applyAutoTune 写入点二次检查
//    (await 期间手势才开始的窗)——否则 silent 写会被 commitTransaction 捕获成幻影撤销步
//    并清 redo。让路一律置 pending,松手后的重排补上。
//  - ★stale(await 窗内容变了→零写入)绝不入账 doneRef:入账=把用户刚写的未调教内容标成
//    「已完成」,最后一次编辑从此静默不被跟随(审查 HIGH)。留空条目+pending 重排=天然重试。
//  - ★busy 撞车不丢趟:置 pending,finally 里补排一轮(autosave pendingFlush 同构)。
//  - ★首见基线(baseline-on-first-sight):没见过的段先按当前内容入账、【不】调教——打开
//    存量工程/粘贴现成内容不发生「开门即群体调教+置脏+烤件全废」的沉默改档;只有此后的
//    编辑触发跟随。新建空段先入 known(0 音符跳过调教),首笔音符即正常跟随。
//  - 成功写入以【写后状态】的 sig 入账(θ 在 contentSig 里)→ 自身触发的下一轮 sig 命中=
//    不自激;undo 触发的重扫零写入收敛(快照含 k+θ 一致,no-op 守卫吸收)。
//  - 后端失败(模型未下载等)→ 熄火 + 一次性 toast 告知;侧栏按钮成功后 resetAutoTuneWatcher()
//    复燃(模型就位信号)。
import { useEffect, useRef } from "react";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { contentSig, inGestureTransaction } from "../../store/history";
import { applyAutoTune, autoTuneScalesOf } from "../../lib/vocal/autoTune";
import i18n from "../../i18n";

const DEBOUNCE_MS = 400;
const RETRY_MS = 500;

let disabled = false;
let killToastShown = false;
/** 侧栏按钮成功跑通(模型在位)后复燃常开跟随。 */
export function resetAutoTuneWatcher(): void {
  disabled = false;
}

export function AutoTuneWatcher() {
  const tracks = useProjectStore((s) => s.tracks);
  const tempo = useProjectStore((s) => s.tempo);
  /** segId → 上次处理完(或首见基线)的 `${contentSig}|${tempo}|${expr}|${vib}`。 */
  const doneRef = useRef(new Map<string, string>());
  /** 见过的段(含空段):首见=只入账不调教(存量保护);known 段的 sig 变化才触发跟随。 */
  const knownRef = useRef(new Set<string>());
  const busyRef = useRef(false);
  const pendingRef = useRef(false);
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (disabled) return;
    const timer = setTimeout(() => {
      void sweep();
    }, DEBOUNCE_MS);
    return () => clearTimeout(timer);

    function scheduleRetry(): void {
      if (retryRef.current) clearTimeout(retryRef.current);
      retryRef.current = setTimeout(() => {
        retryRef.current = null;
        void sweep();
      }, RETRY_MS);
    }

    async function sweep(): Promise<void> {
      if (disabled) return;
      if (busyRef.current || inGestureTransaction()) {
        pendingRef.current = true;
        if (!busyRef.current) scheduleRetry(); // 手势让路:busy 的 finally 无人补排,自排
        return;
      }
      busyRef.current = true;
      try {
        const st = useProjectStore.getState();
        const liveIds = new Set<string>();
        for (const t of st.tracks) {
          if (t.trackType !== "vocal") continue;
          const p = t.vocalParams;
          const follow = p?.autoTuneFollow !== false;
          const scales = autoTuneScalesOf(p);
          for (const seg of t.segments) {
            if (seg.content.type !== "notes") continue;
            liveIds.add(seg.id);
            if (seg.content.notes.length === 0) {
              knownRef.current.add(seg.id); // 空段入 known:首笔音符即正常跟随
              continue;
            }
            const sig = `${contentSig(seg.content)}|${st.tempo}|${scales.expr}|${scales.vib}`;
            if (!knownRef.current.has(seg.id)) {
              // 首见基线:存量/粘贴内容不动,只入账(此后的编辑才触发)
              knownRef.current.add(seg.id);
              doneRef.current.set(seg.id, sig);
              continue;
            }
            if (!follow || doneRef.current.get(seg.id) === sig) continue;
            let res;
            try {
              res = await applyAutoTune(t.id, seg.id, [], scales, "refresh", { silent: true });
            } catch (e) {
              disabled = true;
              console.warn("[autotune] follow disabled after backend failure:", e);
              if (!killToastShown) {
                killToastShown = true;
                useAppStore.getState().showToast(i18n.t("vocalEditor.sidebar.autotunePaused"), "info");
              }
              return;
            }
            if (res.stale) {
              // 在途编辑/手势让路:不入账,pending 重排里自然重试(审查 HIGH:入账=永久漏跟随)
              doneRef.current.delete(seg.id);
              pendingRef.current = true;
              continue;
            }
            // 以写后状态入账(θ 字段进 contentSig;stamp 时点重取 state,含期间的 tempo/k 变化)
            const now = useProjectStore.getState();
            const freshTrack = now.tracks.find((x) => x.id === t.id);
            const fresh = freshTrack?.segments.find((s) => s.id === seg.id);
            if (fresh && fresh.content.type === "notes") {
              const freshScales = autoTuneScalesOf(freshTrack?.vocalParams);
              doneRef.current.set(
                seg.id,
                `${contentSig(fresh.content)}|${now.tempo}|${freshScales.expr}|${freshScales.vib}`,
              );
            }
          }
        }
        // 修剪(删段/换工程的滞留条目;known 同步修剪=段复活按首见基线处理)
        for (const k of [...doneRef.current.keys()]) if (!liveIds.has(k)) doneRef.current.delete(k);
        for (const k of [...knownRef.current]) if (!liveIds.has(k)) knownRef.current.delete(k);
      } finally {
        busyRef.current = false;
        if (pendingRef.current) {
          pendingRef.current = false;
          scheduleRetry();
        }
      }
    }
  }, [tracks, tempo]);

  return null;
}
