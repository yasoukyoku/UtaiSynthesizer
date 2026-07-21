// S73b/c — 自动音高常开 watcher(SynthV Sing 模式同构,用户拍板「未调教段使用自动音高+长开启;
// 最开始也自动化」)。App 常驻挂载(RenderLinkWatcher 先例)。订阅 tracks/tempo,debounce 后扫
// 所有 vocal notes 段:follow 开着的轨,对可调教音符(资格=applyAutoTune,θ 维度的用户调教
// 绕行;pitchDev=用户独立叠加层,机器永不写,S73c 起不参与 θ 资格)静默补 θ(相位保留)。
// 打开存量工程/导入内容也直接补调教(S73c 用户拍板;dirty=真实内容变化,undo 不受染指)。
//
// 收敛与 undo 语义(设计核心,别破坏;S73b 审查修复全在此):
//  - 写入走 applyAutoTune(..., {silent:true}) = history.runSilent:不进撤销栈、不砍 redo。
//  - ★手势事务窗:txnDepth>0 时 sweep 整体让路(入口检查)+ applyAutoTune 写入点二次检查
//    (await 期间手势才开始的窗)——否则 silent 写会被 commitTransaction 捕获成幻影撤销步
//    并清 redo。让路一律置 pending,松手后的重排补上。
//  - ★stale(await 窗内容变了→零写入)绝不入账 doneRef:入账=把用户刚写的未调教内容标成
//    「已完成」,最后一次编辑从此静默不被跟随(审查 HIGH)。删条目+pending 重排=天然重试。
//  - ★busy 撞车不丢趟:置 pending,finally 里补排一轮(autosave pendingFlush 同构)。
//  - 成功写入以【写后状态】的 sig 入账(θ 在 contentSig 里)→ 自身触发的下一轮 sig 命中=
//    不自激;undo 触发的重扫零写入收敛(快照含 k+θ 一致,no-op 守卫吸收)。
//  - 后端失败(模型未下载等)→ 60s 冷却自愈重试(下载完成后自动恢复,无需任何按钮),
//    首次失败一次性 toast + 弹缺模型一键下载对话框(preflightAuxPack 复用,S73c 无手动
//    按钮后这是唯一的发现入口)。
import { useEffect, useRef } from "react";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { contentSig, inGestureTransaction } from "../../store/history";
import { applyAutoTune, autoTuneScalesOf } from "../../lib/vocal/autoTune";
import { preflightAuxPack } from "../../lib/vocal/vocalRender";
import i18n from "../../i18n";

const DEBOUNCE_MS = 400;
const RETRY_MS = 500;
const FAIL_COOLDOWN_MS = 60_000;

let pausedUntil = 0;
let failNoticeShown = false;
/** Retake 成功(模型确认在位)后立刻解除冷却。 */
export function resetAutoTuneWatcher(): void {
  pausedUntil = 0;
}

export function AutoTuneWatcher() {
  const tracks = useProjectStore((s) => s.tracks);
  const tempo = useProjectStore((s) => s.tempo);
  /** segId → 上次处理完的 `${contentSig}|${tempo}|${expr}|${vib}`。 */
  const doneRef = useRef(new Map<string, string>());
  const busyRef = useRef(false);
  const pendingRef = useRef(false);
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      void sweep();
    }, DEBOUNCE_MS);
    return () => clearTimeout(timer);

    function scheduleRetry(delay: number): void {
      if (retryRef.current) clearTimeout(retryRef.current);
      retryRef.current = setTimeout(() => {
        retryRef.current = null;
        void sweep();
      }, delay);
    }

    async function sweep(): Promise<void> {
      const cooldown = pausedUntil - Date.now();
      if (cooldown > 0) {
        scheduleRetry(cooldown + 100); // 冷却期满自动重试=下载完成后无按钮自愈
        return;
      }
      if (busyRef.current || inGestureTransaction()) {
        pendingRef.current = true;
        if (!busyRef.current) scheduleRetry(RETRY_MS); // 手势让路:busy 的 finally 无人补排,自排
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
            if (seg.content.notes.length === 0) continue;
            const sig = `${contentSig(seg.content)}|${st.tempo}|${scales.expr}|${scales.vib}`;
            if (!follow || doneRef.current.get(seg.id) === sig) continue;
            let res;
            try {
              res = await applyAutoTune(t.id, seg.id, [], scales, "refresh", { silent: true });
            } catch (e) {
              pausedUntil = Date.now() + FAIL_COOLDOWN_MS;
              console.warn("[autotune] follow paused after backend failure:", e);
              if (!failNoticeShown) {
                failNoticeShown = true;
                useAppStore.getState().showToast(i18n.t("vocalEditor.sidebar.autotunePaused"), "info");
                void preflightAuxPack("aux-autotune"); // 缺模型 → 一键下载对话框(唯一发现入口)
              }
              scheduleRetry(FAIL_COOLDOWN_MS + 100);
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
        // 修剪(删段/换工程的滞留条目)
        for (const k of [...doneRef.current.keys()]) if (!liveIds.has(k)) doneRef.current.delete(k);
      } finally {
        busyRef.current = false;
        if (pendingRef.current) {
          pendingRef.current = false;
          scheduleRetry(RETRY_MS);
        }
      }
    }
  }, [tracks, tempo]);

  return null;
}
