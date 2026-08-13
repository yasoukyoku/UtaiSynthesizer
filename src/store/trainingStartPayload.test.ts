/**
 * §E2E-M11 —— **冷启动之后**「继续训练」送回去的值,等不等于槽 manifest 里的值。
 *
 * ## 为什么非要一条跨 store 重建的腿
 *
 * store **没有持久化**(全仓无 `persist(`),所以冷启动后每一个 costly 字段都是出厂默认
 * (`augCopies: 0` / `sovitsLoudnorm: false`)。而 `enterProject` 只清 `modelName` ——
 * ⛔ 拿它当「冷启动」是**假的**:它一个 costly 字段都不碰,阴性对照会绿。
 * ⇒ 这条腿从模块加载那一刻抓下来的 `FRESH` 出发,每条用例都重建到那一刻。
 * ⛔ **任何「同一会话里改完就跑」的腿结构上测不到这一格** —— 那时 store 里还留着用户刚输入的值。
 *
 * ## 这条腿覆盖哪三跳,以及**不**覆盖什么
 *
 * 覆盖:`formForSlot(...)` → `updateConfig` → `start_training` 的 request。
 * ⛔ **不覆盖**「按钮走得到 `start()`」:`askRunName` 那个对话框、`trainingDataOk`、空名拦截
 * 三道都在 `start()` 之外。⇒ 别把这条腿说成「继续训练端到端」。
 * ⛔ 也**不覆盖** `formForSlot` 真的被组件调用过 —— 这条腿自己写了那一行,组件哪天改成内联
 * 一份它照样全绿。那一半由 `rowIdentityWiring.test.ts` 的源码闸补。
 * ⛔ 还**不覆盖** request → `run_manifest.json` 那一弧(`try_start` 全仓没有驱动器),
 * 所以 §F2⒝ ④d 给存量池打戳所依赖的「下次请求送的就是它」仍然半边空着。
 *
 * ## ⚠ 判据为什么必须是这个形状
 *
 * ⒜ **loudnorm 成对**(true 与 false 各一条)。只写 true 那条是**空判据**:把请求里的
 *    `loudnorm: config.sovitsLoudnorm` 换成 `config.sovitsVolEmbedding` 之后照样绿 ——
 *    默认 `sovitsVolEmbedding` 是 `true`,而夹具的 `vol_embedding` 一旦是 null 就根本不还原。
 *    夹具因此显式写 `vol_embedding: false`,让两个布尔分得开。
 * ⒝ **五支各用一个互不相同的份数**。`aug_copies` 在请求构造器里是**五份手抄映射**
 *    (`augCopies` / `vocAugCopies` / `diffAugCopies` / `sovitsAugCopies` ×2),五个字段全是
 *    `number` ⇒ 互换过 tsc。共用同一个数值时,一次串线会被上一条用例的残留掩盖过去。
 * ⒞ Rust 侧那两个键都是 `#[serde(default)]` 且 struct 没有 `deny_unknown_fields` ⇒
 *    「表单忘了这个值」与「TS 把键名打错/删掉」在盘上**逐字节同形**(0 / false)。
 *    这条腿是全仓唯一可能看见这两种事故的地方。
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

/** 捕获式的 invoke —— 按命令分派。⛔ 不许写成 `() => Promise.resolve()`:
 *  那样 `training_env_ready` 会收到 `undefined` ⇒ `!undefined` 为真 ⇒ 走进 `showConfirm`,
 *  而那个 Promise 只有被 `resolve(id)` 才 settle ⇒ headless 下**永不返回**,
 *  这条 it 会死在 vitest 超时上,而不是红在断言上。 */
const requests: Record<string, unknown>[] = [];
const invokeMock = vi.fn(async (cmd: string, args?: Record<string, unknown>) => {
  switch (cmd) {
    case "training_env_ready":
      return true;
    case "training_required_assets":
      return [];
    case "start_training":
      requests.push(args!.request as Record<string, unknown>);
      return null;
    case "get_training_status":
      return { state: "idle", project_id: "", model_name: "", backend: "", stage: "", stderr_tail: [] };
    case "get_training_history":
      return [];
    default:
      throw new Error(`这条腿没有预料到的 invoke: ${cmd}`);
  }
});
vi.mock("@tauri-apps/api/core", () => ({ invoke: (c: string, a?: never) => invokeMock(c, a) }));
vi.mock("../i18n", () => ({ default: { t: (k: string) => k } }));
/** ⛔ 对话框要**炸**,不要挂死:一条走进对话框的腿说明前置 mock 不对,那必须是响亮的红。 */
vi.mock("./app", () => ({
  useAppStore: {
    getState: () => ({
      showConfirm: () => {
        throw new Error("这条腿不该走到任何对话框 —— 前置 invoke 的 mock 不对");
      },
      showToast: () => {},
      settingsOpen: false,
      toggleSettings: () => {},
    }),
  },
}));

import { useTrainingStore, type TrainingBackend, type TrainingFormConfig, type WorkspaceInfo } from "./training";
import { formForSlot } from "../lib/training/formForSlot";

/** 冷启动那一刻的 config。⛔ 必须在任何用例动过 store **之前**抓 —— `DEFAULT_CONFIG` 没有导出。 */
const FRESH: TrainingFormConfig = structuredClone(useTrainingStore.getState().config);

function ws(over: Partial<WorkspaceInfo> = {}): WorkspaceInfo {
  return {
    exists: true,
    family: "sovits",
    version: "4.1",
    sample_rate: "44k",
    has_main_progress: true,
    diff_steps: 0,
    best_resume_step: null,
    diff_best_resume_step: null,
    aug_copies: 0,
    // ★ 显式 false,不是 null —— 见头注⒜:null 会让 sovitsVolEmbedding 停在默认的 true,
    //   于是 loudnorm 与 vol_embedding 两个布尔在断言上分不开。
    loudnorm: null,
    has_dataset: true,
    has_preprocessing: true,
    vol_embedding: false,
    n_speakers: 1,
    speakers: [],
    diff_k_step_max: 0,
    ...over,
  };
}

beforeEach(() => {
  requests.length = 0;
  invokeMock.mockClear();
  // ★ 每条用例都回到**冷启动**那一刻:store 是模块级单例,而 `updateConfig` 只加不减。
  useTrainingStore.setState({
    config: structuredClone(FRESH),
    projectDataset: null,
    route: { seg: "detail", projectId: "p1" },
  });
});

/** 走一次「继续训练 → 开始训练」。⚠ 第一行是 `ProjectDetail.startFamily` 那一行的复述 ——
 *  它是这条腿的**已知边界**,由源码闸补(见头注)。 */
async function drive(backend: TrainingBackend, info: WorkspaceInfo): Promise<Record<string, unknown>> {
  useTrainingStore.getState().updateConfig({
    backend,
    modelName: "leg",
    runId: "run-1",
    ...formForSlot(backend, info),
  });
  await useTrainingStore.getState().start(false, false, undefined);
  expect(requests, "start_training 一次都没被调用 —— 这条腿什么也没测").toHaveLength(1);
  return requests[0]!;
}

describe("§E2E-M11 冷启动之后,manifest 的值真的到了 start_training 的载荷上", () => {
  /** ★ 自检 + 哨兵:从 FRESH 直接开始 ⇒ 载荷必须是出厂默认。
   *  少了它,下面每一条「等于 manifest」都可能只是「等于恰好相同的默认值」。 */
  it("★ 自检:不还原任何东西 ⇒ 载荷是出厂默认(0 / false)", async () => {
    useTrainingStore.getState().updateConfig({ backend: "sovits", modelName: "leg", runId: "run-1" });
    await useTrainingStore.getState().start(false, false, undefined);
    expect(requests).toHaveLength(1);
    expect(requests[0]!.aug_copies).toBe(0);
    expect(requests[0]!.loudnorm).toBe(false);
  });

  // ★ 五支各一个**互不相同**的份数 —— 见头注⒝。
  it("rvc:manifest 的 1 份增强送到载荷上", async () => {
    const req = await drive("rvc", ws({ family: "rvc", version: "v2", sample_rate: "40k", aug_copies: 1 }));
    expect(req.aug_copies).toBe(1);
    expect(req.version).toBe("v2");
    expect(req.sample_rate).toBe("40k");
  });

  it("vocoder:2 份", async () => {
    const req = await drive("vocoder", ws({ family: "vocoder", version: "nsf_hifigan", aug_copies: 2 }));
    expect(req.aug_copies).toBe(2);
  });

  it("sovits_diff(diff-first,没有宿主主模型)⇒ 3 份是这个 run 自己的选择", async () => {
    const req = await drive("sovits_diff", ws({ has_main_progress: false, aug_copies: 3 }));
    expect(req.aug_copies).toBe(3);
  });

  it("sovits:4 份 + 响度归一化开着", async () => {
    const req = await drive("sovits", ws({ aug_copies: 4, loudnorm: true }));
    expect(req.aug_copies).toBe(4);
    expect(req.loudnorm).toBe(true);
  });

  /** ★★ 与上一条**成对**,而且是承重的那一条(见头注⒜):
   *  只有「loudnorm=true」那条时,把载荷里的 `config.sovitsLoudnorm` 换成
   *  `config.sovitsVolEmbedding` 照样全绿 —— 默认它就是 true。 */
  it("★ sovits:槽是【关着响度归一化】建的 ⇒ 载荷必须送 false,而不是默认的那个 true", async () => {
    const req = await drive("sovits", ws({ aug_copies: 4, loudnorm: false }));
    expect(req.loudnorm).toBe(false);
    // 同一份载荷里另一个布尔此刻是 true ⇒ 两者分得开(否则上一句可能只是「某个布尔是 false」)
    expect(req.vol_embedding).toBe(false);
  });

  it("sovits_v2:5 份 + loudnorm(v2 同样送、同样折进它的 fp_text)", async () => {
    const req = await drive(
      "sovits_v2",
      ws({ family: "sovits_v2", version: "4.0-v2", aug_copies: 5, loudnorm: true }),
    );
    expect(req.aug_copies).toBe(5);
    expect(req.loudnorm).toBe(true);
    expect(req.version).toBe("4.0-v2");
  });

  /** ★ 槽从没跑过 ⇒ 一个字段都不还原,载荷回到默认。
   *  它同时守着「无条件还原」那个坏法:那会把用户刚在参数页填好的份数清成 0。 */
  it("★ 槽从没跑过(manifest 不存在)⇒ 载荷是默认值,而不是 manifest 里那堆零", async () => {
    const req = await drive("sovits", ws({ version: "", sample_rate: "", aug_copies: 7, loudnorm: true }));
    expect(req.aug_copies).toBe(0);
    expect(req.loudnorm).toBe(false);
  });
});
