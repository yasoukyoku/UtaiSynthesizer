/**
 * 第三条边:**Rust 源码里发出的 CODE → `backendError.ts` 的 CODE_KEYS**(S126 §F2⒝ 批 2 ④ 新增)。
 *
 * `parity.test.ts` 已经闭合了 `CODE_KEYS ↔ backend.*` 两个方向,但那张图缺一条边:**没有任何
 * 一侧去看 Rust**。于是「Rust 新发了一个 CODE、谁都没把它加进表」是一个**零报警**的动作,
 * 代价是用户看到一串带绝对路径的英文 Rust 字符串。实测:§F2⒝ 批 1-3 就这样落下了十条
 * (`RUN_AMBIGUOUS` / `SLOT_MIGRATE_FAILED` / …),而三套测试全绿。
 *
 * ## 扫的是【发射写法】,不是【大写形状】
 *
 * 按「全大写字符串」去猜会把环境变量名(`PYTHONPATH`、`UTAI_MG_*`)和进度阶段常量(`STAGE_*`)
 * 全拉进来 —— 实测那样有 169 条,而一份 169 条的快照就是一座没人清的坟场,只会教人「往里再加
 * 一行」。所以这里只认 CODE **真正被发射出去**的四种写法(见 `EMIT`),`std::env::var("X")`
 * 之类结构上匹配不到。
 *
 * ## 为什么是【棘轮】而不是【全集断言】
 *
 * 仓里确有一批 CODE **不该**进这张表,理由各不相同(`backendError.ts` 的文件头注声明了三类:
 * `VOCAL_*` 由 `vocalRenderErrorMessage` 带载荷自行处理、`CLEANUP_BUSY` 由 Settings 的内联
 * L() 表处理、取消哨兵是**故意**不映射的)。要说「其余那些也没问题」,得逐条判过 —— 而我没有。
 * 所以这里只做**不可增长**:未映射集合必须**恰好等于**下面这份快照。
 *
 * ⇒ 新增一个未映射的 CODE ⇒ 红,报文直接给两条出路;
 * ⇒ 把快照里的某一条映射掉 ⇒ 也红,逼着把它从快照里删掉(否则快照会烂成噪音)。
 *
 * ⚠ 这份快照**不是**「这些都没问题」的背书,而是「S126 当天的现状」。第二组里的每一条都还欠
 *   一次「它到底会不会被用户看见」的判定。
 */
import { describe, it, expect } from "vitest";
import { CODE_KEYS } from "../lib/backendError";

// 前端 tsconfig 无 @types/node —— 同 `resumeLockParity` / `ipcParity`:用「变量说明符的动态
// import」拿 fs(TS 对非字面量说明符不做模块解析,运行时由 node 原生解析)。
type NodeFs = {
  readFileSync(p: string, enc: string): string;
  readdirSync(p: string): string[];
  statSync(p: string): { isDirectory(): boolean };
};
const importFs = (): Promise<NodeFs> => {
  const spec = "node:fs";
  return import(/* @vite-ignore */ spec) as Promise<NodeFs>;
};

const SRC = "src-tauri/src";

/** 一个 CODE 被发射出去的四种写法。带载荷的那种是主力;后三种是不带载荷的裸串。 */
const EMIT = [
  /"([A-Z][A-Z0-9_]{4,}):/g, //                      format!("CODE: {detail}")
  /"([A-Z][A-Z0-9_]{4,})"\s*\.\s*into\(\)/g, //      "CODE".into()
  /"([A-Z][A-Z0-9_]{4,})"\s*\.\s*to_string\(\)/g, // "CODE".to_string()
  /Err\(\s*"([A-Z][A-Z0-9_]{4,})"/g, //              Err("CODE")
];

/** S126 当天未映射的 Rust CODE。分组只在**有依据**时才写理由。 */
const UNMAPPED_SNAPSHOT_S126 = [
  // ── backendError.ts 的文件头注明写「故意不在这张表里」的三类 ──────────────────
  "CANCELLED", // 取消哨兵:映射它会把用户的主动取消变成可见错误
  "CLEANUP_BUSY", // Settings 的内联 L() 表(该文件被承认的惯例)
  "VOCAL_ALIAS",
  "VOCAL_BAD_LANG",
  "VOCAL_DICT_MISSING",
  "VOCAL_EMPTY",
  "VOCAL_ENV_LEN",
  "VOCAL_LYRIC_TOO_LONG",
  "VOCAL_OOV",
  "VOCAL_PHONE_MISSING",
  "VOCAL_UNKNOWN_PHONE", // ↑ `vocalRenderErrorMessage` 带载荷自行处理
  // ── 未判定 —— 留在快照里【不代表】它们没问题,每条都欠一次判定 ────────────────
  "AMD_SERIALIZE_KERNEL",
  "BREATH",
  "CONTROL",
  "CUDA_LAUNCH_BLOCKING",
  "GAME_DELETE_FAILED",
  "GAME_DL_BUSY",
  "GAME_DL_EXTRACT",
  "GAME_DL_FAILED",
  "IMPORT_EMPTY",
  "IMPORT_PARSE_MIDI",
  "IMPORT_PARSE_USTX",
  "IMPORT_PPQ",
  "IMPORT_READ_FAIL",
  "IMPORT_SMPTE",
  "IMPORT_UNSUPPORTED",
  "MIDI_EXTRACT_CANCELLED",
  "MIDI_EXTRACT_FAILED",
  "MIDI_EXTRACT_LOAD_FAILED",
  "MIDI_EXTRACT_NOT_INSTALLED",
  "MIDI_EXTRACT_TOO_SHORT",
  "MIGRATE_FAMILY_AMBIGUOUS",
  "MIGRATE_FAMILY_UNKNOWN",
  "MIRROR_LIST_UNAVAILABLE",
  "PANIC",
  "RANGE_BAD_RECORD",
  "RANGE_CANDIDATE_MISSING",
  "RANGE_EMPTY_SCORE",
  "RANGE_INVALID",
  "SCALE_QUALITY_SHAPE",
  "STRETCH_INPUT_MISSING",
  "STRETCH_JOIN",
  "STRETCH_RATIO_RANGE",
  "TEMPO_JOIN",
  "TEMPO_LOAD_FAILED",
  "TEMPO_NO_BEAT",
  "TEMPO_TOO_SHORT",
  "UPDATE_DOWNLOAD_CANCELLED",
  "UTAI_DIAGNOSTICS",
].sort();

function rsFiles(fs: NodeFs, dir: string): string[] {
  const out: string[] = [];
  for (const name of fs.readdirSync(dir)) {
    const p = `${dir}/${name}`;
    if (fs.statSync(p).isDirectory()) out.push(...rsFiles(fs, p));
    else if (name.endsWith(".rs")) out.push(p);
  }
  return out;
}

async function scanRustCodes(): Promise<Set<string>> {
  const fs = await importFs();
  const found = new Set<string>();
  for (const f of rsFiles(fs, SRC)) {
    const text = fs.readFileSync(f, "utf8");
    for (const re of EMIT) for (const m of text.matchAll(re)) found.add(m[1]!);
  }
  return found;
}

describe("Rust 的 CODE 必须能被本地化", () => {
  /** ⛔ 解析器先死、断言就哑了(S125 血训 M7)。这条是扫描器自己的存活证明,而且它当场买回过
   *  一次真错误:第一版只认带载荷的写法,`TRAINING_WIPE_NOT_CONFIRMED` 那一整族(裸串 +
   *  `.into()`)整体不可见,而「未映射集合」照样是空的 —— 一条全绿的死闸。 */
  it("扫描器本身是活的:带载荷与裸串两种写法都能看见", async () => {
    const found = await scanRustCodes();
    expect(found.size).toBeGreaterThan(300);
    expect(found.has("RUN_AMBIGUOUS")).toBe(true); // format!("CODE: …")
    expect(found.has("TRAINING_WIPE_NOT_CONFIRMED")).toBe(true); // "CODE".into()
    expect(found.has("CONVERT_BUSY")).toBe(true); // Err("CODE".into())
  });

  it("未映射的 CODE 集合不许增长", async () => {
    const found = await scanRustCodes();
    const unmapped = [...found].filter((c) => !(c in CODE_KEYS)).sort();
    expect({
      新增了未映射的CODE_请补三语或加进快照并写明理由: unmapped.filter(
        (c) => !UNMAPPED_SNAPSHOT_S126.includes(c),
      ),
      快照里这些已经被映射了_请从快照删掉: UNMAPPED_SNAPSHOT_S126.filter(
        (c) => !unmapped.includes(c),
      ),
    }).toEqual({
      新增了未映射的CODE_请补三语或加进快照并写明理由: [],
      快照里这些已经被映射了_请从快照删掉: [],
    });
  });
});
