/**
 * 第四条边:**`RANGE_ALGO_VERSION`(前端渲染签名)↔ `audition_cache_tag`(Rust 侧试听缓存名)。**
 *
 * 这两个必须一起动的规矩,从 S82 起就写在记忆里、写在两个文件的注释里 —— 而**没有任何一侧
 * 去看另一侧**。漏掉一处的后果不是报错,是**听到旧缓存**:一次真的引擎换代被读成「我改了,
 * 用户什么都没听见」,也就是 `RANGE_ALGO_VERSION` 这个常量存在的唯一理由本身失效。
 * S146 换引擎(Signalsmith → TD-PSOLA)时这道闸还不存在,所以在这里补上。
 *
 * ⚠ 它只闭合**版本号一致**这一条边,不闭合「该不该 bump」——后者没有机械判据。
 * ⛔ 而 `commands/audio.rs` 的 `_s2r` **不在这条边上**:那是速度滑条的时间伸缩缓存
 *   (key 里只有 time_factor),永远不经过 `apply_inverse`。记忆里「三处成对 bump」
 *   这句话不精确,只有两处在这条路上 —— 写在这里,免得下一个人又去 bump 一个无关的键。
 */
import { describe, it, expect } from "vitest";
import { RANGE_ALGO_VERSION } from "./vocalRender";

type NodeFs = { readFileSync(p: string, enc: string): string };
const importFs = (): Promise<NodeFs> => {
  const spec = "node:fs";
  return import(/* @vite-ignore */ spec) as Promise<NodeFs>;
};

describe("RANGE_ALGO_VERSION ↔ audition_cache_tag", () => {
  it("试听缓存 tag 必须带着当前的 RANGE_ALGO_VERSION", async () => {
    const fs = await importFs();
    const src = fs.readFileSync("src-tauri/src/commands/audition.rs", "utf8");
    // 形如 `"_s85e_ru{:.0}-..."` —— 取 format! 里那个前导字面量
    const m = src.match(/"_([a-z0-9]+)_ru\{/);
    expect(m, "audition.rs 里找不到 audition_cache_tag 的前导字面量(改名了?)").toBeTruthy();
    expect(m![1]).toBe(RANGE_ALGO_VERSION);
  });

  it("速度滑条的 _s2r 不在这条边上(它不经过 apply_inverse)", async () => {
    const fs = await importFs();
    const audio = fs.readFileSync("src-tauri/src/commands/audio.rs", "utf8");
    expect(audio).toContain("_s2r");
    // 它的 key 里只有内容哈希与 time_factor —— 没有 shift、没有 kappa、没有 range 记录。
    const key = audio.match(/let key = format!\("([^"]+)"/);
    expect(key, "audio.rs 的 stretch 缓存 key 变了形状,请重判这条边").toBeTruthy();
    expect(key![1]).not.toMatch(/shift|kappa|range/i);
  });
});
