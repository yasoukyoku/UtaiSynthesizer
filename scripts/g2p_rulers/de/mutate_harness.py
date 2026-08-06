# -*- coding: utf-8 -*-
"""S110 §C24 —— gate 的变异探针。契约与 §G14 那份相同(见 `s110_g14_2_callsite\\mutate_s110_g14.py`):
先抓字节 · 每次变异后**先断言变异真的发生了** · finally 必还原并复核 sha256 · 逐条记**红在哪一组**。

⚠ 判「编译失败」不能用 `^error` —— cargo 在测试失败时也打 `error: test failed`。上一份脚本第一次跑
   就因此把八条全误判成编译错。这里用 `could not compile` / `error[E\\d+]`。

★ 探针设计原则(S107/S108):每一组都要有一个**只打得到它**的变异。第 (6) 组(血径)是**兜底**,
   按设计任何改动都会动它 —— 所以它没有隔离探针,这一点明写,不装作有。
"""
import hashlib
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[3] / "src-tauri"   # scripts/g2p_rulers/de → 仓库根
G2P = REPO / "src" / "inference" / "g2p.rs"
OUT = Path(__file__).parent
GATE = "inference::g2p::tests::s110_de_literal_j_gate"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compile_failed(log):
    return ("could not compile" in log) or bool(re.search(r"^error\[E\d+\]", log, re.M))


def assertion(log):
    m = re.search(r"panicked at ([^\n]+):\n(.*?)\n(?:note:|stack backtrace|test |failures)", log, re.S)
    if not m:
        return "(没抽到 panic 正文 —— 去读原始 log)"
    return " ".join(m.group(2).split())[:230]


MUTATIONS = [
    # (id, 变异, 瞄的组)
    ("C1_clusters_removed",
     lambda s: s.replace('"t s j", "z j", "ɡ j", "b j", "n j",', '"t s j", "z j", "ɡ j", "b j",'),
     "(3) 切法 —— 拿掉新加的 n j,⟨i⟩ 派生那一半应当退回旧切法"),

    ("C2_glide_block_not_applied",
     lambda s: s.replace("            if g >= a + 1 && g < b {\n                lo = lo.max(g - (a + 1));\n            }\n",
                         "            let _ = g;\n"),
     "(3) 切法 —— 谓词照旧算得对,但 `constraint` 不用它(只打得到管道那一层)"),

    ("C3_override_immunity_dropped",
     lambda s: s.replace(
         "        if !self\n            .pronunciations(&key)\n            .any(|p| p.split_whitespace().eq(phones.iter().map(String::as_str)))\n        {\n            return Vec::new(); // override / merged fragment / composed rung — the letters do not describe these phones\n        }\n",
         ""),
     "(2) 覆盖免疫 —— 手打音素也会被拼写约束"),

    ("C4_abstain_becomes_guess",
     lambda s: s.replace("            if literal >= positions.len() {", "            if literal > 0 {"),
     "(1) 谓词 —— 歧义词不再弃权而是猜"),

    ("C5_spelling_table_widened",
     lambda s: s.replace('    ("v", "wj"), ("v", "vj"),', '    ("v", "wj"), ("v", "vj"), ("ç", "chj"),'),
     "(5) 覆盖面 —— 表里多一个辅音,而它不改变任何切法"),
]


def main():
    only = set(sys.argv[1:])
    orig = G2P.read_bytes()
    base = sha(G2P)
    print(f"基线 g2p.rs sha256 = {base}\n")
    rows = []
    try:
        for mid, fn, aimed in MUTATIONS:
            if only and mid not in only:
                continue
            src = orig.decode("utf-8")
            mut = fn(src)
            if mut == src:
                rows.append((mid, "⛔探针无效", "变异没改动文件 —— 锚字符串对不上", aimed))
                print(f"[{mid}] ⛔ 变异未生效,跳过")
                continue
            G2P.write_text(mut, encoding="utf-8", newline="")
            p = subprocess.run(["cargo", "test", "--lib", GATE, "--", "--nocapture"],
                               cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
            log = p.stdout + p.stderr
            (OUT / f"mut_{mid}.log").write_text(log, encoding="utf-8")
            G2P.write_bytes(orig)
            if sha(G2P) != base:
                raise SystemExit("⛔⛔ 还原失败,立刻停手")
            if p.returncode == 0:
                v, act = "绿(闸没抓到)", ""
            elif compile_failed(log):
                v, act = "⛔无效(编译失败)", "红的原因是编译不过,与闸无关"
            else:
                v, act = "红", assertion(log)
            rows.append((mid, v, act, aimed))
            print(f"[{mid}] {v}  ← 瞄 {aimed}")
            if act:
                print(f"      实际: {act}")
    finally:
        G2P.write_bytes(orig)
        assert sha(G2P) == base
        print("\ng2p.rs 已还原,sha256 复核一致。")

    print("\n" + "=" * 100)
    for mid, v, act, aimed in rows:
        print(f"{mid:32s} {v:16s} {aimed}")
        if act:
            print(f"{'':32s} {'':16s} → {act}")
    print("\n⚠ 第 (6) 组(血径 431)按设计是兜底的:任何改动都会动它 ⇒ **没有隔离探针**,不假装有。"
          "\n   它的职责不是隔离,是「重生成词典后有人必须来看一眼」。")


if __name__ == "__main__":
    main()
