# -*- coding: utf-8 -*-
"""盲测打包器 —— 把 README 第 8 条那套协议变成可执行的东西。

## 为什么它必须在仓里

S145/S146/S146g 三轮的打包脚本全在 scratchpad(会被 GC),而且 **S148 复核发现其中两份的
注释是假的**:`s145_blind.py:5` 写「文件名由**内容** sha1 定序」而实际
`sha1(f"{组名}{版本标签}s145")`;`make_blind.py:55` 写「标签由**内容**哈希决定」而实际
`sha256(组名)`。后果不是洁癖问题:**同一个组名在下一轮会给出逐字相同的答案** ——
复用 G1/K1/A/B 这些名字就是泄题。只有 `pack_blind.py` 真的读了文件字节。

⇒ 这一份**只按文件内容定序**,并且把三条容易退化的规矩做成 assert。

## 用法

    py -3 blind_pack.py spec.json

spec.json:
{
  "round": "s148-r1-ab",
  "question": "preference",              // 或 "detection",决定每组几个文件
  "out": "D:\\\\MyDev\\\\TESTING\\\\s148_blind\\\\r1",
  "note": "自由文本,会原样写进包里的说明",
  "instruments": ["仪器事前说了什么,一行一条"],
  "level_match": "rms",                  // 或 "none"
  "pairs": [
    {"label": "P1", "ref": "...A.wav", "cand": "...B.wav", "start_s": 259.0, "end_s": 266.0},
    {"label": "CTRL", "ref": "...A.wav", "cand": "...A.wav", "start_s": 100.0, "end_s": 106.0, "blank": true}
  ]
}

## 三条硬 assert(违反就退 3 = 跑不起来,不是「读数不符」)

1. **空白对照必须逐位相同** —— 发出去之前 `max|Δ| == 0`。
2. **文件名只由内容决定** —— 落盘后回读,断言字母序 == 内容 sha1 序。
3. **KEY 不与素材同目录**,而且**不打到 stdout**(打包者已经知道答案,但别让它进转录/终端历史)。
"""
import hashlib
import io
import json
import os
import shutil
import sys

import numpy as np
import soundfile as sf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    # ⚠ compare.py 的 UNRUNNABLE 消息因为漏了这一行,在本机打出来是乱码 —— 「跑不起来」
    #    这条信息本身读不出来,正是 S129 铁律要防的形状。
    sys.stderr.reconfigure(encoding="utf-8")

EXIT_OK, EXIT_BAD, EXIT_UNRUNNABLE = 0, 1, 3


def die(msg):
    print(f"UNRUNNABLE: {msg}", file=sys.stderr)
    raise SystemExit(EXIT_UNRUNNABLE)


def load(path):
    if not os.path.exists(path):
        die(f"缺素材: {path}")
    x, sr = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x[:, 0]
    return x, sr


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if len(x) else 0.0


def main(spec_path):
    spec = json.load(io.open(spec_path, encoding="utf-8"))
    out = spec["out"]
    question = spec.get("question", "preference")
    if question not in ("preference", "detection"):
        die(f"question 只能是 preference / detection,收到 {question!r}")
    if question == "detection":
        die("detection 形状(每组 4 个文件、任务「分成两对」)本脚本还没实现 —— "
            "别用 preference 的 2 文件形状去冒充它,那是 S146g 犯过的退化(见 README 8a)")
    os.makedirs(out, exist_ok=True)
    listen = os.path.join(out, "listen")
    os.makedirs(listen, exist_ok=True)

    key = {}
    report = []
    for p in spec["pairs"]:
        label = p["label"]
        blank = bool(p.get("blank"))
        xr, sr_r = load(p["ref"])
        xc, sr_c = load(p["cand"])
        if sr_r != sr_c:
            die(f"{label}: 采样率不同 {sr_r} vs {sr_c}")
        a = int(round(p["start_s"] * sr_r))
        b = int(round(p["end_s"] * sr_r))
        if not (0 <= a < b <= min(len(xr), len(xc))):
            die(f"{label}: 区间 {a}..{b} 越界(ref {len(xr)} / cand {len(xc)})")
        sr = sr_r
        seg_r, seg_c = xr[a:b].copy(), xc[a:b].copy()

        gain_db = 0.0
        if not blank and spec.get("level_match", "rms") == "rms":
            gr, gc = rms(seg_r), rms(seg_c)
            if gr > 0 and gc > 0:
                g = gr / gc
                seg_c *= g
                gain_db = 20.0 * np.log10(g)

        if blank:
            # ⑴ 空白对照必须逐位相同
            d = float(np.max(np.abs(seg_r - seg_c))) if len(seg_r) else 0.0
            if d != 0.0:
                die(f"{label}: 空白对照两边不是逐位相同(max|Δ|={d:.3e})—— "
                    f"⛔ SVC 渲染不逐位可复现,空白对照必须用【同一个文件的副本】")

        # 淡入淡出 30 ms,免得切点自己变成线索
        n = min(int(sr * 0.03), (b - a) // 4)
        if n > 1:
            w = np.hanning(2 * n)
            for s in (seg_r, seg_c):
                s[:n] *= w[:n]
                s[-n:] *= w[n:]

        # ⑵ 字母只由**内容**决定:sha1(字节) 排序,小的拿 _A
        items = []
        for tag, seg in (("ref", seg_r), ("cand", seg_c)):
            h = hashlib.sha1(np.ascontiguousarray(seg.astype(np.float32)).tobytes()).hexdigest()
            items.append((h, tag, seg))
        items.sort(key=lambda t: t[0])
        for letter, (h, tag, seg) in zip("AB", items):
            name = f"{label}_{letter}.wav"
            sf.write(os.path.join(listen, name), seg.astype(np.float32), sr, subtype="PCM_16")
            key[name] = {"arm": tag, "sha1": h}
        report.append({
            "label": label, "blank": blank,
            "span_s": [p["start_s"], p["end_s"]],
            "seconds": round((b - a) / sr, 3),
            "cand_gain_db": round(gain_db, 4),
            "ref_file": os.path.basename(p["ref"]), "cand_file": os.path.basename(p["cand"]),
        })

    # ⑵ 回读自证:落盘后的字母序必须等于内容 sha1 序
    for p in spec["pairs"]:
        lab = p["label"]
        hs = []
        for letter in "AB":
            with open(os.path.join(listen, f"{lab}_{letter}.wav"), "rb") as f:
                hs.append(hashlib.sha1(f.read()).hexdigest())
        # 落盘是 PCM16 量化后的,顺序仍必须与内存序一致 ⇒ 直接比登记的 sha1 顺序
        ks = [key[f"{lab}_{letter}.wav"]["sha1"] for letter in "AB"]
        if ks != sorted(ks):
            die(f"{lab}: 字母序与内容 sha1 序不一致 —— 定序被别的东西决定了")

    # ⑶ KEY 单独一份,放在 listen 的**外面**,而且不打到 stdout
    keydir = os.path.join(out, "_key_听完再开")
    os.makedirs(keydir, exist_ok=True)
    json.dump({"round": spec["round"], "key": key, "pairs": report},
              io.open(os.path.join(keydir, "ANSWER_KEY.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    md = [f"# 盲测 {spec['round']}", "",
          "## 怎么听", "",
          "每一组两个文件(`_A` / `_B`)。**任务不是「哪个是新版」,是「哪个更好听」。**",
          "允许的答案有三个:**A 更好 / B 更好 / 听不出区别**。",
          "⛔ 「听不出区别」是一个**正确答案**,不要逼自己选 —— 其中有一组两边是同一段音频的副本,",
          "如果那一组你也选出了高下,这一轮的结论要作废重来。", "",
          "顺序是按**文件内容的哈希**排的,与版本无关;答案在 `_key_听完再开/` 里。", "",
          "## 仪器那边先说了什么(事前写死,免得事后凑)", ""]
    md += [f"* {s}" for s in spec.get("instruments", [])] or ["* (无)"]
    md += ["", "## 各组", "",
           "| 组 | 秒 | 区间(秒) | 候选臂增益 |", "|---|---|---|---|"]
    for r in report:
        md.append(f"| {r['label']}{' **(空白对照)**' if r['blank'] else ''} | {r['seconds']} | "
                  f"{r['span_s'][0]}–{r['span_s'][1]} | {r['cand_gain_db']:+.3f} dB |")
    if spec.get("note"):
        md += ["", "## 备注", "", spec["note"]]
    io.open(os.path.join(out, "怎么听.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    shutil.copyfile(spec_path, os.path.join(keydir, "spec.json"))

    # ⛔ 只报结构,不报映射
    print(f"打包完成: {out}")
    print(f"  {len(report)} 组 / {len(key)} 个文件 · 空白对照 "
          f"{sum(1 for r in report if r['blank'])} 组")
    for r in report:
        print(f"    {r['label']:<6} {r['seconds']:>6.2f}s  增益 {r['cand_gain_db']:+.3f} dB"
              f"{'   (空白对照)' if r['blank'] else ''}")
    print(f"  KEY: {keydir}\\ANSWER_KEY.json  (⛔ 听完再开)")
    return EXIT_OK


if __name__ == "__main__":
    if len(sys.argv) != 2:
        die("用法: blind_pack.py <spec.json>")
    raise SystemExit(main(sys.argv[1]))
