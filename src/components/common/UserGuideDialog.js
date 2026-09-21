import { jsxs as _jsxs, jsx as _jsx } from "react/jsx-runtime";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { t18 } from "../../lib/models/msst-catalog";
import "./UserGuideDialog.css";
const SECTIONS = [
    {
        id: "quickstart",
        icon: "🚀",
        title: { zh: "快速上手（3 步开唱）", en: "Quick Start (3 Steps)", ja: "クイックスタート（3ステップ）" },
        blocks: [
            {
                kind: "p",
                text: {
                    zh: "第一次打开软件只需要做三件事：下载组件 → 导入音频 → 开始创作。下面按顺序做完，所有核心功能就都能用了。",
                    en: "On first launch: download components → import audio → create. Finish these in order and every core feature unlocks.",
                    ja: "初回起動時は：コンポーネント入手 → 音声インポート → 制作開始。この順で進めれば全機能が使えます。",
                },
            },
            {
                kind: "steps",
                items: [
                    {
                        zh: "① 点顶部「资源管理」→「MIDI 模型」页 → 点「⚡ 一键下载必备组件」。它会自动把推理引擎、FFmpeg、FluidSynth、默认音源、默认转谱模型、Whisper 歌词模型等全部下好（约 3-4 GB，请耐心等待，支持断点续传）。",
                        en: "① Open Resource Manager → MIDI Models tab → click “⚡ One-click Essentials”. It downloads the engine, FFmpeg, FluidSynth, default soundfont, default models and Whisper automatically (~3-4 GB, resumable).",
                        ja: "① リソース管理 → MIDIモデルタブ →「⚡ 一括DL」をクリック。エンジン等を自動入手します（約3-4GB、再開対応）。",
                    },
                    {
                        zh: "② 把一首歌（MP3 / FLAC / WAV 都可以）拖进软件窗口，或用「文件 → 导入」。",
                        en: "② Drag a song (MP3 / FLAC / WAV) into the window, or use File → Import.",
                        ja: "② 楽曲（MP3 / FLAC / WAV）をウィンドウにドラッグ、またはファイル → インポート。",
                    },
                    {
                        zh: "③ 在音频上点右键试各种功能：「提取人声」「干声转音符（MIDI）」「自动识别歌词」……结果都能试听、编辑、导出。",
                        en: "③ Right-click the audio to try features: extract vocals, voice-to-MIDI, auto lyrics… All results can be previewed, edited and exported.",
                        ja: "③ 音声を右クリックで各機能を試せます：ボーカル抽出、音声→MIDI、歌詞認識…結果は試聴・編集・書き出し可能です。",
                    },
                ],
            },
            {
                kind: "tip",
                text: {
                    zh: "小提示：功能用到哪个组件没装时，软件会弹提示告诉你去资源管理里下载哪一项，不会让你猜。",
                    en: "Tip: when a feature needs a missing component, the app tells you exactly what to download.",
                    ja: "ヒント：必要コンポーネントがない場合、何を入手すべきかアプリが案内します。",
                },
            },
        ],
    },
    {
        id: "download",
        icon: "📦",
        title: { zh: "资源下载与网络问题", en: "Downloads & Network", ja: "ダウンロードとネットワーク" },
        blocks: [
            {
                kind: "p",
                text: {
                    zh: "模型和工具都从 HuggingFace / GitHub 下载。国内网络直连可能很慢或失败，软件内置了镜像加速：",
                    en: "Models and tools download from HuggingFace / GitHub. Mainland users can use built-in mirror acceleration:",
                    ja: "モデルは HuggingFace / GitHub から入手。内蔵ミラー加速が使えます：",
                },
            },
            {
                kind: "steps",
                items: [
                    {
                        zh: "① 打开「设置」→ 下载线路：HuggingFace 可切换 hf-mirror.com（国内直连推荐）；GitHub 可选 gh-proxy.com 等加速前缀。",
                        en: "① Settings → Download routes: switch HuggingFace to hf-mirror.com (recommended in mainland China); pick a GitHub proxy prefix such as gh-proxy.com.",
                        ja: "① 設定 → ダウンロード経路：HuggingFace は hf-mirror.com、GitHub は gh-proxy.com 等に切替。",
                    },
                    {
                        zh: "② 资源列表里标注「需魔法」的项目（如 Zenodo、Meta CDN 的个别模型）没有任何镜像线路——如果所有线路都连不上，需要开启外网代理（魔法上网）再下载。",
                        en: "② Items badged “VPN” have no mirror route (Zenodo, Meta CDN). If every route fails, enable external network access (VPN) first.",
                        ja: "②「要VPN」表示の項目はミラー経路がありません。全経路失敗時は外部ネットワーク（VPN）が必要です。",
                    },
                    {
                        zh: "③ 下载中断不用怕：重新点下载会从断点继续（.part 续传），已完成并通过校验的文件会自动跳过。",
                        en: "③ Interrupted downloads resume from the breakpoint; completed verified files are skipped automatically.",
                        ja: "③ 中断したダウンロードは再開可能。検証済みファイルは自動スキップされます。",
                    },
                ],
            },
            {
                kind: "warn",
                text: {
                    zh: "「一键下载必备组件」包含：AMT 推理引擎（约1.8GB）、FFmpeg、FluidSynth、MuseScore 默认音源、YourMT3+ 默认转谱模型、Beat-This 节拍模型、Whisper Small 歌词模型、MuseScore 4 乐谱运行时。全程约 3-4 GB，建议在网络空闲时进行。",
                    en: "“One-click Essentials” covers the AMT engine (~1.8 GB), FFmpeg, FluidSynth, default soundfont, YourMT3+ model, Beat-This, Whisper Small and MuseScore 4 — about 3-4 GB total.",
                    ja: "「一括DL」には AMT エンジン（約1.8GB）、FFmpeg、FluidSynth、音源、YourMT3+、Beat-This、Whisper Small、MuseScore 4 が含まれます（合計約3-4GB）。",
                },
            },
        ],
    },
    {
        id: "vocal",
        icon: "🎤",
        title: { zh: "人声合成（干音 → 修音 → 成品）", en: "Vocal Synthesis", ja: "ボーカル合成" },
        blocks: [
            {
                kind: "p",
                text: {
                    zh: "把歌手干音（无伴奏的人声）拖进软件后，双击音频块进入人声编辑器：音符、音高、歌词全部可视化编辑，像拼积木一样改歌。",
                    en: "Drop an a-cappella vocal into the app and double-click the clip to open the vocal editor: notes, pitch and lyrics are all editable visually.",
                    ja: "無伴奏ボーカルを投入し、クリップをダブルクリックするとボーカルエディタが開き、音符・ピッチ・歌詞を編集できます。",
                },
            },
            {
                kind: "steps",
                items: [
                    { zh: "① 顶部「歌词」按钮打开歌词总编辑：整段粘贴歌词，自动一字一音符分配（中文）或按空格分词（英文）。", en: "① The Lyrics button opens bulk editing: paste the whole lyric and it maps onto notes automatically.", ja: "①「歌詞」ボタンで一括編集：歌詞全体を貼り付けるだけで音符に自動マッピング。" },
                    { zh: "② 点任意音符可单独改它的歌词、音高、时值；拖拽移动，边框拉伸改长度。", en: "② Click any note to edit its lyric/pitch/duration; drag to move, stretch edges to resize.", ja: "② 音符クリックで歌詞・ピッチ・長さを個別編集。ドラッグで移動。" },
                    { zh: "③ 改完点「确定并试听」立刻听到新歌词的演唱效果。", en: "③ Click “Apply & Preview” to hear the new lyrics instantly.", ja: "③「適用して試聴」で新しい歌詞をすぐ試聴。" },
                ],
            },
            {
                kind: "tip",
                text: {
                    zh: "默认语言已是中文——中文歌直接显示中文歌词和中文轨道名；日文/英文歌在轨道设置里把语言切到对应语种即可。",
                    en: "The default language is Chinese. For Japanese/English songs, switch the track language accordingly.",
                    ja: "既定言語は中国語です。日本語・英語の曲はトラック設定で言語を切替えてください。",
                },
            },
        ],
    },
    {
        id: "amt",
        icon: "🎹",
        title: { zh: "干声转 MIDI + 歌词（核心玩法）", en: "Voice-to-MIDI + Lyrics", ja: "音声→MIDI＋歌詞" },
        blocks: [
            {
                kind: "p",
                text: {
                    zh: "在干音上点右键 →「提取 MIDI / 人声转音符」，软件会把人声转成带音符的 MIDI。如果是人声干音，还会自动用 Whisper 识别歌词并写到每个音符上（中文歌显示中文）。",
                    en: "Right-click a vocal clip → “Extract MIDI / voice to notes”. For dry vocals, Whisper automatically attaches lyrics to notes (Chinese songs get Chinese lyrics).",
                    ja: "ボーカルクリップを右クリック →「MIDI抽出」。乾声の場合、Whisper が自動で歌詞を音符に割り当てます。",
                },
            },
            {
                kind: "steps",
                items: [
                    { zh: "① 双击 MIDI 结果打开编辑工作台：钢琴卷帘里每个音符上直接写着它唱的字。", en: "① Double-click the MIDI result: every note shows its syllable on the piano roll.", ja: "① MIDI結果をダブルクリック：ピアノロールの音符に歌詞が表示されます。" },
                    { zh: "② 想改某个字：双击那个音符，输入新歌词，回车确认。", en: "② To change a syllable: double-click the note, type the new lyric, press Enter.", ja: "② 歌詞変更：音符をダブルクリックして入力。" },
                    { zh: "③ 想改整首：右侧「歌词」面板 →「文本模式」，一行一句粘贴整段歌词 →「应用」→「写回 MIDI」。", en: "③ To rewrite the whole song: Lyrics panel → Text mode, paste one line per phrase → Apply → Write back to MIDI.", ja: "③ 全体変更：歌詞パネル → テキストモードで全体を貼り付け → 適用 → MIDIに書き戻し。" },
                    { zh: "④ 试听就是新歌词；「写回 MIDI」后音符上的字也会同步刷新，导出的 MIDI 在别的软件（如 FL Studio、Cubase）里打开同样带着歌词。", en: "④ Preview sings the new lyrics; after write-back the notes update too, and exported MIDI keeps lyrics in other DAWs.", ja: "④ 試聴は新しい歌詞で再生。書き戻し後、書き出したMIDIも歌詞を保持します。" },
                ],
            },
            {
                kind: "tip",
                text: {
                    zh: "干音轨的试听就是人声本身；乐器轨用 FluidSynth + 音源发声。轨道名默认中文化（人声/伴奏/鼓…）。",
                    en: "Vocal stems preview with the real voice; instrument stems play via FluidSynth + soundfont. Track names are localized (人声/伴奏/鼓…).",
                    ja: "ボーカルトラックは実音で試聴可。楽器は FluidSynth＋音源で再生。トラック名は中国語表示されます。",
                },
            },
        ],
    },
    {
        id: "separation",
        icon: "🔀",
        title: { zh: "音频分离（人声 / 伴奏 / 乐器）", en: "Audio Separation", ja: "音声分離" },
        blocks: [
            {
                kind: "p",
                text: {
                    zh: "在任意音频上点右键 → 分离，可以把歌拆成人声+伴奏，或鼓/贝斯/吉他/钢琴等多轨。「资源管理 → 音频分离」页提供几十种模型：提取人声、去混响、去噪声、卡拉OK（去和声）等，每个都有中文说明和推荐标记。",
                    en: "Right-click audio → separate into vocals/instrumental or drums/bass/guitar/piano etc. The Separation tab offers dozens of models with descriptions and recommendations.",
                    ja: "右クリック → 分離でボーカル/伴奏やドラム/ベース等に分割。分離タブに多数のモデルがあります。",
                },
            },
            {
                kind: "tip",
                text: {
                    zh: "新手首选「BS-Roformer 12.98」（提取人声最准）和「MelBand 伴奏 V2」（伴奏最干净），下载时保持默认 fp32 精度即可。",
                    en: "Start with BS-Roformer 12.98 (best vocals) and MelBand Inst V2 (cleanest instrumental).",
                    ja: "まずは BS-Roformer 12.98（ボーカル）と MelBand 伴奏 V2（伴奏）から。",
                },
            },
        ],
    },
    {
        id: "soundfont",
        icon: "🎨",
        title: { zh: "更换音源（让 MIDI 更好听）", en: "Changing Soundfonts", ja: "音源の変更" },
        blocks: [
            {
                kind: "steps",
                items: [
                    { zh: "① 把任意 .sf2 / .sf3 音源文件直接拖进软件窗口，即自动安装进音源列表。", en: "① Drag any .sf2 / .sf3 file onto the window to install it.", ja: "① .sf2 / .sf3 をウィンドウにドロップでインストール。" },
                    { zh: "② 在 MIDI 工作台顶部的音源下拉里切换；所有试听和导出立即使用新音源。", en: "② Switch in the workbench soundfont dropdown; previews and exports use it immediately.", ja: "② ワークベンチの音源ドロップダウンで切替。試聴・書き出しに即反映。" },
                    { zh: "③ 资源管理里预置了 FluidR3（经典温暖）、GeneralUser GS（仅30MB）、Arachno（复古游戏风）等可一键下载。", en: "③ Preset downloads: FluidR3 (classic warm), GeneralUser GS (30 MB), Arachno (retro).", ja: "③ FluidR3・GeneralUser GS・Arachno 等がプリセットで入手可能。" },
                ],
            },
        ],
    },
    {
        id: "export",
        icon: "💾",
        title: { zh: "导出成品", en: "Exporting", ja: "書き出し" },
        blocks: [
            {
                kind: "p",
                text: {
                    zh: "MIDI 工作台的「下载」菜单五件套：MIDI 文件（每乐器一个 .mid，带歌词）、乐谱（PDF/MusicXML，需 MuseScore 组件）、转录音频（WAV 24bit/48kHz）、分轨音频（每乐器一个 WAV 的 zip）、立体声 A/B（左耳原声右耳 MIDI，对比神器）。",
                    en: "The workbench Download menu: per-instrument MIDI (with lyrics), sheet music (PDF/MusicXML), rendered WAV (24-bit/48kHz), stem WAVs (zip), and stereo A/B (left=original, right=MIDI).",
                    ja: "ダウンロードメニュー：楽器別MIDI（歌詞付き）、楽譜（PDF/MusicXML）、WAV、ステム一式、A/Bステレオ。",
                },
            },
            {
                kind: "p",
                text: {
                    zh: "歌词面板还能单独导出 .lrc 卡拉OK字幕文件；「文件 → 导出音频 / 导出乐谱」导出整个人声工程。",
                    en: "The lyrics panel exports .lrc karaoke files; File → Export covers the whole vocal project.",
                    ja: "歌詞パネルから .lrc も書き出せます。",
                },
            },
        ],
    },
    {
        id: "faq",
        icon: "❓",
        title: { zh: "常见问题", en: "FAQ", ja: "よくある質問" },
        blocks: [
            {
                kind: "p",
                text: {
                    zh: "Q：下载一直失败 / 速度为 0？\nA：先到设置里切换镜像线路；带「需魔法」标记的必须开外网代理。下载中断直接重点，会断点续传。",
                    en: "Q: Downloads keep failing?\nA: Switch mirror routes in Settings; “VPN”-badged items need external access. Interrupted downloads resume on retry.",
                    ja: "Q：ダウンロードが失敗する？\nA：設定でミラー経路を切替。「要VPN」項目は外部ネットワークが必要。再試行で再開します。",
                },
            },
            {
                kind: "p",
                text: {
                    zh: "Q：MIDI 播放没声音？\nA：检查资源管理里 FluidSynth 和 MuseScore 音源是否已安装（必备区前两项）。",
                    en: "Q: No sound when playing MIDI?\nA: Check that FluidSynth and the soundfont are installed (first two essentials).",
                    ja: "Q：MIDI再生で音が出ない？\nA：FluidSynth と音源がインストール済みか確認してください。",
                },
            },
            {
                kind: "p",
                text: {
                    zh: "Q：歌词识别不出来 / 全是空的？\nA：需要先下载 Whisper 模型（必备区「Whisper Small 歌词识别」）。识别源会优先用分离出的干音，比整曲混音准得多。",
                    en: "Q: Lyrics come out empty?\nA: Install the Whisper model first (Whisper Small in Essentials). Extraction prefers the separated dry vocal for accuracy.",
                    ja: "Q：歌詞が認識されない？\nA：Whisper モデル（Whisper Small）を先に入手してください。",
                },
            },
            {
                kind: "p",
                text: {
                    zh: "Q：转 MIDI 很慢？\nA：默认模型精度最高但最慢；急用时可在模型库里换轻量模型（如 MuScriptor Small）。GPU（N卡）会比 CPU 快很多倍。",
                    en: "Q: Transcription is slow?\nA: The default model is the most accurate but slowest; switch to a lighter model (e.g. MuScriptor Small) for drafts. GPU is many times faster than CPU.",
                    ja: "Q：変換が遅い？\nA：軽量モデルへの切替か GPU 利用を推奨します。",
                },
            },
            {
                kind: "p",
                text: {
                    zh: "Q：闪退 / 功能异常？\nA：「日志」菜单里可以查看运行日志，把报错内容发给开发者（QQ：202112）能最快定位。",
                    en: "Q: Crash or misbehavior?\nA: Check the Log menu and send the error text to the developer (QQ: 202112).",
                    ja: "Q：異常終了？\nA：ログメニューからエラー内容を開発者（QQ: 202112）へ。",
                },
            },
        ],
    },
];
/** 新手使用说明 —— 帮助菜单打开。左侧目录 + 右侧图文步骤，Esc / 点击背景关闭。 */
export function UserGuideDialog({ onClose }) {
    const { i18n } = useTranslation();
    const lang = i18n.language;
    const [active, setActive] = useState(SECTIONS[0].id);
    useEffect(() => {
        const onKey = (e) => {
            if (e.key === "Escape")
                onClose();
        };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [onClose]);
    return (_jsx("div", { className: "ug-overlay", onClick: onClose, children: _jsxs("div", { className: "ug-panel", onClick: (e) => e.stopPropagation(), children: [_jsxs("div", { className: "ug-head", children: [_jsxs("span", { className: "ug-title", children: ["\uD83D\uDCD6 ", t18({ zh: "使用说明（新手友好版）", en: "User Guide (Beginner-Friendly)", ja: "使い方ガイド（初心者向け）" }, lang)] }), _jsx("button", { className: "ug-close", onClick: onClose, children: "\u2715" })] }), _jsxs("div", { className: "ug-body", children: [_jsx("nav", { className: "ug-nav", children: SECTIONS.map((s) => (_jsxs("button", { className: `ug-nav-item ${active === s.id ? "active" : ""}`, onClick: () => {
                                    setActive(s.id);
                                    document.getElementById(`ug-${s.id}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
                                }, children: [_jsx("span", { className: "ug-nav-icon", children: s.icon }), t18(s.title, lang)] }, s.id))) }), _jsx("div", { className: "ug-content", children: SECTIONS.map((s) => (_jsxs("section", { id: `ug-${s.id}`, className: "ug-section", children: [_jsxs("h3", { className: "ug-section-title", children: [_jsx("span", { className: "ug-section-icon", children: s.icon }), t18(s.title, lang)] }), s.blocks.map((b, i) => {
                                        if (b.kind === "p") {
                                            return (_jsx("p", { className: "ug-p", children: t18(b.text, lang).split("\n").map((line, j) => (_jsx("span", { className: "ug-line", children: line }, j))) }, i));
                                        }
                                        if (b.kind === "steps") {
                                            return (_jsx("ol", { className: "ug-steps", children: b.items.map((it, j) => (_jsx("li", { className: "ug-step", children: t18(it, lang) }, j))) }, i));
                                        }
                                        return (_jsxs("div", { className: `ug-callout ${b.kind === "warn" ? "warn" : "tip"}`, children: [_jsx("span", { className: "ug-callout-icon", children: b.kind === "warn" ? "⚠️" : "💡" }), _jsx("span", { children: t18(b.text, lang) })] }, i));
                                    })] }, s.id))) })] }), _jsx("div", { className: "ug-foot", children: t18({ zh: "按 Esc 或点击空白处关闭 · 更多帮助：QQ 202112", en: "Esc or click outside to close · More help: QQ 202112", ja: "Esc または外側クリックで閉じる · QQ 202112" }, lang) })] }) }));
}
