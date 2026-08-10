//! §F2⒝ 批 2 ④d(R3)—— **五条训练链的池身份公式**,头一次有东西看着。
//!
//! ## 它守的是什么
//!
//! `dataset.fingerprint` 的**文本**决定这次运行的预处理产物落在哪个池(`utai_train/pool.py`
//! 的 `open_pool` 按内容选,不匹配就铸一个兄弟池并**重跑整份预处理**,唯一的痕迹是一行
//! `logger.info`)。这条文本由五条链各自拼出来,而它们**不共用一份实现**:
//!
//! * sovits / sovits_diff / sovits_v2 走同一个 helper `extract_cache_fp_text`;
//! * sovits_v2 在 helper 的返回值**之后**再条件追加一段;
//! * rvc 与 vocoder **完全内联自建**。
//!
//! ⇒ 一次跨五条链的编辑,「改了 3 条漏了 2 条」在 ④d 之前**没有任何东西会红**:
//! `converter/verify/training/gate_pool_table.py` 的七组检查没有一组读公式的文本,而且它本身
//! 不在任何自动闸里(本仓无 CI、无 git hook,自动闸只有 `cargo test` 与 `vitest run`)。
//!
//! ## 它为什么长这个样子
//!
//! ⛔ **判据是【值】不是【顺序】**。「追加语句排在 `open_pool` 之前」这类顺序断言会被邻居行
//! 满足,而且 sovits_v2 已经有一段条件追加(`|f0=`)—— 新 token 加在它之前还是之后是**两个
//! 不同的字符串**,顺序判据对这个差别一无所知。所以这里比的是每条链**实际发射的 token 集合**。
//!
//! ⛔ **它必须能抓「漏了一条链」而不只是「改了一条链」**。④d 之前靠的是
//! [`NOT_YET_IN_THE_FORMULA`] 那张待办清单(一条条目管全部五条链);④d 落地之后清单清空,
//! 接手的是**共享后缀**本身:`|sr=` / `|aug=` 住在 `pool.identity_suffix` 一个函数里,一条链
//! 的 token 集合是「它自己那段的字面量」∪「它**调不调**那个 helper、传不传 `sample_rate=`」——
//! 于是「漏了一条链」表现为那条链少两个 token,当场红。
//!
//! ⛔ **它自己必须证明活着**。一个读空的解析器会让两个空集恰好相等。所以:每条链的 token 集合
//! 逐条写死对拍(不是「非空」这种下限),外加一组合成坏源码的阴性对照。
//!
//! `include_str!` 是有意的:文件被挪走 = **编译失败**,而不是解析出一个空串。

use std::collections::BTreeSet;

use utai_lib::training::resume_lock::{resume_locked_fields, LockTier};
use utai_lib::training::tpool;

const SOVITS_PY: &str = include_str!("../../training/utai_train/sovits/pipeline.py");
const DIFF_PY: &str = include_str!("../../training/utai_train/sovits/diff_pipeline.py");
const V2_PY: &str = include_str!("../../training/utai_train/sovits_v2/pipeline.py");
const RVC_PY: &str = include_str!("../../training/utai_train/rvc/pipeline.py");
const VOC_PY: &str = include_str!("../../training/utai_train/vocoder/pipeline.py");
const POOL_PY: &str = include_str!("../../training/utai_train/pool.py");
const FLIST_PY: &str = include_str!("../../training/utai_train/sovits/flist.py");
const CKPT_PY: &str = include_str!("../../training/utai_train/ckpt_guard.py");

/// 一个顶层 python 函数的函数体(到下一个顶格 `def ` 为止)。
fn top_level_fn(src: &'static str, name: &str, what: &str) -> &'static str {
    let at = src
        .find(&format!("def {name}("))
        .unwrap_or_else(|| panic!("{what}:找不到 `def {name}(` —— {name} 没了或改名了"));
    let rest = &src[at..];
    let end = rest[1..].find("\ndef ").map(|i| i + 1).unwrap_or(rest.len());
    &rest[..end]
}

/// 共享 helper 的函数体 —— sovits 家三条链的公式主体都在这里。
fn helper_body() -> &'static str {
    top_level_fn(
        SOVITS_PY,
        "extract_cache_fp_text",
        "共享的池身份 helper 没了 —— 三条 sovits 链的公式从此各写各的",
    )
}

/// 五条链**共用**的尾巴 —— `|sr=` 与 `|aug=` 住在这里,而且只住在这里。
fn suffix_body() -> &'static str {
    top_level_fn(POOL_PY, "identity_suffix", "④d 的共享尾巴没了 —— 五条链又各拼各的了")
}

/// 一条链**自己**那段拼串代码:从第一次给 `fp_text`/`fp_src` 赋值,到 `open_pool(` 为止。
///
/// ⛔ 上界取 `open_pool(` 而不是「往下数 N 行」:窗口必须由**被测结构自己**收尾。固定宽度的
/// 窗口会吞掉邻居(S127 血训 ⒜:一个 500 字符的窗口吞掉了下一个调用点,于是把被测的那处改坏
/// 之后,断言被邻居满足、探针报绿)。
fn own_region(src: &'static str, chain: &str) -> &'static str {
    let start = src
        .find("fp_text = ")
        .or_else(|| src.find("fp_src = "))
        .unwrap_or_else(|| panic!("{chain}: 找不到池身份串的赋值"));
    let rest = &src[start..];
    let end = rest
        .find("open_pool(")
        .unwrap_or_else(|| panic!("{chain}: 赋值之后没有 open_pool —— 这段代码的形状变了"));
    &rest[..end]
}

/// 这段 python 源码里的**字符串字面量**,按出现顺序。
///
/// 只收字面量:`"%s|enc=%s|loudnorm=%d"` 里的 `enc`/`loudnorm` 是真 token,而注释或 docstring
/// 里写的 `|aug=` 不是。少了这一条过滤,一句「TODO: 加 |aug=」的注释就能让这道闸提前变绿。
///
/// ⚠ 单引号与双引号**都收**:python 两种都合法,只认一种就等于给未来的编辑留了一个
/// 「换个引号这道闸就瞎了」的洞。
fn literals_of(src: &str) -> Vec<String> {
    let ch: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let (mut i, mut in_doc, mut comment) = (0usize, false, false);
    let mut in_str: Option<char> = None;
    let mut doc_q = '"';
    let triple = |ch: &Vec<char>, i: usize, q: char| {
        ch.get(i) == Some(&q) && ch.get(i + 1) == Some(&q) && ch.get(i + 2) == Some(&q)
    };
    let mut lit = String::new();
    while i < ch.len() {
        let c = ch[i];
        // ⛔ docstring 必须整段跳过,而**不是**当成一个字符串字面量。这几个文件的函数头
        // 全是 `"""…"""`,而 python 里没有块注释 —— 一句「格式是 `%s|enc=%s|aug=%d`」的文档
        // 会被当成真公式采进来,于是这道闸对着一段散文变绿。
        if in_doc {
            if triple(&ch, i, doc_q) {
                in_doc = false;
                i += 3;
                continue;
            }
            i += 1;
            continue;
        }
        if c == '\n' {
            comment = false;
            in_str = None;
            lit.clear();
        } else if comment {
            // 行注释:整行剩下的部分都不算
        } else if in_str.is_none() && (triple(&ch, i, '"') || triple(&ch, i, '\'')) {
            in_doc = true;
            doc_q = c;
            i += 3;
            continue;
        } else if in_str.is_none() && c == '#' {
            comment = true;
        } else if c == '"' || c == '\'' {
            match in_str {
                // 另一种引号在字符串里就是普通字符(`"it's"`)
                Some(q) if q != c => lit.push(c),
                Some(_) => {
                    out.push(std::mem::take(&mut lit));
                    in_str = None;
                }
                None => in_str = Some(c),
            }
        } else if in_str.is_some() {
            lit.push(c);
        }
        i += 1;
    }
    out
}

/// 这段源码里的字面量会发射出哪些 `|key=` token。
fn tokens_of(src: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for lit in literals_of(src) {
        for piece in lit.split('|').skip(1) {
            if let Some(eq) = piece.find('=') {
                let key = &piece[..eq];
                if !key.is_empty() && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                    out.insert(key.to_string());
                }
            }
        }
    }
    out
}

/// 一条链:名字 · 它自己那段源码 · 它**实际**会发射的 token 全集。
///
/// ★ 全集 = 自己那段的字面量 ∪ 共享尾巴给它的那些。后者**不是声明出来的**,是从源码里
/// 「它调不调 `identity_suffix(`」「传不传 `sample_rate=`」读出来的 —— 所以一条链忘了调那个
/// helper,它的全集当场少两个 token,而不是悄悄少一段字符串。
struct Chain {
    name: &'static str,
    region: &'static str,
    /// 这条链的整份源码 —— `region` 在 `open_pool(` 处收尾,所以它**看不到那次调用的实参**。
    src: &'static str,
    emitted: BTreeSet<String>,
}

fn chains() -> Vec<Chain> {
    let h = helper_body();
    let raw: [(&'static str, &'static str, &'static str); 5] = [
        ("sovits", h, SOVITS_PY),
        ("sovits_diff", h, DIFF_PY),
        ("sovits_v2", h, V2_PY),
        ("rvc", "", RVC_PY),
        ("vocoder", "", VOC_PY),
    ];
    raw.into_iter()
        .map(|(name, shared, src)| {
            let region = own_region(src, name);
            let mut emitted = tokens_of(&format!("{shared}\n{region}"));
            if region.contains("identity_suffix(") {
                emitted.insert("aug".into());
                if region.contains("sample_rate=") {
                    emitted.insert("sr".into());
                }
            }
            Chain { name, region, src, emitted }
        })
        .collect()
}

/// 今天每条链**实际**发射的 token,逐条写死。
///
/// ⛔ 不是「非空」这种下限:一个读空的解析器会让每一条都是空集,而空集与空集相等。
const CHAIN_TOKENS: &[(&str, &[&str])] = &[
    ("sovits", &["aug", "enc", "loudnorm"]),
    ("sovits_diff", &["aug", "enc", "loudnorm"]),
    // v2 在 helper 之后条件追加 `|f0=<method>`(非 rmvpe 时,只有 gate 会传 dio),**再**接尾巴
    ("sovits_v2", &["aug", "enc", "f0", "loudnorm"]),
    // rvc 的自建部分是**裸的**数据集指纹(多说话人是一串无 `=` 的 hash 用 `|` 拼);
    // 五条链里只有它的采样率是用户选的,所以只有它传 `sample_rate=`
    ("rvc", &["aug", "sr"]),
    // vocoder 自建部分的尾巴 `|vocoder-v3` 是手工版本 tag,不是 key=value
    ("vocoder", &["aug"]),
];

/// 池级(锁表 `scope` 说它作废预处理产物)且用户**改得动**(`Costly`)的字段,在公式里怎么承载。
///
/// `Fingerprint` = 由 `dataset_fingerprint` 本身承载(数据集内容变了指纹就变),不需要单独的
/// token;`Token(k)` = 公式里必须出现 `|k=`。
enum CarriedBy {
    Token(&'static str),
    Fingerprint,
}

fn carrier(id: &str) -> Option<CarriedBy> {
    match id {
        "dataset" => Some(CarriedBy::Fingerprint),
        "loudnorm" => Some(CarriedBy::Token("loudnorm")),
        // ★④d:这一行从 `NOT_YET_IN_THE_FORMULA` 挪到这里,就是这一批做完了的意思。
        "augCopies" => Some(CarriedBy::Token("aug")),
        _ => None,
    }
}

/// ⛔ **不是白名单,是待办清单。**
///
/// 每一条都是「这个池级字段今天**不在**公式里,所以改它会静默复用另一套配方算出来的产物」。
/// ④d 之前它有一条(`augCopies`);现在它是空的,而空**本身**是被下面那条断言钉住的 ——
/// 往里加一项就必须说清「为什么一个池级字段可以不进身份」。
const NOT_YET_IN_THE_FORMULA: &[(&str, &str)] = &[];

#[test]
fn every_chain_emits_the_pool_identity_tokens_it_declares() {
    // ⑴ 存活证明:解析器真的读到了东西,而且每条链**逐条**相等。
    for c in chains() {
        let want: BTreeSet<String> = CHAIN_TOKENS
            .iter()
            .find(|(n, _)| *n == c.name)
            .unwrap_or_else(|| panic!("{} 没有声明它的 token 集合", c.name))
            .1
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            c.emitted, want,
            "{}: 公式发射的 token 与声明不一致 —— 改公式必须同时改这张表",
            c.name
        );
    }
    assert_eq!(CHAIN_TOKENS.len(), 5, "五条链,一条都不许从这张表上消失");

    // ⑵ 阴性对照:解析器必须真的会说「不一样」,而且不许把注释/docstring 读成 token。
    assert_eq!(tokens_of(r#"x = "%s|enc=%s|aug=%d""#), ["aug", "enc"].map(String::from).into());
    // 单引号同样要收 —— 换一种引号就让这道闸变瞎,是最省事也最难发现的一种改法。
    assert_eq!(tokens_of("x = '%s|enc=%s'"), ["enc"].map(String::from).into());
    // ⚠ 这两条阴性对照**必须带引号**。第一版写的是 `"# TODO: 以后加 |aug=<n>"` —— 那条输入
    // 里一个引号都没有,采集器压根走不到注释这条分支,于是「把注释屏蔽关掉」这个变异**存活**了:
    // 一条测不到被测分支的阴性对照,就是一条空判据。
    assert!(
        tokens_of("x = 1  # 以后要加 \"%s|aug=%d\"\n").is_empty(),
        "行注释里的 token 不算 —— 否则一句 TODO 就能让这道闸提前变绿"
    );
    assert!(
        tokens_of("def f():\n    \"\"\"格式是 %s|aug=%d 这样\"\"\"\n    return 1\n").is_empty(),
        "docstring 里的 token 不算 —— python 没有块注释,函数头那一大段散文全在三引号里"
    );
    assert!(
        tokens_of("def f():\n    '''格式是 %s|aug=%d 这样'''\n    return 1\n").is_empty(),
        "单引号 docstring 同理"
    );
    assert!(tokens_of(r#"x = "%s|vocoder-v3""#).is_empty(), "没有 `=` 的尾巴不是 key=value");
}

/// ★④d 的**结构**判据:五条链共用一条尾巴,而且都把它接在**最后**。
///
/// 「接在最后」不是洁癖:sovits_v2 在共享 helper 之后还有一段自己的 `|f0=`,所以把新 token 折进
/// helper 会让同一组 token 在 v2 上排在 `|f0=` 之前、在另外两条 sovits 链上排在末尾 —— 一组
/// token 两种拼接顺序,而 Rust 那边的迁移器要重现哪一种取决于它知不知道自己在看哪条链。
#[test]
fn every_chain_appends_the_one_shared_suffix_last() {
    let body = suffix_body();

    // ⑴ 尾巴只发射这两个 token,而且是这个顺序。顺序是**值**的一部分:`|sr=` 与 `|aug=`
    //    换个先后就是另一个字符串,也就是另一个池。
    let lits: Vec<String> =
        literals_of(body).into_iter().filter(|s| s.starts_with('|')).collect();
    assert_eq!(
        lits,
        vec!["|sr=%d".to_string(), "|aug=%d".to_string()],
        "共享尾巴发射的字面量与顺序变了 —— 同批必须改 `tpool::identity_suffix`"
    );

    // ⑵ 三条守卫都在:版本闸、`sample_rate` 可选、`n>0` 才追加。
    //    (`n=0` 不追加正是「现存未增强的池全部继续匹配」那句话的落点。)
    for guard in ["if identity_version(cfg) < 2:", "if sample_rate is not None:", "> 0:"] {
        assert!(body.contains(guard), "共享尾巴少了守卫 `{guard}`");
    }

    // ⑶ 五条链**都**调它,传的是**这次运行的份数变量**,而且调完之后不许再动那个串。
    //
    // ⛔ 「传的是什么」必须一起钉。`emitted` 的推导只看「调没调 / 传没传 `sample_rate=`」,
    //    所以 `identity_suffix(cfg, 0)`(份数写死)或 `identity_suffix(cfg, seed)`(传错变量)
    //    对上面那条集合断言是**全绿**的 —— 而它们的后果是所有 run 落进同一个池、或者按 seed 分池。
    //    这里钉的是实参的**文本形状**;真正跑一遍那个值的是行为腿。
    for c in chains() {
        assert!(
            c.region.contains("identity_suffix(cfg, aug_copies"),
            "{}: 这条链没有用【这次运行的份数变量】接共享尾巴 —— \
             「改 3 漏 2」与「传错变量」都从这里开始",
            c.name
        );
        let last = c.region.rfind("identity_suffix(").unwrap();
        let tail = &c.region[last..];
        for later in ["fp_text +=", "fp_src +=", "fp_text = ", "fp_src = "] {
            assert!(
                !tail.contains(later),
                "{}: 追加共享尾巴之后又动了池身份串(`{later}`)—— 尾巴必须是最后一步",
                c.name
            );
        }
    }

    // ⑷ 五条链都把 `run_dir` 交给 `open_pool` —— 那是 run↔pool 这条边**唯一**的记录点
    //    (④e 的池回收离了它就没有答案,而 Rust 算不出池名:池由数据集指纹命名)。
    //    ⚠ 漏掉一条链是**静默**的:`run_dir` 是可选参数,不传就是不记,没有任何东西会说话。
    for c in chains() {
        assert!(
            c.src.contains("open_pool(slot_dir, fp_text, run_dir)")
                || c.src.contains("open_pool(slot_dir, fp_src, run_dir)"),
            "{}: 这条链没有把 run_dir 交给 open_pool —— 它选的池不会被记下来",
            c.name
        );
    }

    // ⑸ 只有 rvc 传采样率。它是五条链里唯一一条采样率是用户选的;另外四条硬编 44100,
    //    给它们发一个恒定的 token 只会让每一个存量池当场换身份。
    let with_sr: Vec<&str> = chains()
        .iter()
        .filter(|c| c.region.contains("sample_rate="))
        .map(|c| c.name)
        .collect();
    assert_eq!(with_sr, vec!["rvc"], "`|sr=` 的归属变了");
    // …而 rvc 传的必须是**这次运行请求的那个采样率**,不是一个写死的数字。写死 44100 会让
    // 每一个 rvc 池都算出同一个 `|sr=`,于是这个 token 什么都不分开 —— 而集合断言照样绿。
    let rvc = chains().into_iter().find(|c| c.name == "rvc").unwrap();
    assert!(
        rvc.region.contains("sample_rate=SR_MAP[sr_str]"),
        "rvc 传给共享尾巴的采样率不再是这次请求的那个"
    );
}

/// ★④d:两种语言必须拼出**同一个**字符串,否则整槽的池被判陌生、用户重跑几小时。
///
/// ⚠ 这条钉的是 Rust 那半的**值** + python 那半的**字面量与顺序**(上一条)。两条合起来才等于
/// 「两边同串」;单独任何一条都能被一个空实现满足。真正跨语言逐字节对拍的是行为腿(它让 python
/// 真跑一次再和这里算出的串比),这道闸的职责是让那条腿**跑之前**就不可能改歪。
#[test]
fn the_two_languages_build_the_same_identity_suffix() {
    use tpool::identity_suffix as sfx;
    assert_eq!(sfx(2, 0, None), "", "n=0 且无采样率 ⇒ 空尾巴 = 存量池继续匹配");
    assert_eq!(sfx(2, 2, None), "|aug=2");
    assert_eq!(sfx(2, 0, Some(40_000)), "|sr=40000", "rvc 的 sr 是无条件的(它必填)");
    assert_eq!(sfx(2, 3, Some(48_000)), "|sr=48000|aug=3", "sr 在前,aug 在后");
    assert_eq!(sfx(1, 3, Some(48_000)), "", "v1 = ④d 之前的公式,一个 token 都不许多");
    // 版本号两边必须同步;python 那半在 `pool.POOL_IDENTITY_VERSION`。
    assert!(
        POOL_PY.contains(&format!("POOL_IDENTITY_VERSION = {}", tpool::POOL_IDENTITY_VERSION)),
        "两种语言的池身份版本号对不上"
    );
    assert!(
        POOL_PY.contains(&format!("SOLE_SPEAKER_DIR = \"{}\"", tpool::SOLE_SPEAKER_DIR)),
        "单说话人切片目录名两种语言对不上 —— 它同时是 `config.spk` 的键与 loader 的反查键"
    );
    // ★承载版本号的那个 `run.json` 键。⛔ 这是整批唯一一个**改错了不会报错**的地方:python 对
    //   ABSENT 的键回落到 1(旧 run.json 描述的确实是 v1 的盘),所以一次改名不会炸,只会让 ④d
    //   **永远不生效** —— 迁移器照样重新打戳,而 python 一直算旧公式,一次静默的空转。
    assert!(
        POOL_PY.contains(&format!("cfg.get(\"{}\")", tpool::IDENTITY_VERSION_KEY)),
        "python 读的 run.json 键与 `tpool::IDENTITY_VERSION_KEY` 对不上"
    );
    // ⛔★S130 —— 左值与**右值**要一起钉。只钉 `run_config[…KEY]` 的版本(S129 那版)对
    //   「把右值写死成 `json!(2)`」是**瞎的**,而那个改动的后果最重:每个槽都被告知「你已经
    //   迁完了」,于是 python 对**没迁过**的槽也算 v2 公式 ⇒ 全量重跑到一个兄弟池 ——
    //   恰恰是版本闸存在的全部理由(「Rust 拒绝迁移 ⇒ python 照新公式算」)反过来发生一遍。
    //   ⇒ 值必须是**那个槽的**纯函数,不是常量。
    const VERSION_WRITE: &str =
        "run_config[tpool::IDENTITY_VERSION_KEY] = serde_json::json!(tpool::identity_version(";
    assert!(
        include_str!("../src/training/mod.rs").contains(VERSION_WRITE),
        "run.json 里那个版本号不再是**这个槽的** `identity_version(...)` 了 —— 写成常量的话,\
         没迁过的槽也会被告知自己是 v2,python 于是按新公式算一个盘上不存在的池"
    );
    // 固定名的三条硬规矩(`pool.py` 写明了每一条背后的故障)。
    let d = tpool::SOLE_SPEAKER_DIR;
    assert!(!d.contains(".wav"), "路径级 replace 会改写目录段");
    assert!(!d.eq_ignore_ascii_case("nul"), "Windows 上 makedirs 成功却什么也没建");
    assert!(!d.contains('_'), "slugify 产出的 slug 一律带 `_`+8 hex ⇒ 无 `_` 就撞不上真说话人");
}

/// ★★§F2⒝ ④e 笔 1 —— 「这个 run 是新铸的」这条载体,两侧各钉一条。
///
/// ## 它守的是什么
///
/// ④e 要把「重训」从**删掉整个槽**改成**在旁边铸一个新 run**。那一刀能造出的故障不是崩溃:
/// 只要铸造哪天把一个**已经在用**的目录交回来(id 是 family 的纯函数、迁移半途、手搭的 cfg),
/// python **无从分辨** —— `fresh` 根本不过语言边界,空的 `resume_from` 在上游被规范成 `latest`,
/// 而 `plan_load` 只看文件在不在。于是它续训**别人那个 run** 并覆盖它,而 UI 说新 run 开始了。
///
/// ⚠ 诚实边界(交接里写的「没有任何现存判据看得见」是**夸大**,侦察核验推翻了绝对形式):
/// 五条链**都**有 `steps_run == 0` 的事后抛,所以撞上一个**已经练完**的 run 今天是响亮的
/// —— 代价是几小时预处理已经付掉。**静默的是撞上一个还没练完的 run**,那才是「再训一个」的常态。
///
/// ## ⛔ 为什么这条判据必须钉**右值**
///
/// `run_worker` 的作用域里 `req` 就在手边,所以 `json!(req.fresh)` 是最顺手的一笔,而它在
/// `diff_partial_wipe` 那条臂上是**错的**(那条臂 `fresh==true` 却只删 `<run>/diffusion/`,
/// run 根下主模型的 `G_*.pth` 全在)⇒ python 会拒绝一次完全正常的「重训(仅扩散)」,
/// 而且**今天不会有任何判据变红**(扩散链根本不走 `plan_load`)。
/// ⇒ 右值必须是 `try_start` 里那个具名绑定;绑定本身的右值由 `mod.rs` 的源序棘轮钉。
#[test]
fn the_two_languages_agree_on_the_fresh_run_carrier() {
    use utai_lib::training::trun;
    // ⑴ 键名逐字相同。⛔ 与 ④d 的版本号键同一种失效:python 对 ABSENT 回落 False(旧 run.json
    //    描述的确实是旧世界的 start),所以**改名不会报错**,只会让这道闸永远不生效。
    assert!(
        CKPT_PY.contains(&format!("FRESH_RUN_KEY = \"{}\"", trun::FRESH_RUN_KEY)),
        "python 读的 run.json 键与 `trun::FRESH_RUN_KEY` 对不上 —— 改名是**静默**失效"
    );
    // ⑵ 缺键必须回落成「不是新铸的」。反过来(缺键⇒True)会让每一次续训都被拒。
    let reader = top_level_fn(CKPT_PY, "declares_fresh_run", "④e 的载体读点没了");
    assert!(
        reader.contains("if raw is None:\n        return False"),
        "缺键不再回落成「不是新铸的」—— 存量 run.json 会被当成新铸,续训当场被拒"
    );
    // ⑶ 非布尔值必须**响亮拒绝**,不许 `bool(raw)`:`\"false\"` 会是 True(拒绝每一次续训),
    //    `0`/`\"\"` 会是 False(这道闸永远关着)。两种都静默。
    assert!(
        reader.contains("if not isinstance(raw, bool):"),
        "载体的值不再被校验类型 —— 一个写错的写者会静默地把这道闸常开或常关"
    );
    // ⑷ Rust 那半的**右值**。照 ④d 的教训:只钉左值对「写死成常量」是瞎的,而写死 `true`
    //    会让**每一次续训**都声称自己是新铸的 ⇒ 只要盘上有存档就当场拒,功能整个不可用。
    const FRESH_WRITE: &str =
        "run_config[trun::FRESH_RUN_KEY] = serde_json::json!(ctx.mints_fresh_run);";
    assert!(
        include_str!("../src/training/mod.rs").contains(FRESH_WRITE),
        "run.json 里那个「是不是新铸的」不再是 `ctx.mints_fresh_run` —— 写成常量或写成 \
         `req.fresh` 的后果全在 `trun::FRESH_RUN_KEY` 的 doc 里"
    );
}

/// ★★§F2⒝ ④e 笔 1 —— 那条闸挂在**五条链共用的那一个**读点上,一条都不许漏。
///
/// ⛔ 挂在 `runner.main` 上覆盖面一样,但 `gate_aug0_driver.py` 是 `importlib` 进来直接调
/// `pipeline.run(cfg, …)` 的 —— 绕开 runner ⇒ **行为腿结构上永远驱不到那条闸**。
/// 这道断言钉的就是「它挂在腿够得着的那一层」。
///
/// ⚠ 与上面那条 `open_pool(…, run_dir)` 的五链断言同形,理由也同:漏掉一条链是**静默**的。
#[test]
fn every_chain_routes_its_run_dir_through_the_fresh_run_guard() {
    for c in chains() {
        assert!(
            c.src.contains("checked_run_dir(cfg, slot_dir)"),
            "{}: 这条链不再经过共用的 run 目录解析器 —— ④e 的闸(以及 RUN_DIR_NOT_IN_SLOT)\
             对它零覆盖",
            c.name
        );
    }
    // …而那个解析器**真的**调闸。⛔ 少了这一句,上面五条断言就退化成「五条链都调了一个不做事的
    //   函数」——「测试自己是空的」那一族。
    let body = top_level_fn(POOL_PY, "checked_run_dir", "共用的 run 目录解析器没了");
    assert!(
        body.contains("ckpt_guard.refuse_to_resume_into_a_fresh_run(cfg, run_dir)"),
        "共用读点不再问「这个新铸的 run 是不是已经有人练过了」"
    );
    // 闸自己必须把三类产物都看一遍:只查 GAN 那对,等于扩散与声码器两条链**结构上没被盖到**,
    // 而报告读起来像盖全了。
    let guard = top_level_fn(CKPT_PY, "refuse_to_resume_into_a_fresh_run", "④e 的闸没了");
    for arm in ["_main_resume_point(", "_diffusion_resume_point(", "_vocoder_resume_point("] {
        assert!(guard.contains(arm), "④e 的闸少了一类续训点:{arm}");
    }
}

/// ★④d 去名字化的**两个守卫**,一条一条钉住。
///
/// `resolve_speakers` 的返回值里那个 `slug` 同时是三样东西 —— `dataset_44k` 的子目录、
/// `config.spk` 的键、两个 DataLoader 反查 spk id 的键 —— 所以它是这一批里改坏了后果最大的
/// 一行。两个守卫各守一种灾难,而它们**都是删一个词就没了**的形状:
///
/// * 掉了 `len(out) == 1` ⇒ 多说话人的 N 个 slug 全塌成一个常量:N 棵切片树合成一棵、
///   `config.spk` 只剩一个键、而且那些 slug 是折进 `extract_cache_fp_text` 的 blake2b 的
///   ⇒ 每一个多说话人池当场换身份并全量重跑,而且说话人映射是错的;
/// * 掉了 `identity_version(cfg) >= 2` ⇒ 每一个**存量**单说话人池的切片目录当场对不上,
///   重切一棵新树、重算全部伴生特征,旧树永远留在盘上被 `extract_all` 每轮扫一遍。
///
/// ⚠ 这是一条**字面量**判据,而且是有意的:它守的是一行代码的**语义**,而 `cargo test` 够不到
/// python 的运行时(本仓自动闸只有 `cargo test` 与 `vitest run`)。真跑一遍那一行的是行为腿。
#[test]
fn the_sole_speaker_slice_dir_is_constant_only_for_a_sole_speaker() {
    let body = top_level_fn(FLIST_PY, "resolve_speakers", "说话人解析没了");
    assert!(
        body.contains("if len(out) == 1 and identity_version(cfg) >= 2:"),
        "去名字化的两个守卫必须一字不差:少一个是「多说话人全塌成一个目录」,\
         少另一个是「每个存量单说话人池重切重抽」"
    );
    assert!(body.contains("out[0][\"slug\"] = SOLE_SPEAKER_DIR"), "固定名不再是那个常量了");
    // 守卫用的 arity 判据必须和**公式自己**用的是同一个:公式在 `len(speakers) == 1` 时根本
    // 不把 slug 折进指纹,那正是「换掉单说话人的 slug 不会重新命名任何池」的理由。
    assert!(
        helper_body().contains("if len(speakers) == 1:"),
        "公式的单说话人分支变了 —— 去名字化「不改变任何池身份」这句话就要重新论证"
    );
}

#[test]
fn every_pool_scoped_knob_is_in_the_formula_or_declared_as_not_yet() {
    let pending: BTreeSet<&str> = NOT_YET_IN_THE_FORMULA.iter().map(|(id, _)| *id).collect();
    for c in chains() {
        let backend = c.name;
        for f in resume_locked_fields(backend) {
            if !f.scope.invalidates_pool() || f.tier != LockTier::Costly {
                continue;
            }
            if pending.contains(f.id) {
                continue;
            }
            match carrier(f.id) {
                Some(CarriedBy::Fingerprint) => {}
                Some(CarriedBy::Token(k)) => assert!(
                    c.emitted.contains(k),
                    "{backend}: 锁表说 `{}` 是池级的,但这条链的 fp_text 里没有 `|{k}=` —— \
                     改它会静默复用另一套配方算出来的预处理产物",
                    f.id
                ),
                None => panic!(
                    "{backend}: 锁表新增了池级字段 `{}`,但没人说它在公式里怎么承载。\
                     要么给它一个 carrier,要么把它写进 NOT_YET_IN_THE_FORMULA 并写清理由。",
                    f.id
                ),
            }
        }
    }

    // ★ 待办清单本身:④d 把它清空了,而「空」是被钉住的 —— 往里加一项要先说清为什么一个
    //   池级字段可以不进身份。
    assert!(
        pending.is_empty(),
        "④d 已经把这张清单清空了。要重新往里加,先说清为什么一个池级字段可以不进身份:{pending:?}"
    );
    for (id, why) in NOT_YET_IN_THE_FORMULA {
        assert!(why.len() > 40, "{id}: 待办条目必须写清代价,不许只留一个编号");
    }

    // ★ rvc 的 `sampleRate` 是 `Locked` 不是 `Costly`,所以上面那条按 tier 过滤的循环**看不见
    //   它** —— 它单独钉在这里。这条断言在 ④d 落地那天从 `!contains` 翻成了 `contains`。
    //
    //   ⚠ 不能靠把 Locked 也拉进那条循环来补:`LockScope` 对「这个 family 恒定不变的项」填的是
    //   **假设性**答案(sovits 家的 44k、vocoder 的 version 都写着 `Both`/`Run` 却根本改不动),
    //   拉进去就会要求给一堆永远不会变的常量各发一个 token。
    //   ⚠ 也不许把这条写成「一段散文里含 `|sr=`」那种形状 —— 那种断言只在有人改散文时红,
    //   对被测代码零覆盖。
    let rvc = chains().into_iter().find(|c| c.name == "rvc").unwrap();
    assert!(
        rvc.emitted.contains("sr"),
        "rvc 的公式里没有 `|sr=` —— 换采样率会静默沿用另一个采样率算出来的特征"
    );
}
