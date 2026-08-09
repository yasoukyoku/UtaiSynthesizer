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
//! ⇒ 一次跨五条链的编辑,「改了 3 条漏了 2 条」在今天**没有任何东西会红**:
//! `converter/verify/training/gate_pool_table.py` 的七组检查没有一组读公式的文本,而且它本身
//! 不在任何自动闸里(本仓无 CI、无 git hook,自动闸只有 `cargo test` 与 `vitest run`)。
//!
//! ## 它为什么长这个样子
//!
//! ⛔ **判据是【值】不是【顺序】**。「追加语句排在 `open_pool` 之前」这类顺序断言会被邻居行
//! 满足,而且 sovits_v2 已经有一段条件追加(`|f0=`)—— 新 token 加在它之前还是之后是**两个
//! 不同的字符串**,顺序判据对这个差别一无所知。所以这里比的是每条链**实际发射的 token 集合**。
//!
//! ⛔ **它必须能抓「漏了一条链」而不只是「改了一条链」**。前者靠 [`NOT_YET_IN_THE_FORMULA`]:
//! 那不是白名单,是 ④d 第 1 件的**待办清单**,而且一条条目管**全部五条链** —— 想把它删掉,
//! 五条链就必须**同时**带上那个 token,少一条当场红。
//!
//! ⛔ **它自己必须证明活着**。一个读空的解析器会让两个空集恰好相等。所以:每条链的 token 集合
//! 逐条写死对拍(不是「非空」这种下限),外加一条合成坏源码的阴性对照。
//!
//! `include_str!` 是有意的:文件被挪走 = **编译失败**,而不是解析出一个空串。

use std::collections::BTreeSet;

use utai_lib::training::resume_lock::{resume_locked_fields, LockTier};

const SOVITS_PY: &str = include_str!("../../training/utai_train/sovits/pipeline.py");
const DIFF_PY: &str = include_str!("../../training/utai_train/sovits/diff_pipeline.py");
const V2_PY: &str = include_str!("../../training/utai_train/sovits_v2/pipeline.py");
const RVC_PY: &str = include_str!("../../training/utai_train/rvc/pipeline.py");
const VOC_PY: &str = include_str!("../../training/utai_train/vocoder/pipeline.py");

/// 共享 helper 的函数体 —— sovits 家三条链的公式主体都在这里。
fn helper_body() -> &'static str {
    let at = SOVITS_PY
        .find("def extract_cache_fp_text(")
        .expect("共享的池身份 helper 没了 —— 三条 sovits 链的公式从此各写各的");
    let rest = &SOVITS_PY[at..];
    // 到下一个顶格 `def ` 为止(python 顶层函数)
    let end = rest[1..].find("\ndef ").map(|i| i + 1).unwrap_or(rest.len());
    &rest[..end]
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

/// 这段源码里的字符串字面量会发射出哪些 `|key=` token。
///
/// 只看**字面量**:`"%s|enc=%s|loudnorm=%d"` 里的 `enc`/`loudnorm` 是真 token,而注释里写的
/// `|aug=` 不是。少了这一条过滤,一句「TODO: 加 |aug=」的注释就能让这道闸提前变绿。
fn tokens_of(src: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let ch: Vec<char> = src.chars().collect();
    let (mut i, mut in_doc, mut in_str, mut comment) = (0usize, false, false, false);
    let triple = |ch: &Vec<char>, i: usize| {
        ch.get(i) == Some(&'"') && ch.get(i + 1) == Some(&'"') && ch.get(i + 2) == Some(&'"')
    };
    let mut lit = String::new();
    while i < ch.len() {
        let c = ch[i];
        // ⛔ docstring 必须整段跳过,而**不是**当成一个字符串字面量。这五个文件的函数头
        // 全是 `"""…"""`,而 python 里没有块注释 —— 一句「格式是 `%s|enc=%s|aug=%d`」的文档
        // 会被当成真公式采进来,于是这道闸对着一段散文变绿。
        if in_doc {
            if triple(&ch, i) {
                in_doc = false;
                i += 3;
                continue;
            }
            i += 1;
            continue;
        }
        if c == '\n' {
            comment = false;
            in_str = false;
            lit.clear();
        } else if comment {
            // 行注释:整行剩下的部分都不算
        } else if !in_str && triple(&ch, i) {
            in_doc = true;
            i += 3;
            continue;
        } else if !in_str && c == '#' {
            comment = true;
        } else if c == '"' {
            if in_str {
                harvest(&lit, &mut out);
                lit.clear();
            }
            in_str = !in_str;
        } else if in_str {
            lit.push(c);
        }
        i += 1;
    }
    out
}

fn harvest(lit: &str, out: &mut BTreeSet<String>) {
    for piece in lit.split('|').skip(1) {
        if let Some(eq) = piece.find('=') {
            let key = &piece[..eq];
            if !key.is_empty() && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                out.insert(key.to_string());
            }
        }
    }
}

/// 每条链 + 它的公式源码。sovits 家的三条都要带上共享 helper。
fn chains() -> Vec<(&'static str, String)> {
    let h = helper_body();
    vec![
        ("sovits", format!("{h}\n{}", own_region(SOVITS_PY, "sovits"))),
        ("sovits_diff", format!("{h}\n{}", own_region(DIFF_PY, "sovits_diff"))),
        ("sovits_v2", format!("{h}\n{}", own_region(V2_PY, "sovits_v2"))),
        ("rvc", own_region(RVC_PY, "rvc").to_string()),
        ("vocoder", own_region(VOC_PY, "vocoder").to_string()),
    ]
}

/// 今天每条链**实际**发射的 token,逐条写死。
///
/// ⛔ 不是「非空」这种下限:一个读空的解析器会让每一条都是空集,而空集与空集相等。
const CHAIN_TOKENS: &[(&str, &[&str])] = &[
    ("sovits", &["enc", "loudnorm"]),
    ("sovits_diff", &["enc", "loudnorm"]),
    // v2 在 helper 之后条件追加 `|f0=<method>`(非 rmvpe 时,只有 gate 会传 dio)
    ("sovits_v2", &["enc", "f0", "loudnorm"]),
    // rvc 是**裸的**数据集指纹(多说话人是一串无 `=` 的 hash 用 `|` 拼)
    ("rvc", &[]),
    // vocoder 的尾巴 `|vocoder-v3` 是手工版本 tag,不是 key=value
    ("vocoder", &[]),
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
        _ => None,
    }
}

/// ⛔ **不是白名单,是 ④d 第 1 件的待办清单。**
///
/// 每一条都是「这个池级字段今天**不在**公式里,所以改它会静默复用另一套配方算出来的产物」。
/// 清空这张表 = ④d 第 1 件做完了;而一条条目管**全部五条链** —— 想删掉 `augCopies` 这一条,
/// 五条链就必须**同时**带上 `|aug=`,少改一条当场红。这就是「改 3 漏 2 必须红」的落点。
const NOT_YET_IN_THE_FORMULA: &[(&str, &str)] = &[(
    "augCopies",
    "④d 第 1 件:五条链都要追加 `|aug=<n>`(仅 n>0)。今天改它走的是 `augment_slices` 的\
     就地增删(`idx > copies` 直接删 wav + 全部伴生),所以锁表那句「会重新指纹化」还是预告。",
)];

/// rvc 的 `sampleRate` 是 `Locked` 不是 `Costly`,所以上面那条按 tier 过滤的循环看不到它 ——
/// 它单独列在这里,理由同样是 ④d 第 1 件。
const RVC_SAMPLE_RATE_NOT_YET: &str =
    "④d 第 1 件:rvc 的 `|sr=<x>`。`1_16k_wavs` 的内容依赖目标采样率,而 f0/特征按名 skip \
     ⇒ 换 sr 会静默沿用另一个采样率算出来的特征。";

#[test]
fn every_chain_emits_the_pool_identity_tokens_it_declares() {
    // ⑴ 存活证明:解析器真的读到了东西,而且每条链**逐条**相等。
    for (name, src) in chains() {
        let want: BTreeSet<String> = CHAIN_TOKENS
            .iter()
            .find(|(n, _)| *n == name)
            .unwrap_or_else(|| panic!("{name} 没有声明它的 token 集合"))
            .1
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            tokens_of(&src),
            want,
            "{name}: 公式发射的 token 与声明不一致 —— 改公式必须同时改这张表"
        );
    }
    assert_eq!(CHAIN_TOKENS.len(), 5, "五条链,一条都不许从这张表上消失");

    // ⑵ 阴性对照:解析器必须真的会说「不一样」,而且不许把注释读成 token。
    assert_eq!(tokens_of(r#"x = "%s|enc=%s|aug=%d""#), ["aug", "enc"].map(String::from).into());
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
    assert!(tokens_of(r#"x = "%s|vocoder-v3""#).is_empty(), "没有 `=` 的尾巴不是 key=value");
}

#[test]
fn every_pool_scoped_knob_is_in_the_formula_or_declared_as_not_yet() {
    let pending: BTreeSet<&str> = NOT_YET_IN_THE_FORMULA.iter().map(|(id, _)| *id).collect();
    for (name, src) in chains() {
        let emitted = tokens_of(&src);
        let backend = if name == "sovits_diff" { "sovits_diff" } else { name };
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
                    emitted.contains(k),
                    "{name}: 锁表说 `{}` 是池级的,但这条链的 fp_text 里没有 `|{k}=` —— \
                     改它会静默复用另一套配方算出来的预处理产物",
                    f.id
                ),
                None => panic!(
                    "{name}: 锁表新增了池级字段 `{}`,但没人说它在公式里怎么承载。\
                     要么给它一个 carrier,要么把它写进 NOT_YET_IN_THE_FORMULA 并写清理由。",
                    f.id
                ),
            }
        }
    }

    // ★ 待办清单本身:它是 ④d 第 1 件的进度条,不许悄悄变长,也不许没有理由。
    assert_eq!(
        pending.iter().copied().collect::<Vec<_>>(),
        ["augCopies"],
        "这张表只能因为 ④d 第 1 件而变短。要往里加一项,先说清为什么一个池级字段可以不进身份"
    );
    for (id, why) in NOT_YET_IN_THE_FORMULA {
        assert!(why.len() > 40, "{id}: 待办条目必须写清代价,不许只留一个编号");
    }
    assert!(RVC_SAMPLE_RATE_NOT_YET.contains("|sr="), "rvc 那一条的落点必须写明");
    // …而它今天确实还不在 rvc 的公式里。这条断言在 ④d 第 1 件落地当天**必须翻面**。
    let rvc = chains().into_iter().find(|(n, _)| *n == "rvc").unwrap().1;
    assert!(
        !tokens_of(&rvc).contains("sr"),
        "rvc 已经有 `|sr=` 了 ⇒ 把 RVC_SAMPLE_RATE_NOT_YET 删掉,并把这条断言翻成 contains"
    );
}
