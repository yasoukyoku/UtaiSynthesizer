//! What a 续训 may and may not change — ONE table, and the guard that enforces it.
//!
//! ## Why a table
//!
//! The rule lived as eight `if` blocks inside `try_start` and, in three other places, as
//! somebody's memory of them: the run-step's pre-start diff dialog, the project page's
//! form-restore, and (from S78) the parameters page, which has to render the locked fields
//! read-only. Four copies of one rule is four chances to disagree — and the failure mode is
//! the worst kind: a dialog that promises「继续训练」and a start that refuses it, or a field the
//! UI lets you edit that silently makes the slot unresumable.
//!
//! So: [`resume_locked_fields`] is the table, [`check_resume_locks`] is the only enforcement,
//! and a unit test drives ONE through the OTHER — for every `Locked` row, a request differing
//! in exactly that field must be refused with exactly that CODE; for every `Costly` row it must
//! NOT be refused. A field added to the guard without a table row (or vice versa) fails there.
//! `src/lib/resumeLockParity.test.ts` extends the same rule across the language boundary.
//!
//! ## Two tiers, because there are two different truths
//!
//! * **Locked** — the value is baked into artifacts that already exist (graph shape, wire
//!   inputs, emb_g rows, the cached ContentVec space). Changing it cannot be reconciled, so the
//!   start is refused and 重训 is the only way through.
//! * **Costly** — changing it is legitimate but re-fingerprints the dataset, so the next run
//!   redoes slicing and feature extraction. Nothing is lost and no progress is destroyed. These
//!   are NOT refused; the UI says what it will cost.
//!
//! Making the Costly ones Locked would be the easy-looking mistake: it converts today's
//! "slow but fine" into a refusal for a case users legitimately hit (adding augmentation, or
//! adding a few takes to an existing singer).

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum LockTier {
    /// Refused outright on resume — only 重训 (which wipes the slot) unlocks it.
    Locked,
    /// Allowed; invalidates the extraction caches, so the next run re-preprocesses.
    Costly,
}

/// WHICH LAYER's artifacts this field names — the second axis, and it is orthogonal to the tier.
///
/// `LockTier` answers「续训改这一项会不会被拒」. That is a question about the GUARD. It says
/// nothing about the thing ④d needs to know, which is a question about the DISK: 改了它之后,
/// **哪一层的产物作废**。The two really are independent — `version` on SoVITS is Locked *and*
/// pool-invalidating, `volEmbedding` is Locked and pool-neutral, `augCopies` is Costly and
/// pool-invalidating.
///
/// ⛔ 它回答的是「**如果这一项变了**,哪一层作废」。对某个 family 恒定不变的项(sovits 家的
/// 44k、sovits_v2 与 vocoder 的 version)这个答案是**假设性**的 —— 仍然照「变了会怎样」填,
/// 因为那才是这一列的语义。「它能不能变」是另一个问题,由 `TRAINING_SR_FIXED_44K` 那一族
/// 枚举闸回答,不在这张表里。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum LockScope {
    /// 只烧进这个 **run** 的产物:图形状、线上输入、emb_g 行、已有权重。池不受影响。
    Run,
    /// 只决定**池**里那堆预处理产物:切片、f0、特征、检索资产。图不受影响。
    Pool,
    /// 两者都。三行是这样的,而且它们是这一列最容易被填错的三行。
    Both,
}

impl LockScope {
    /// 改了这一项之后,**这个槽已有的预处理产物还能不能用**。
    ///
    /// ★§F2⒝ ④d 的池身份不变量就写在这个谓词上:**凡答 true 且这个 family 真能改的字段,
    /// 都必须出现在那条链的 `fp_text` 公式里** —— 否则改它会静默复用另一套配方算出来的特征。
    /// 今天 `augCopies`(五条链)与 rvc 的 `sampleRate` 正好是两个反例,而它们就是 ④d 要补的。
    pub fn invalidates_pool(self) -> bool {
        matches!(self, LockScope::Pool | LockScope::Both)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LockedField {
    /// Stable id shared with the frontend's rendering table (parity-tested).
    pub id: &'static str,
    pub tier: LockTier,
    /// The CODE `check_resume_locks` returns for a `Locked` field; "" for `Costly`.
    pub code: &'static str,
    /// 这一项命名的是哪一层的产物 —— 见 [`LockScope`]。
    pub scope: LockScope,
}

const fn locked(id: &'static str, code: &'static str, scope: LockScope) -> LockedField {
    LockedField { id, tier: LockTier::Locked, code, scope }
}
const fn costly(id: &'static str, scope: LockScope) -> LockedField {
    LockedField { id, tier: LockTier::Costly, code: "", scope }
}

/// The fields a resume of `backend` may not (or may not cheaply) change.
///
/// `version`/`sample_rate` are Locked everywhere: they choose the graph, the sample rate of
/// every cached slice and — for a diffusion run — the ContentVec space of the `.soft.pt` files
/// AND of the main model the result will be attached to.
pub fn resume_locked_fields(backend: &str) -> Vec<LockedField> {
    // A diffusion run inside a live sovits slot reports its own CODE: 重训(仅扩散) cannot
    // unlock the version there — the main model pins it — so the text must not suggest it.
    // (With no main model in the workspace it is an ordinary resume mismatch; the dedicated
    // test below covers both branches.)
    let ver_code = if backend == "sovits_diff" {
        "DIFF_VERSION_MISMATCH"
    } else {
        "RESUME_PARAMS_MISMATCH"
    };
    // `sampleRate` 无论哪一家都同时决定图与每一个切片的内容 ⇒ Both。
    let mut v = vec![locked("sampleRate", ver_code, LockScope::Both)];
    // `version` 的 scope 按 family 分,所以它是**两条互斥的 push** 而不是一个三目 ——
    // 跨语言对拍闸逐行走这个函数并按「这一行处在哪个条件里」判归属,一个算出来的 scope 变量
    // 它读不出来(而读不出来的表现是**少一行**,也就是两个集合恰好相等)。
    // sovits 家的版本选的是 ContentVec 空间(`|enc=` 就在 fp_text 里,`.soft.pt` 维度跟着变)
    // ⇒ Both;rvc 的 v1/v2 只是在**同一个池**里切 `3_feature256` / `3_feature768` 两个**共存**
    // 的子目录、不进 fp ⇒ Run;vocoder 的版本是常量标记(它 fp 尾巴上那个 `|vocoder-v3` 是
    // 手工版本 tag,不是它)⇒ Run。
    if matches!(backend, "sovits" | "sovits_diff" | "sovits_v2") {
        v.push(locked("version", ver_code, LockScope::Both));
    }
    if matches!(backend, "rvc" | "vocoder") {
        v.push(locked("version", ver_code, LockScope::Run));
    }
    match backend {
        // 响度嵌入 changes the generator's inputs, so it is part of the graph.
        // ⚠ 只在图里:`.vol.npy` 是**无条件**产出的伴生文件,翻这个开关不作废任何池产物。
        "sovits" => {
            v.push(locked("volEmbedding", "RESUME_VOL_EMBEDDING_MISMATCH", LockScope::Run));
        }
        _ => {}
    }
    if matches!(backend, "sovits" | "rvc" | "sovits_v2") {
        // count and ORDER both: the position IS the emb_g row.
        // ⚠ 两栖:位置是 emb_g 行(run),而说话人集合与顺序**也**是 fp_text 的输入
        // (sovits 家把 slug 与各自的数据集指纹按序折进 blake2b;rvc 按序 `|` 拼)⇒ Both。
        v.push(locked("speakerCount", "RESUME_SPEAKER_COUNT_MISMATCH", LockScope::Both));
        v.push(locked("speakerSet", "RESUME_SPEAKER_SET_MISMATCH", LockScope::Both));
    }
    if backend == "sovits_diff" {
        // pins the training distribution t ~ [0,k) and the exported sidecar contract — but only
        // once there IS diffusion progress; before that the slot is free.
        v.push(locked("kStepMax", "RESUME_KSTEP_MISMATCH", LockScope::Run));
    }
    // ── Costly ────────────────────────────────────────────────────────────────────────────
    // 折进 dataset fingerprint(它**重写每一个 wav**),扩散 run 继承而不是自己选。
    // ⛔ §F2⒝ ④d 笔 1 补上 sovits_v2:它同样送 loudnorm、同样经 `extract_cache_fp_text` 折进
    // 自己的 fp_text(`sovits_v2/pipeline.py`),所以它也是 Pool 级的 —— 两侧**一致地**漏了
    // 这一行,于是跨语言对拍永远绿、守卫也不拒(Costly 本来就不拒),从来没有可见故障。
    if matches!(backend, "sovits" | "sovits_v2") {
        v.push(costly("loudnorm", LockScope::Pool));
    }
    // Every backend. For a diffusion run it is honoured only when the sovits slot holds no main
    // model (diff-first); when it IS inherited the request field is simply ignored, and "allowed"
    // remains the truthful answer either way.
    // ⚠ scope=Pool 是**今天就成立的事实**,而 fp_text 里还没有它 —— 那正是 ④d 第 1 件。今天
    // 改它走的是 `augment_slices` 的就地增删(`idx > copies` 直接删 wav + 全部伴生),所以
    // 这一档的文案「会重新指纹化」今天仍然是**预告**而不是描述。
    v.push(costly("augCopies", LockScope::Pool));
    // Every backend: adding or removing audio re-fingerprints the shared dataset.
    v.push(costly("dataset", LockScope::Pool));
    v
}

/// Everything the guard needs to know that is not in the request.
pub struct ResumeState<'a> {
    /// The slot's `run_manifest.json`, or None when it has never run.
    pub manifest: Option<&'a serde_json::Value>,
    /// A main-model checkpoint shares this workspace (decides the diff wording).
    pub has_main: bool,
    /// Max numbered diffusion checkpoint; 0/None = no diffusion progress yet.
    pub max_diffusion_step: Option<u64>,
    /// The slot's frozen `(slug, name)` pairs — see `training::frozen_speakers`.
    pub frozen_speakers: &'a [super::dsmanifest::DsSpeaker],
}

/// THE resume guard. Returns the CODE (with its payload) to refuse with, or None to allow.
///
/// `enforce` is `!req.fresh || diff_partial_wipe`: a 重训 trains into a run where nothing is baked
/// in yet — but the diffusion partial wipe KEEPS the manifest, so a mismatched version could never
/// train afterwards; deleting first would destroy hours of diffusion progress and only THEN refuse.
///
/// ⚠★§F2⒝ ④e — the CONCLUSION is unchanged and the REASON is not. Until the flip this said 「a
/// full 重训 wipes the slot, so nothing is baked in any more」, and that sentence died the moment
/// 「重训」 stopped erasing anything. What makes a fresh start unguardable now is that it MINTS a
/// new run (`trun::run_dir_for_start` with `mint`), so the fields are being chosen for a directory
/// that is empty by construction rather than for one that was just emptied.
/// ⛔ Worth spelling out because the guard would have stayed GREEN either way: a criterion whose
/// stated reason is dead is the shape that survives a refactor and then licenses the wrong edit.
pub fn check_resume_locks(
    req: &super::StartTrainingRequest,
    st: &ResumeState<'_>,
    enforce: bool,
) -> Option<String> {
    if !enforce {
        return None;
    }
    let old = st.manifest?;
    let old_ver = old["version"].as_str().unwrap_or("");
    let old_sr = old["sample_rate"].as_str().unwrap_or("");
    // An ABSENT key fails open on purpose: a pre-S37 workspace records neither, and demanding a
    // match would refuse every one of them forever.
    if (!old_ver.is_empty() && old_ver != req.version)
        || (!old_sr.is_empty() && old_sr != req.sample_rate)
    {
        return Some(if req.backend == "sovits_diff" && st.has_main {
            // 重训(仅扩散) cannot unlock the version — it is pinned by the main model, so
            // don't suggest it.
            format!(
                "DIFF_VERSION_MISMATCH: {}/{} -> {}/{}",
                old_ver, old_sr, req.version, req.sample_rate
            )
        } else {
            format!(
                "RESUME_PARAMS_MISMATCH: {}/{} -> {}/{}",
                old_ver, old_sr, req.version, req.sample_rate
            )
        });
    }
    if req.backend == "sovits" {
        if let Some(old_vol) = old["vol_embedding"].as_bool() {
            if old_vol != req.vol_embedding {
                return Some(format!(
                    "RESUME_VOL_EMBEDDING_MISMATCH: {} -> {}",
                    if old_vol { "on" } else { "off" },
                    if req.vol_embedding { "on" } else { "off" }
                ));
            }
        }
    }
    // ①c: n_speakers + the ordered speaker set are baked into the emb_g rows — resuming with a
    // different count / order / set would silently mis-assign every speaker's timbre. (Old
    // single-speaker manifests have no n_speakers key -> 1, which matches a single-speaker
    // resume = no false rejection.)
    if matches!(req.backend.as_str(), "sovits" | "rvc" | "sovits_v2") {
        let old_n = old["n_speakers"].as_u64().unwrap_or(1);
        let cur_n = if req.speakers.len() > 1 { req.speakers.len() as u64 } else { 1 };
        if old_n != cur_n {
            return Some(format!("RESUME_SPEAKER_COUNT_MISMATCH: {} -> {}", old_n, cur_n));
        }
        if cur_n > 1 {
            // Compare DISPLAY NAMES by position, not recomputed slugs: the slug is a
            // `DefaultHasher` derivative, so judging identity by it means a toolchain bump
            // reports every existing co-trained project as "different speakers".
            let named = st.frozen_speakers.len() == cur_n as usize
                && st.frozen_speakers.iter().all(|s| !s.name.is_empty());
            let same = if named {
                st.frozen_speakers
                    .iter()
                    .zip(req.speakers.iter())
                    .all(|(f, s)| f.name == s.name)
            } else {
                // No name survives anywhere (pre-`speaker_names` AND a diff run rewrote run.json
                // without the key). Fall back to the slug comparison this guard always did — the
                // same toolchain exposure, but it is the only identity left.
                let old_slugs: Vec<String> = old["speakers"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                old_slugs
                    == super::assign_speaker_slugs(&req.speakers)
                        .into_iter()
                        .map(|(_, s)| s)
                        .collect::<Vec<_>>()
            };
            if !same {
                return Some("RESUME_SPEAKER_SET_MISMATCH".into());
            }
        }
    }
    // k_step_max pins the diffusion TRAINING distribution and the exported sidecar contract.
    // The fresh partial-wipe path resets the progress, so it may change there.
    if req.backend == "sovits_diff" && !req.fresh {
        if let (Some(old_k), Some(max_step)) = (old["diff_k_step_max"].as_u64(), st.max_diffusion_step)
        {
            if max_step > 0 && old_k != req.k_step_max as u64 {
                let show = |k: u64| if k == 0 { "full-diffusion".to_string() } else { k.to_string() };
                return Some(format!(
                    "RESUME_KSTEP_MISMATCH: {} -> {}",
                    show(old_k),
                    show(req.k_step_max as u64)
                ));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::{dsmanifest::DsSpeaker, StartTrainingRequest};

    fn req(json: serde_json::Value) -> StartTrainingRequest {
        serde_json::from_value(json).expect("request fixture")
    }

    fn base(backend: &str) -> serde_json::Value {
        serde_json::json!({
            "model_name": "m", "backend": backend,
            "version": if backend == "rvc" { "v2" } else { "4.1" },
            "sample_rate": if backend == "rvc" { "40k" } else { "44k" },
            // must MATCH `manifest()` — an "unchanged" fixture that is not actually
            // unchanged makes every assertion below vacuous (caught by the guard on first run)
            "k_step_max": 100,
            "dataset_files": [], "total_epoch": 1, "batch_size": 1,
        })
    }

    fn manifest(backend: &str) -> serde_json::Value {
        serde_json::json!({
            "backend": backend,
            "version": if backend == "rvc" { "v2" } else { "4.1" },
            "sample_rate": if backend == "rvc" { "40k" } else { "44k" },
            "vol_embedding": false,
            "loudnorm": false,
            "aug_copies": 0,
            "diff_k_step_max": 100,
        })
    }

    fn state<'a>(m: &'a serde_json::Value, frozen: &'a [DsSpeaker]) -> ResumeState<'a> {
        ResumeState {
            manifest: Some(m),
            has_main: true,
            max_diffusion_step: Some(500),
            frozen_speakers: frozen,
        }
    }

    /// 造一个「**只有这一项**与 manifest 不同」的请求。`None` = 这一项不是请求字段
    /// (`dataset` 由指纹决定),调用方跳过。
    ///
    /// ⛔ 抽出来是为了让**两个方向**共用同一个突变器:表里**有**的行要按档位表现,表里
    /// **没有**的行必须被放行。两个方向若各写各的突变,它们比较的就不是同一件事,而
    /// 「表里没有的行也被拒」这条路今天完全没有判据(把一行从表里删掉却忘了改守卫 = 全绿)。
    fn only_this_differs(
        backend: &str,
        id: &str,
    ) -> Option<(serde_json::Value, serde_json::Value, bool)> {
        let two = |a: &str, b: &str| {
            serde_json::json!([{"name": a, "files": []}, {"name": b, "files": []}])
        };
        let mut j = base(backend);
        let mut mf = manifest(backend);
        let mut use_frozen = false;
        match id {
            "version" => j["version"] = serde_json::json!("v1"),
            "sampleRate" => j["sample_rate"] = serde_json::json!("32k"),
            "volEmbedding" => j["vol_embedding"] = serde_json::json!(true),
            "speakerCount" => j["speakers"] = two("A", "B"),
            "speakerSet" => {
                // same COUNT as the manifest, different order
                mf["n_speakers"] = serde_json::json!(2);
                mf["speakers"] = serde_json::json!(["a_1", "b_2"]);
                j["speakers"] = two("B", "A");
                use_frozen = true;
            }
            "kStepMax" => j["k_step_max"] = serde_json::json!(200),
            "loudnorm" => j["loudnorm"] = serde_json::json!(true),
            "augCopies" => j["aug_copies"] = serde_json::json!(3),
            "dataset" => return None, // not a request field — the fingerprint decides
            other => panic!("table row `{other}` has no test case"),
        }
        Some((j, mf, use_frozen))
    }

    /// ★ THE anti-drift test: drive the TABLE through the GUARD, **in both directions**.
    ///
    /// Every `Locked` row must actually refuse — with its own CODE — when only that field
    /// differs; every `Costly` row must actually be allowed. A guard added without a row, or a
    /// row without a guard, fails here rather than in a user's half-trained slot.
    ///
    /// ★§F2⒝ ④d 笔 1 补上**反方向**:表里**没有**这一行的 family,守卫必须**放行**它。
    /// 此前只测了「锁着的真拒」,所以「把 rvc 的 sampleRate 从表里删掉而忘了改守卫」是全绿的 ——
    /// 而那恰好是 ④d 第 5 件要做的动作。放行方向失守的形态与锁定方向相反:UI 让人改,
    /// 到开始训练才被拒,正是这个模块存在要消灭的那种矛盾。
    #[test]
    fn every_locked_field_refuses_and_every_costly_field_does_not() {
        use std::collections::BTreeSet;
        const BACKENDS: [&str; 5] = ["rvc", "sovits", "sovits_v2", "sovits_diff", "vocoder"];
        // 全集从表本身算出来 ⇒ 加一行新字段自动进入两个方向,不必手抄第二份清单。
        // ⚠ 这个取法有一条**实测过的上限**:一个字段若从**每一个** family 都被删掉,它同时
        // 离开全集 ⇒ 反方向对它零覆盖(变异 L5 实测:删掉 speakerCount 那一行,红的是 scope
        // 那条断言而不是这里)。守住那种全域删除的是跨语言对拍与 scope 锚点,不是这一段。
        // 这一段守的是**按 family 的**差异 —— 也就是 ④d 第 5 件「只放开 rvc 的 sampleRate」
        // 那个形状(变异 L8 实测在这里红:`rvc/version` 表里没了而守卫还在拒)。
        let universe: BTreeSet<&str> =
            BACKENDS.iter().flat_map(|b| resume_locked_fields(b)).map(|f| f.id).collect();

        for backend in BACKENDS {
            let m = manifest(backend);
            let frozen = vec![
                DsSpeaker { slug: "a_1".into(), name: "A".into() },
                DsSpeaker { slug: "b_2".into(), name: "B".into() },
            ];
            // the unchanged request must ALWAYS pass, or every assertion below is vacuous
            let ok = req(base(backend));
            assert_eq!(
                check_resume_locks(&ok, &state(&m, &[]), true),
                None,
                "{backend}: an unchanged resume must be allowed"
            );

            let mine = resume_locked_fields(backend);
            for f in &mine {
                let Some((j, mf, use_frozen)) = only_this_differs(backend, f.id) else { continue };
                let frz: &[DsSpeaker] = if use_frozen { &frozen } else { &[] };
                let got = check_resume_locks(&req(j), &state(&mf, frz), true);
                match f.tier {
                    LockTier::Locked => {
                        let msg = got.unwrap_or_else(|| {
                            panic!("{backend}/{}: table says Locked but the guard allowed it", f.id)
                        });
                        assert!(
                            msg.starts_with(f.code),
                            "{backend}/{}: expected {}, got {msg}",
                            f.id,
                            f.code
                        );
                    }
                    LockTier::Costly => assert_eq!(
                        got, None,
                        "{backend}/{}: table says Costly but the guard refused it",
                        f.id
                    ),
                }
            }

            // ── 反方向 ────────────────────────────────────────────────────────────────
            let ours: BTreeSet<&str> = mine.iter().map(|f| f.id).collect();
            let absent: Vec<&&str> = universe.difference(&ours).collect();
            // 存活证明:这个循环必须真的有东西可跑。每个 family 的表都少几行别人有的
            // (rvc 没有 volEmbedding/kStepMax/loudnorm,vocoder 连说话人那两行都没有)。
            assert!(!absent.is_empty(), "{backend}: 补集是空的,反方向这一段什么也没测");
            for id in absent {
                let Some((j, mf, use_frozen)) = only_this_differs(backend, id) else { continue };
                let frz: &[DsSpeaker] = if use_frozen { &frozen } else { &[] };
                assert_eq!(
                    check_resume_locks(&req(j), &state(&mf, frz), true),
                    None,
                    "{backend}/{id}: 表里没有这一行,守卫却拒了它 —— UI 会让人改,\
                     然后在「开始训练」那一刻拒绝,而用户什么都没做错"
                );
            }
        }
    }

    /// `scope` 这一列的锚点。
    ///
    /// ⛔ 它不能只被「有没有这一列」验:一张**全填 `Run`** 的表会让每一个消费者
    /// (costly 提示、④d 的池身份不变量)静静地什么都不做。所以钉的是**这一列真的分得开**:
    /// 同一个档位下两种 scope 都存在,而且几个最容易填错的格子逐个写死。
    #[test]
    fn the_scope_column_distinguishes_the_two_layers() {
        let scope_of = |backend: &str, id: &str| {
            resume_locked_fields(backend).into_iter().find(|f| f.id == id).map(|f| f.scope)
        };

        // Locked 这一档里两种 scope 都有 —— 否则 scope 只是 tier 的别名。
        assert_eq!(scope_of("sovits", "version"), Some(LockScope::Both), "|enc= 就在 fp_text 里");
        assert_eq!(scope_of("rvc", "version"), Some(LockScope::Run), "v1/v2 是池内两个共存子目录");
        assert_eq!(
            scope_of("sovits", "volEmbedding"),
            Some(LockScope::Run),
            ".vol.npy 是无条件产出的,翻这个开关不作废任何池产物"
        );
        // Costly 这一档今天全是 Pool —— 那正是「允许改」与「会重跑预处理」的交集。
        assert_eq!(scope_of("sovits", "loudnorm"), Some(LockScope::Pool));
        assert_eq!(scope_of("sovits_v2", "loudnorm"), Some(LockScope::Pool), "④d 笔 1 补的那一行");
        assert_eq!(scope_of("rvc", "loudnorm"), None, "rvc 不送这个字段");
        assert_eq!(scope_of("vocoder", "augCopies"), Some(LockScope::Pool));

        // ⚠ 第一版这里写的是「每个 family 既有池级项也有 run 级项」—— **那是我自己编的性质,
        // 当场就红了**:`sovits_v2` 的每一行都影响池(它没有 volEmbedding、没有 kStepMax,
        // 剩下的 version/sampleRate/说话人/loudnorm/augCopies/dataset 全是池级)。那是事实不是
        // 缺陷,而且它顺带解释了为什么 costly 提示对 v2 最有用:loudnorm 是它唯一能改的
        // 重预处理开关。真正该钉的是**这一列没有被塌成一个常量**:
        let all: Vec<LockScope> = ["rvc", "sovits", "sovits_v2", "sovits_diff", "vocoder"]
            .iter()
            .flat_map(|b| resume_locked_fields(b))
            .map(|f| f.scope)
            .collect();
        for want in [LockScope::Run, LockScope::Pool, LockScope::Both] {
            assert!(all.contains(&want), "{want:?} 一个都没有 ⇒ 这一列被塌成常量了");
        }

        // ★ `version` 是**两条互斥的 push**(见函数体),所以「两条都没命中」是一个新 family
        // 会静默掉进去的洞:那个 family 从此没有版本锁,而守卫照拒。
        for backend in ["rvc", "sovits", "sovits_v2", "sovits_diff", "vocoder"] {
            let n = resume_locked_fields(backend).iter().filter(|f| f.id == "version").count();
            assert_eq!(n, 1, "{backend}: version 这一行必须**恰好**一条");
        }

        // ★ 池级集合逐个写死两个 family。④d 第 3 步(R3)要拿的就是这个集合去比五条链的
        // fp_text —— 集合悄悄少一项,那道闸就会漏掉一条链而照样绿。
        let pool_ids = |backend: &str| {
            let mut v: Vec<&str> = resume_locked_fields(backend)
                .into_iter()
                .filter(|f| f.scope.invalidates_pool())
                .map(|f| f.id)
                .collect();
            v.sort_unstable();
            v
        };
        assert_eq!(
            pool_ids("sovits"),
            ["augCopies", "dataset", "loudnorm", "sampleRate", "speakerCount", "speakerSet", "version"]
        );
        assert_eq!(
            pool_ids("vocoder"),
            ["augCopies", "dataset", "sampleRate"],
            "vocoder 的 version 是常量标记,它的 fp 尾巴 `|vocoder-v3` 是手工版本 tag、不是它"
        );
        assert_eq!(
            pool_ids("rvc"),
            ["augCopies", "dataset", "sampleRate", "speakerCount", "speakerSet"],
            "rvc 的 version 是池内两个共存的特征子目录,不进身份"
        );
    }

    /// 重训 unlocks everything — that is the ONLY way out of a Locked field, so it must work.
    ///
    /// ⚠★§F2⒝ ④e — this test passes `false` as a LITERAL, so it pins the function's contract and
    /// not the call site. That means the flip could not make it red, and it did not: what changed
    /// is the reason (see `check_resume_locks`'s doc — 「the slot was wiped」 became 「the run is
    /// newly minted and therefore empty」). The call site itself (`training::try_start`) has no
    /// unit driver at all; the source-order ratchet
    /// `training::tests::a_start_re_resolves_its_run_after_the_wipe` is what pins it instead.
    #[test]
    fn a_fresh_run_is_never_guarded() {
        let m = manifest("sovits");
        let mut j = base("sovits");
        j["version"] = serde_json::json!("4.0");
        j["fresh"] = serde_json::json!(true);
        assert_eq!(check_resume_locks(&req(j), &state(&m, &[]), false), None);
    }

    /// A manifest that predates a field must not be read as「不匹配」— every one of those
    /// workspaces would become unresumable forever.
    #[test]
    fn absent_manifest_keys_fail_open() {
        let bare = serde_json::json!({ "backend": "sovits" });
        let j = base("sovits");
        assert_eq!(check_resume_locks(&req(j), &state(&bare, &[]), true), None);
        // …and so does a slot that never ran at all
        let none = ResumeState {
            manifest: None,
            has_main: false,
            max_diffusion_step: None,
            frozen_speakers: &[],
        };
        assert_eq!(check_resume_locks(&req(base("rvc")), &none, true), None);
    }

    /// Diffusion depth is only pinned once there IS progress to contradict.
    #[test]
    fn k_step_is_free_until_the_diffusion_has_run() {
        let m = manifest("sovits_diff");
        let mut j = base("sovits_diff");
        j["k_step_max"] = serde_json::json!(200);
        let fresh_slot = ResumeState {
            manifest: Some(&m),
            has_main: true,
            max_diffusion_step: Some(0),
            frozen_speakers: &[],
        };
        assert_eq!(check_resume_locks(&req(j.clone()), &fresh_slot, true), None);
        assert!(check_resume_locks(&req(j), &state(&m, &[]), true).is_some());
    }

    /// The diff run's version refusal names a different CODE, because 重训(仅扩散) cannot
    /// unlock it — the main model pins it.
    #[test]
    fn a_diffusion_version_mismatch_says_so_specifically() {
        let m = manifest("sovits_diff");
        let mut j = base("sovits_diff");
        j["version"] = serde_json::json!("4.0");
        let msg = check_resume_locks(&req(j.clone()), &state(&m, &[]), true).unwrap();
        assert!(msg.starts_with("DIFF_VERSION_MISMATCH"), "{msg}");
        // without a main model in the workspace it is an ordinary resume mismatch
        let no_main = ResumeState {
            manifest: Some(&m),
            has_main: false,
            max_diffusion_step: Some(500),
            frozen_speakers: &[],
        };
        let msg = check_resume_locks(&req(j), &no_main, true).unwrap();
        assert!(msg.starts_with("RESUME_PARAMS_MISMATCH"), "{msg}");
    }
}
