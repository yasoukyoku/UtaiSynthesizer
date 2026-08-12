//! S141 —— 「再训一个」**在一个真实形状的未迁移槽上**到底会不会铸出第二个 run。
//!
//! ## 为什么这条腿存在
//!
//! 实机第一次开窗口(S141)之后,用户看到的是「per-run 那一段 UI 没有出来」。查下来 UI 没错:
//! 那个槽**没有 `runs/` 容器**,只有一条 `id === ""` 的行,而删除按钮**按设计**只在 `id` 非空时
//! 才画。真正的原因是**从来没有铸出过第二个 run**(那一次走的是续训)。
//!
//! 而「铸新 run」这条链此前**只有源码闸**在守(`try_start` 吃 `&self` 与一大堆状态,仓内驱不动):
//! 折叠(`migrate_one_slot`)与铸造(`run_dir_for_start`)本身是两个普通的 `pub fn`,
//! 把它们按 `try_start` 的**同一个顺序**驱一遍,就是这条链行为那一半能拿到的最真的判据。
//!
//! ⛔ **诚实边界**:它驱的是那两步,不是 `try_start` 整个函数 —— 「try_start 真的按这个顺序调
//! 它们」由 `training::tests::minting_a_new_run_never_wipes_the_slot_the_old_runs_live_in` 那道
//! 源码闸守着。两条一起才是完整的一圈;单独任何一条都不够。
//!
//! ## 夹具照的是**盘上真实的形状**,不是我想象的形状
//!
//! 取自 S141 实机那个槽:`run.json`(带 `model_name` + 冻结的 `model_slug`)· `run_manifest.json` ·
//! `weights/<slug>_*.pth` · `audition/<slug>_*/model.json`(里面那份**没人会重测**的音域)·
//! `pools/<id>/` · 槽根的 `G_*.pth`/`D_*.pth`/`train.log`/`events.out.tfevents.*`。
//! S133 记过一条同族的血训:**夹具造了一个现实里不存在的形状 ⇒ 测试为一个错的理由红。**

use std::path::{Path, PathBuf};

use utai_lib::training::{migrate_one_slot, trun};

fn w(p: &Path, body: &str) {
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(p, body).unwrap();
}

/// 一个**真的能被读出采样率**的 PCM wav 头。
///
/// ⛔ 第一版的池只有 `dataset.fingerprint`、没有切片,折叠当场拒绝:
/// `POOL_SAMPLE_RATE_UNKNOWN: has an identity but no rvc slices to read the rate from`。
/// 那是 S132 笔 2 买回来的守卫**正确工作**,而我的夹具造了一个**现实里不存在的形状** ——
/// 正是 S133 §4-3 记的那条(夹具造了不存在的形状 ⇒ 测试为一个错的理由红)。真池里
/// `0_gt_wavs/` 是有 wav 的,采样率就从它的 RIFF 头读。
fn wav_48k(samples: usize) -> Vec<u8> {
    let data = samples * 2;
    let mut b = Vec::with_capacity(44 + data);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&((36 + data) as u32).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    b.extend_from_slice(&1u16.to_le_bytes()); // PCM
    b.extend_from_slice(&1u16.to_le_bytes()); // mono
    b.extend_from_slice(&48_000u32.to_le_bytes());
    b.extend_from_slice(&96_000u32.to_le_bytes()); // byte rate
    b.extend_from_slice(&2u16.to_le_bytes()); // block align
    b.extend_from_slice(&16u16.to_le_bytes()); // bits
    b.extend_from_slice(b"data");
    b.extend_from_slice(&(data as u32).to_le_bytes());
    b.resize(44 + data, 0);
    b
}

/// 逐文件 (相对路径, sha256) —— 「一个字节都没动」只能这样说。
fn fingerprint(root: &Path) -> Vec<(String, String)> {
    fn walk(dir: &Path, base: &Path, out: &mut Vec<(String, String)>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, base, out);
            } else if let Ok(bytes) = std::fs::read(&p) {
                use sha2::Digest;
                let mut h = sha2::Sha256::new();
                h.update(&bytes);
                out.push((
                    p.strip_prefix(base).unwrap().to_string_lossy().replace('\\', "/"),
                    format!("{:x}", h.finalize()),
                ));
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, &mut out);
    out.sort();
    out
}

#[test]
fn retraining_a_flat_slot_folds_it_and_mints_a_second_run_beside_the_first() {
    let base = std::env::temp_dir().join(format!("utai_retrain_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let data = base.join("data");
    let project = "test_9f86d081";
    let family = "rvc";
    let slot: PathBuf = data.join("training").join(project).join(family);

    // ── 未迁移的槽,照实机那个的形状铺 ────────────────────────────────────────────
    const SLUG: &str = "test-rvc_ea3c92d9";
    w(&slot.join("run.json"), r#"{"model_name":"test-rvc","model_slug":"test-rvc_ea3c92d9"}"#);
    w(&slot.join("run_manifest.json"), r#"{"backend":"rvc","version":"v2","sample_rate":"48k","aug_copies":2}"#);
    w(&slot.join("weights").join(format!("{SLUG}_best.pth")), "WEIGHTS-BEST");
    w(&slot.join("weights").join(format!("{SLUG}_e2_s29.pth")), "WEIGHTS-E2");
    // ⛔ 音域那份是承重的:它是**实测出来的**,没有任何东西会重测它。丢了就永远丢了。
    w(
        &slot.join("audition").join(format!("{SLUG}_best")).join("model.json"),
        r#"{"vocal_range":{"low":45,"high":80}}"#,
    );
    w(&slot.join("G_2333333.pth"), "G");
    w(&slot.join("D_2333333.pth"), "D");
    w(&slot.join("train.log"), "log");
    w(&slot.join("events.out.tfevents.1786545330.host.1.0"), "tb");
    // 池:照真池的形状(id 与目录名都取自实机那个槽),而且 `0_gt_wavs` 里要有**真的能读出
    // 采样率**的 wav —— 否则折叠会正确地拒绝,而这条腿就为一个错的理由红。
    let pool = slot.join("pools").join("p5a9f31b9ceb8");
    w(&pool.join("dataset.fingerprint"), "abc|sr=48000|aug=2");
    std::fs::create_dir_all(pool.join("0_gt_wavs")).unwrap();
    std::fs::write(pool.join("0_gt_wavs").join("0_0.wav"), wav_48k(480)).unwrap();
    std::fs::create_dir_all(pool.join("3_feature768")).unwrap();
    std::fs::write(pool.join("3_feature768").join("0_0.npy"), b"feat").unwrap();
    assert!(!trun::runs_root(&slot).exists(), "前置:这个槽还没有 runs/ 容器");

    let before = fingerprint(&slot);
    assert!(before.len() >= 9, "夹具太薄,证明不了「一个字节都没动」:{}", before.len());

    // ── try_start 在铸新时做的两步,同一个顺序 ────────────────────────────────────
    // ⑴ 先折叠。⛔ 顺序是硬的:`tpool::slot_facts` 拒绝一个持有两份 run_manifest 的槽,而 3→4
    //    的迁移只在开机跑 —— 一个在 layout 3 上长出第二个 run 的槽**永远折不动了**。
    migrate_one_slot(&data, project, family).expect("按需折叠这个槽");
    // ⑵ 再铸。`mint = true` ⇒ 无视 requested,一定是一个新目录。
    let minted = trun::run_dir_for_start(&slot, family, None, true).expect("铸一个新 run");

    // ── 判决 ─────────────────────────────────────────────────────────────────────
    let runs = trun::list_runs(&slot).expect("列举 run");
    assert_eq!(runs.len(), 2, "折叠 + 铸新之后应当恰好有两个 run,实得 {:?}", runs.iter().map(|r| &r.id).collect::<Vec<_>>());

    // 旧 run:folded 之后它有了自己的 id,而它的产物**一个字节都没动**。
    // `RunDir` 有意不暴露 id(唯一的取得方式是解析器)⇒ 按**目录**认它,而不是按 id 串。
    let minted_dir = minted.path().to_path_buf();
    let old_id = runs
        .iter()
        .map(|r| r.id.clone())
        .find(|id| trun::resolve_run_dir(&slot, Some(id)).map(|d| d.path() != minted_dir).unwrap_or(false))
        .expect("旧 run 还在");
    let old = trun::resolve_run_dir(&slot, Some(&old_id)).expect("解析旧 run");
    let old_fp = fingerprint(old.path());
    for want in [
        format!("weights/{SLUG}_best.pth"),
        format!("weights/{SLUG}_e2_s29.pth"),
        format!("audition/{SLUG}_best/model.json"),
        "run.json".to_string(),
    ] {
        let hit = old_fp.iter().find(|(rel, _)| *rel == want);
        let was = before.iter().find(|(rel, _)| *rel == want);
        assert!(hit.is_some(), "旧 run 的 {want} 在折叠 + 铸新之后不见了");
        assert_eq!(
            hit.map(|(_, h)| h),
            was.map(|(_, h)| h),
            "旧 run 的 {want} 内容变了 —— 「铸新 run」这个动作不许改旧 run 的任何一个字节\
             (音域那一份尤其:它是实测出来的,没有任何东西会重测它)"
        );
    }
    assert_eq!(
        utai_lib::training::tproject::run_model_name(&old).as_deref(),
        Some("test-rvc"),
        "旧 run 的名字被动了 —— 那正是实机撞到的那条:新名字盖到了旧 run 头上"
    );

    // 新 run:是**另一个目录**,而且是空的(它继承不到任何东西)。
    assert_ne!(minted.path(), old.path(), "铸出来的 run 和旧 run 是同一个目录 —— 那就不是铸,是续训");
    assert!(minted.path().starts_with(trun::runs_root(&slot)), "新 run 不在 runs/ 容器里");
    assert!(
        !minted.join("run.json").exists() && !minted.join("weights").exists(),
        "刚铸出来的 run 里已经有东西了 —— 它必须什么也继承不到(§F2⒝ ④e:a minted run inherits NOTHING)"
    );

    // 池是**槽级**的,不跟着任何一个 run 走 —— 那是 layout 2 的全部意义。
    assert!(
        pool.join("dataset.fingerprint").is_file() && pool.join("0_gt_wavs").join("0_0.wav").is_file(),
        "预处理池被折叠/铸新带走了 —— 那是几小时的预处理,而它属于整个槽"
    );

    let _ = std::fs::remove_dir_all(&base);
}
