//! Drive the §F2⒝ ④d layout 3→4 identity migration against REAL python-built pools.
//!
//! ## Why an `#[ignore]`d integration test
//!
//! `tpool`'s unit tests build their own fixtures, so they only ever prove the migrator correct on
//! shapes I invented — and this migration's whole job is to agree, BYTE FOR BYTE, with a string
//! another language computes from a directory neither of us made up. The pools under
//! `TESTING/utai-v2-testing/gate_aug/ws_pipe_*` were produced by the real python pipelines; this
//! is what drives the production function against them.
//!
//! Same shape and same reason as `tests/tpool_migrate.rs`: ignored by default, path from the
//! environment, and it MOVES files — point it at a copy.
//!
//! ```text
//! set UTAI_IDENTITY_DATA=C:\...\leg_root      :: a data root: <root>/training/<proj>/<family>/
//! cargo test --test tpool_identity_migrate -- --ignored --nocapture identity_migrate_data_root
//! ```
//!
//! It prints one JSON line per slot so the python leg reads the OUTCOME from Rust rather than
//! re-deriving it — the same anti-second-opinion rule `tpool_layout_constants` follows.

use std::path::Path;

use utai_lib::training::{tpool, tproject};

#[test]
#[ignore]
fn identity_migrate_data_root() {
    let Ok(root) = std::env::var("UTAI_IDENTITY_DATA") else {
        eprintln!("set UTAI_IDENTITY_DATA to a data root (a COPY — this rewrites files)");
        return;
    };
    let data = Path::new(&root);

    // ★S130 —— ⛔ 两件事必须在**跑之前**说清楚,否则这条闸违反 S129 立的那条铁律
    // (「闸跑不起来」与「被测的东西不对」不许报成同一种红)。
    //
    // ⑴ **机器级的静默停机开关**:开机链的四步每一步都先问 `other_instance_alive()`,为真就
    //    直接 `return` —— 什么都不做、不报错、不返回计数(tproject.rs / tpool.rs / trun.rs 各一处,
    //    外加 `migrate_identity_all`)。它扫的是日志目录下的 `session.<pid>.alive` 哨兵,是**整机**
    //    状态,与 `UTAI_IDENTITY_DATA` 指的这份副本无关。⇒ 机器上开着一个 dev build 时,这条闸会
    //    打出 layout=0 / identity_version=1,而 python 那半读到的红与「④d 真的坏了」**一模一样**。
    //    所以把它作为一条**事实**打出来,让读者不必猜。
    // ⑵ **拒绝的理由此前根本不进转录**:`IdentityOutcome::Refused(why)` 只经 `tracing::warn!` 出场,
    //    而这个测试从来没装过 subscriber ⇒ 那些行去了虚空,python 侧只剩 `layout` 一个数字,
    //    而 layout<4 同时对应【被 refuse】【被 other_instance 跳过】【前面几折失败】三种因。
    let blocked = utai_lib::crashlog::other_instance_alive();
    println!("IDENTITY_ENV {{\"other_instance_alive\":{blocked}}}");
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_test_writer()
        .try_init();

    // The WHOLE chain, not just the last step: a slot arrives here at layout 0 (python built its
    // pools without any marker), and the identity step must refuse anything the earlier folds have
    // not committed. Driving only the last step would prove the one thing that cannot happen.
    utai_lib::training::migrate_layouts(data);

    let training = tproject::training_root(data);
    let Ok(rd) = std::fs::read_dir(&training) else {
        panic!("no training root at {}", training.display());
    };
    for e in rd.flatten() {
        let proj = e.path();
        if !proj.join(tproject::PROJECT_META).is_file() {
            continue;
        }
        for family in tproject::FAMILIES {
            let slot = proj.join(family);
            if !slot.is_dir() {
                continue;
            }
            let layout = tpool::read_slot_meta(&slot).map(|m| m.layout).unwrap_or(0);
            let pools: Vec<String> = tpool::list_pools(&slot)
                .into_iter()
                .map(|p| format!("{{\"id\":{:?},\"fp\":{:?}}}", p.id, p.fp_text))
                .collect();
            let runs: Vec<String> = utai_lib::training::trun::run_dirs(&slot)
                .iter()
                .map(|r| {
                    let pool = utai_lib::training::trun::pool_of_run(r)
                        .map(|s| format!("{s:?}"))
                        .unwrap_or_else(|| "null".to_string());
                    format!(
                        "{{\"dir\":{:?},\"pool\":{pool}}}",
                        r.path().file_name().unwrap_or_default().to_string_lossy()
                    )
                })
                .collect();
            println!(
                "IDENTITY_LEG {{\"project\":{:?},\"family\":{:?},\"layout\":{},\
                 \"identity_version\":{},\"pools\":[{}],\"runs\":[{}]}}",
                proj.file_name().unwrap_or_default().to_string_lossy(),
                family,
                layout,
                tpool::identity_version(&slot),
                pools.join(","),
                runs.join(",")
            );
        }
    }
}
