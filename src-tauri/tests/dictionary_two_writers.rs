//! S141 §E2E-D3 —— 词典目录有**两个写者**,而它们在开机时的先后顺序不是确定的。
//!
//! `lib.rs` setup() 先 **spawn** 数据根回收线程,五行之后才**同步**调
//! `sync_bundled_dictionaries`。于是回收完全可能落在 sync **之后** —— 而它对
//! `<old root>/dictionaries` 走的是「大小不同 ‖ 源更新 ⇒ 拷过去」的增量同步。没有跳过名单的
//! 话,它会把一份 update 之前的 fr.tsv 盖回刚刚被 sync 治好的活动根:
//!
//! * **这次会话**按旧音素渲染(载入器是 first-call-wins + `Box::leak`,一装载就定了整场);
//! * **下一次启动** sync 又把文件刷回来 ⇒ **事后去盘上看什么都看不出**。
//!
//! ## 这条腿与已有的两条各自守什么(别把它读成第三份重复)
//!
//! | 已有 | 它证明的 | 它**结构上**看不见的 |
//! |---|---|---|
//! | `settings.rs` 的 `reclaim_never_carries_…` | 回收**单独**跑时不碰活动根的 bundled 文件 | sync 已经跑过之后的世界;载入器 |
//! | `tests/dictionary_distribution.rs` | sync **单独**跑之后,`GlobalDicts` 唱的是新音素 | 回收这个第二写者 |
//! | **本文件** | 两个写者按**危险顺序**跑完之后,这个会话唱的是新音素 | 真正的竞态(两条线程交错)—— 只钉顺序,不赌调度 |
//!
//! ⛔ 为什么必须是**自己的 test binary**:`g2p::set_dict_dir` 是 first-call-wins 的 `OnceLock`,
//! 词典 `Box::leak` 到进程生命周期。lib 测试二进制里已有三处抢了它(指向仓库真词典),
//! `dictionary_distribution.rs` 里也已被它的第一条测试占掉。一个进程只有一次机会。
//!
//! ⛔ 驱的是 `reclaim_one_root` 而**不是** `spawn_pending_data_dir_delete`:后者被五道读**真机**
//! 状态的前置挡着(`crashlog::other_instance_alive()` 读真实的 `%LOCALAPPDATA%`),维护者开着
//! app 的时候整条链空转 —— 那样这条腿会「通过而什么都没断言」(S129 铁律 / M22 同族)。
//!
//! Skip-if-absent:`data/dictionaries` 是 gitignored 的生成物,裸 checkout 不许因此变红。
//!
//! Run:  cargo test --test dictionary_two_writers -- --nocapture

use std::path::{Path, PathBuf};

use utai_lib::commands::settings::{reclaim_one_root, sync_bundled_dictionaries};
use utai_lib::inference::g2p::{self, GlobalDicts, Lang, ResolvedKind, ScoreEvt};
use utai_lib::inference::g2p_alias::PhonemeSet;

const SHIPPED: [&str; 8] = [
    "en.tsv", "de.tsv", "fr.tsv", "es.tsv", "it.tsv",
    "zh_syllables.tsv", "zh_chars.tsv", "zh_phrases.tsv",
];

/// fr 的 D6 镜像定义的那个词:上游读 `a p s t ə ɲ i ʁ`,训练标注(以及出货的词典)读
/// `a p s t ə n i ʁ`。
const WORD: &str = "abstenir";

fn repo_dictionaries() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/dictionaries")
}

#[test]
fn a_reclaim_landing_after_the_sync_cannot_make_this_session_sing_the_old_phones() {
    let shipped = repo_dictionaries();
    if !shipped.join("fr.tsv").is_file() {
        eprintln!(
            "[dict-two-writers] SKIPPED — {} not present (gitignored generated asset)",
            shipped.display()
        );
        return;
    }
    let base = std::env::temp_dir().join(format!("utai_dicttwo_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let app_dir = base.join("install");
    let data_dir = base.join("data");
    // ⛔ 旧根**不能**放在 `app_dir/data` 下:那一格 `reclaim_one_root` 有专门的自我保护
    //    (默认根的 dictionaries 不删),会让这条腿测到另一件事。
    let old = base.join("oldroot");
    let install_dicts = app_dir.join("data").join("dictionaries");
    let root_dicts = data_dir.join("dictionaries");
    let old_dicts = old.join("dictionaries");
    for d in [&install_dicts, &root_dicts, &old_dicts] {
        std::fs::create_dir_all(d).unwrap();
    }

    // 1. install 树 = 出货的那一份。
    for name in SHIPPED {
        std::fs::copy(shipped.join(name), install_dicts.join(name)).unwrap();
    }
    // 2. 旧根 = update **之前**的那一代,从出货文件反向撤掉这把刀重建 ——
    //    所以陈的那一臂是「真词典减这一处改动」,不是一份永远走不到 vote/syllabify 的玩具文件。
    let fresh_fr = std::fs::read_to_string(install_dicts.join("fr.tsv")).unwrap();
    let stale_fr = fresh_fr.replace(" n i", " \u{0272} i");
    assert_ne!(stale_fr, fresh_fr, "陈的那一臂没造出来 —— 这条腿会什么也证明不了");
    std::fs::write(old_dicts.join("fr.tsv"), &stale_fr).unwrap();
    std::fs::write(old_dicts.join("notes.txt"), "user parked this here").unwrap();
    // 3. 活动根开局也是**陈的**(= 迁移过一次、之后每次 update 只刷新了 install 那一份:S83 那条)。
    for name in SHIPPED {
        if name == "fr.tsv" {
            std::fs::write(root_dicts.join(name), &stale_fr).unwrap();
        } else {
            std::fs::copy(shipped.join(name), root_dicts.join(name)).unwrap();
        }
    }

    // ── 危险顺序:sync 先跑(把活动根治好),回收线程**之后**才落地 ────────────────
    sync_bundled_dictionaries(&app_dir, &data_dir);
    // 前置自检:sync 真的治好了。少了它,下面那条断言在「sync 压根没生效」时会红得
    // 指向错误的组件(S129:一条红必须能被归因)。
    assert_eq!(
        std::fs::read_to_string(root_dicts.join("fr.tsv")).unwrap(),
        fresh_fr,
        "前置不成立:sync 没有把活动根治好,后面测的就不是「回收会不会把它弄脏」了"
    );

    let processed = reclaim_one_root(&app_dir, &data_dir, old.to_str().unwrap());
    assert!(processed, "回收没有处理这条队列项 —— 它连跑都没跑,这条腿是空的");

    // ── 载入器第一次读:这就是**这个会话**要唱的东西 ───────────────────────────────
    g2p::set_dict_dir(root_dicts.clone());
    let evts = [ScoreEvt {
        lyric: WORD,
        note_num: 62,
        frames: 64,
        lang: Lang::Fr,
        phoneme_input: None,
        phoneme_set: PhonemeSet::Words,
    }];
    let resolved = g2p::resolve_score(&evts, &GlobalDicts).expect("strict resolve of a French word");
    let ResolvedKind::Phones(phones) = &resolved[0].kind else {
        panic!("expected sung phones, got {:?}", resolved[0].kind)
    };
    assert!(
        phones.windows(2).any(|w| w[0] == "n" && w[1] == "i"),
        "这个会话唱的不是 `n i`:{phones:?} —— 回收落在 sync 之后,把 update 之前的 fr.tsv 盖了回去。\
         ⚠ 盘上事后看不出来:下一次启动 sync 会再把文件刷回新的。"
    );
    assert!(
        !phones.windows(2).any(|w| w[0] == "\u{0272}" && w[1] == "i"),
        "这个会话仍然唱着未经训练的那个二元组:{phones:?}"
    );

    // 顺带钉住「跳过名单只跳 bundled 的名字,不是整棵子树」—— 否则用户放在那里的文件会跟着
    // 旧根一起被删掉,而这条保护与上面那条是**互相拉扯**的两个方向。
    assert_eq!(
        std::fs::read_to_string(root_dicts.join("notes.txt")).ok().as_deref(),
        Some("user parked this here"),
        "旧根 dictionaries 目录里的用户文件被丢掉了 —— 跳过名单必须只点名 bundled 的那几个"
    );

    let _ = std::fs::remove_dir_all(&base);
}
