//! # 分配器审计件(S92k)—— 「哪些音素被丢了 / 哪些没吃满自己的目标」的**记账核对**,不是启发式
//!
//! 用户 2026-07-31 立项时的两条要求,本文件的全部设计都是从它们推出来的:
//!   ①「建出来的仪器至少你得保证它既不是瞎的也不是错的」
//!   ②「仪器的更大意义是发现未知的问题,但是还不能引起误导」
//!
//! ## 这台仪器**是什么**
//! 对我们自己的数据做加减法:`g2p::resolve_score` 说该唱哪些音素(**分配之前**的真值)
//! vs `build_arrays_daw` 实际发出了什么。两边是同一个数据结构 ⇒ 丢音检测**零假阳、零假阴**,
//! 不是估计。目标同理:`onset/coda_target_frames` 就在隔壁,不用猜。
//!
//! ## 这台仪器**不是什么**(写在最前,免得它被当成判决)
//! 1. **不判好听**。cv 域一切度量与耳朵解耦是本仓铁律;它只报「与我们自己的目标 / 与训练分布不一致」。
//! 2. **抓不到「我们和真人一样、但两者都不好」**(例:decoder 侧的外国腔 —— 盘上根本没有英语音源,
//!    那条不在这条链上)。
//! 3. **抓不到「时长对但落点错」**,除非你也看 `displacement` 与位置列。A(coda-first 切分)正是
//!    这一类:每个音素时长都正常,辅音却落进了长音中间。
//! 4. 对 zh 延音那两条特殊发射路径只做**显式建模**;模型一旦与生产漂移,`unmodelled` 会响
//!    (见 `audit_models_the_zh_sustain_paths`),而不是静默吐假丢音。
//!
//! ## 为什么它不会**瞎**(六条,全部从本仓踩过的坑反推,不是清单)
//! - **单一真源**:音节切分走生产的 `syllable_split`,目标走生产的 `*_target_frames`,zh 延音走
//!   生产的 `zh_hold_phone`。**审计件里没有一行是对生产规则的第二次实现**。
//! - **真值取自分配之前**(`resolve_score`),不是泳道 —— S92 血训:一个丢了辅音的音符在泳道里
//!   正好显示成「它本来就只有一个 coda」,拿泳道当真值 = 拿被过滤过的产物证明「不会被过滤」。
//! - **未建模事件必须为 0 且会被打印**(`unmodelled`)。仪器不许对自己没覆盖的路径宣称干净。
//! - **阳性对照**:`audit()` 是 `(score, resolved, arr)` 的**纯函数**,可以喂一份人工挖掉音素的
//!   `arr`,断言它恰好报出来。没有阳性对照的检查器,「零发现」既可能是干净也可能是没接上。
//! - **每条 finding 点到具体音符**(evt / 歌词 / 音素 / 实测 / 目标);聚合数字只是它们的和。
//!   S92j 实测过一次「分项对得上、聚合对不上」=聚合口径有分歧,能点到音符的数字才可证伪。
//! - **两个目标分开报**:`target_effective`(分配器自己瞄的,含 fr≤5 封顶等策略)与
//!   `target_measured`(训练数据的原始先验)。两者之差 = **我们主动放弃的发音时长**;S92i 那条
//!   「`thing` 的 θ 被砍到 40 ms」就是这个差暴露的。只报一个数会漏掉一整类问题。

use super::*;
use super::super::score2cv_audit_ref as ref_tbl;

/// 音素在音节里的位置 —— 决定该读哪张目标表。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Position {
    Onset,
    Medial,
    Nucleus,
    Coda,
}

impl Position {
    pub(crate) fn code(self) -> &'static str {
        match self {
            Position::Onset => "onset",
            Position::Medial => "medial",
            Position::Nucleus => "nucleus",
            Position::Coda => "coda",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    /// g2p 说该唱,泳道里没有 —— 静默丢音(用户听到的「唱不出原词」)。
    Dropped,
    /// 发了,但短于**分配器自己**的目标。
    Starved,
    /// 分配器的目标本身低于训练先验 —— 策略主动放弃的发音时长(S92i 那一类)。
    PolicyCapped,
    /// 短于训练管线的硬下限:真英语数据 **0 / 5608** 个辅音短于 3 帧
    /// (`scripts/realign_mindur.py` 的 `DCONS = 3`,训练侧 S57 用耳朵定的)。
    BelowTrainingFloor,
    /// 核落进 S84 实测的 2 帧 cv/decoder 塌陷区。
    NucleusCollapse,
    /// ★★**出了真人的分布** —— 这个音素的时长低于真人歌手在同一格(语言 × 音素 × 音节位置 ×
    /// 音符长度桶)里的 **p05**。参照来自 `score2cv_audit_ref.rs`(与时长目标同一批录音)。
    ///
    /// 这是仪器**发现未知问题**的那一半:上面五条轴都在问「我们有没有做到自己定的目标」,只有这条
    /// 在问「**我们和真人到底像不像**」。S92 那批根因全是这么挖出来的(「真英语数据 0/5608 个辅音
    /// 短于 3 帧,而我们 41%」),它把那种一次性的手工比对变成常驻的。
    /// ⚠ 每条都带 `n=样本量` —— **没有样本量的偏离度不可读**(`ʒ` 全量英语训练数据只有约 14 秒,
    /// 报它「出格」毫无意义,那是数据枯竭不是缺陷)。样本不足的格子**不发参照**,于是「没量过」
    /// 与「量过没问题」永远分得开。
    OutOfDistribution,
    /// ★**这个音符的元音被别的音符借走了帧** —— 「另一个词伸过来把这个词的元音剪短」,
    /// 用户耳判的拼接味 / 开口元音变闭口。数字来自生产的借帧账本(`ScoreArrays::borrow_ledger`),
    /// 是**精确值**:从最终数组反推不出来,因为一个音符可以同时借进和借出,净额里两者不可分离。
    /// ⚠ 与「元音被**自己**的词首辅音吃掉」是两回事 —— 后者是每个音节都要付的正常代价。
    NucleusLentAway,
}

impl Kind {
    pub(crate) fn code(self) -> &'static str {
        match self {
            Kind::Dropped => "DROPPED",
            Kind::Starved => "STARVED",
            Kind::PolicyCapped => "POLICY_CAPPED",
            Kind::BelowTrainingFloor => "BELOW_TRAINING_FLOOR",
            Kind::NucleusCollapse => "NUCLEUS_COLLAPSE",
            Kind::NucleusLentAway => "NUCLEUS_LENT_AWAY",
            Kind::OutOfDistribution => "OUT_OF_DISTRIBUTION",
        }
    }
    pub(crate) const ALL: [Kind; 7] = [
        Kind::Dropped,
        Kind::OutOfDistribution,
        Kind::Starved,
        Kind::PolicyCapped,
        Kind::BelowTrainingFloor,
        Kind::NucleusCollapse,
        Kind::NucleusLentAway,
    ];
}

#[derive(Debug, Clone)]
pub(crate) struct Finding {
    pub kind: Kind,
    pub evt: usize,
    pub lyric: String,
    pub note_frames: i64,
    pub lang: &'static str,
    pub phone: &'static str,
    pub position: Position,
    /// 实发帧数(0 = 被丢弃)。
    pub actual: i64,
    /// 分配器自己瞄的目标(含策略封顶)。核没有目标,记 −1。
    pub target_effective: i64,
    /// 训练数据的原始先验(不含任何策略封顶)。核没有目标,记 −1。
    pub target_measured: i64,
    /// **这个音符自己的帧数放不下全部音素的最低下限**(核 `NUCLEUS_KEEP_MIN` + 其余每个
    /// `CODA_MIN_FRAMES`,两个都是生产常量;这是一个**下界**,不是对分配器的模拟)。
    ///
    /// ★它把「谱面写得太短」与「分配器算错了」分开 —— 快歌里 2 帧音符的核只能是 2 帧,那是谱面
    /// 本身;把它和一个 20 帧音符上的 2 帧核混在同一个计数里,读数的人就会被误导(实测:日文快歌
    /// 报 110 个核塌陷,其中 105 个落在 ≤4 帧的音符上)。
    /// ⚠**它不等于「不可修」**:`even` 的 /n/ 当初正是在一个 4 帧音符上被丢掉的,而 S92b 靠改规则
    /// 把它救了回来。所以这只是**排序与分组**的依据,**任何一条都不会因此被隐藏**。
    pub score_forced: bool,
    /// 这个音素所在**音符组**的总帧数(`ScoreArrays.note_dur` = 生产按 (音高, 语言) 连续段的分组)。
    /// ★与 `note_frames`(音符自己的帧数)**是两个口径**,分布参照表按前者量,所以两个都要显示 ——
    /// 只显示一个,读数的人无从判断查表查得对不对。
    pub group_frames: i64,
    /// 只对 `OutOfDistribution` 有意义:这一格参照的**观测样本量**。0 = 与分布无关的其它类。
    /// **必须随判决一起显示** —— 没有样本量的偏离度不可读(`ʒ` 全量英语只有约 14 秒)。
    pub ref_count: i64,
}

impl Finding {
    /// 严重度 = 缺了多少帧,排序用。丢音按「本该拿到的目标」算,所以它天然排在最前。
    pub(crate) fn deficit(&self) -> i64 {
        match self.kind {
            Kind::Dropped => self.target_measured.max(CODA_MIN_FRAMES),
            Kind::PolicyCapped => self.target_measured - self.target_effective,
            Kind::NucleusCollapse => NUCLEUS_KEEP_MIN - self.actual,
            // 参照的 p05 记在 target_effective 里 ⇒ 差 = 比真人的下界还短多少。
            Kind::OutOfDistribution => self.target_effective - self.actual,
            // 借走的帧数直接记在 target_effective 里(= 实发 + 被借走),差就是被借走的量。
            Kind::NucleusLentAway => self.target_effective - self.actual,
            _ => (self.target_effective.max(0) - self.actual).max(0),
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct Report {
    pub findings: Vec<Finding>,
    pub events: usize,
    pub phones_expected: usize,
    pub phones_emitted: usize,
    /// **必须为空**。审计件没有建模的发射路径落在这里 —— 仪器不许对自己没覆盖的东西宣称干净。
    pub unmodelled: Vec<usize>,
    /// 逐事件的跨界位移(Σ 该事件音素帧 − 该音符自己的帧数):借帧把时间轴挪了多远。
    pub displacement: Vec<(usize, i64)>,
    /// (谱面总帧, 实发总帧) —— 必须相等。
    pub conservation: (i64, i64),
    /// 这份报告是对当前构建还是对存档泳道做的 —— 决定了哪几条轴可用(见 `Source`)。
    pub source: Source,
}

impl Default for Source {
    fn default() -> Self {
        Source::Live
    }
}

impl Report {
    pub(crate) fn of_kind(&self, k: Kind) -> impl Iterator<Item = &Finding> {
        self.findings.iter().filter(move |f| f.kind == k)
    }
    pub(crate) fn count(&self, k: Kind) -> usize {
        self.of_kind(k).count()
    }
    /// 该类里**音符本来放得下**的那些 —— 也就是真正可能是分配器算错的那一批。
    pub(crate) fn count_actionable(&self, k: Kind) -> usize {
        self.of_kind(k).filter(|f| !f.score_forced).count()
    }
    /// 位移超过 `n` 帧的音符数(S92j 用这根轴抓到了耳朵听得见的「拼接味」)。
    /// ⚠ 它是**逐事件净额**:一条链上每个中间音符既借进又借出时会互相抵消,所以它**单独不足以**
    /// 覆盖「时长对但落点错」—— 真正的元音损伤轴是下面这个。
    pub(crate) fn displaced_beyond(&self, n: i64) -> usize {
        self.displacement.iter().filter(|(_, d)| d.abs() > n).count()
    }
    /// ★**元音总损失帧数** —— 分配器发给核的帧,有多少在之后被别人借走了。
    /// 这就是 S92j 那一轮必须先量、而当时只能靠外部脚本临时拼出来的那个数;它是「另一个词伸过来
    /// 把这个词的元音剪短」(用户耳判的拼接味/开口元音变闭口)唯一的量化出口。
    /// ⚠ 位移轴在一条借帧链上会自己抵消掉,**这个数不会**。
    pub(crate) fn vowel_frames_lost(&self) -> i64 {
        self.of_kind(Kind::NucleusLentAway).map(|f| f.deficit()).sum()
    }
    /// 一行摘要 + 按严重度排序的明细。**永远带上 `unmodelled`** —— 它非空时,上面的数字都不算数。
    pub(crate) fn render(&self, top: usize) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let _ = writeln!(
            s,
            "[audit] events={} phones {}→{} 守恒 {}=={} 位移>4帧的音符={} \
             **元音总损失帧数={}** 未建模={:?}",
            self.events,
            self.phones_expected,
            self.phones_emitted,
            self.conservation.0,
            self.conservation.1,
            self.displaced_beyond(4),
            self.vowel_frames_lost(),
            self.unmodelled,
        );
        if self.source == Source::ArchivedLane {
            let _ = writeln!(
                s,
                "  ⚠存档泳道模式:借帧账本与音符内分配无法重建 ⇒ STARVED / POLICY_CAPPED /\n\
                 \x20  NUCLEUS_LENT_AWAY 三条轴**已关闭**(上面的 0 不是「没问题」,是「没量」)。\n\
                 \x20  DROPPED / BELOW_TRAINING_FLOOR / NUCLEUS_COLLAPSE 三条在这个模式下仍然成立。"
            );
        }
        for k in Kind::ALL {
            let _ = writeln!(
                s, "  {:<20} {:>4}   其中音符本来放得下 {:>4}",
                k.code(), self.count(k), self.count_actionable(k)
            );
        }
        let _ = writeln!(
            s,
            "  (POLICY_CAPPED = 分配器**主动**把目标砍到先验以下,是策略不是缺陷 —— 但它是\n\
             \x20  「我们放弃了多少发音时长」的唯一读数;S92i 那一类问题就是靠它才看得见。\n\
             \x20  NUCLEUS_LENT_AWAY 同理 = Auto 臂的**机制本身**(前一个音符的元音供养下一个词的\n\
             \x20  词首辅音,好让元音落在拍点上),不是缺陷 —— 要看的是**量**:它的总和就是抬头那个\n\
             \x20  「元音总损失帧数」,改动前后对比才有意义(S92j 的 207→194 就是这个数)。\n\
             \x20  末列 `谱短` = 这个音符自己的帧数放不下全部音素的最低下限,**不等于不可修**。)"
        );
        for f in self.findings.iter().take(top) {
            let _ = writeln!(
                s,
                "  {:<20} [{:>4}] {:<14} {:<3} {:<8} {:>3}fr  实发 {:>2}  目标 {:>2}(先验 {:>2})  缺 {:>2} {}{}",
                f.kind.code(), f.evt, f.lyric, f.phone, f.position.code(),
                f.note_frames, f.actual, f.target_effective, f.target_measured, f.deficit(),
                if f.score_forced { "谱短 " } else { "" },
                // ★样本量与判决同行 —— 没有它,偏离度不可读。
                if f.ref_count > 0 { format!("[真人 p05={} p50={} n={}]", f.target_effective, f.target_measured, f.ref_count) } else { String::new() },
            );
        }
        s
    }
}

/// 查真人分布参照:先查这门语言自己的格子,没有(或样本不足,生成器根本不发)就退到跨语言池化格。
/// 两条都没有 ⇒ `None` = **没量过**,与「量过没问题」严格区分 —— 审计件不对没参照的格子下判决。
///
/// ⚠ 桶键必须是 **note group 总帧**(`ScoreArrays.note_dur`),因为参照表就是按那个口径量的;
/// 拿单音符帧数去查是另一个口径(那正是队列 B 项的不一致,别在这里再制造一次)。
fn dist_cell(lang: &str, token: &str, position: Position, group_frames: i64) -> Option<&'static ref_tbl::DurCell> {
    let bucket = if group_frames <= 7 {
        0u8
    } else if group_frames <= 15 {
        1
    } else {
        2
    };
    let pos = match position {
        Position::Onset => 0u8,
        Position::Medial => 1,
        Position::Nucleus => 2,
        Position::Coda => 3,
    };
    let find = |lg: &str| {
        ref_tbl::PHONE_DUR_DIST
            .iter()
            .find(|c| c.lang == lg && c.token == token && c.position == pos && c.bucket == bucket)
    };
    find(lang).or_else(|| find(""))
}

/// 训练语料里辅音的最短时长 —— **单一真源在生产侧**,这里只转发。
/// ★S97 澄清:这是「模型见过的最短辅音」(`realign_mindur.py` 的 DCONS=3 造成的),所以它是
/// **分布外**判据、可以继续用;但它**不是**「真人不会更短」的证据 —— 那句话是循环论证
/// (参照表的 en p05 全部等于 3 正是这条 DP 的产物)。生产侧的**真人**地板另有其人,
/// 见 `score2cv::chaining_coda_floor`(取自 GTSinger 上游标注,逐音素)。
use super::TRAINING_MIN_FRAMES as TRAINING_CONSONANT_FLOOR;

/// 一个事件「该发什么」。**只描述发射契约,不复制分配规则** —— 分配规则的产物是 `arr`,
/// 拿它当真值就是自证。
enum Expect {
    /// 一个休止/呼吸 token。
    Single(&'static str),
    /// 走 `allocate_in_note` 的普通音符。
    Allocated(Vec<&'static str>),
    /// zh 变音高延音:**不经分配器**,直接发一个变形后的载体音素、拿整个音符的帧。
    ZhHold(&'static str),
    /// zh 同音高延音:帧并进**前一条**,本事件**不发任何音素** —— 这不是丢音。
    MergedIntoPrev,
    /// 审计件没有建模的路径。必须为空集,否则响亮报出来。
    Unmodelled,
}

/// 复刻 `assemble_arrays` 的**发射契约**(不是分配算法):哪些事件发几个音素、发的是哪个 token。
/// 只有 zh 延音那两条分支需要特判,而它们调用生产的 `zh_hold_phone`,不是第二份实现。
fn expectation(
    res: &g2p::ResolvedNote,
    prev_sung: bool,
    prev_pitch: Option<i64>,
    note_num: i64,
) -> Expect {
    match &res.kind {
        g2p::ResolvedKind::Rest => Expect::Single("SP"),
        g2p::ResolvedKind::Breath => Expect::Single("AP"),
        g2p::ResolvedKind::Unknown => Expect::Unmodelled,
        g2p::ResolvedKind::Phones(ph) => {
            if res.is_sustain && res.run_lang == g2p::Lang::Zh && prev_sung {
                if prev_pitch == Some(note_num) {
                    return Expect::MergedIntoPrev;
                }
                if let Some(&carrier) = ph.last() {
                    return Expect::ZhHold(zh_hold_phone(carrier));
                }
            }
            Expect::Allocated(ph.clone())
        }
    }
}

/// 一个音素的位置 + 两个目标。
///
/// ★**`target_effective` 不再由我算,而是由生产的 `allocate_in_note` 给** —— 这是 S92k 对抗审查
/// 抓出的三条 major 的共同正解。我原先手抄了 medial 元音的份额公式并读了 `note_frames`,而分配器
/// 读的是 `spendable`(InNote 臂上两者不等)⇒ 幻影 STARVED;同样地,coda 的 `fr*2/5` 预算天花板、
/// 丛下限、核的余量全都不在我的算式里。直接调用它,这一整类失配**由构造消失**。
///
/// 于是两条轴的语义变得干净且互不重叠:
///   • `target_effective` = **分配器当场发给它的帧数**(音符内的账)⇒ `Starved` = 分配之后又被
///     **借走**了多少(这正是「元音总损失」那根轴,核第一次有了目标);
///   • `target_measured` = **训练先验**(与分配器无关)⇒ `PolicyCapped` = 分配器自己就比真人短
///     多少(`fr*2/5` 的 coda 天花板、fr≤5 快段封顶……都在这里现形,清单 ① 正是这一类)。
///
/// onset 是唯一的例外:Auto 臂上它由**借帧**供给,分配器留 0,所以它的 effective 仍是那个
/// `target` 闭包(生产的 `onset_target_frames` + fr≤5 封顶)。
fn targets(
    ph: &[&'static str],
    in_note: &[i64],
    i: usize,
    note_frames: i64,
    onset_capped_to_2: bool,
) -> (Position, i64, i64) {
    let (onset_end, nuc) = syllable_split(ph);
    if i < onset_end {
        let measured = onset_target_frames(ph[i], note_frames);
        let effective = if onset_capped_to_2 { measured.min(2) } else { measured };
        (Position::Onset, effective, measured)
    } else if i < nuc {
        let measured = if is_nucleus_phone(ph[i]) { in_note[i] } else { onset_target_frames(ph[i], note_frames) };
        (Position::Medial, in_note[i], measured)
    } else if i == nuc {
        // 核在音符内的分配额(信息列)。它**不是**判据 —— 判据走借帧账本,见 `audit()`。
        (Position::Nucleus, in_note[i], -1)
    } else {
        (Position::Coda, in_note[i], coda_target_frames(ph[i], note_frames))
    }
}

/// `arr` 是**这次构建**产出的,还是一份**存档泳道**?
///
/// ★存档泳道里 `borrow_ledger` 与 `in_note_alloc` 是重建不出来的(借进/借出在净额里不可分离,
/// 音符内分配也早已被借帧覆盖),于是依赖它们的三条轴 —— 饥饿 / 策略封顶 / 元音借出 —— **不可用**。
/// 拿当前代码的账去配历史的分配会静默出错数,所以那种模式下它们被**关掉并声明**,而不是照算。
/// 丢音 / 训练下限 / 核塌陷三条只看最终帧数与 g2p 真值,两种模式下都成立。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Source {
    /// `arr` 来自本次 `build_arrays_daw` —— 六条轴全可用。
    Live,
    /// `arr` 的 phone/dur/evt 来自一份存档泳道 —— 只有三条轴可用。
    ArchivedLane,
}

/// **纯函数** —— 这是阳性对照能存在的原因:可以喂一份人工损坏的 `arr`,断言它恰好报出来。
pub(crate) fn audit(
    score: &[g2p::ScoreEvt],
    resolved: &[g2p::ResolvedNote],
    arr: &ScoreArrays,
    source: Source,
) -> Report {
    assert_eq!(score.len(), resolved.len(), "score/resolved 长度不一致 —— 喂错了");
    assert_eq!(arr.phon.len(), arr.phone_dur.len(), "arr 自身不自洽");
    assert_eq!(arr.phon.len(), arr.evt.len(), "arr 自身不自洽");
    let live = source == Source::Live;
    let mut rep = Report { events: score.len(), source, ..Default::default() };

    // (arr 里的绝对下标, token, 帧数) —— 绝对下标是借帧账本的键。
    let mut emitted: Vec<Vec<(usize, &'static str, i64)>> = vec![Vec::new(); score.len()];
    for i in 0..arr.phon.len() {
        let e = arr.evt[i];
        assert!(e < score.len(), "evt 下标越界");
        emitted[e].push((i, arr.phon[i], arr.phone_dur[i]));
    }
    rep.phones_emitted = arr.phon.len();
    // 生产记的借帧账本:出借音素下标 → 被借走的净帧数。
    let mut lent_by: std::collections::HashMap<usize, i64> = std::collections::HashMap::new();
    for &(idx, d) in &arr.borrow_ledger {
        *lent_by.entry(idx).or_insert(0) += d;
    }
    // 生产在借帧之前的音符内分配快照:事件下标 → 逐音素帧数。
    let alloc_of: std::collections::HashMap<usize, &Vec<i64>> =
        arr.in_note_alloc.iter().map(|(k, d)| (*k, d)).collect();
    rep.conservation =
        (score.iter().map(|e| e.frames).sum(), arr.phone_dur.iter().sum());

    let mut prev_sung = false;
    let mut prev_pitch: Option<i64> = None;
    let mut prev_phone: Option<&'static str> = None;
    // 上一个**真正发了音素**的事件下标 —— zh 同音高延音把帧并进它,位移账要记在它头上。
    let mut last_emitting: Option<usize> = None;

    for (k, (evt, res)) in score.iter().zip(resolved.iter()).enumerate() {
        let got = &emitted[k];
        let chaining = consonant_chaining_language(res.run_lang);
        let mut findings: Vec<Finding> = Vec::new();
        // 这个音符自己的帧数能不能放下全部音素的最低下限(核 3 + 其余各 2,都是生产常量)。
        // 下界,不是模拟 —— 说「放不下」时一定真的放不下。
        let n_phones = match &res.kind {
            g2p::ResolvedKind::Phones(p) => p.len() as i64,
            _ => 1,
        };
        let score_forced =
            evt.frames < NUCLEUS_KEEP_MIN + (n_phones - 1).max(0) * CODA_MIN_FRAMES;
        let mk = |kind: Kind, phone: &'static str, position: Position, actual: i64,
                  target_effective: i64, target_measured: i64| Finding {
            kind,
            evt: k,
            lyric: evt.lyric.to_string(),
            note_frames: evt.frames,
            lang: res.run_lang.code(),
            phone,
            position,
            actual,
            target_effective,
            target_measured,
            score_forced,
            group_frames: 0,
            ref_count: 0,
        };

        let expectation = expectation(res, prev_sung, prev_pitch, evt.note_num);
        // ★位移账:zh 同音高延音把自己的帧并进**前一个发过音素的事件**,时间轴上什么都没动 ——
        //   按逐事件净额算会同时记出 −fr 和 +fr 两笔假位移(审查抓到的 major),所以把它的帧
        //   记回那个事件的「自有帧」里,两笔一起消掉。
        if matches!(expectation, Expect::MergedIntoPrev) {
            rep.displacement.push((k, 0));
            if let Some(owner) = last_emitting {
                if let Some(e) = rep.displacement.iter_mut().find(|(i, _)| *i == owner) {
                    e.1 -= evt.frames;
                }
            }
        } else {
            rep.displacement.push((k, got.iter().map(|(_, _, d)| *d).sum::<i64>() - evt.frames));
        }

        match expectation {
            Expect::Unmodelled => rep.unmodelled.push(k),
            Expect::MergedIntoPrev => {
                // 建模说「不该发」,却发了 ⇒ 我的模型与生产漂移了,响亮报出来。
                if !got.is_empty() {
                    rep.unmodelled.push(k);
                }
            }
            Expect::Single(tok) => {
                rep.phones_expected += 1;
                if got.len() != 1 || got[0].1 != tok {
                    rep.unmodelled.push(k);
                }
            }
            Expect::ZhHold(tok) => {
                // 分配器根本没参与:生产直接 push 一个音素、给它整个音符的帧。
                rep.phones_expected += 1;
                if got.len() != 1 || got[0].1 != tok {
                    rep.unmodelled.push(k);
                } else {
                    let want = evt.frames.max(1);
                    let pos = if is_nucleus_phone(tok) { Position::Nucleus } else { Position::Coda };
                    if got[0].2 < want {
                        findings.push(mk(Kind::Starved, tok, pos, got[0].2, want, -1));
                    }
                }
            }
            Expect::Allocated(ph) => {
                rep.phones_expected += ph.len();
                let (_, nuc) = syllable_split(&ph);
                // 分配器的 fr≤5 封顶谓词:下一个事件是 sustain,且它的第一个音素就是本音符的核。
                let held_by_next = nuc < ph.len()
                    && resolved.get(k + 1).is_some_and(|r| {
                        r.is_sustain
                            && matches!(&r.kind, g2p::ResolvedKind::Phones(np)
                                        if np.first() == Some(&ph[nuc]))
                    });
                let cap2 = evt.frames <= 5 && !held_by_next;
                // ★S92b 的「核是被延长而非起音」判据 —— 走生产的 `nucleus_is_held`,因为它决定了
                //   一个 2 帧的核到底算不算 S84 的塌陷区(生产明文:那 2 帧延续的是已在唱的元音)。
                let held = nucleus_is_held(prev_phone.as_ref(), &ph, nuc, res.is_sustain);
                // ★目标 = 生产**实际发出**的音符内分配(借帧之前的快照),不是我算的、也不是我
                //   再调一次分配器算的 —— 两条臂的 `spendable` 不同(InNote 先预留 onset),
                //   自己再算一次就只能在一条臂上正确。缺快照 ⇒ 响亮记未建模,不猜。
                // 存档泳道模式下没有快照,依赖它的三条轴已经关掉,给一份全 −1 的占位。
                let placeholder = vec![-1i64; ph.len()];
                let in_note: &Vec<i64> = match alloc_of.get(&k) {
                    Some(v) => v,
                    None if !live => &placeholder,
                    None => {
                        rep.unmodelled.push(k);
                        continue;
                    }
                };
                assert_eq!(in_note.len(), ph.len(), "分配快照与音素表长度不符 —— 喂错了");

                // 实发是期望的**子序列**(发射只跳过 d<=0,永不重排、永不新增)。
                let mut gi = 0usize;
                for (i, &p) in ph.iter().enumerate() {
                    let (position, eff, measured) = targets(&ph, in_note, i, evt.frames, cap2);
                    let actual = if gi < got.len() && got[gi].1 == p {
                        gi += 1;
                        got[gi - 1].2
                    } else {
                        0
                    };
                    if actual == 0 {
                        findings.push(mk(Kind::Dropped, p, position, 0, eff, measured));
                        continue;
                    }
                    // ★核不比 `eff`:分配器留给它的额度里,本来就包含要付给自己词首辅音的那份
                    //   (Auto 臂把 onset 槽留成 0,等着从别处供给)。核的判据走账本,见下。
                    if live && position != Position::Nucleus && actual < eff {
                        findings.push(mk(Kind::Starved, p, position, actual, eff, measured));
                    }
                    if live && position == Position::Nucleus {
                        let lent = lent_by.get(&got[gi - 1].0).copied().unwrap_or(0);
                        if lent > 0 {
                            findings.push(mk(
                                Kind::NucleusLentAway, p, position, actual, actual + lent, -1,
                            ));
                        }
                    }
                    if live && measured >= 0 && eff < measured {
                        findings.push(mk(Kind::PolicyCapped, p, position, actual, eff, measured));
                    }
                    // ★塌陷区只对**起音**的核成立。被延长的核(S92b)生产明文写着那 2 帧是安全的
                    //   ——「它们延续的是模型已经在唱的元音,不是 S84 量到的 2 帧起音」。
                    if position == Position::Nucleus && actual <= CODA_MIN_FRAMES && !held {
                        findings.push(mk(Kind::NucleusCollapse, p, position, actual, eff, -1));
                    }
                    // ★与真人分布对拍 —— 仪器**发现未知问题**的那一半。桶键走生产自己的
                    //   note group 分组(`note_dur`),与参照表同口径。
                    // ★桶键 = **这个音符自己的帧数**,不是 `note_dur` 的音符组。
                    //   实测走过一次弯路:生产的 `note_dur` 把整段同音高音符并成一组(日文快歌里
                    //   一个 4 帧的「さ」报出 86 帧的组),而训练数据的分组**同样会合并**(20% 的组
                    //   含 2 个以上的核,最长 252 帧)⇒ 桶键根本不控制**密度**:同样 86 帧,可能是
                    //   真人的一个长音,也可能是我们这首快歌的二十个 4 帧音符,拿它们对比毫无意义。
                    //   按音符自己的帧数查,短音符就与短音符组比;这也与生产查时长目标的口径一致。
                    if let Some(cell) =
                        dist_cell(res.run_lang.code(), p, position, evt.frames)
                    {
                        if actual < cell.p05 {
                            let mut f =
                                mk(Kind::OutOfDistribution, p, position, actual, cell.p05, cell.p50);
                            f.ref_count = cell.count;
                            f.group_frames = arr.note_dur[got[gi - 1].0];
                            // ★可负担性判据 —— 桶很宽(≤7 帧是一档),拿一个 3 帧音符的元音去比
                            //   「真人在这一档里的 p05=5」是不公平的:那个音符物理上就放不下 5 帧。
                            //   所以这里的 `score_forced` 用**参照值本身**重算(比逐事件那个更精确):
                            //   这个音符组能不能在其余音素各守下限的前提下,给它 p05 帧?
                            //   ⚠ 仍然只是**分类**:这些条目一条不少地留在总表里,只是不进「可动」计数,
                            //   也不排在前面 —— 隐藏才会误导。
                            let others = (ph.len() as i64 - 1).max(0) * CODA_MIN_FRAMES;
                            f.score_forced = evt.frames < cell.p05 + others;
                            findings.push(f);
                        }
                    }
                    // 训练下限是**辅音**的下限,且只对成丛语言成立。
                    if chaining && !is_nucleus_phone(p) && actual < TRAINING_CONSONANT_FLOOR {
                        findings.push(mk(
                            Kind::BelowTrainingFloor, p, position, actual, eff, measured,
                        ));
                    }
                }
                if gi != got.len() {
                    // 实发里有期望序列外的东西 ⇒ 我漏了一条发射路径。
                    rep.unmodelled.push(k);
                }
            }
        }
        rep.findings.append(&mut findings);

        if let Some(&(_, p, _)) = got.last() {
            prev_phone = Some(p);
            prev_sung = !matches!(p, "SP" | "AP");
            prev_pitch = if prev_sung { Some(evt.note_num) } else { None };
            last_emitting = Some(k);
        }
    }

    // 「音符本来放得下」的排在前面 —— 那一批才可能是分配器算错;谱面写太短的仍然全部保留。
    rep.findings.sort_by_key(|f| (f.score_forced, -f.deficit(), f.evt));
    rep
}

/// 把一个音素字符串还原成 vocab 里的 `&'static str`(读**存档泳道**时需要:JSON 里是 String,
/// 而 `ScoreArrays.phon` 是 'static)。不在 210 token 词表里 ⇒ None,响亮失败,不静默替换。
pub(crate) fn intern(p: &str) -> Option<&'static str> {
    tbl::PHONE_TO_ID.iter().find(|(tok, _)| *tok == p).map(|(tok, _)| *tok)
}

/// 端到端入口:一份谱 → 一份报告。走生产路径,零复刻。
pub(crate) fn audit_score(
    score: &[g2p::ScoreEvt],
    dicts: &dyn g2p::DictSource,
    timing: ArticulationTiming,
) -> Result<Report> {
    let resolved = g2p::resolve_score(score, dicts)?;
    let arr = build_arrays_daw(score, dicts, timing)?;
    Ok(audit(score, &resolved, &arr, Source::Live))
}

// ─── 仪器自己的 gate ──────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod selfcheck {
    use super::*;
    use super::super::tests::{en_dicts, raw};
    use crate::inference::g2p_alias::PhonemeSet;

    fn en(p: &'static str, fr: i64) -> g2p::ScoreEvt<'static> {
        g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames: fr, lang: g2p::Lang::En,
            phoneme_input: Some(p), phoneme_set: PhonemeSet::Words,
        }
    }

    /// ★阳性对照 #1 —— 没有它,「零发现」既可能是干净也可能是仪器没接上。
    /// 人工从 arr 里挖掉一个音素,审计必须**恰好**报出那一个 `Dropped`,并点得到是哪个。
    #[test]
    fn audit_positive_control_catches_an_injected_drop() {
        let d = en_dicts();
        let score = vec![en("M AY1 N D", 20)];
        let resolved = g2p::resolve_score(&score, &d).unwrap();
        let good = build_arrays_daw(&score, &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(good.phon, vec!["m", "aɪ", "n", "d"]);
        assert_eq!(audit(&score, &resolved, &good, Source::Live).count(Kind::Dropped), 0, "干净的分配不该报丢音");

        // 挖掉 /n/,帧还给核(守恒不破 —— 否则报出来的可能是守恒而不是丢音)。
        let mut broken = build_arrays_daw(&score, &d, ArticulationTiming::Auto).unwrap();
        let n_at = broken.phon.iter().position(|&p| p == "n").unwrap();
        let freed = broken.phone_dur[n_at];
        broken.phon.remove(n_at);
        broken.phone_dur.remove(n_at);
        broken.evt.remove(n_at);
        let nuc_at = broken.phon.iter().position(|&p| p == "aɪ").unwrap();
        broken.phone_dur[nuc_at] += freed;

        let rep = audit(&score, &resolved, &broken, Source::Live);
        assert_eq!(rep.count(Kind::Dropped), 1, "注入的丢音必须被抓到");
        let f = rep.of_kind(Kind::Dropped).next().unwrap();
        assert_eq!((f.phone, f.position, f.evt), ("n", Position::Coda, 0));
        assert!(f.target_measured >= CODA_MIN_FRAMES, "丢音要带着它本该拿到的目标");
        assert!(rep.unmodelled.is_empty(), "这条路径是建模过的");
        assert_eq!(rep.conservation.0, rep.conservation.1, "注入没有破坏守恒");
    }

    /// ★阳性对照 #2 —— 饥饿轴。把一个辅音压到 1 帧,必须同时报 STARVED 与 BELOW_TRAINING_FLOOR,
    /// 且**不该**报成丢音(它还在)。
    #[test]
    fn audit_positive_control_catches_an_injected_starvation() {
        let d = en_dicts();
        let score = vec![en("S IY1", 16)];
        let resolved = g2p::resolve_score(&score, &d).unwrap();
        let mut broken = build_arrays_daw(&score, &d, ArticulationTiming::Auto).unwrap();
        let s_at = broken.phon.iter().position(|&p| p == "s").unwrap();
        let i_at = broken.phon.iter().position(|&p| p == "i").unwrap();
        broken.phone_dur[i_at] += broken.phone_dur[s_at] - 1;
        broken.phone_dur[s_at] = 1;

        let rep = audit(&score, &resolved, &broken, Source::Live);
        assert_eq!(rep.count(Kind::Dropped), 0, "它还在,不是丢音");
        assert_eq!(rep.count(Kind::Starved), 1);
        assert_eq!(rep.count(Kind::BelowTrainingFloor), 1);
        let f = rep.of_kind(Kind::Starved).next().unwrap();
        assert_eq!((f.phone, f.actual), ("s", 1));
        assert!(f.target_effective >= 4, "s 的长音桶目标是量出来的,不是 2");

        // ★分布轴的阳性对照 —— 变异实测:没有这一条,把与真人分布的对拍**整个关掉**,
        //   全套自检照样绿(别的测试只「允许」它出现,没有一条「要求」它出现)。
        let ood: Vec<_> = rep.of_kind(Kind::OutOfDistribution).collect();
        assert_eq!(ood.len(), 1, "1 帧的 /s/ 必须被真人分布判为出格:\n{}", rep.render(20));
        assert_eq!(ood[0].phone, "s");
        assert!(ood[0].ref_count >= 50, "判决必须带样本量");
        assert!(
            ood[0].target_effective >= TRAINING_CONSONANT_FLOOR,
            "真英语 /s/ 的 p05 不该低于训练管线下限 3(参照表是不是没生成全?): {:?}", ood[0]
        );
        // ★判别器:未被破坏的同一份分配里,`s` 拿到自己的目标 ⇒ 分布轴必须**沉默**。
        let clean = audit_score(&score, &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(
            clean.of_kind(Kind::OutOfDistribution).filter(|f| f.phone == "s").count(), 0,
            "正常时长的 /s/ 被判出格 ⇒ 判据恒亮:\n{}", clean.render(20)
        );
    }

    /// ★已知正确样本 —— 干净的一条 ja 线。
    ///
    /// ⚠**这条测试的第一版是空的**(S92k 对抗审查抓出):它只断言 DROPPED / NUCLEUS_COLLAPSE /
    /// BELOW_TRAINING_FLOOR 为 0,而那三类在这个夹具上**结构上就不可能触发**(无 coda、无 medial、
    /// 无丛、非成丛语言),同时夹具真产出的 STARVED 它一条都没看。
    /// **规矩:「已知正确样本」必须断言 findings 的全集,不能只断言自己挑的几类** ——
    /// 否则它证明的是「我没看的地方没问题」。
    #[test]
    fn audit_reports_only_what_it_should_on_clean_ja() {
        let ja = vec![raw("k a", 12), raw("s a", 12), raw("t a", 12)];
        let rep = audit_score(&ja, &NoDicts, ArticulationTiming::Auto).unwrap();
        assert!(rep.unmodelled.is_empty(), "ja 普通音符必须全部建模: {:?}", rep.unmodelled);
        assert_eq!(rep.conservation.0, rep.conservation.1);
        // 全集断言:每一条 finding 都必须落在这三类**已知机制**上 ——
        //   ①词首辅音没借够(ja 不走 S92c 的补齐,onset 只拿到借帧给的量;谱首那个更是**没有
        //     出借者**,按构造只能落进 2 帧兜底);
        //   ②同一个 onset 也会被分布轴看见 —— 真日语歌手在 12 帧音符上给 `k` 中位 5 帧、下界 3,
        //     我们给 2。**这是真事,不是判据坏**:两条轴用不同参照系看同一个现象(我们自己的目标
        //     vs 真人的分布),它们同时响才是对的;
        //   ③前一个音符的元音供养了下一个词的词首辅音 = Auto 臂的立命之本,不是缺陷。
        // 任何第四类冒出来都说明判据坏了。
        for f in &rep.findings {
            assert!(
                matches!(
                    (f.kind, f.position),
                    (Kind::Starved, Position::Onset)
                        | (Kind::OutOfDistribution, Position::Onset)
                        | (Kind::NucleusLentAway, Position::Nucleus)
                ),
                "ja 干净线冒出了预期外的 finding ⇒ 判据坏了:\n{}", rep.render(20)
            );
        }
        // 分布轴必须带着样本量出现 —— 没有 n 的偏离度不可读。
        for f in rep.of_kind(Kind::OutOfDistribution) {
            assert!(f.ref_count >= 50, "分布判决没带样本量: {f:?}");
        }
        assert_eq!(rep.count(Kind::Dropped), 0, "{}", rep.render(20));
        assert_eq!(rep.count(Kind::NucleusCollapse), 0, "{}", rep.render(20));
        assert_eq!(
            rep.count(Kind::BelowTrainingFloor), 0,
            "训练下限只对成丛语言成立,ja 不该吃这条:\n{}", rep.render(20)
        );
        // ⚠ 元音总损失在 ja 上**本来就非零**(那正是前借机制),所以这里钉的是「它是个数,不是
        //   缺陷计数」—— 钉成 0 就等于把这根轴关掉。
        assert!(rep.vowel_frames_lost() > 0, "ja 的前借机制在跑,这个数不该是 0");
    }

    /// ★★S92k 对抗审查抓到的那条 MAJOR 的回归钉:**核原先没有目标**,于是前借从**邻居音符的
    /// 元音**里抽走的每一帧都结构性不可见 —— 而那正是用户耳判「`hurt` 抢掉 `might`」的形状,
    /// 也正是 S92j 必须先量的那个数。审查给的复现链:三个音符,每个中间音符既借进又借出,
    /// 于是连 `displacement` 轴都被抵消掉,旧版报告是一张满分的白卷。
    #[test]
    fn audit_sees_a_neighbour_vowel_being_borrowed_away() {
        let d = en_dicts();
        let score = vec![en("N OW1", 12), en("S W EY1", 12), en("M AY1", 12)];
        let rep = audit_score(&score, &d, ArticulationTiming::Auto).unwrap();
        assert!(rep.unmodelled.is_empty());
        assert!(
            rep.vowel_frames_lost() > 0,
            "邻居的元音被借走了,元音总损失却是 0 ⇒ 这条轴没接上:\n{}", rep.render(20)
        );
        // ★字面 golden —— 变异实测证明:不钉数字的话,把生产的 `DEEP_LENDER_SHARE` 从 4 改回 2
        //   (即 S92j 修掉的那个退化)整套自检照样全绿 = 这根轴对生产改动无感,等于装饰。
        //   手推:`[n oʊ]@12` → n3 oʊ9,`[s w eɪ]` 借 (9−3).min(9/4)=… 见下方逐条断言。
        assert_eq!(
            rep.vowel_frames_lost(), 9,
            "元音总损失的字面值变了 —— 若是有意改分配规则请更新这个数,并想清楚它该升还是该降:\n{}",
            rep.render(20)
        );
        let vowels: Vec<_> = rep.of_kind(Kind::NucleusLentAway).collect();
        assert!(
            vowels.iter().any(|f| f.evt == 0) && vowels.iter().any(|f| f.evt == 1),
            "链上前两个音符的元音都被借过,必须逐个点名:\n{}", rep.render(20)
        );
        // ★中间那个音符**同时借进又借出**,`displacement` 净额把它抹平了(审查那位质疑者的加强
        //   论据)—— 这条断言把「位移轴不够用、必须有独立的元音轴」钉死。
        let disp1 = rep.displacement.iter().find(|(k, _)| *k == 1).unwrap().1;
        assert!(
            disp1.abs() <= 1 && vowels.iter().any(|f| f.evt == 1 && f.deficit() >= 4),
            "中间音符位移={disp1} 却真的丢了 ≥4 帧元音 —— 这正是位移轴看不见的那一类:\n{}",
            rep.render(20)
        );
        // ★★走深的形状 —— 上面那条链只借到**深度 1**,碰不到 `DEEP_LENDER_SHARE`。变异实测:
        //   只有上面那条时,把生产的深借钳位从 1/4 改回 1/2(= S92j 修掉的那个退化)整套自检
        //   照样全绿 = 这根轴对生产改动无感。这里用 S92d 那个夹具:`d`(2帧)和 `n`(辅音)都给
        //   不出,走到**深度 3** 才够到 `mind` 的 aɪ。
        let deep = vec![en("M AY1 N D", 20), en("S IY1", 16)];
        let rep3 = audit_score(&deep, &d, ArticulationTiming::Auto).unwrap();
        let ai: Vec<_> = rep3.of_kind(Kind::NucleusLentAway).filter(|f| f.phone == "aɪ").collect();
        assert_eq!(ai.len(), 1, "深度 3 的元音出借没被记账:\n{}", rep3.render(20));
        assert_eq!(
            (ai[0].actual, ai[0].deficit()), (6, 2),
            "aɪ 应当 8→6(让出 8/DEEP_LENDER_SHARE=2 帧);若这个数变了,说明深借钳位被改动过:\n{}",
            rep3.render(20)
        );

        // ★判别器:同样三个音符,但中间那个不需要借帧(它以元音起头)⇒ 元音一帧不该少。
        let calm = vec![en("N OW1", 12), en("AY1", 12), en("OW1", 12)];
        let rep2 = audit_score(&calm, &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(
            rep2.vowel_frames_lost(), 0,
            "没有词首辅音要借帧,元音总损失却非 0 ⇒ 判据恒亮:\n{}", rep2.render(20)
        );
    }

    /// ★zh 延音的两条特殊发射路径必须被**建模**,而不是被当成丢音 —— 否则中文轨会淹在假阳里。
    /// 同音高延音 = 帧并进前一条(本事件零音素);变音高延音 = 发一个变形后的载体音素。
    ///
    /// ★夹具形态:**只手工构造 `resolve` 的输出**(那一步才需要 zh 词典,而词典是 gitignored 资产,
    /// 单测不该依赖它),**发射本身交给生产的 `assemble_arrays`**。所以这条仍然是真的漂移检测器 ——
    /// 如果哪天生产改了 zh 延音的发射规则而审计件没跟上,它会红,而不是静默吐假丢音。
    #[test]
    fn audit_models_the_zh_sustain_paths() {
        let zh = |lyric: &'static str, nn: i64| g2p::ScoreEvt {
            lyric, note_num: nn, frames: 20, lang: g2p::Lang::Zh,
            phoneme_input: None, phoneme_set: PhonemeSet::Words,
        };
        let score = vec![zh("x", 60), zh("-", 60), zh("-", 62)];
        let note = |ph: Vec<&'static str>, sustain: bool| g2p::ResolvedNote {
            kind: g2p::ResolvedKind::Phones(ph),
            run_lang: g2p::Lang::Zh,
            is_sustain: sustain,
            nucleus_stress: None,
        };
        // `wang` = [w, uɑŋ](zh 韵母是原子 token);两个延音各自 re-emit 载体。
        let resolved = vec![
            note(vec!["w", "uɑŋ"], false),
            note(vec!["uɑŋ"], true), // 同音高 ⇒ 帧并进前一条,本事件零音素
            note(vec!["uɑŋ"], true), // 变音高 ⇒ 发 zh_hold_phone(载体)
        ];
        let arr = assemble_arrays(&score, &resolved, Assembly::Daw(ArticulationTiming::Auto)).unwrap();
        // 先钉住「生产确实走了那两条特殊路径」——否则这条测试可能在测一个普通发射。
        assert_eq!(arr.evt, vec![0, 0, 2], "同音高延音必须零发射、变音高延音必须发一个");
        assert_ne!(arr.phon[2], "uɑŋ", "变音高延音发的应是变形后的载体(glide 被剥掉)");

        let rep = audit(&score, &resolved, &arr, Source::Live);
        assert!(rep.unmodelled.is_empty(), "zh 延音路径没建模全: {:?}\n{}", rep.unmodelled, rep.render(10));
        assert_eq!(rep.count(Kind::Dropped), 0, "zh 延音被误报成丢音:\n{}", rep.render(10));
        assert_eq!(rep.conservation.0, rep.conservation.1, "守恒");
    }

    /// ★同一条路的反面:延音是**普通** `Phones` 发射的语言(ja)也必须零未建模、零假丢音。
    /// `raw()` 夹具是 `Lang::Ja`,不碰任何词典。
    #[test]
    fn audit_models_ja_sustains() {
        let hold = |nn: i64| g2p::ScoreEvt {
            lyric: "-", note_num: nn, frames: 16, lang: g2p::Lang::Ja,
            phoneme_input: None, phoneme_set: PhonemeSet::Words,
        };
        let score = vec![raw("k a", 16), hold(60), hold(62)];
        let rep = audit_score(&score, &NoDicts, ArticulationTiming::Auto).unwrap();
        assert!(rep.unmodelled.is_empty(), "ja 延音没建模: {:?}\n{}", rep.unmodelled, rep.render(10));
        assert_eq!(rep.count(Kind::Dropped), 0, "ja 延音被误报成丢音:\n{}", rep.render(10));
        assert_eq!(rep.conservation.0, rep.conservation.1);
    }

    /// ★两个目标必须真的分得开 —— S92i 那一类(分配器主动把目标砍掉)只有这个差看得见。
    /// `fr ≤ 5` 且后面没有延音 ⇒ onset 目标被封到 2,而先验说的更长。
    #[test]
    fn audit_separates_policy_cap_from_measured_prior() {
        let d = en_dicts();
        let score = vec![en("TH IH1 NG", 4)];
        let rep = audit_score(&score, &d, ArticulationTiming::Auto).unwrap();
        let capped: Vec<_> = rep.of_kind(Kind::PolicyCapped).collect();
        assert!(!capped.is_empty(), "策略封顶没被报出来 —— S92i 那一类问题就是这么藏住的");
        let f = capped[0];
        assert!(
            f.target_measured > f.target_effective,
            "{}: 先验 {} 必须严格大于 effective {}", f.phone, f.target_measured, f.target_effective
        );
        // ★判别器 —— 证明这条判据真在分流,而不是恒亮:一个**分配器足额发得出**的音符上,
        // POLICY_CAPPED 必须一条都没有。`t` 的 coda 目标只有 3 帧,长音符的预算绰绰有余。
        // ⚠ 这里原本用 `TH IH1 NG`@20 当判别器,S92n 放开 coda 钳位后它**合法地**变红了:
        //   `ŋ` 的先验涨到 10,而 `fr*2/5` 只发得出 8 ⇒ 那道预算第一次以 POLICY_CAPPED 现形。
        //   那正是这条轴该干的事(它是「我们放弃了多少发音时长」的读数),所以换夹具、不放宽断言。
        let uncapped = vec![en("AA1 T", 28)];
        let rep2 = audit_score(&uncapped, &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(
            rep2.count(Kind::PolicyCapped), 0,
            "预算发得出目标的音符却报了封顶 ⇒ 判据恒亮:\n{}", rep2.render(10)
        );
    }

    /// ★分布参照表本身必须**结构自洽** —— 它是生成物,而生成器一旦口径写错,整条「发现未知问题」
    /// 的轴就会安静地给出错数。这条只查形状(值的正确性由生成器的真值链保证:同一份 split、同一份
    /// npz、同一张 id→token 表)。
    #[test]
    fn audit_reference_table_is_wellformed() {
        use super::ref_tbl::PHONE_DUR_DIST as D;
        assert!(D.len() > 200, "参照表太小 —— 是不是只跑了 --limit 冒烟版? n={}", D.len());
        let vocab: std::collections::HashSet<&str> =
            tbl::PHONE_TO_ID.iter().map(|(t, _)| *t).collect();
        let mut seen = std::collections::HashSet::new();
        let mut langs = std::collections::HashSet::new();
        for c in D {
            assert!(seen.insert((c.lang, c.token, c.position, c.bucket)), "重复格 {c:?}");
            assert!(vocab.contains(c.token), "词表外的音素 {c:?}");
            assert!(c.position <= 3 && c.bucket <= 2, "位置/桶越界 {c:?}");
            assert!(c.p05 <= c.p50 && c.p50 <= c.p95, "分位数不单调 {c:?}");
            assert!(c.p05 >= 1, "时长下界必须为正 {c:?}");
            assert!(c.count >= 50, "样本不足的格子不该被发出来(生成器该丢掉它) {c:?}");
            langs.insert(c.lang);
        }
        assert!(langs.contains(""), "缺跨语言池化兜底行");
        for lg in ["zh", "en", "ja"] {
            assert!(langs.contains(lg), "缺 {lg} 的自有格子 —— 语料没读全?");
        }
        // 核(元音)必须在表里 —— 目标表完全没有它们,这正是本表存在的理由之一。
        assert!(D.iter().any(|c| c.position == 2), "参照表没有核 ⇒ 元音轴无从判断");
        assert!(D.iter().any(|c| c.position == 1), "参照表没有 medial");
    }

    /// ★查表的回退与「没量过」必须分得开:本语言的格 → 池化格 → `None`。
    /// `None` **不是**「没问题」,所以它绝不能产生 finding。
    #[test]
    fn audit_reference_lookup_falls_back_then_gives_up() {
        // 一个词表里根本不存在的 token ⇒ 两级都查不到 ⇒ None(而不是拿别人的格子凑合)
        assert!(dist_cell("en", "zzz-not-a-phone", Position::Onset, 20).is_none());
        // 一门参照表里没有自有格子的语言 ⇒ 落到池化行(不是 None)
        let pooled_exists = ref_tbl::PHONE_DUR_DIST.iter().any(|c| c.lang.is_empty());
        assert!(pooled_exists);
        // ★本语言有自己的格子时**必须**用它,不能拿池化行凑合 —— 整个 S92 的根因就是「英语是唯一
        //   付账的语言」,池化行被中文主导,拿它判英语等于把要找的东西平均掉。
        let ja_a = dist_cell("ja", "a", Position::Nucleus, 10).expect("ja/a/nucleus/mid 该有自有格");
        assert_eq!(ja_a.lang, "ja", "回退顺序反了:本语言有格却用了池化行 {ja_a:?}");
        assert!(ja_a.count >= 50);
        // 一门参照表里没有自有格子的语言 ⇒ 落到池化行(而不是 None)
        let de = dist_cell("de", "a", Position::Nucleus, 10);
        assert!(de.is_some(), "缺池化兜底");
    }

    /// ★★MEDIAL 位置的目标必须来自**生产实际发出的分配**,而不是我手抄的公式 —— S92k 对抗审查
    /// 有三路独立报了同一条:我原先手抄了 medial 元音的份额式并读 `note_frames`,而分配器读的是
    /// `spendable`,**InNote 臂上两者不相等** ⇒ 每个带 medial 的音符都会冒出幻影 STARVED。
    /// 变异实测:在补这条测试之前,把目标改回手抄公式,整套自检**全绿** = 那一整个位置零覆盖。
    ///
    /// 夹具 = 一个音符上塞多音节词(medial 就是这么产生的),**两条臂都测**。
    #[test]
    fn audit_medial_targets_come_from_the_real_allocation_on_both_arms() {
        for timing in [ArticulationTiming::Auto, ArticulationTiming::InNote] {
            // [s i f o]:i 是 medial 元音(最后一个核是 o),f 是 medial 辅音。
            let score = vec![raw("s i f o", 20)];
            let rep = audit_score(&score, &NoDicts, timing).unwrap();
            assert!(rep.unmodelled.is_empty(), "{timing:?}: {:?}", rep.unmodelled);
            let arr = build_arrays_daw(&score, &NoDicts, timing).unwrap();
            // 没有邻居可借 ⇒ medial 的实发就等于分配额 ⇒ 一条 medial STARVED 都不该有。
            let medial_starved: Vec<_> = rep
                .findings
                .iter()
                .filter(|f| f.position == Position::Medial && f.kind == Kind::Starved)
                .collect();
            assert!(
                medial_starved.is_empty(),
                "{timing:?} 臂上冒出 medial 幻影饥饿(目标算错了,不是分配器的问题):\n{}\n实际分配 {:?}",
                rep.render(20), arr.phone_dur
            );
        }
    }

    /// ★`score_forced` 必须真的在分流 —— 否则「其中音符本来放得下 N」那一列是装饰。
    /// 同一批音素:短音符上放不下(谱短),长音符上放得下。**两边都仍然被报出来**,只是分组不同。
    #[test]
    fn audit_separates_score_forced_from_allocator_fault() {
        let d = en_dicts();
        // [f aɪ n d] 4 个音素 ⇒ 下限 = 3 + 3*2 = 9 帧。
        let short = audit_score(&[en("F AY1 N D", 8)], &d, ArticulationTiming::Auto).unwrap();
        assert!(
            !short.findings.is_empty() && short.findings.iter().all(|f| f.score_forced),
            "8 帧放不下 4 个音素的下限(9),应当整批标成谱短:\n{}", short.render(10)
        );
        let long = audit_score(&[en("F AY1 N D", 30)], &d, ArticulationTiming::Auto).unwrap();
        assert!(
            long.findings.iter().all(|f| !f.score_forced),
            "30 帧显然放得下,不该标谱短:\n{}", long.render(10)
        );
        // 分组不等于隐藏:两边的 findings 都还在总表里。
        assert_eq!(short.count(Kind::Dropped), short.of_kind(Kind::Dropped).count());
    }

    /// ★S92k 审查抓到的第二条:**被延长的核**(S92b `nucleus_continues`)上 2 帧是生产**明文
    /// 声明安全**的("那 2 帧延续的是模型已经在唱的元音,不是 S84 量到的 2 帧起音"),旧版却把它
    /// 报成塌陷 —— 而这个形状正是英文归韵谱上每个带词尾辅音的 `+` 延音,即本仪器的主战场。
    #[test]
    fn audit_does_not_flag_a_held_nucleus_as_collapsed() {
        // [a] @10 然后 [a n] @4:第二个音符以同一个元音起头 ⇒ nucleus_is_held ⇒ 核可以是 2 帧。
        let score = vec![raw("a", 10), raw("a n", 4)];
        let rep = audit_score(&score, &NoDicts, ArticulationTiming::Auto).unwrap();
        assert!(rep.unmodelled.is_empty());
        assert_eq!(
            rep.count(Kind::NucleusCollapse), 0,
            "被延长的核被误报成塌陷(S92b 明文安全):\n{}", rep.render(20)
        );
        // ★判别器:一个**全新起音**的 2 帧核仍然必须被报出来 —— 否则上面那条等于把检查关掉了。
        //   前一个音符以辅音收尾 ⇒ `nucleus_is_held` 为假 ⇒ 这是真起音。
        let attack = vec![raw("a k", 10), raw("a", 2)];
        let rep2 = audit_score(&attack, &NoDicts, ArticulationTiming::Auto).unwrap();
        assert!(
            rep2.count(Kind::NucleusCollapse) > 0,
            "全新起音的 2 帧核必须仍然被报:\n{}", rep2.render(20)
        );
    }

    /// ★守恒与位移轴 —— 「时长对但落点错」那一类(A: coda-first)唯一看得见的地方。
    #[test]
    fn audit_reports_displacement_and_conservation() {
        let d = en_dicts();
        let score = vec![en("M AY1 N D", 20), en("S IY1", 16)];
        let rep = audit_score(&score, &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(rep.conservation.0, rep.conservation.1, "守恒");
        assert_eq!(rep.displacement.iter().map(|(_, v)| *v).sum::<i64>(), 0, "位移必须零和");
        assert!(rep.displaced_beyond(0) >= 2, "前借发生了,位移轴却是平的 = 轴没接上");
    }

    /// ★仪器必须看得见 S92 那批真实缺陷的**形状**。这里用 S92 立案时的原始算术:
    /// `fined` = [f aɪ n d] @ 10 帧,修复前 /n/ 整个消失(唱成 "fide")。今天它不该再丢,
    /// 所以断言的是**双向**的:要么它在泳道里,要么审计报了它 —— 绝不会静默消失。
    /// 再把它人工挖掉,审计必须立刻点名 `n` —— 这条才是「仪器看得见这一类」的证明。
    #[test]
    fn audit_sees_the_s92_cluster_shape() {
        let d = en_dicts();
        let score = vec![en("F AY1 N D", 10)];
        let resolved = g2p::resolve_score(&score, &d).unwrap();
        let arr = build_arrays_daw(&score, &d, ArticulationTiming::Auto).unwrap();
        let rep = audit(&score, &resolved, &arr, Source::Live);
        assert!(rep.unmodelled.is_empty());
        assert!(
            arr.phon.contains(&"n") || rep.of_kind(Kind::Dropped).any(|f| f.phone == "n"),
            "词尾 /n/ 既不在泳道里也没被报出来 = 仪器瞎了"
        );
        let mut broken = build_arrays_daw(&score, &d, ArticulationTiming::Auto).unwrap();
        let at = broken.phon.iter().position(|&p| p == "n").expect("S92 之后 n 应当还在");
        let freed = broken.phone_dur[at];
        broken.phon.remove(at);
        broken.phone_dur.remove(at);
        broken.evt.remove(at);
        let nuc = broken.phon.iter().position(|&p| p == "aɪ").unwrap();
        broken.phone_dur[nuc] += freed;
        let rep2 = audit(&score, &resolved, &broken, Source::Live);
        assert!(
            rep2.of_kind(Kind::Dropped).any(|f| f.phone == "n"),
            "把 S92 修的那个洞挖回去,审计必须点名 n:\n{}", rep2.render(10)
        );
    }
}
