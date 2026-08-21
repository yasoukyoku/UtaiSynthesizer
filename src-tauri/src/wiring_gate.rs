//! Shared parts for **wiring gates** — the test-only checks that read production SOURCE and ask
//! "is this call actually there", as opposed to driving the function.
//!
//! ⛔ Why this module exists (S141): these two helpers were written for
//! `models::tests::s120_every_attachment_installer_asks_about_the_vocoder` and lived private to
//! that test module. `src-tauri` already carries **six** hand-rolled comment strippers in as many
//! test modules; adding a seventh for the audition tripwire would have been the drift this repo
//! keeps paying for. They are moved here verbatim, with their blood lessons attached, and every
//! wiring gate calls the same two.
//!
//! ⚠ Honest boundary, stated here rather than implied: a wiring gate proves an identifier appears
//! **in code** inside a function. It cannot prove the call is on a reachable path, nor that its
//! arguments are right. Those need a behaviour test — see the pairs in `models/mod.rs`
//! (`s120_…asks_about_the_vocoder` + `s120_package_import_surfaces_the_vocoder_hint`) and in
//! `commands/audition.rs` (`every_audition_command_…` + `the_audition_tripwire_…`).

/// Strip comments before asking "is this identifier wired in".
///
/// ⛔ S119 blood lesson, and the reason this helper exists at all: a bare
/// `src.contains("enable_version_counter=False")` stayed GREEN when the probe commented that
/// line out — the substring lives in the corpse too. A wiring assertion that cannot tell code
/// from a comment is not an assertion.
///
/// Full-line `//` goes; a trailing `//` goes only on lines with no `"` (so a `//` inside a
/// string literal is never mistaken for a comment). That is deliberately conservative: the
/// mutation this must survive is "comment the call out", which always produces a full-line
/// comment.
pub(crate) fn strip_comments_for_wiring(src: &str) -> String {
    src.lines()
        .map(|l| {
            let t = l.trim_start();
            if t.starts_with("//") {
                return "";
            }
            if !l.contains('"') {
                if let Some(i) = l.find("//") {
                    return &l[..i];
                }
            }
            l
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Split a Rust source into (fn name, fn body-ish chunk) pairs by `fn` headers.
/// Crude on purpose: a chunk runs to the next `fn` header, which is all a wiring check needs.
///
/// ⚠ A chunk therefore ENDS with whatever precedes the next header — the next function's
/// attributes (`#[tauri::command]`, doc comments) land in the PREVIOUS chunk. Key a gate on the
/// signature (which is at the head of its own chunk), never on the attribute above it.
pub(crate) fn split_by_fn(code: &str) -> Vec<(String, String)> {
    const HEADS: [&str; 5] = ["fn ", "pub fn ", "pub(crate) fn ", "async fn ", "pub async fn "];
    let mut out: Vec<(String, String)> = Vec::new();
    let mut name = String::from("<preamble>");
    let mut body = String::new();
    for line in code.lines() {
        let t = line.trim_start();
        if HEADS.iter().any(|h| t.starts_with(h)) {
            out.push((std::mem::take(&mut name), std::mem::take(&mut body)));
            let after = t.split("fn ").nth(1).unwrap_or("");
            name = after
                .split(|c| c == '(' || c == '<')
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
        }
        body.push_str(line);
        body.push('\n');
    }
    out.push((name, body));
    out
}

/// The signature region of a chunk produced by [`split_by_fn`]: from the `fn` header through the
/// line that opens the body.
///
/// ⛔ Why not `body.lines().take(N)`: a fixed window is a SILENT truncation. The day a command
/// grows a longer parameter list, a gate keyed on "is `workspace: String` in the first 14 lines"
/// simply stops seeing that command — it goes green by not looking, which is the failure mode
/// every gate in this repo is written to avoid.
pub(crate) fn signature_of(body: &str) -> String {
    let mut sig = String::new();
    for line in body.lines() {
        sig.push_str(line);
        sig.push(' ');
        if line.contains('{') {
            break;
        }
    }
    sig
}

/// S159 —— **日志字面量里不许出现 CJK。**
///
/// 返回 `(相对路径, 行号, 命中的那几个汉字)`。`roots` 是要扫的目录。
///
/// ## ⛔ 它为什么是一条闸而不是一条规矩
/// 仓里早就有「Rust 不硬编用户可见串」这条规矩,而日志这一族**一直是英文的** ——
/// 全后端只有 4 处例外,其中 3 处是同一场(S159)一口气写进去的。⇒ 失效点不在「不知道规矩」,
/// 在**动手那一刻没有任何东西拦我**(与记忆钩子那条血训同形)。
///
/// ## 边界(明写,别当它比实际强)
/// * 它只看 `info!/warn!/error!/debug!/trace!`(含 `tracing::` 前缀)这五个宏;
/// * 判据是**注释剥掉之后**的宏体 —— 中文**注释**照写不误,那正是这个仓的风格;
/// * 它证明不了英文写得好,只证明没有汉字。
fn cjk_in_log_macros(roots: &[std::path::PathBuf]) -> Vec<(String, usize, String)> {
    const MACROS: [&str; 5] = ["info!(", "warn!(", "error!(", "debug!(", "trace!("];
    let mut hits = Vec::new();
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push(p);
            }
        }
    }
    for r in roots {
        walk(r, &mut files);
    }
    files.sort();
    for f in files {
        let Ok(raw) = std::fs::read_to_string(&f) else { continue };
        let code = strip_comments_for_wiring(&raw);
        let lines: Vec<&str> = code.lines().collect();
        let mut i = 0usize;
        while i < lines.len() {
            let t = lines[i].trim_start();
            let is_macro = MACROS.iter().any(|m| {
                t.starts_with(m) || t.starts_with(&format!("tracing::{m}"))
            });
            if !is_macro {
                i += 1;
                continue;
            }
            // 吃到括号配平为止 —— 这些宏几乎全是多行的。
            let (mut open, mut close, mut j) = (0usize, 0usize, i);
            let mut body = String::new();
            loop {
                open += lines[j].matches('(').count();
                close += lines[j].matches(')').count();
                body.push_str(lines[j]);
                body.push('\n');
                if open <= close || j + 1 >= lines.len() {
                    break;
                }
                j += 1;
            }
            let han: String = body.chars().filter(|c| ('\u{4e00}'..='\u{9fff}').contains(c)).collect();
            if !han.is_empty() {
                let rel = f.file_name().map_or_else(|| f.display().to_string(), |n| {
                    format!("{}/{}", f.parent().and_then(|p| p.file_name()).map_or(String::new(), |p| p.to_string_lossy().into()), n.to_string_lossy())
                });
                hits.push((rel, i + 1, han.chars().take(24).collect()));
            }
            i = j + 1;
        }
    }
    hits
}


#[cfg(test)]
mod tests {
    use super::*;

    /// S89: a checker is a program too, and this one is load-bearing for four gates. Feed it a
    /// sample whose right answer is not in dispute before trusting anything it says.
    ///
    /// Each assertion below names a way the stripper can be wrong that ONLY it catches:
    /// the full-line arm (the mutation gates must survive), the string-literal arm (a `//` inside
    /// a URL or a doc string is not a comment), and the trailing arm.
    #[test]
    fn the_comment_stripper_can_tell_code_from_its_corpse() {
        let src = "\
let a = call_me();
// let b = call_me();
    // let c = call_me();
let d = 1; // call_me()
let e = \"https://example.com/call_me()\";";
        let out = strip_comments_for_wiring(src);

        assert_eq!(
            out.matches("call_me()").count(),
            2,
            "expected the live call and the one inside the string literal to survive, and the \
             three commented ones to go. Got:\n{out}"
        );
        assert!(out.contains("let a = call_me();"), "the live call was eaten");
        assert!(
            out.contains("https://example.com/call_me()"),
            "a `//` inside a string literal was mistaken for a comment — every URL in the file \
             would truncate and a gate could go green on a half-read line"
        );
        assert!(
            !out.contains("let b") && !out.contains("let c"),
            "a commented-out call survived: this is exactly the S119 hole"
        );
        assert!(!out.contains("let d = 1; //"), "the trailing comment survived");
    }

    /// The splitter's own non-vacuity: it must return the chunks a gate then iterates, and it must
    /// put a signature at the HEAD of its own chunk (the property `signature_of` relies on).
    #[test]
    fn the_splitter_puts_each_signature_at_the_head_of_its_own_chunk() {
        let src = "\
use x;

#[tauri::command]
pub async fn alpha(
    state: State,
    workspace: String,
) -> Result<(), String> {
    guard(&workspace)?;
    Ok(())
}

fn beta(a: u8) -> u8 { a }";
        let fns = split_by_fn(&strip_comments_for_wiring(src));
        let names: Vec<&str> = fns.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["<preamble>", "alpha", "beta"]);

        let alpha = &fns.iter().find(|(n, _)| n == "alpha").unwrap().1;
        assert!(
            signature_of(alpha).contains("workspace: String"),
            "the signature region must carry the parameter list — a gate keyed on a parameter \
             would see nothing"
        );
        assert!(
            !signature_of(alpha).contains("guard(&workspace)"),
            "the signature region ran past the opening brace and into the body — a gate keyed on \
             the signature would start matching body text"
        );
        // ⚠ The attribute of `alpha` belongs to the PREAMBLE chunk, not to alpha's. This is the
        // property that makes "key on the attribute" wrong, and it is asserted so that a future
        // change to the splitter cannot quietly invalidate the gates that rely on it.
        assert!(fns[0].1.contains("#[tauri::command]"));
        assert!(!alpha.contains("#[tauri::command]"));
        // A one-line body must not swallow the next function.
        assert!(fns.iter().any(|(n, b)| n == "beta" && b.contains("-> u8 { a }")));
    }

    /// S159 —— ⛔ **后端日志里不许出现汉字。**
    ///
    /// 全后端本来只有 **1 处**例外(`tpool.rs` 的「续训」),而 S159 一场就写进去 **3 处** ——
    /// 因为这条规矩只活在一份记忆文件里,**动手那一刻没有任何东西拦我**(与「钩子存在但是不读」
    /// 那条血训同形:失效点不在知不知道)。
    ///
    /// ⚠ 边界写在 [`cjk_in_log_macros`] 的 doc 里:它只看那五个宏、只看**注释剥掉之后**的宏体
    /// (中文**注释**是这个仓的风格,照写),而且它证明不了英文写得好,只证明没有汉字。
    #[test]
    fn no_chinese_in_backend_log_lines() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let hits = cjk_in_log_macros(&[root.join("src"), root.join("crates")]);
        assert!(
            hits.is_empty(),
            "日志宏里出现了汉字({} 处)—— 后端日志一律英文,中文只写在注释里:\n{}",
            hits.len(),
            hits.iter()
                .map(|(f, n, h)| format!("  {f}:{n}  {h}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    /// ⛔ 上面那条的**非空性**:它得真的分得出「宏体里的汉字」与「注释里的汉字」,
    /// 否则「零命中」既可能是干净、也可能是这把尺子瞎。
    ///
    /// ⚠ 用临时目录而不是硬编一段字符串对着私有函数跑 —— 那个函数吃的是**路径**,
    /// 拿字符串测等于测了另一个东西。
    #[test]
    fn the_chinese_log_gate_can_tell_a_log_line_from_a_comment() {
        let dir = std::env::temp_dir().join(format!("utai_cjk_gate_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // ⑴ 阴性:汉字全在注释里 + 一条纯英文日志 ⇒ 必须零命中。
        std::fs::write(
            dir.join("clean.rs"),
            "// 这一行是中文注释,照写不误\nfn a() {\n    tracing::info!(\"range-extend: all good\");\n}\n",
        )
        .unwrap();
        assert!(
            cjk_in_log_macros(&[dir.clone()]).is_empty(),
            "注释里的汉字被当成了日志 —— 这把尺子会把整个仓库判红"
        );
        // ⑵ 阳性:一条**跨行**的日志宏里塞汉字 ⇒ 必须抓到(生产里那三处正是多行的)。
        std::fs::write(
            dir.join("dirty.rs"),
            "fn b() {\n    tracing::warn!(\n        \"range-extend: 窗被忽略 {}\",\n        1\n    );\n}\n",
        )
        .unwrap();
        let hits = cjk_in_log_macros(&[dir.clone()]);
        assert_eq!(hits.len(), 1, "跨行的日志宏没被抓到:{hits:?}");
        assert!(hits[0].2.contains('窗'), "抓到了但没报出是哪几个字:{hits:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
