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
}
