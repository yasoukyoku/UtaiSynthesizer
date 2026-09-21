import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * 训练页 第 1 段 · 项目列表(S76 批 4)。
 *
 * 在此之前训练页没有「项目」这个概念:身份完全由用户敲进去的模型名承担,归属靠同名推断,
 * 关掉 app 之后盘上有什么只能靠猜。这一页是那件事的反面 —— 磁盘上有什么就列什么。
 *
 * 样式基底一律复用既有的(NO-Duplication):列表/卡片=`.msst-model-list` 族(资源管理器),
 * 搜索框=`.log-search`(全项目唯一先例,日志面板),筛选 chip=`.msst-filter`。只借类名,
 * **不借文案层** —— MsstModelManager 里是 33 处 `lang==="zh"?…:…` 的双语三元(没有日文),
 * 这一页的文案全部走 i18n JSON 三语。
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { invoke } from "@tauri-apps/api/core";
import { useAppStore } from "../../store/app";
import { useTrainingStore } from "../../store/training";
import { backendErrorMessage } from "../../lib/backendError";
import { fmtSize } from "../../lib/constants";
import "./TrainingProjects.css";
/** The same set `clean_project_name` (tproject.rs) refuses. Tested by CODE POINT rather than
 *  with a character-class regex: a literal control character inside a regex is invisible in
 *  every diff and every review, which is exactly how one gets silently edited into something
 *  else. */
function hasControlChar(s) {
    for (const ch of s) {
        const c = ch.codePointAt(0) ?? 0;
        if (c < 0x20 || c === 0x7f)
            return true;
    }
    return false;
}
/** Architecture ids are NOT localized, on purpose: they are the same proper nouns the training
 *  page and the storage page already print verbatim (`famLabel`, Settings.tsx), and inventing a
 *  translated vocabulary here would make it the SECOND one. */
function famLabel(f) {
    return f;
}
export function ProjectsStep() {
    const { t } = useTranslation();
    const showConfirm = useAppStore((s) => s.showConfirm);
    const showToast = useAppStore((s) => s.showToast);
    // Subscribe narrowly: the other steps take the WHOLE store, so every training-step event
    // already re-renders them. This list must not join that crowd.
    const enterProject = useTrainingStore((s) => s.enterProject);
    const [rows, setRows] = useState([]);
    const [loaded, setLoaded] = useState(false);
    const [query, setQuery] = useState("");
    const [sort, setSort] = useState("recent");
    /** `measure` walks every project directory (tens of GB) — the page paints from the cache
     *  first and asks for real figures afterwards, so opening it is never a stall. */
    const load = useCallback(async (measure) => {
        try {
            const list = await invoke("list_training_projects", { refresh: measure });
            setRows(list);
            return list;
        }
        catch (e) {
            console.error("list_training_projects failed", e);
            return [];
        }
        finally {
            setLoaded(true);
        }
    }, []);
    useEffect(() => {
        let alive = true;
        void (async () => {
            const cached = await load(false);
            if (!alive)
                return;
            // Re-walk only when the figures could plausibly have changed. This component re-mounts on
            // every trip back to the list, and a full `dir_size` sweep of tens of GB per visit would
            // be pure waste — while never re-walking would show yesterday's sizes after a training run.
            // A minute is longer than any navigation round-trip and shorter than anything that writes
            // GBs. (`computedMs === 0` = never measured, which always measures.)
            const newest = cached.reduce((m, p) => Math.max(m, p.computedMs), 0);
            if (Date.now() - newest > 60_000)
                await load(true);
        })();
        return () => {
            alive = false;
        };
    }, [load]);
    const visible = useMemo(() => {
        const q = query.trim().toLowerCase();
        const matched = q
            ? rows.filter((p) => p.name.toLowerCase().includes(q) || p.note.toLowerCase().includes(q))
            : rows;
        const sorted = [...matched];
        if (sort === "name")
            sorted.sort((a, b) => a.name.localeCompare(b.name));
        else if (sort === "size")
            sorted.sort((a, b) => b.totalBytes - a.totalBytes);
        else
            sorted.sort((a, b) => b.updatedMs - a.updatedMs);
        return sorted;
    }, [rows, query, sort]);
    /** Reject client-side what the backend would reject anyway — a dialog that closes and THEN
     *  toasts an error is a worse way to learn you typed a duplicate name. The backend keeps its
     *  own checks: this one cannot see a project another instance just created. */
    const nameProblem = useCallback((value, exceptId) => {
        const v = value.trim();
        if (!v)
            return t("backend.PROJECT_NAME_EMPTY");
        if ([...v].length > 80)
            return t("backend.PROJECT_NAME_TOO_LONG");
        if (hasControlChar(v))
            return t("backend.PROJECT_NAME_INVALID");
        if (rows.some((p) => !p.missing && p.id !== exceptId && p.name === v)) {
            return t("backend.PROJECT_NAME_EXISTS");
        }
        return null;
    }, [rows, t]);
    const onCreate = async () => {
        const name = await showConfirm({
            title: t("training.projectNewTitle"),
            body: t("training.projectNewBody"),
            buttons: [
                { id: "ok", label: t("training.projectCreate"), kind: "primary" },
                // "__cancel": with input mode the PRIMARY resolves the typed VALUE while other buttons
                // resolve their id, so a project literally named "cancel" would otherwise read as one.
                { id: "__cancel", label: t("training.cancel") },
            ],
            input: { placeholder: t("training.projectNamePlaceholder"), invalid: (v) => nameProblem(v) },
        });
        if (!name || name === "__cancel")
            return;
        try {
            const id = await invoke("create_training_project", { name, note: "" });
            await load(false);
            enterProject(id);
        }
        catch (e) {
            showToast(backendErrorMessage(e) ?? String(e), "error");
        }
    };
    const onRename = async (p) => {
        const name = await showConfirm({
            title: t("training.projectRenameTitle"),
            body: t("training.projectRenameBody"),
            buttons: [
                { id: "ok", label: t("training.projectRename"), kind: "primary" },
                { id: "__cancel", label: t("training.cancel") },
            ],
            input: { initial: p.name, invalid: (v) => nameProblem(v, p.id) },
        });
        if (!name || name === "__cancel")
            return;
        try {
            await invoke("update_training_project", { projectId: p.id, name, note: p.note });
            await load(false);
        }
        catch (e) {
            showToast(backendErrorMessage(e) ?? String(e), "error");
        }
    };
    /** Only ever offered for a MISSING row, and it touches nothing on disk — by definition there
     *  is nothing there. Deleting a real project stays in Settings → 存储占用, where the three
     *  destructive training actions (and their guards) already live. */
    const onForget = async (p) => {
        const choice = await showConfirm({
            title: t("training.projectForgetTitle"),
            body: t("training.projectForgetBody", { name: p.name }),
            buttons: [
                { id: "cancel", label: t("training.cancel") },
                { id: "forget", label: t("training.projectForget"), kind: "danger" },
            ],
        });
        if (choice !== "forget")
            return;
        try {
            await invoke("forget_training_project", { projectId: p.id });
            await load(false);
        }
        catch (e) {
            showToast(backendErrorMessage(e) ?? String(e), "error");
        }
    };
    return (_jsxs("div", { className: "tproj-page", children: [_jsxs("div", { className: "tproj-toolbar", children: [_jsx("input", { className: "log-search", type: "text", value: query, placeholder: t("training.projectSearch"), onChange: (e) => setQuery(e.target.value) }), _jsx("div", { className: "msst-filter tproj-sort", children: ["recent", "name", "size"].map((k) => (_jsx("button", { className: sort === k ? "active" : "", onClick: () => setSort(k), children: t(`training.projectSort.${k}`) }, k))) }), _jsx("button", { className: "training-btn primary", onClick: () => void onCreate(), children: t("training.projectNew") })] }), loaded && rows.length === 0 && (_jsx("div", { className: "training-empty", children: t("training.projectEmpty") })), loaded && rows.length > 0 && visible.length === 0 && (_jsx("div", { className: "training-empty", children: t("training.projectNoMatch") })), _jsx("div", { className: "msst-model-list tproj-list", children: visible.map((p) => (_jsxs("div", { className: `msst-model-card-wrap ${p.missing ? "tproj-gone" : ""}`, children: [(p.missing || p.needsAttention !== "TRAINING_LAYOUT_MIGRATION_PENDING") && (_jsx("div", { className: "msst-model-card-slide", children: p.missing ? (_jsx("button", { title: t("training.projectForget"), onClick: () => void onForget(p), children: "\u2715" })) : (_jsx("button", { title: t("training.projectRename"), onClick: () => void onRename(p), children: "\u270E" })) })), _jsxs("button", { className: "msst-model-card tproj-card", 
                            // A project whose layout migration has not run yet has no `project.json`, so the
                            // detail page could only greet it with PROJECT_META_UNREADABLE. It is listed —
                            // never silently dropped — but not enterable until the next launch folds it.
                            disabled: p.missing || p.needsAttention === "TRAINING_LAYOUT_MIGRATION_PENDING", onClick: () => {
                                // The card carries selectable text (project name / note / paths are exactly
                                // what goes into a bug report). Finishing a drag-select inside a <button>
                                // still fires its click, so without this, highlighting a name navigates away
                                // and takes the selection with it.
                                if (window.getSelection()?.toString())
                                    return;
                                enterProject(p.id);
                            }, children: [_jsxs("div", { className: "model-card-header", children: [_jsx("span", { className: "tproj-name", children: p.name }), _jsx("span", { className: "tproj-size", children: p.computedMs > 0 ? fmtSize(p.totalBytes) : "—" })] }), _jsxs("div", { className: "tproj-meta", children: [p.families.length > 0 ? (p.families.map((f) => (_jsx("span", { className: "tproj-fam", children: famLabel(f) }, f)))) : (_jsx("span", { className: "tproj-fam empty", children: t("training.projectNoSlots") })), _jsx("span", { className: "tproj-dot", children: "\u00B7" }), _jsx("span", { children: p.hasDataset
                                                ? t("training.projectHasData", { size: fmtSize(p.datasetBytes) })
                                                : t("training.projectNoData") })] }), p.note && _jsx("div", { className: "tproj-note", children: p.note }), p.needsAttention && (
                                // The value is a stable CODE — run it through the shared mapper first so
                                //「待迁移」and「判不出架构」read as the different situations they are; the
                                // generic line is the fallback for a code that has no text of its own.
                                _jsx("div", { className: "tproj-flag", children: backendErrorMessage(p.needsAttention) ?? t("training.projectNeedsAttention") })), p.missing && _jsx("div", { className: "tproj-flag", children: t("training.projectMissing") })] })] }, p.id))) })] }));
}
