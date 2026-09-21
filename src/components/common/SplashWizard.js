import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useEffect, useState, useCallback, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useTranslation } from "react-i18next";
import { List } from "react-window";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useHistoryStore } from "../../store/history";
import { fitTimelineToContent } from "../../lib/timeline/fitView";
import { newProjectFile, openProjectFromPath, restoreAutosave } from "../../lib/project/projectFile";
import { readAutosave, clearAutosave, setRecoveryPending, markAutosaveBaseline } from "../../lib/project/autosave";
import { getRecentProjects, removeRecentProject, updateRecentProjectName } from "../../lib/project/recentProjects";
import { isTauri } from "../../lib/tauri";
import "./Wizard.css";
/** 折叠阈值：右侧只直接展示前 N 条，其余收进「查看全部」弹窗。 */
const MAX_VISIBLE_RECENT = 4;
/** 虚拟列表行高(px)：卡片 58 + 行间距 8，必须与 .startpage-vrow/.startpage-recent-item 的盒模型一致。 */
const VROW_HEIGHT = 66;
/** 虚拟列表视口最大高度(px)。List 需要一个确定高度(父级只有 max-height,
 *  给 flex:1 + height:100% 会塌成 0),所以按条目数算出确定像素高度。 */
const VLIST_MAX_HEIGHT = 396;
function fmtDate(at) {
    return new Date(at).toLocaleString(undefined, {
        year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
    });
}
/** react-window 行组件：style 由 List 注入(绝对定位),必须原样透传到最外层节点。 */
function RecentRow({ index, style, items, render }) {
    const item = items[index];
    if (!item)
        return null;
    return (_jsx("div", { className: "startpage-vrow", style: style, children: render(item) }));
}
/**
 * 启动欢迎页：左边是新建区域，右边是历史记录/保存的工程
 * 简洁设计，移除示例工程卡片
 */
export function SplashWizard({ onClose }) {
    const { t } = useTranslation();
    const projectName = useProjectStore((s) => s.name);
    const [recent, setRecent] = useState(() => getRecentProjects());
    const [autosave, setAutosave] = useState(null);
    const [openingPath, setOpeningPath] = useState(null);
    const [busy, setBusy] = useState(false);
    const [contextMenu, setContextMenu] = useState({ visible: false, x: 0, y: 0, project: null });
    const [renaming, setRenaming] = useState(null);
    const [showMore, setShowMore] = useState(false);
    const contextMenuRef = useRef(null);
    // 启动时读 autosave
    useEffect(() => {
        let cancelled = false;
        void (async () => {
            const env = await readAutosave();
            if (cancelled || !env)
                return;
            setAutosave({
                filePath: env.filePath,
                name: env.name,
                savedAt: env.savedAt,
                dirty: true,
            });
            setRecoveryPending(true);
        })();
        return () => { cancelled = true; };
    }, []);
    // 工程被别处加载时自动关闭
    useEffect(() => {
        const onLoaded = () => { setRecoveryPending(false); onClose(); };
        window.addEventListener("utai:project-loaded", onLoaded);
        return () => { window.removeEventListener("utai:project-loaded", onLoaded); };
    }, [onClose]);
    // 点击外部关闭右键菜单
    useEffect(() => {
        const handleClickOutside = (e) => {
            if (contextMenuRef.current && !contextMenuRef.current.contains(e.target)) {
                setContextMenu({ visible: false, x: 0, y: 0, project: null });
            }
        };
        if (contextMenu.visible) {
            document.addEventListener("mousedown", handleClickOutside);
            return () => document.removeEventListener("mousedown", handleClickOutside);
        }
    }, [contextMenu.visible]);
    /** 新建空白工程，直接进入主界面 */
    const pickBlank = useCallback(async () => {
        if (busy)
            return;
        setBusy(true);
        try {
            const s = useProjectStore.getState();
            const isAlreadyEmpty = (!s.name || s.name === "Untitled") && s.tracks.length === 0 && !s.dirty;
            if (isAlreadyEmpty) {
                useAppStore.getState().clearSelection();
                useHistoryStore.getState().reset();
                useHistoryStore.getState().markSaved();
                fitTimelineToContent();
                if (isTauri())
                    await markAutosaveBaseline();
                setRecoveryPending(false);
                onClose();
            }
            else {
                if (isTauri()) {
                    await newProjectFile();
                }
                // 浏览器环境下直接关闭欢迎页
                setRecoveryPending(false);
                onClose();
            }
        }
        catch (e) {
            // 不能静默失败：否则欢迎页既不关闭也无任何提示，用户会以为按钮坏了。
            useAppStore.getState().showToast(e instanceof Error ? e.message : String(e), "error");
        }
        finally {
            setBusy(false);
        }
    }, [busy, onClose]);
    const openRecent = useCallback(async (p) => {
        if (!isTauri() || busy || openingPath)
            return;
        setBusy(true);
        setOpeningPath(p.path);
        try {
            const exists = await invoke("path_exists", { path: p.path });
            if (!exists) {
                removeRecentProject(p.path);
                setRecent(getRecentProjects());
                useAppStore.getState().showToast(t("splash.recentMissing"), "warning");
                return;
            }
            const ok = await openProjectFromPath(p.path);
            if (ok) {
                setRecoveryPending(false);
                onClose();
            }
        }
        catch (e) {
            useAppStore.getState().showToast(e instanceof Error ? e.message : String(e), "error");
        }
        finally {
            setOpeningPath(null);
            setBusy(false);
        }
    }, [busy, openingPath, onClose, t]);
    // 右键菜单：删除
    const handleDeleteProject = useCallback((path) => {
        removeRecentProject(path);
        setRecent(getRecentProjects());
        setContextMenu({ visible: false, x: 0, y: 0, project: null });
        useAppStore.getState().showToast(t("splash.recentRemoved"), "info");
    }, [t]);
    // 右键菜单：重命名
    const handleRenameProject = useCallback((project) => {
        setRenaming({ path: project.path, name: project.name, original: project.name });
        setContextMenu({ visible: false, x: 0, y: 0, project: null });
    }, []);
    const handleRenameSubmit = useCallback(() => {
        if (!renaming)
            return;
        const newName = renaming.name.trim();
        // 对比进入编辑时的名字(original),不是文件路径 —— 否则任何改名都会被判定为"有变化"/永远写入。
        if (newName && newName !== renaming.original) {
            updateRecentProjectName(renaming.path, newName);
            setRecent(getRecentProjects());
            useAppStore.getState().showToast(t("splash.renamed"), "success");
        }
        setRenaming(null);
    }, [renaming, t]);
    const handleRenameCancel = useCallback(() => {
        setRenaming(null);
    }, []);
    // 显示右键菜单
    const handleContextMenu = useCallback((e, project) => {
        e.preventDefault();
        e.stopPropagation();
        const menuWidth = 160;
        const menuHeight = 80;
        let x = e.clientX;
        let y = e.clientY;
        if (x + menuWidth > window.innerWidth) {
            x = window.innerWidth - menuWidth - 10;
        }
        if (y + menuHeight > window.innerHeight) {
            y = window.innerHeight - menuHeight - 10;
        }
        setContextMenu({
            visible: true,
            x,
            y,
            project,
        });
    }, []);
    // autosave 恢复
    const recoverAutosave = useCallback(async () => {
        if (!autosave || busy)
            return;
        setBusy(true);
        try {
            const env = await readAutosave();
            if (!env) {
                setAutosave(null);
                setRecoveryPending(false);
                return;
            }
            await restoreAutosave(env);
            setAutosave(null);
            setRecoveryPending(false);
            onClose();
        }
        catch (e) {
            useAppStore.getState().showToast(e instanceof Error ? e.message : String(e), "error");
        }
        finally {
            setBusy(false);
        }
    }, [autosave, busy, onClose]);
    const discardAutosave = useCallback(async () => {
        if (!autosave || busy)
            return;
        setBusy(true);
        try {
            await clearAutosave();
            setAutosave(null);
            setRecoveryPending(false);
            useAppStore.getState().showToast(t("splash.autosaveDiscarded"), "info");
        }
        catch (e) {
            // 与 recoverAutosave 对齐：缺 catch 会变成未处理的 Promise rejection。
            useAppStore.getState().showToast(e instanceof Error ? e.message : String(e), "error");
        }
        finally {
            setBusy(false);
        }
    }, [autosave, busy, t]);
    const title = projectName && projectName !== "Untitled"
        ? `${t("splash.titleWithName")} ${projectName}`
        : t("splash.title");
    /** 用 path 而非 index 作为虚拟行 key：删除/重命名后行内状态不会错位。 */
    const rowKey = useCallback((index, { items }) => items[index]?.path ?? index, []);
    /** 弹窗内虚拟列表的实际高度(px)：取 (条目数 × 行高) 和 (视口上限) 中较小值。 */
    const vlistHeight = Math.min(recent.length * VROW_HEIGHT, VLIST_MAX_HEIGHT);
    /** 渲染单个历史工程卡片（主列表与「查看全部」弹窗共用）。 */
    const renderRecentItem = useCallback((p) => {
        const isCrashed = autosave?.filePath === p.path;
        const isRenaming = renaming?.path === p.path;
        const isOpening = openingPath === p.path;
        return (_jsxs("button", { className: `startpage-recent-item ${isOpening ? "opening" : ""} ${isRenaming ? "renaming" : ""} ${isCrashed ? "crashed" : ""}`, onClick: () => { if (!isRenaming)
                void openRecent(p); }, onContextMenu: (e) => handleContextMenu(e, p), disabled: busy, title: p.path, children: [_jsx("span", { className: "startpage-recent-icon", children: isCrashed ? "⚠" : "🎼" }), isRenaming ? (_jsx("input", { type: "text", className: "startpage-rename-input", value: renaming.name, onChange: (e) => setRenaming({ ...renaming, name: e.target.value }), onKeyDown: (e) => {
                        if (e.key === "Enter")
                            handleRenameSubmit();
                        if (e.key === "Escape")
                            handleRenameCancel();
                    }, onBlur: handleRenameSubmit, autoFocus: true, onClick: (e) => e.stopPropagation() })) : (_jsxs("span", { className: "startpage-recent-info", children: [_jsxs("span", { className: "startpage-recent-name", children: [p.name, isCrashed && _jsx("span", { className: "startpage-recent-crash-badge", children: t("splash.crashBadge") })] }), _jsxs("span", { className: "startpage-recent-meta", children: [_jsx("span", { children: fmtDate(p.at) }), _jsx("span", { className: "startpage-recent-path", children: p.path })] })] }))] }, p.path));
    }, [autosave?.filePath, renaming, openingPath, busy, t, handleContextMenu, openRecent, handleRenameSubmit, handleRenameCancel]);
    return (_jsxs("div", { className: "startpage-backdrop", onClick: (e) => e.preventDefault(), children: [_jsxs("div", { className: "startpage-wrap", children: [_jsxs("div", { className: "startpage-brand-header", children: [_jsx("img", { className: "startpage-brand-logo", src: "/logo-user.png", alt: "MunoAI" }), _jsx("div", { className: "startpage-brand-name", children: "MunoAI" }), _jsx("div", { className: "startpage-brand-sep", children: "\u00B7" }), _jsx("div", { className: "startpage-brand-tagline", children: "\u9020\u4E50\u4E4B\u5730" })] }), _jsxs("div", { className: "startpage-left", children: [_jsx("button", { className: "startpage-hero", onClick: () => void pickBlank(), disabled: busy, children: _jsxs("div", { className: "startpage-hero-inner", children: [_jsx("div", { className: "startpage-hero-emoji", children: "\uD83C\uDFB5" }), _jsx("div", { className: "startpage-hero-title", children: t("splash.createBlank") }), _jsx("div", { className: "startpage-hero-desc", children: t("splash.createBlankDesc") }), _jsx("div", { className: "startpage-hero-arrow", "aria-hidden": true, children: "\u2192" })] }) }), autosave && (_jsxs("div", { className: "startpage-autosave", children: [_jsx("div", { className: "startpage-autosave-icon", children: "\u26A0" }), _jsxs("div", { className: "startpage-autosave-body", children: [_jsx("div", { className: "startpage-autosave-title", children: t("splash.autosaveTitle") }), _jsxs("div", { className: "startpage-autosave-meta", children: [autosave.name || t("splash.untitled"), " \u00B7 ", fmtDate(autosave.savedAt)] }), _jsxs("div", { className: "startpage-autosave-actions", children: [_jsx("button", { className: "startpage-autosave-recover", onClick: () => void recoverAutosave(), disabled: busy, children: t("splash.autosaveRecover") }), _jsx("button", { className: "startpage-autosave-discard", onClick: () => void discardAutosave(), disabled: busy, children: t("splash.autosaveDiscardBtn") })] })] })] }))] }), _jsxs("div", { className: "startpage-right", children: [_jsx("div", { className: "startpage-title", children: title }), _jsx("div", { className: "startpage-subtitle", children: t("splash.recentProjects") }), recent.length === 0 ? (_jsxs("div", { className: "startpage-empty", children: [_jsx("div", { className: "startpage-empty-art", "aria-hidden": true, children: "\uD83C\uDFBC" }), _jsx("div", { className: "startpage-empty-title", children: t("splash.emptyTitle") }), _jsx("div", { className: "startpage-empty-desc", children: t("splash.recentEmpty") })] })) : (_jsxs(_Fragment, { children: [_jsx("div", { className: "startpage-recent", children: recent.slice(0, MAX_VISIBLE_RECENT).map((p) => renderRecentItem(p)) }), recent.length > MAX_VISIBLE_RECENT && (_jsxs("button", { className: "startpage-recent-more", onClick: () => setShowMore(true), title: t("splash.recentTotal", { n: recent.length }), children: [_jsx("span", { className: "startpage-recent-more-label", children: t("splash.viewAll") }), _jsxs("span", { className: "startpage-recent-more-num", children: ["+", recent.length - MAX_VISIBLE_RECENT] }), _jsx("span", { className: "startpage-recent-more-arrow", "aria-hidden": true, children: "\u2192" })] })), showMore && (_jsx("div", { className: "startpage-modal-mask", onClick: () => setShowMore(false), children: _jsxs("div", { className: "startpage-modal", role: "dialog", "aria-modal": "true", "aria-label": t("splash.allProjects", { n: recent.length }), onClick: (e) => e.stopPropagation(), children: [_jsxs("div", { className: "startpage-modal-header", children: [_jsx("div", { className: "startpage-modal-title", children: t("splash.allProjects", { n: recent.length }) }), _jsx("button", { className: "startpage-modal-close", onClick: () => setShowMore(false), "aria-label": t("splash.closeModal"), title: t("splash.closeModal"), children: "\u2715" })] }), _jsx("div", { className: "startpage-modal-list", children: _jsx(List, { className: "startpage-vlist", rowCount: recent.length, rowHeight: VROW_HEIGHT, rowKey: rowKey, rowComponent: RecentRow, rowProps: { items: recent, render: renderRecentItem }, style: { height: vlistHeight, width: "100%" } }) })] }) }))] }))] })] }), contextMenu.visible && contextMenu.project && (_jsxs("div", { ref: contextMenuRef, className: "context-menu", style: { left: contextMenu.x, top: contextMenu.y }, children: [_jsxs("button", { className: "context-menu-item", onClick: () => contextMenu.project && handleRenameProject(contextMenu.project), children: ["\u270F\uFE0F ", t("splash.rename")] }), _jsxs("button", { className: "context-menu-item danger", onClick: () => contextMenu.project && handleDeleteProject(contextMenu.project.path), children: ["\uD83D\uDDD1\uFE0F ", t("splash.delete")] })] }))] }));
}
