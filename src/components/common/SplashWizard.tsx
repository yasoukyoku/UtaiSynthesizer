import { useEffect, useState, useCallback, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useTranslation } from "react-i18next";
import { List, type RowComponentProps } from "react-window";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useHistoryStore } from "../../store/history";
import { fitTimelineToContent } from "../../lib/timeline/fitView";
import { newProjectFile, openProjectFromPath, restoreAutosave } from "../../lib/project/projectFile";
import { readAutosave, clearAutosave, setRecoveryPending, markAutosaveBaseline } from "../../lib/project/autosave";
import { getRecentProjects, removeRecentProject, updateRecentProjectName, type RecentProject } from "../../lib/project/recentProjects";
import { isTauri } from "../../lib/tauri";
import "./Wizard.css";

interface Props { onClose: () => void; }

interface AutosaveEntry {
  filePath: string | null;
  name: string;
  savedAt: number;
  dirty: boolean;
  handled?: boolean;
}

interface ContextMenuState {
  visible: boolean;
  x: number;
  y: number;
  project: RecentProject | null;
}

/** 折叠阈值：右侧只直接展示前 N 条，其余收进「查看全部」弹窗。 */
const MAX_VISIBLE_RECENT = 4;

/** 虚拟列表行高(px)：卡片 58 + 行间距 8，必须与 .startpage-vrow/.startpage-recent-item 的盒模型一致。 */
const VROW_HEIGHT = 66;

/** 虚拟列表视口最大高度(px)。List 需要一个确定高度(父级只有 max-height,
 *  给 flex:1 + height:100% 会塌成 0),所以按条目数算出确定像素高度。 */
const VLIST_MAX_HEIGHT = 396;

/** 重命名会话：original 保留进入编辑时的名字，用于判断是否真的改动过。 */
interface RenameState {
  path: string;
  name: string;
  original: string;
}

function fmtDate(at: number): string {
  return new Date(at).toLocaleString(undefined, {
    year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
  });
}

/** 虚拟列表的行载荷：列表数据 + 复用主组件的卡片渲染函数。 */
interface RecentRowProps {
  items: RecentProject[];
  render: (p: RecentProject) => React.ReactNode;
}

/** react-window 行组件：style 由 List 注入(绝对定位),必须原样透传到最外层节点。 */
function RecentRow({ index, style, items, render }: RowComponentProps<RecentRowProps>) {
  const item = items[index];
  if (!item) return null;
  return (
    <div className="startpage-vrow" style={style}>
      {render(item)}
    </div>
  );
}

/**
 * 启动欢迎页：左边是新建区域，右边是历史记录/保存的工程
 * 简洁设计，移除示例工程卡片
 */
export function SplashWizard({ onClose }: Props) {
  const { t } = useTranslation();
  const projectName = useProjectStore((s) => s.name);
  const [recent, setRecent] = useState<RecentProject[]>(() => getRecentProjects());
  const [autosave, setAutosave] = useState<AutosaveEntry | null>(null);
  const [openingPath, setOpeningPath] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [contextMenu, setContextMenu] = useState<ContextMenuState>({ visible: false, x: 0, y: 0, project: null });
  const [renaming, setRenaming] = useState<RenameState | null>(null);
  const [showMore, setShowMore] = useState(false);
  const contextMenuRef = useRef<HTMLDivElement>(null);

  // 启动时读 autosave
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const env = await readAutosave();
      if (cancelled || !env) return;
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
    const handleClickOutside = (e: MouseEvent) => {
      if (contextMenuRef.current && !contextMenuRef.current.contains(e.target as Node)) {
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
    if (busy) return;
    setBusy(true);
    try {
      const s = useProjectStore.getState();
      const isAlreadyEmpty = (!s.name || s.name === "Untitled") && s.tracks.length === 0 && !s.dirty;
      if (isAlreadyEmpty) {
        useAppStore.getState().clearSelection();
        useHistoryStore.getState().reset();
        useHistoryStore.getState().markSaved();
        fitTimelineToContent();
        if (isTauri()) await markAutosaveBaseline();
        setRecoveryPending(false);
        onClose();
      } else {
        if (isTauri()) {
          await newProjectFile();
        }
        // 浏览器环境下直接关闭欢迎页
        setRecoveryPending(false);
        onClose();
      }
    } catch (e) {
      // 不能静默失败：否则欢迎页既不关闭也无任何提示，用户会以为按钮坏了。
      useAppStore.getState().showToast(e instanceof Error ? e.message : String(e), "error");
    } finally {
      setBusy(false);
    }
  }, [busy, onClose]);

  const openRecent = useCallback(async (p: RecentProject) => {
    if (!isTauri() || busy || openingPath) return;
    setBusy(true);
    setOpeningPath(p.path);
    try {
      const exists = await invoke<boolean>("path_exists", { path: p.path });
      if (!exists) {
        removeRecentProject(p.path);
        setRecent(getRecentProjects());
        useAppStore.getState().showToast(t("splash.recentMissing"), "warning");
        return;
      }
      const ok = await openProjectFromPath(p.path);
      if (ok) { setRecoveryPending(false); onClose(); }
    } catch (e) {
      useAppStore.getState().showToast(e instanceof Error ? e.message : String(e), "error");
    } finally {
      setOpeningPath(null);
      setBusy(false);
    }
  }, [busy, openingPath, onClose, t]);

  // 右键菜单：删除
  const handleDeleteProject = useCallback((path: string) => {
    removeRecentProject(path);
    setRecent(getRecentProjects());
    setContextMenu({ visible: false, x: 0, y: 0, project: null });
    useAppStore.getState().showToast(t("splash.recentRemoved"), "info");
  }, [t]);

  // 右键菜单：重命名
  const handleRenameProject = useCallback((project: RecentProject) => {
    setRenaming({ path: project.path, name: project.name, original: project.name });
    setContextMenu({ visible: false, x: 0, y: 0, project: null });
  }, []);

  const handleRenameSubmit = useCallback(() => {
    if (!renaming) return;
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
  const handleContextMenu = useCallback((e: React.MouseEvent, project: RecentProject) => {
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
    if (!autosave || busy) return;
    setBusy(true);
    try {
      const env = await readAutosave();
      if (!env) { setAutosave(null); setRecoveryPending(false); return; }
      await restoreAutosave(env);
      setAutosave(null);
      setRecoveryPending(false);
      onClose();
    } catch (e) {
      useAppStore.getState().showToast(e instanceof Error ? e.message : String(e), "error");
    } finally {
      setBusy(false);
    }
  }, [autosave, busy, onClose]);

  const discardAutosave = useCallback(async () => {
    if (!autosave || busy) return;
    setBusy(true);
    try {
      await clearAutosave();
      setAutosave(null);
      setRecoveryPending(false);
      useAppStore.getState().showToast(t("splash.autosaveDiscarded"), "info");
    } catch (e) {
      // 与 recoverAutosave 对齐：缺 catch 会变成未处理的 Promise rejection。
      useAppStore.getState().showToast(e instanceof Error ? e.message : String(e), "error");
    } finally {
      setBusy(false);
    }
  }, [autosave, busy, t]);

  const title = projectName && projectName !== "Untitled"
    ? `${t("splash.titleWithName")} ${projectName}`
    : t("splash.title");

  /** 用 path 而非 index 作为虚拟行 key：删除/重命名后行内状态不会错位。 */
  const rowKey = useCallback(
    (index: number, { items }: RecentRowProps) => items[index]?.path ?? index,
    [],
  );

  /** 弹窗内虚拟列表的实际高度(px)：取 (条目数 × 行高) 和 (视口上限) 中较小值。 */
  const vlistHeight = Math.min(recent.length * VROW_HEIGHT, VLIST_MAX_HEIGHT);

  /** 渲染单个历史工程卡片（主列表与「查看全部」弹窗共用）。 */
  const renderRecentItem = useCallback((p: RecentProject) => {
    const isCrashed = autosave?.filePath === p.path;
    const isRenaming = renaming?.path === p.path;
    const isOpening = openingPath === p.path;

    return (
      <button
        key={p.path}
        className={`startpage-recent-item ${isOpening ? "opening" : ""} ${isRenaming ? "renaming" : ""} ${isCrashed ? "crashed" : ""}`}
        onClick={() => { if (!isRenaming) void openRecent(p); }}
        onContextMenu={(e) => handleContextMenu(e, p)}
        disabled={busy}
        title={p.path}
      >
        <span className="startpage-recent-icon">{isCrashed ? "⚠" : "🎼"}</span>
        {isRenaming ? (
          <input
            type="text"
            className="startpage-rename-input"
            value={renaming.name}
            onChange={(e) => setRenaming({ ...renaming, name: e.target.value })}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleRenameSubmit();
              if (e.key === "Escape") handleRenameCancel();
            }}
            onBlur={handleRenameSubmit}
            autoFocus
            onClick={(e) => e.stopPropagation()}
          />
        ) : (
          <span className="startpage-recent-info">
            <span className="startpage-recent-name">
              {p.name}
              {isCrashed && <span className="startpage-recent-crash-badge">{t("splash.crashBadge")}</span>}
            </span>
            <span className="startpage-recent-meta">
              <span>{fmtDate(p.at)}</span>
              <span className="startpage-recent-path">{p.path}</span>
            </span>
          </span>
        )}
      </button>
    );
  }, [autosave?.filePath, renaming, openingPath, busy, t, handleContextMenu, openRecent, handleRenameSubmit, handleRenameCancel]);

  return (
    <div className="startpage-backdrop" onClick={(e) => e.preventDefault()}>
      <div className="startpage-wrap">
        {/* 顶部品牌 LOGO 区 */}
        <div className="startpage-brand-header">
          <img className="startpage-brand-logo" src="/logo-user.png" alt="MunoAI" />
          <div className="startpage-brand-name">MunoAI</div>
          <div className="startpage-brand-sep">·</div>
          <div className="startpage-brand-tagline">造乐之地</div>
        </div>

        {/* 左边：新建区域 */}
        <div className="startpage-left">
          <button className="startpage-hero" onClick={() => void pickBlank()} disabled={busy}>
            <div className="startpage-hero-inner">
              <div className="startpage-hero-emoji">🎵</div>
              <div className="startpage-hero-title">{t("splash.createBlank")}</div>
              <div className="startpage-hero-desc">{t("splash.createBlankDesc")}</div>
              <div className="startpage-hero-arrow" aria-hidden>→</div>
            </div>
          </button>

          {/* autosave 恢复卡（如果存在）*/}
          {autosave && (
            <div className="startpage-autosave">
              <div className="startpage-autosave-icon">⚠</div>
              <div className="startpage-autosave-body">
                <div className="startpage-autosave-title">{t("splash.autosaveTitle")}</div>
                <div className="startpage-autosave-meta">
                  {autosave.name || t("splash.untitled")} · {fmtDate(autosave.savedAt)}
                </div>
                <div className="startpage-autosave-actions">
                  <button className="startpage-autosave-recover" onClick={() => void recoverAutosave()} disabled={busy}>
                    {t("splash.autosaveRecover")}
                  </button>
                  <button className="startpage-autosave-discard" onClick={() => void discardAutosave()} disabled={busy}>
                    {t("splash.autosaveDiscardBtn")}
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* 右边：历史记录/保存的工程 */}
        <div className="startpage-right">
          <div className="startpage-title">{title}</div>
          <div className="startpage-subtitle">{t("splash.recentProjects")}</div>

          {recent.length === 0 ? (
            <div className="startpage-empty">
              <div className="startpage-empty-art" aria-hidden>🎼</div>
              <div className="startpage-empty-title">{t("splash.emptyTitle")}</div>
              <div className="startpage-empty-desc">{t("splash.recentEmpty")}</div>
            </div>
          ) : (
            <>
              <div className="startpage-recent">
                {recent.slice(0, MAX_VISIBLE_RECENT).map((p) => renderRecentItem(p))}
              </div>

              {recent.length > MAX_VISIBLE_RECENT && (
                <button
                  className="startpage-recent-more"
                  onClick={() => setShowMore(true)}
                  title={t("splash.recentTotal", { n: recent.length })}
                >
                  <span className="startpage-recent-more-label">{t("splash.viewAll")}</span>
                  <span className="startpage-recent-more-num">+{recent.length - MAX_VISIBLE_RECENT}</span>
                  <span className="startpage-recent-more-arrow" aria-hidden>→</span>
                </button>
              )}

              {/* 更多历史弹窗：条目多时用 react-window 虚拟滚动，只渲染可视行 */}
              {showMore && (
                <div className="startpage-modal-mask" onClick={() => setShowMore(false)}>
                  <div
                    className="startpage-modal"
                    role="dialog"
                    aria-modal="true"
                    aria-label={t("splash.allProjects", { n: recent.length })}
                    onClick={(e) => e.stopPropagation()}
                  >
                    <div className="startpage-modal-header">
                      <div className="startpage-modal-title">{t("splash.allProjects", { n: recent.length })}</div>
                      <button
                        className="startpage-modal-close"
                        onClick={() => setShowMore(false)}
                        aria-label={t("splash.closeModal")}
                        title={t("splash.closeModal")}
                      >
                        ✕
                      </button>
                    </div>
                    <div className="startpage-modal-list">
                      <List
                        className="startpage-vlist"
                        rowCount={recent.length}
                        rowHeight={VROW_HEIGHT}
                        rowKey={rowKey}
                        rowComponent={RecentRow}
                        rowProps={{ items: recent, render: renderRecentItem }}
                        style={{ height: vlistHeight, width: "100%" }}
                      />
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* 右键菜单 */}
      {contextMenu.visible && contextMenu.project && (
        <div
          ref={contextMenuRef}
          className="context-menu"
          style={{ left: contextMenu.x, top: contextMenu.y }}
        >
          <button
            className="context-menu-item"
            onClick={() => contextMenu.project && handleRenameProject(contextMenu.project)}
          >
            ✏️ {t("splash.rename")}
          </button>
          <button
            className="context-menu-item danger"
            onClick={() => contextMenu.project && handleDeleteProject(contextMenu.project.path)}
          >
            🗑️ {t("splash.delete")}
          </button>
        </div>
      )}
    </div>
  );
}

