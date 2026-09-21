import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { t18 } from "../../lib/models/msst-catalog";
import "./ShortcutsDialog.css";

const S11 = (zh: string, en: string, ja: string, keys: string): [string, string, string, string] => [zh, en, ja, keys];

/** 快捷键帮助面板 —— 汇总常用键盘操作。Esc / 点击背景关闭。 */
export function ShortcutsDialog({ onClose }: { onClose: () => void }) {
  const { i18n } = useTranslation();
  const lang = i18n.language;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const ROWS: [string, string, string, string][] = [
    S11("保存工程", "Save project", "プロジェクトを保存", "Ctrl / ⌘ + S"),
    S11("另存为", "Save as", "名前を付けて保存", "Ctrl / ⌘ + Shift + S"),
    S11("打开工程", "Open project", "開く", "Ctrl / ⌘ + O"),
    S11("新建工程", "New project", "新規", "Ctrl / ⌘ + N"),
    S11("撤销", "Undo", "元に戻す", "Ctrl / ⌘ + Z"),
    S11("重做", "Redo", "やり直し", "Ctrl / ⌘ + Y"),
    S11("全选", "Select all", "すべて選択", "Ctrl / ⌘ + A"),
    S11("复制", "Copy", "コピー", "Ctrl / ⌘ + C"),
    S11("剪切", "Cut", "カット", "Ctrl / ⌘ + X"),
    S11("粘贴", "Paste", "ペースト", "Ctrl / ⌘ + V"),
    S11("删除所选", "Delete selection", "選択を削除", "Delete"),
    S11("切片 / 分割", "Slice clip", "分割", "Ctrl / ⌘ + K"),
    S11("横向缩放", "Zoom horizontally", "横方向ズーム", "滑块 / 滚轮"),
    S11("纵向缩放", "Zoom vertically", "縦方向ズーム", "Alt + 滚轮"),
    S11("播放 / 暂停", "Play / Pause", "再生 / 一時停止", "空格"),
  ];

  return (
    <div className="shortcuts-overlay" onClick={onClose}>
      <div className="shortcuts-panel" onClick={(e) => e.stopPropagation()}>
        <div className="shortcuts-head">
          <span className="shortcuts-title">{t18({ zh: "快捷键帮助", en: "Keyboard Shortcuts", ja: "ショートカット" }, lang)}</span>
          <button className="shortcuts-close" onClick={onClose}>✕</button>
        </div>
        <div className="shortcuts-body">
          {ROWS.map(([zh, en, ja, keys]) => (
            <div className="shortcuts-row" key={`${zh}-${keys}`}>
              <span className="shortcuts-desc">{t18({ zh, en, ja }, lang)}</span>
              <span className="shortcuts-keys">
                {keys.split(" / ").map((k) => (
                  <kbd key={k}>{k}</kbd>
                ))}
              </span>
            </div>
          ))}
        </div>
        <div className="shortcuts-foot">{t18({ zh: "按 Esc 或点击空白处关闭", en: "Press Esc or click outside to close", ja: "Esc キーまたは外側クリックで閉じます" }, lang)}</div>
      </div>
    </div>
  );
}