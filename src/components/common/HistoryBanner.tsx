import { useEffect, useState } from "react";
import { useAppStore } from "../../store/app";
import type { BannerKind } from "../../store/app";
import "./HistoryBanner.css";

/**
 * Transient top-right corner banner. Tells the user what just happened (undo/redo of WHAT, save/load
 * confirmation, …) without disturbing the view/flow — replaced the old reveal-on-undo viewport scroll.
 * A single element that re-triggers in place on each event (keyed on the store `seq`); the inner block
 * is keyed by `seq` so the slide-in + countdown line restart on a rapid retrigger (no stacking).
 */

const KIND_COLOR: Record<BannerKind, string> = {
  undo: "var(--accent-primary)",
  redo: "var(--accent-secondary)",
  save: "var(--color-success)",
  load: "var(--accent-tertiary)",
  info: "var(--accent-primary)",
};

const svg = { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };

function Icon({ kind }: { kind: BannerKind }) {
  switch (kind) {
    case "undo":
      return <svg {...svg}><path d="M9 14 4 9l5-5" /><path d="M4 9h11a5 5 0 0 1 0 10h-3" /></svg>;
    case "redo":
      return <svg {...svg}><path d="m15 14 5-5-5-5" /><path d="M20 9H9a5 5 0 0 0 0 10h3" /></svg>;
    case "save":
      return <svg {...svg}><path d="M12 3v12" /><path d="m7 10 5 5 5-5" /><path d="M5 21h14" /></svg>;
    case "load":
      return <svg {...svg}><path d="M12 21V9" /><path d="m7 14 5-5 5 5" /><path d="M5 3h14" /></svg>;
    default:
      return <svg {...svg}><circle cx="12" cy="12" r="9" /></svg>;
  }
}

export function HistoryBanner() {
  const banner = useAppStore((s) => s.banner);
  const [shown, setShown] = useState(false);

  useEffect(() => {
    if (!banner) return;
    setShown(true);
    const id = setTimeout(() => setShown(false), 1800);
    return () => clearTimeout(id);
  }, [banner?.seq]);

  if (!banner) return null;
  return (
    <div
      className={`app-banner ${shown ? "show" : ""}`}
      style={{ ["--ab-color" as string]: KIND_COLOR[banner.kind] }}
      aria-live="polite"
    >
      <div className="app-banner-inner" key={banner.seq}>
        <span className="app-banner-icon"><Icon kind={banner.kind} /></span>
        <span className="app-banner-text">{banner.message}</span>
      </div>
    </div>
  );
}
