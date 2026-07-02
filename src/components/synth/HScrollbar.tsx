import { useRef, useCallback, useEffect } from "react";
import { useAppStore } from "../../store/app";
import "./HScrollbar.css";

interface Props {
  totalWidth: number;
  viewWidth: number;
  onChange: (x: number) => void;
}

export function HScrollbar({ totalWidth, viewWidth, onChange }: Props) {
  // Self-subscribe scrollX so horizontal scroll re-renders ONLY this tiny component
  // (two styled divs), not the whole DawView subtree.
  const scrollX = useAppStore((s) => s.scrollX);
  const trackRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);
  const justDragged = useRef(false);
  const dragStartMouseX = useRef(0);
  const dragStartScrollX = useRef(0);

  // Keep latest values in refs so event listeners always see fresh data
  const onChangeRef = useRef(onChange);
  const totalWidthRef = useRef(totalWidth);
  const maxScrollRef = useRef(0);
  onChangeRef.current = onChange;
  totalWidthRef.current = totalWidth;
  maxScrollRef.current = Math.max(0, totalWidth - viewWidth);

  const maxScroll = maxScrollRef.current;
  const thumbRatio = Math.min(1, viewWidth / Math.max(1, totalWidth));
  const thumbLeft = maxScroll > 0 ? (scrollX / maxScroll) * (1 - thumbRatio) * 100 : 0;

  const handleTrackClick = useCallback((e: React.MouseEvent) => {
    if (justDragged.current) {
      justDragged.current = false;
      return;
    }
    const track = trackRef.current;
    if (!track) return;
    const rect = track.getBoundingClientRect();
    const clickRatio = (e.clientX - rect.left) / rect.width;
    onChangeRef.current(Math.round(clickRatio * maxScrollRef.current));
  }, []);

  const handleThumbDown = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    isDragging.current = true;
    dragStartMouseX.current = e.clientX;
    dragStartScrollX.current = useAppStore.getState().scrollX;
  }, []);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!isDragging.current || !trackRef.current) return;
      const trackWidth = trackRef.current.getBoundingClientRect().width;
      const dx = e.clientX - dragStartMouseX.current;
      const scrollDelta = (dx / trackWidth) * totalWidthRef.current;
      const newScroll = Math.max(0, Math.min(maxScrollRef.current, dragStartScrollX.current + scrollDelta));
      onChangeRef.current(Math.round(newScroll));
    };

    const onUp = () => {
      if (isDragging.current) {
        isDragging.current = false;
        justDragged.current = true;
        setTimeout(() => { justDragged.current = false; }, 0);
      }
    };

    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
  }, []);

  return (
    <div className="hscrollbar-track" ref={trackRef} onClick={handleTrackClick}>
      <div
        className="hscrollbar-thumb"
        style={{
          left: `${thumbLeft}%`,
          width: `${Math.max(5, thumbRatio * 100)}%`,
        }}
        onMouseDown={handleThumbDown}
      />
    </div>
  );
}
