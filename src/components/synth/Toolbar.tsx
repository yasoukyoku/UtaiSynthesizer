import { useState, useEffect, useRef } from "react";
import { useProjectStore } from "../../store/project";
import { useAudioStore } from "../../store/audio";
import { useAppStore } from "../../store/app";
import { useTranslation } from "react-i18next";
import { open } from "@tauri-apps/plugin-dialog";
import * as playback from "../../lib/audio/playback";
import { TICKS_PER_BEAT } from "../../lib/constants";
import { OverviewMap } from "./OverviewMap";
import { importAudioToTrack } from "../../lib/audio/import";
import i18n from "../../i18n";
import "./Toolbar.css";

export function Toolbar() {
  const { t } = useTranslation();
  const { addTrack, tempo, setTempo, playheadTick, setPlayhead, timeSignature, tracks, updateTrack } =
    useProjectStore();
  const { loadAudioFile, audioFiles, isPlaying, setPlaying, setPlayStart } = useAudioStore();
  const { selectedSegment, clearSelection } = useAppStore();
  const { splitSegment, deleteSegment } = useProjectStore();
  const [showAddMenu, setShowAddMenu] = useState(false);
  const animRef = useRef<number>(0);
  const playStartTickRef = useRef(0);
  const animatingRef = useRef(false);
  const addMenuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isPlaying) {
      animatingRef.current = false;
      cancelAnimationFrame(animRef.current);
      return;
    }

    playStartTickRef.current = playheadTick;
    animatingRef.current = true;
    const startTime = playback.getContextTime();

    const animate = () => {
      if (!animatingRef.current) return;
      const elapsed = playback.getContextTime() - startTime;
      const ticksElapsed = playback.secondsToTicks(elapsed, tempo);
      setPlayhead(Math.round(playStartTickRef.current + ticksElapsed));
      animRef.current = requestAnimationFrame(animate);
    };
    animRef.current = requestAnimationFrame(animate);

    return () => {
      animatingRef.current = false;
      cancelAnimationFrame(animRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPlaying, tempo]);

  const handleTogglePlay = async () => {
    if (isPlaying) {
      animatingRef.current = false;
      playback.stopPlayback();
      setPlaying(false);
      return;
    }

    const started = await playback.playAllTracks(
      tracks,
      audioFiles,
      playheadTick,
      tempo,
      () => setPlaying(false),
    );
    if (started) {
      setPlaying(true);
      setPlayStart(playback.getContextTime(), playheadTick);
    }
  };

  const handleReturnToStart = () => {
    animatingRef.current = false;
    playback.stopPlayback();
    setPlaying(false);
    setPlayhead(0);
  };

  const handleImportAudio = async () => {
    setShowAddMenu(false);
    try {
      const path = await open({
        title: t("toolbar.importAudio"),
        filters: [{ name: "Audio", extensions: ["wav", "mp3", "flac", "ogg"] }],
      });
      if (!path) return;
      await importAudioToTrack(
        path as string, tempo, playheadTick, tracks, addTrack, loadAudioFile, updateTrack,
      );
    } catch (e) {
      console.error("Import failed:", e);
    }
  };

  const handleAddMidiTrack = () => {
    setShowAddMenu(false);
    addTrack({
      id: `track-${Date.now()}`,
      name: `Vocal ${tracks.filter((t) => t.trackType === "vocal").length + 1}`,
      trackType: "vocal",
      segments: [],
      volumeDb: 0,
      pan: 0,
      muted: false,
      solo: false,
    });
  };

  const handleSplit = () => {
    if (!selectedSegment) return;
    splitSegment(selectedSegment.trackId, selectedSegment.segmentId, playheadTick);
  };

  const handleDelete = () => {
    if (!selectedSegment) return;
    deleteSegment(selectedSegment.trackId, selectedSegment.segmentId);
    clearSelection();
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === " ") {
        e.preventDefault();
        handleTogglePlay();
      } else if (e.key === "Delete" && selectedSegment) {
        handleDelete();
      } else if (e.key === "k" && e.ctrlKey && selectedSegment) {
        e.preventDefault();
        handleSplit();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  });

  useEffect(() => {
    if (!showAddMenu) return;
    const onClick = (e: MouseEvent) => {
      if (addMenuRef.current && !addMenuRef.current.contains(e.target as Node)) {
        setShowAddMenu(false);
      }
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, [showAddMenu]);

  const cycleLang = () => {
    const langs = ["zh", "en", "ja"];
    const cur = langs.indexOf(i18n.language);
    i18n.changeLanguage(langs[(cur + 1) % langs.length]!);
  };

  return (
    <div className="toolbar">
      <div className="toolbar-section transport">
        <button
          className="transport-btn"
          onClick={handleReturnToStart}
          data-tooltip={t("transport.returnToStart")}
        >
          <span className="transport-icon icon-return" />
        </button>
        <button
          className={`transport-btn play ${isPlaying ? "playing" : ""}`}
          onClick={handleTogglePlay}
          data-tooltip={isPlaying ? t("transport.pause") : t("transport.play")}
        >
          {isPlaying
            ? <span className="transport-icon icon-pause" />
            : <span className="transport-icon icon-play" />
          }
        </button>
      </div>

      <OverviewMap />

      <div className="toolbar-divider" />

      <div className="toolbar-section tempo-section">
        <label className="toolbar-label">{t("toolbar.bpm")}</label>
        <input
          type="number"
          className="tempo-input mono"
          value={tempo}
          min={20}
          max={400}
          step={1}
          onChange={(e) => setTempo(Number(e.target.value))}
        />
      </div>

      <div className="toolbar-section time-sig">
        <span className="mono time-display">
          {timeSignature[0]}/{timeSignature[1]}
        </span>
      </div>

      <div className="toolbar-divider" />

      <div className="toolbar-section position-section">
        <label className="toolbar-label">{t("toolbar.position")}</label>
        <span className="mono position-display">
          {formatPosition(playheadTick, timeSignature)}
        </span>
      </div>

      <div className="toolbar-divider" />

      <div className="toolbar-section snap-section">
        <label className="toolbar-label">{t("toolbar.snap")}</label>
        <select className="snap-select" defaultValue="16">
          <option value="4">1/4</option>
          <option value="8">1/8</option>
          <option value="16">1/16</option>
          <option value="32">1/32</option>
          <option value="triplet">{t("toolbar.triplet")}</option>
          <option value="free">{t("toolbar.free")}</option>
        </select>
      </div>

      <div className="toolbar-divider" />

      <div className="toolbar-section edit-section">
        <button
          className="toolbar-btn"
          onClick={handleSplit}
          disabled={!selectedSegment}
          data-tooltip={`${t("toolbar.split")} [Ctrl+K]`}
        >
          {t("toolbar.split")}
        </button>
        <button
          className="toolbar-btn"
          onClick={handleDelete}
          disabled={!selectedSegment}
          data-tooltip={`${t("toolbar.delete")} [Del]`}
        >
          {t("toolbar.delete")}
        </button>
      </div>

      <div className="toolbar-spacer" />

      <div className="toolbar-section" style={{ position: "relative" }} ref={addMenuRef}>
        <button className="toolbar-btn" onClick={() => setShowAddMenu(!showAddMenu)}>
          + {t("toolbar.addTrack")}
        </button>
        {showAddMenu && (
          <div className="add-track-menu">
            <button className="add-track-option" onClick={handleImportAudio}>
              {t("toolbar.importAudio")}
            </button>
            <button className="add-track-option" onClick={handleAddMidiTrack}>
              {t("toolbar.addMidi")}
            </button>
          </div>
        )}
      </div>

      <div className="toolbar-divider" />

      <div className="toolbar-section">
        <button className="toolbar-btn lang-btn mono" onClick={cycleLang}>
          {i18n.language.toUpperCase()}
        </button>
      </div>
    </div>
  );
}

function formatPosition(tick: number, timeSig: [number, number]): string {
  const ticksPerBar = TICKS_PER_BEAT * timeSig[0];
  const bar = Math.floor(tick / ticksPerBar) + 1;
  const beat = Math.floor((tick % ticksPerBar) / TICKS_PER_BEAT) + 1;
  const sub = Math.floor(((tick % ticksPerBar) % TICKS_PER_BEAT) / (TICKS_PER_BEAT / 4));
  return `${bar}:${beat}:${sub.toString().padStart(2, "0")}`;
}
