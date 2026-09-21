import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";
import { useProjectStore } from "../../store/project";
import { useAudioStore } from "../../store/audio";
import { importAudioToExistingTrack, importAudioToNewTrack } from "./import";
import { isTauri } from "../tauri";
let mediaRecorder = null;
let mediaStream = null;
let chunks = [];
/** 浏览器录音 Blob(通常 webm/ogg)→ 16-bit PCM WAV ArrayBuffer */
async function blobToWav(blob) {
    const ctx = new AudioContext();
    try {
        const buf = await ctx.decodeAudioData(await blob.arrayBuffer());
        const ch = buf.numberOfChannels;
        const len = buf.length;
        const pcm = new ArrayBuffer(44 + len * ch * 2);
        const view = new DataView(pcm);
        const wStr = (o, s) => { for (let i = 0; i < s.length; i++)
            view.setUint8(o + i, s.charCodeAt(i)); };
        wStr(0, "RIFF");
        view.setUint32(4, 36 + len * ch * 2, true);
        wStr(8, "WAVEfmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, ch, true);
        view.setUint32(24, buf.sampleRate, true);
        view.setUint32(28, buf.sampleRate * ch * 2, true);
        view.setUint16(32, ch * 2, true);
        view.setUint16(34, 16, true);
        wStr(36, "data");
        view.setUint32(40, len * ch * 2, true);
        let o = 44;
        const chans = [];
        for (let c = 0; c < ch; c++)
            chans.push(buf.getChannelData(c));
        for (let i = 0; i < len; i++) {
            for (let c = 0; c < ch; c++) {
                const s = Math.max(-1, Math.min(1, chans[c][i]));
                view.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
                o += 2;
            }
        }
        return pcm;
    }
    finally {
        void ctx.close();
    }
}
export const useRecording = create((set, get) => ({
    recordingTrackId: null,
    start: async (trackId) => {
        if (!navigator.mediaDevices?.getUserMedia) {
            useProjectStore.getState();
            throw new Error("浏览器不支持录音");
        }
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        chunks = [];
        mediaRecorder = new MediaRecorder(mediaStream);
        mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0)
            chunks.push(e.data); };
        mediaRecorder.start();
        set({ recordingTrackId: trackId });
    },
    stop: async () => {
        const trackId = get().recordingTrackId;
        set({ recordingTrackId: null });
        if (!mediaRecorder || !trackId)
            return;
        const rec = mediaRecorder;
        mediaRecorder = null;
        const done = new Promise((resolve) => {
            rec.onstop = () => resolve(new Blob(chunks, { type: rec.mimeType || "audio/webm" }));
        });
        rec.stop();
        for (const t of mediaStream?.getTracks() ?? [])
            t.stop();
        mediaStream = null;
        const blob = await done;
        if (blob.size === 0)
            return;
        const wav = await blobToWav(blob);
        let filePath;
        if (isTauri()) {
            filePath = await invoke("save_recording", { data: Array.from(new Uint8Array(wav)) });
        }
        else {
            // 浏览器预览: 无后端,退回对象 URL 不落盘 — 直接放 toast 提示
            useAudioStore.getState();
            filePath = URL.createObjectURL(blob);
        }
        // 导入到录音所在的轨道(播放头位置);轨道已删则新建
        const proj = useProjectStore.getState();
        const tick = proj.playheadTick;
        if (proj.tracks.some((t) => t.id === trackId)) {
            await importAudioToExistingTrack(filePath, trackId, tick);
        }
        else {
            await importAudioToNewTrack(filePath, tick);
        }
    },
    toggle: async (trackId) => {
        if (get().recordingTrackId === trackId) {
            await get().stop();
        }
        else {
            if (get().recordingTrackId)
                await get().stop();
            await get().start(trackId);
        }
    },
}));
