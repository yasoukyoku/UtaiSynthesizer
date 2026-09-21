import { invoke } from '@tauri-apps/api/core';
export async function mergeAudioFiles(inputPaths, outputPath, options = {}) {
    if (inputPaths.length === 0) {
        throw new Error('至少需要一个输入文件');
    }
    if (inputPaths.length === 1) {
        await invoke('audio_copy', {
            input: inputPaths[0],
            output: outputPath
        });
        return;
    }
    await invoke('mix_audio_files', {
        paths: inputPaths,
        output: outputPath,
        volumes: options.volumes,
        pans: options.pans,
        normalization: options.normalization || 'none',
        targetLufs: options.targetLUFS || -14.0
    });
}
