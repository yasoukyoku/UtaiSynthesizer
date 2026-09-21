export function buildMenuItems(options) {
    const { isAudioTrack, isMidiTrack, hasNotes, audioPath, onAmt, onStemSeparation, onQuantize, onHumanize, onVelocityCurve, onSoundfontRender, onArrange, onChordMidi, onDetectChords, onDetectDrums, onRename, onCopy, onPaste, onDelete, canPaste = false } = options;
    const arrangeDisabled = !hasNotes;
    const chordMidiDisabled = !hasNotes;
    const chordAnalyzeDisabled = !hasNotes && !(isAudioTrack && !!audioPath);
    const detectDrumsDisabled = !isAudioTrack || !audioPath;
    const tipNoNotes = "需要有音符内容的旋律/乐器轨";
    const tipNoAudio = "需要音频轨 (带音频文件)";
    const items = [];
    if (isAudioTrack && audioPath) {
        items.push({
            type: 'submenu',
            label: '🎵 转MIDI (AI 转谱)',
            icon: '🎵',
            title: '选择转换质量预设',
            items: [
                {
                    label: '⚡ 快速模式 (Fast)',
                    title: '最快速度，适合预览和初步转换',
                    onClick: () => onAmt?.('fast')
                },
                {
                    label: '📊 标准模式 (Standard)',
                    title: '平衡速度和质量，适合大多数场景',
                    onClick: () => onAmt?.('standard')
                },
                {
                    label: '✨ 最佳质量 (Best)',
                    title: '最高质量，处理时间较长',
                    onClick: () => onAmt?.('best')
                },
                {
                    label: '🎸 吉他专用模式',
                    title: '针对吉他音色优化的转换',
                    onClick: () => onAmt?.('guitar')
                },
                {
                    label: '🎹 钢琴专用模式',
                    title: '针对钢琴音色优化的转换',
                    onClick: () => onAmt?.('piano')
                }
            ]
        });
        items.push({
            type: 'submenu',
            label: '🎛️ 一键分轨 (Stem Separation)',
            icon: '🎛️',
            title: '选择分离质量预设',
            items: [
                {
                    label: '⚡ 快速分离 (Fast)',
                    title: 'HTDemucs 快速模式，适合预览',
                    onClick: () => onStemSeparation?.('fast')
                },
                {
                    label: '📊 标准分离 (Standard)',
                    title: 'HTDemucs 标准模式，平衡质量',
                    onClick: () => onStemSeparation?.('standard')
                },
                {
                    label: '🎯 专业分离 (Pro)',
                    title: 'HTDemucs FT 模型，高质量分离',
                    onClick: () => onStemSeparation?.('pro')
                },
                {
                    label: '✨ 最佳分离 (Best)',
                    title: 'HTDemucs 6s 模型，最高质量',
                    onClick: () => onStemSeparation?.('best')
                }
            ]
        });
    }
    if (isMidiTrack && hasNotes) {
        items.push({
            type: 'submenu',
            label: '🎹 MIDI 编辑',
            icon: '🎹',
            title: 'MIDI 音符编辑工具',
            items: [
                {
                    type: 'submenu',
                    label: '📐 量化 (Quantize)',
                    items: [
                        {
                            label: '1/4 音符',
                            onClick: () => onQuantize?.(0.25)
                        },
                        {
                            label: '1/8 音符',
                            onClick: () => onQuantize?.(0.125)
                        },
                        {
                            label: '1/16 音符',
                            onClick: () => onQuantize?.(0.0625)
                        },
                        {
                            label: '1/32 音符',
                            onClick: () => onQuantize?.(0.03125)
                        }
                    ]
                },
                {
                    label: '🎭 人性化 (Humanize)',
                    title: '添加随机时间和力度变化',
                    onClick: onHumanize
                },
                {
                    type: 'submenu',
                    label: '📈 力度曲线',
                    items: [
                        {
                            label: '📈 渐强 (Crescendo)',
                            onClick: () => onVelocityCurve?.('crescendo')
                        },
                        {
                            label: '📉 渐弱 (Diminuendo)',
                            onClick: () => onVelocityCurve?.('diminuendo')
                        },
                        {
                            label: '🌊 指数曲线',
                            onClick: () => onVelocityCurve?.('exponential')
                        }
                    ]
                },
                { label: '', separator: true },
                {
                    label: '🔊 Soundfont 渲染为音频',
                    title: '使用 Soundfont 将 MIDI 渲染为音频轨',
                    onClick: onSoundfontRender
                }
            ]
        });
    }
    if (items.length > 0) {
        items.push({ label: '', separator: true });
    }
    items.push({
        label: '✨ 智能编曲…',
        disabled: arrangeDisabled,
        title: arrangeDisabled ? tipNoNotes : '打开编曲面板:11 类乐器自由组合,实时试听后一键生成多轨',
        onClick: onArrange
    }, {
        label: '🎹 和弦MIDI…',
        disabled: chordMidiDisabled,
        title: chordMidiDisabled ? tipNoNotes : '生成一条和弦轨 (C Am F G …)',
        onClick: onChordMidi
    }, {
        label: '🎼 识别和弦',
        disabled: chordAnalyzeDisabled,
        title: chordAnalyzeDisabled ? (isAudioTrack ? tipNoAudio : tipNoNotes) : '分析这条轨的和弦进行 (显示在和弦轨)',
        onClick: onDetectChords
    }, {
        label: '🥁 识别鼓点',
        disabled: detectDrumsDisabled,
        title: detectDrumsDisabled ? tipNoAudio : '把这条音频轨的鼓点变成 MIDI 鼓轨',
        onClick: onDetectDrums
    });
    items.push({ label: '', separator: true });
    items.push({
        label: '重命名',
        onClick: onRename
    }, {
        label: '复制轨道',
        onClick: onCopy
    }, {
        label: '粘贴轨道',
        disabled: !canPaste,
        onClick: onPaste
    }, {
        label: '删除',
        danger: true,
        onClick: onDelete
    });
    return items;
}
