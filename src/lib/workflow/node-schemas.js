export const NODE_SCHEMAS = {
    input: {
        type: 'input',
        inputs: [],
        outputs: [
            { id: 'audio', type: 'audio', label: '音频输出' }
        ]
    },
    output: {
        type: 'output',
        inputs: [
            { id: 'audio', type: 'audio', label: '音频输入' }
        ],
        outputs: []
    },
    rvc: {
        type: 'rvc',
        inputs: [
            { id: 'audio', type: 'audio', label: '输入音频' },
            { id: 'f0_curve', type: 'report', label: 'F0曲线', optional: true }
        ],
        outputs: [
            { id: 'audio', type: 'audio', label: '变声后音频' }
        ]
    },
    sovits: {
        type: 'sovits',
        inputs: [
            { id: 'audio', type: 'audio', label: '输入音频' }
        ],
        outputs: [
            { id: 'audio', type: 'audio', label: 'SoVITS音频' }
        ]
    },
    transpose: {
        type: 'transpose',
        inputs: [
            { id: 'audio', type: 'audio', label: '输入音频' }
        ],
        outputs: [
            { id: 'audio', type: 'audio', label: '变调后音频' }
        ]
    },
    msstSeparation: {
        type: 'msstSeparation',
        inputs: [
            { id: 'audio', type: 'audio', label: '输入音频' }
        ],
        outputs: [
            { id: 'vocals', type: 'audio', label: '人声' },
            { id: 'instrumental', type: 'audio', label: '伴奏' },
            { id: 'drums', type: 'audio', label: '鼓组', optional: true },
            { id: 'bass', type: 'audio', label: '贝斯', optional: true }
        ]
    },
    split: {
        type: 'split',
        inputs: [
            { id: 'audio', type: 'audio', label: '输入音频' }
        ],
        outputs: [
            { id: 'out1', type: 'audio', label: '输出1' },
            { id: 'out2', type: 'audio', label: '输出2' },
            { id: 'out3', type: 'audio', label: '输出3', optional: true },
            { id: 'out4', type: 'audio', label: '输出4', optional: true }
        ]
    },
    merge: {
        type: 'merge',
        inputs: [
            { id: 'in1', type: 'audio', label: '输入1' },
            { id: 'in2', type: 'audio', label: '输入2' },
            { id: 'in3', type: 'audio', label: '输入3', optional: true },
            { id: 'in4', type: 'audio', label: '输入4', optional: true }
        ],
        outputs: [
            { id: 'audio', type: 'audio', label: '混音输出' }
        ]
    },
    complianceCheck: {
        type: 'complianceCheck',
        inputs: [
            { id: 'audio', type: 'audio', label: '输入音频' }
        ],
        outputs: [
            { id: 'audio', type: 'audio', label: '音频透传' },
            { id: 'report', type: 'report', label: '合规报告' }
        ]
    },
    lufsNormalize: {
        type: 'lufsNormalize',
        inputs: [
            { id: 'audio', type: 'audio', label: '输入音频' }
        ],
        outputs: [
            { id: 'audio', type: 'audio', label: '归一化音频' }
        ]
    },
    dither: {
        type: 'dither',
        inputs: [
            { id: 'audio', type: 'audio', label: '输入音频' }
        ],
        outputs: [
            { id: 'audio', type: 'audio', label: '抖动量化音频' }
        ]
    },
    busEq: {
        type: 'busEq',
        inputs: [
            { id: 'audio', type: 'audio', label: '输入音频' }
        ],
        outputs: [
            { id: 'audio', type: 'audio', label: 'EQ处理音频' }
        ]
    },
    stereoWidth: {
        type: 'stereoWidth',
        inputs: [
            { id: 'audio', type: 'audio', label: '输入音频' }
        ],
        outputs: [
            { id: 'audio', type: 'audio', label: '立体声宽度音频' }
        ]
    },
    saturate: {
        type: 'saturate',
        inputs: [
            { id: 'audio', type: 'audio', label: '输入音频' }
        ],
        outputs: [
            { id: 'audio', type: 'audio', label: '饱和音频' }
        ]
    },
    phaseRotate: {
        type: 'phaseRotate',
        inputs: [
            { id: 'audio', type: 'audio', label: '输入音频' }
        ],
        outputs: [
            { id: 'audio', type: 'audio', label: '相位旋转音频' }
        ]
    },
    dcRemove: {
        type: 'dcRemove',
        inputs: [
            { id: 'audio', type: 'audio', label: '输入音频' }
        ],
        outputs: [
            { id: 'audio', type: 'audio', label: 'DC去除音频' }
        ]
    },
    spectrogram: {
        type: 'spectrogram',
        inputs: [
            { id: 'audio', type: 'audio', label: '输入音频' }
        ],
        outputs: [
            { id: 'audio', type: 'audio', label: '音频透传' },
            { id: 'image', type: 'image', label: '频谱图' }
        ]
    },
    f0Curve: {
        type: 'f0Curve',
        inputs: [
            { id: 'audio', type: 'audio', label: '输入音频' }
        ],
        outputs: [
            { id: 'audio', type: 'audio', label: '音频透传' },
            { id: 'report', type: 'report', label: 'F0曲线数据' }
        ]
    },
    timbreMetrics: {
        type: 'timbreMetrics',
        inputs: [
            { id: 'audio', type: 'audio', label: '输入音频' }
        ],
        outputs: [
            { id: 'audio', type: 'audio', label: '音频透传' },
            { id: 'report', type: 'report', label: '音色指标' }
        ]
    },
    harmonicityCheck: {
        type: 'harmonicityCheck',
        inputs: [
            { id: 'audio', type: 'audio', label: '输入音频' }
        ],
        outputs: [
            { id: 'audio', type: 'audio', label: '音频透传' },
            { id: 'report', type: 'report', label: '谐波度报告' }
        ]
    },
    spectralCompare: {
        type: 'spectralCompare',
        inputs: [
            { id: 'audio1', type: 'audio', label: '参考音频' },
            { id: 'audio2', type: 'audio', label: '对比音频' }
        ],
        outputs: [
            { id: 'report', type: 'report', label: '频谱对比报告' }
        ]
    },
    dtwAlign: {
        type: 'dtwAlign',
        inputs: [
            { id: 'audio1', type: 'audio', label: '参考音频' },
            { id: 'audio2', type: 'audio', label: '待对齐音频' }
        ],
        outputs: [
            { id: 'audio', type: 'audio', label: '对齐后音频' },
            { id: 'report', type: 'report', label: '对齐报告' }
        ]
    },
    abCompare: {
        type: 'abCompare',
        inputs: [
            { id: 'audioA', type: 'audio', label: '方案A' },
            { id: 'audioB', type: 'audio', label: '方案B' }
        ],
        outputs: [
            { id: 'report', type: 'report', label: 'AB对比报告' }
        ]
    },
    lufsAnalyze: {
        type: 'lufsAnalyze',
        inputs: [
            { id: 'audio', type: 'audio', label: '输入音频' }
        ],
        outputs: [
            { id: 'audio', type: 'audio', label: '音频透传' },
            { id: 'report', type: 'report', label: 'LUFS报告' }
        ]
    },
    amtMidi: {
        type: 'amtMidi',
        inputs: [
            { id: 'audio', type: 'audio', label: '输入音频' }
        ],
        outputs: [
            { id: 'midi', type: 'midi', label: 'MIDI输出' },
            { id: 'report', type: 'report', label: '转换报告', optional: true }
        ]
    },
    speedShift: {
        type: 'speedShift',
        inputs: [
            { id: 'audio', type: 'audio', label: '输入音频' }
        ],
        outputs: [
            { id: 'audio', type: 'audio', label: '变速音频' }
        ]
    },
    chordDetect: {
        type: 'chordDetect',
        inputs: [
            { id: 'audio', type: 'audio', label: '输入音频' }
        ],
        outputs: [
            { id: 'chords', type: 'chords', label: '和弦序列' }
        ]
    },
    autoArrange: {
        type: 'autoArrange',
        inputs: [
            { id: 'midi', type: 'midi', label: '旋律MIDI' },
            { id: 'chords', type: 'chords', label: '和弦序列', optional: true }
        ],
        outputs: [
            { id: 'midi', type: 'midi', label: '编曲MIDI' }
        ]
    },
    deepOriginal: {
        type: 'deepOriginal',
        inputs: [
            { id: 'midi', type: 'midi', label: '输入MIDI' }
        ],
        outputs: [
            { id: 'midi', type: 'midi', label: '原创MIDI' }
        ]
    },
    midiFileIn: {
        type: 'midiFileIn',
        inputs: [],
        outputs: [
            { id: 'midi', type: 'midi', label: 'MIDI输出' }
        ]
    },
    chordBlockIn: {
        type: 'chordBlockIn',
        inputs: [],
        outputs: [
            { id: 'chords', type: 'chords', label: '和弦输出' }
        ]
    },
    harmonizer: {
        type: 'harmonizer',
        inputs: [
            { id: 'midi', type: 'midi', label: '主旋律' },
            { id: 'chords', type: 'chords', label: '和弦序列', optional: true }
        ],
        outputs: [
            { id: 'midi', type: 'midi', label: '和声MIDI' }
        ]
    },
    melodyGen: {
        type: 'melodyGen',
        inputs: [
            { id: 'chords', type: 'chords', label: '和弦序列' }
        ],
        outputs: [
            { id: 'midi', type: 'midi', label: '旋律MIDI' }
        ]
    },
    soundfontRender: {
        type: 'soundfontRender',
        inputs: [
            { id: 'midi', type: 'midi', label: 'MIDI输入' }
        ],
        outputs: [
            { id: 'audio', type: 'audio', label: '音频输出' }
        ]
    },
    melodySimilarity: {
        type: 'melodySimilarity',
        inputs: [
            { id: 'midi1', type: 'midi', label: '旋律1' },
            { id: 'midi2', type: 'midi', label: '旋律2' }
        ],
        outputs: [
            { id: 'report', type: 'report', label: '相似度报告' }
        ]
    },
    songLyrics: {
        type: 'songLyrics',
        inputs: [
            { id: 'text', type: 'lyrics', label: '歌词输入', optional: true }
        ],
        outputs: [
            { id: 'lyrics', type: 'lyrics', label: '歌词输出' }
        ]
    },
    songPrompt: {
        type: 'songPrompt',
        inputs: [],
        outputs: [
            { id: 'lyrics', type: 'lyrics', label: '提示词输出' }
        ]
    },
    songGen: {
        type: 'songGen',
        inputs: [
            { id: 'lyrics', type: 'lyrics', label: '歌词' },
            { id: 'midi', type: 'midi', label: '旋律', optional: true }
        ],
        outputs: [
            { id: 'audio', type: 'audio', label: '生成歌曲' }
        ]
    },
    songCover: {
        type: 'songCover',
        inputs: [
            { id: 'audio', type: 'audio', label: '原歌曲' }
        ],
        outputs: [
            { id: 'audio', type: 'audio', label: '翻唱版本' }
        ]
    },
    songRepaint: {
        type: 'songRepaint',
        inputs: [
            { id: 'audio', type: 'audio', label: '原音频' },
            { id: 'lyrics', type: 'lyrics', label: '新歌词', optional: true }
        ],
        outputs: [
            { id: 'audio', type: 'audio', label: '重绘音频' }
        ]
    },
    songComplete: {
        type: 'songComplete',
        inputs: [
            { id: 'audio', type: 'audio', label: '片段音频' }
        ],
        outputs: [
            { id: 'audio', type: 'audio', label: '完整歌曲' }
        ]
    },
    songExtract: {
        type: 'songExtract',
        inputs: [
            { id: 'audio', type: 'audio', label: '输入歌曲' }
        ],
        outputs: [
            { id: 'midi', type: 'midi', label: '提取MIDI' },
            { id: 'lyrics', type: 'lyrics', label: '提取歌词' }
        ]
    },
    songLego: {
        type: 'songLego',
        inputs: [
            { id: 'audio1', type: 'audio', label: '片段1' },
            { id: 'audio2', type: 'audio', label: '片段2' },
            { id: 'audio3', type: 'audio', label: '片段3', optional: true }
        ],
        outputs: [
            { id: 'audio', type: 'audio', label: '组合歌曲' }
        ]
    },
    songStems: {
        type: 'songStems',
        inputs: [
            { id: 'audio', type: 'audio', label: '输入歌曲' }
        ],
        outputs: [
            { id: 'vocals', type: 'audio', label: '人声' },
            { id: 'instrumental', type: 'audio', label: '伴奏' }
        ]
    },
    songSheet: {
        type: 'songSheet',
        inputs: [
            { id: 'audio', type: 'audio', label: '输入歌曲', optional: true },
            { id: 'midi', type: 'midi', label: '输入MIDI', optional: true }
        ],
        outputs: [
            { id: 'image', type: 'image', label: '乐谱图片' }
        ]
    },
    midiHumanize: {
        type: 'midiHumanize',
        inputs: [
            { id: 'midi', type: 'midi', label: '输入MIDI' }
        ],
        outputs: [
            { id: 'midi', type: 'midi', label: '人性化MIDI' }
        ]
    },
    velocityCurve: {
        type: 'velocityCurve',
        inputs: [
            { id: 'midi', type: 'midi', label: '输入MIDI' }
        ],
        outputs: [
            { id: 'midi', type: 'midi', label: '力度曲线MIDI' }
        ]
    },
    swingQuantize: {
        type: 'swingQuantize',
        inputs: [
            { id: 'midi', type: 'midi', label: '输入MIDI' }
        ],
        outputs: [
            { id: 'midi', type: 'midi', label: '摇摆量化MIDI' }
        ]
    },
    melodyReharm: {
        type: 'melodyReharm',
        inputs: [
            { id: 'midi', type: 'midi', label: '旋律MIDI' }
        ],
        outputs: [
            { id: 'midi', type: 'midi', label: '重新和声MIDI' },
            { id: 'chords', type: 'chords', label: '新和弦序列' }
        ]
    },
    rhythmRestructure: {
        type: 'rhythmRestructure',
        inputs: [
            { id: 'midi', type: 'midi', label: '输入MIDI' }
        ],
        outputs: [
            { id: 'midi', type: 'midi', label: '重构节奏MIDI' }
        ]
    },
    contourMorph: {
        type: 'contourMorph',
        inputs: [
            { id: 'midi1', type: 'midi', label: '源旋律' },
            { id: 'midi2', type: 'midi', label: '目标旋律' }
        ],
        outputs: [
            { id: 'midi', type: 'midi', label: '形变旋律' }
        ]
    },
    motifDevelop: {
        type: 'motifDevelop',
        inputs: [
            { id: 'midi', type: 'midi', label: '动机MIDI' }
        ],
        outputs: [
            { id: 'midi', type: 'midi', label: '发展MIDI' }
        ]
    },
    reharmonize: {
        type: 'reharmonize',
        inputs: [
            { id: 'midi', type: 'midi', label: '旋律MIDI' },
            { id: 'chords', type: 'chords', label: '原和弦', optional: true }
        ],
        outputs: [
            { id: 'chords', type: 'chords', label: '新和弦序列' }
        ]
    },
    rhythmVariation: {
        type: 'rhythmVariation',
        inputs: [
            { id: 'midi', type: 'midi', label: '输入MIDI' }
        ],
        outputs: [
            { id: 'midi', type: 'midi', label: '节奏变化MIDI' }
        ]
    },
    structureEdit: {
        type: 'structureEdit',
        inputs: [
            { id: 'midi', type: 'midi', label: '输入MIDI' }
        ],
        outputs: [
            { id: 'midi', type: 'midi', label: '结构编辑MIDI' }
        ]
    },
    breathPlanner: {
        type: 'breathPlanner',
        inputs: [
            { id: 'midi', type: 'midi', label: '旋律MIDI' },
            { id: 'lyrics', type: 'lyrics', label: '歌词', optional: true }
        ],
        outputs: [
            { id: 'midi', type: 'midi', label: '换气规划MIDI' }
        ]
    }
};
