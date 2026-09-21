/**
 * GM（General MIDI）乐器名三语言映射 + 名称→program 反查。
 *
 * 数据源与后端 `data/amt/python/src/models/models/gm_instruments.py` 保持一致：
 * 128 个标准 GM program + YourMT3 扩展名（Drums / 人声等）。
 * 用途：
 *   1. MIDI 转写结果的乐器轨道名（如 "Guitar (clean)"）在编辑器标题等
 *      纯显示位置翻译为「清音电吉他」——不改动轨道存储名（导出文件名/
 *      WAV 匹配键都以原名工作）。
 *   2. 乐器轨道试听时由轨道名反猜 GM program（鼓→channel 10）。
 */

export interface GmInstrument {
  program: number;
  en: string;
  zh: string;
  ja: string;
}

/** 标准 GM 128 乐器（0-127），与 gm_instruments.py 一一对应。 */
export const GM_INSTRUMENTS: readonly GmInstrument[] = [
  { program: 0, en: "Acoustic Grand Piano", zh: "大钢琴", ja: "アコースティック・ピアノ" },
  { program: 1, en: "Bright Acoustic Piano", zh: "明亮钢琴", ja: "ブライト・ピアノ" },
  { program: 2, en: "Electric Grand Piano", zh: "电钢琴", ja: "エレクトリック・グランドピアノ" },
  { program: 3, en: "Honky-tonk Piano", zh: "酒吧钢琴", ja: "ホンキートンク・ピアノ" },
  { program: 4, en: "Electric Piano 1", zh: "电子钢琴1", ja: "エレクトリック・ピアノ1" },
  { program: 5, en: "Electric Piano 2", zh: "电子钢琴2", ja: "エレクトリック・ピアノ2" },
  { program: 6, en: "Harpsichord", zh: "拨弦古钢琴", ja: "ハープシコード" },
  { program: 7, en: "Clavinet", zh: "克拉维内特琴", ja: "クラビネット" },
  { program: 8, en: "Celesta", zh: "钢片琴", ja: "チェレスタ" },
  { program: 9, en: "Glockenspiel", zh: "钟琴", ja: "グロッケンシュピール" },
  { program: 10, en: "Music Box", zh: "八音盒", ja: "ミュージックボックス" },
  { program: 11, en: "Vibraphone", zh: "颤音琴", ja: "ヴィブラフォン" },
  { program: 12, en: "Marimba", zh: "马林巴琴", ja: "マリンバ" },
  { program: 13, en: "Xylophone", zh: "木琴", ja: "シロフォン" },
  { program: 14, en: "Tubular Bells", zh: "管钟", ja: "チューブラー・ベル" },
  { program: 15, en: "Dulcimer", zh: "扬琴", ja: "ダルシマー" },
  { program: 16, en: "Drawbar Organ", zh: "拉杆风琴", ja: "ドローバー・オルガン" },
  { program: 17, en: "Percussive Organ", zh: "打击风琴", ja: "パーカッシブ・オルガン" },
  { program: 18, en: "Rock Organ", zh: "摇滚风琴", ja: "ロック・オルガン" },
  { program: 19, en: "Church Organ", zh: "教堂风琴", ja: "チャーチ・オルガン" },
  { program: 20, en: "Reed Organ", zh: "簧风琴", ja: "リード・オルガン" },
  { program: 21, en: "Accordion", zh: "手风琴", ja: "アコーディオン" },
  { program: 22, en: "Harmonica", zh: "口琴", ja: "ハーモニカ" },
  { program: 23, en: "Tango Accordion", zh: "探戈手风琴", ja: "タンゴ・アコーディオン" },
  { program: 24, en: "Acoustic Guitar (nylon)", zh: "尼龙弦吉他", ja: "アコースティック・ギター（ナイロン）" },
  { program: 25, en: "Acoustic Guitar (steel)", zh: "钢弦吉他", ja: "アコースティック・ギター（スチール）" },
  { program: 26, en: "Electric Guitar (jazz)", zh: "爵士电吉他", ja: "エレクトリック・ギター（ジャズ）" },
  { program: 27, en: "Electric Guitar (clean)", zh: "清音电吉他", ja: "エレクトリック・ギター（クリーン）" },
  { program: 28, en: "Electric Guitar (muted)", zh: "闷音电吉他", ja: "エレクトリック・ギター（ミュート）" },
  { program: 29, en: "Overdriven Guitar", zh: "过载吉他", ja: "オーバードライブ・ギター" },
  { program: 30, en: "Distortion Guitar", zh: "失真吉他", ja: "ディストーション・ギター" },
  { program: 31, en: "Guitar Harmonics", zh: "吉他泛音", ja: "ギター・ハーモニクス" },
  { program: 32, en: "Acoustic Bass", zh: "原声贝斯", ja: "アコースティック・ベース" },
  { program: 33, en: "Electric Bass (finger)", zh: "指弹电贝斯", ja: "エレクトリック・ベース（フィンガー）" },
  { program: 34, en: "Electric Bass (pick)", zh: "拨片电贝斯", ja: "エレクトリック・ベース（ピック）" },
  { program: 35, en: "Fretless Bass", zh: "无品贝斯", ja: "フレットレス・ベース" },
  { program: 36, en: "Slap Bass 1", zh: "拍击贝斯1", ja: "スラップ・ベース1" },
  { program: 37, en: "Slap Bass 2", zh: "拍击贝斯2", ja: "スラップ・ベース2" },
  { program: 38, en: "Synth Bass 1", zh: "合成贝斯1", ja: "シンセ・ベース1" },
  { program: 39, en: "Synth Bass 2", zh: "合成贝斯2", ja: "シンセ・ベース2" },
  { program: 40, en: "Violin", zh: "小提琴", ja: "ヴァイオリン" },
  { program: 41, en: "Viola", zh: "中提琴", ja: "ヴィオラ" },
  { program: 42, en: "Cello", zh: "大提琴", ja: "チェロ" },
  { program: 43, en: "Contrabass", zh: "低音提琴", ja: "コントラバス" },
  { program: 44, en: "Tremolo Strings", zh: "弦乐震音", ja: "トレモロ・ストリングス" },
  { program: 45, en: "Pizzicato Strings", zh: "弦乐拨奏", ja: "ピチカート・ストリングス" },
  { program: 46, en: "Orchestral Harp", zh: "竖琴", ja: "オーケストラル・ハープ" },
  { program: 47, en: "Timpani", zh: "定音鼓", ja: "ティンパニ" },
  { program: 48, en: "String Ensemble 1", zh: "弦乐合奏1", ja: "ストリング・アンサンブル1" },
  { program: 49, en: "String Ensemble 2", zh: "弦乐合奏2", ja: "ストリング・アンサンブル2" },
  { program: 50, en: "Synth Strings 1", zh: "合成弦乐1", ja: "シンセ・ストリングス1" },
  { program: 51, en: "Synth Strings 2", zh: "合成弦乐2", ja: "シンセ・ストリングス2" },
  { program: 52, en: "Choir Aahs", zh: "人声合唱", ja: "コーラス・アー" },
  { program: 53, en: "Voice Oohs", zh: "人声", ja: "ボイス・ウー" },
  { program: 54, en: "Synth Voice", zh: "合成人声", ja: "シンセ・ボイス" },
  { program: 55, en: "Orchestra Hit", zh: "管弦乐打击", ja: "オーケストラ・ヒット" },
  { program: 56, en: "Trumpet", zh: "小号", ja: "トランペット" },
  { program: 57, en: "Trombone", zh: "长号", ja: "トロンボーン" },
  { program: 58, en: "Tuba", zh: "大号", ja: "チューバ" },
  { program: 59, en: "Muted Trumpet", zh: "弱音小号", ja: "ミュート・トランペット" },
  { program: 60, en: "French Horn", zh: "圆号", ja: "フレンチ・ホルン" },
  { program: 61, en: "Brass Section", zh: "铜管组", ja: "ブラス・セクション" },
  { program: 62, en: "Synth Brass 1", zh: "合成铜管1", ja: "シンセ・ブラス1" },
  { program: 63, en: "Synth Brass 2", zh: "合成铜管2", ja: "シンセ・ブラス2" },
  { program: 64, en: "Soprano Sax", zh: "高音萨克斯", ja: "ソプラノ・サックス" },
  { program: 65, en: "Alto Sax", zh: "中音萨克斯", ja: "アルト・サックス" },
  { program: 66, en: "Tenor Sax", zh: "次中音萨克斯", ja: "テナー・サックス" },
  { program: 67, en: "Baritone Sax", zh: "上低音萨克斯", ja: "バリトン・サックス" },
  { program: 68, en: "Oboe", zh: "双簧管", ja: "オーボエ" },
  { program: 69, en: "English Horn", zh: "英国管", ja: "イングリッシュホルン" },
  { program: 70, en: "Bassoon", zh: "巴松管", ja: "バスーン" },
  { program: 71, en: "Clarinet", zh: "单簧管", ja: "クラリネット" },
  { program: 72, en: "Piccolo", zh: "短笛", ja: "ピッコロ" },
  { program: 73, en: "Flute", zh: "长笛", ja: "フルート" },
  { program: 74, en: "Recorder", zh: "竖笛", ja: "リコーダー" },
  { program: 75, en: "Pan Flute", zh: "排箫", ja: "パン・フルート" },
  { program: 76, en: "Blown Bottle", zh: "吹瓶", ja: "ブロウン・ボトル" },
  { program: 77, en: "Shakuhachi", zh: "尺八", ja: "尺八" },
  { program: 78, en: "Whistle", zh: "口哨", ja: "ホイッスル" },
  { program: 79, en: "Ocarina", zh: "陶笛", ja: "オカリナ" },
  { program: 80, en: "Lead 1 (square)", zh: "方波主音", ja: "リード1（スクエア）" },
  { program: 81, en: "Lead 2 (sawtooth)", zh: "锯齿波主音", ja: "リード2（ソートゥース）" },
  { program: 82, en: "Lead 3 (calliope)", zh: "汽笛风琴主音", ja: "リード3（カリオペ）" },
  { program: 83, en: "Lead 4 (chiff)", zh: "吹管主音", ja: "リード4（チフ）" },
  { program: 84, en: "Lead 5 (charang)", zh: "吉他主音", ja: "リード5（チャランゴ）" },
  { program: 85, en: "Lead 6 (voice)", zh: "人声主音", ja: "リード6（ボイス）" },
  { program: 86, en: "Lead 7 (fifths)", zh: "五度主音", ja: "リード7（フィフス）" },
  { program: 87, en: "Lead 8 (bass + lead)", zh: "贝斯主音", ja: "リード8（ベース+リード）" },
  { program: 88, en: "Pad 1 (new age)", zh: "新世纪铺底", ja: "パッド1（ニューエイジ）" },
  { program: 89, en: "Pad 2 (warm)", zh: "温暖铺底", ja: "パッド2（ウォーム）" },
  { program: 90, en: "Pad 3 (polysynth)", zh: "复音合成铺底", ja: "パッド3（ポリシンセ）" },
  { program: 91, en: "Pad 4 (choir)", zh: "合唱铺底", ja: "パッド4（クワイア）" },
  { program: 92, en: "Pad 5 (bowed)", zh: "弓弦铺底", ja: "パッド5（ボウ）" },
  { program: 93, en: "Pad 6 (metallic)", zh: "金属铺底", ja: "パッド6（メタリック）" },
  { program: 94, en: "Pad 7 (halo)", zh: "光环铺底", ja: "パッド7（ハロー）" },
  { program: 95, en: "Pad 8 (sweep)", zh: "扫频铺底", ja: "パッド8（スイープ）" },
  { program: 96, en: "FX 1 (rain)", zh: "雨声", ja: "FX1（レイン）" },
  { program: 97, en: "FX 2 (soundtrack)", zh: "原声带", ja: "FX2（サウンドトラック）" },
  { program: 98, en: "FX 3 (crystal)", zh: "水晶", ja: "FX3（クリスタル）" },
  { program: 99, en: "FX 4 (atmosphere)", zh: "氛围", ja: "FX4（アトモスフィア）" },
  { program: 100, en: "FX 5 (brightness)", zh: "明亮", ja: "FX5（ブライトネス）" },
  { program: 101, en: "FX 6 (goblins)", zh: "鬼魅", ja: "FX6（ゴブリン）" },
  { program: 102, en: "FX 7 (echoes)", zh: "回声", ja: "FX7（エコーズ）" },
  { program: 103, en: "FX 8 (sci-fi)", zh: "科幻", ja: "FX8（SF）" },
  { program: 104, en: "Sitar", zh: "西塔尔琴", ja: "シタール" },
  { program: 105, en: "Banjo", zh: "班卓琴", ja: "バンジョー" },
  { program: 106, en: "Shamisen", zh: "三味线", ja: "三味線" },
  { program: 107, en: "Koto", zh: "古筝", ja: "琴" },
  { program: 108, en: "Kalimba", zh: "卡林巴琴", ja: "カリンバ" },
  { program: 109, en: "Bag pipe", zh: "风笛", ja: "バグパイプ" },
  { program: 110, en: "Fiddle", zh: "小提琴", ja: "フィドル" },
  { program: 111, en: "Shanai", zh: "唢呐", ja: "シャナイ" },
  { program: 112, en: "Tinkle Bell", zh: "铃铛", ja: "ティンクルベル" },
  { program: 113, en: "Agogo", zh: "阿哥哥鼓", ja: "アゴゴ" },
  { program: 114, en: "Steel Drums", zh: "钢鼓", ja: "スチールドラム" },
  { program: 115, en: "Woodblock", zh: "木鱼", ja: "ウッドブロック" },
  { program: 116, en: "Taiko Drum", zh: "太鼓", ja: "太鼓" },
  { program: 117, en: "Melodic Tom", zh: "旋律嗵鼓", ja: "メロディック・タム" },
  { program: 118, en: "Synth Drum", zh: "合成鼓", ja: "シンセ・ドラム" },
  { program: 119, en: "Reverse Cymbal", zh: "反镲", ja: "リバース・シンバル" },
  { program: 120, en: "Guitar Fret Noise", zh: "吉他品噪音", ja: "ギター・フレットノイズ" },
  { program: 121, en: "Breath Noise", zh: "呼吸声", ja: "ブレスノイズ" },
  { program: 122, en: "Seashore", zh: "海浪", ja: "シーショア" },
  { program: 123, en: "Bird Tweet", zh: "鸟鸣", ja: "バードツイート" },
  { program: 124, en: "Telephone Ring", zh: "电话铃", ja: "テレフォン・リング" },
  { program: 125, en: "Helicopter", zh: "直升机", ja: "ヘリコプター" },
  { program: 126, en: "Applause", zh: "掌声", ja: "拍手" },
  { program: 127, en: "Gunshot", zh: "枪声", ja: "ガンショット" },
];

/**
 * 名称别名 → GM program。覆盖 YourMT3 / 常见转写器的口语化乐器名
 * （"Guitar (clean)"、"Drums"、"Vocals"、"Piano"…），这些不是标准 GM 全名。
 * key 一律先过 normalizeKey 再查。
 */
const NAME_ALIASES: Record<string, { program: number; zh: string; ja: string; en: string }> = {
  "drums": { program: 0, zh: "鼓组", ja: "ドラム", en: "Drums" },
  "drum": { program: 0, zh: "鼓组", ja: "ドラム", en: "Drums" },
  "drum kit": { program: 0, zh: "鼓组", ja: "ドラム", en: "Drum Kit" },
  "percussion": { program: 0, zh: "打击乐", ja: "パーカッション", en: "Percussion" },
  "vocals": { program: 53, zh: "人声", ja: "ボーカル", en: "Vocals" },
  "vocal": { program: 53, zh: "人声", ja: "ボーカル", en: "Vocal" },
  "singing voice": { program: 53, zh: "人声", ja: "ボーカル", en: "Singing Voice" },
  "voice": { program: 53, zh: "人声", ja: "ボーカル", en: "Voice" },
  "人声": { program: 53, zh: "人声", ja: "ボーカル", en: "Vocals" },
  "piano": { program: 0, zh: "大钢琴", ja: "ピアノ", en: "Piano" },
  "electric piano": { program: 4, zh: "电子钢琴", ja: "エレクトリック・ピアノ", en: "Electric Piano" },
  "guitar": { program: 25, zh: "吉他", ja: "ギター", en: "Guitar" },
  "guitar (clean)": { program: 27, zh: "清音电吉他", ja: "クリーンギター", en: "Guitar (clean)" },
  "guitar (distorted)": { program: 30, zh: "失真吉他", ja: "ディストーションギター", en: "Guitar (distorted)" },
  "guitar (jazz)": { program: 26, zh: "爵士电吉他", ja: "ジャズギター", en: "Guitar (jazz)" },
  "acoustic guitar": { program: 25, zh: "钢弦吉他", ja: "アコースティックギター", en: "Acoustic Guitar" },
  "electric guitar": { program: 27, zh: "清音电吉他", ja: "エレクトリックギター", en: "Electric Guitar" },
  "bass": { program: 33, zh: "电贝斯", ja: "ベース", en: "Bass" },
  "electric bass": { program: 33, zh: "电贝斯", ja: "エレクトリックベース", en: "Electric Bass" },
  "strings": { program: 48, zh: "弦乐", ja: "ストリングス", en: "Strings" },
  "string ensemble": { program: 48, zh: "弦乐合奏", ja: "ストリングアンサンブル", en: "String Ensemble" },
  "organ": { program: 16, zh: "风琴", ja: "オルガン", en: "Organ" },
  "flute": { program: 73, zh: "长笛", ja: "フルート", en: "Flute" },
  "piccolo": { program: 72, zh: "短笛", ja: "ピッコロ", en: "Piccolo" },
  "clarinet": { program: 71, zh: "单簧管", ja: "クラリネット", en: "Clarinet" },
  "sax": { program: 66, zh: "萨克斯", ja: "サックス", en: "Sax" },
  "saxophone": { program: 66, zh: "萨克斯", ja: "サックス", en: "Saxophone" },
  "trumpet": { program: 56, zh: "小号", ja: "トランペット", en: "Trumpet" },
  "trombone": { program: 57, zh: "长号", ja: "トロンボーン", en: "Trombone" },
  "horn": { program: 60, zh: "圆号", ja: "ホルン", en: "Horn" },
  "violin": { program: 40, zh: "小提琴", ja: "ヴァイオリン", en: "Violin" },
  "cello": { program: 42, zh: "大提琴", ja: "チェロ", en: "Cello" },
  "viola": { program: 41, zh: "中提琴", ja: "ヴィオラ", en: "Viola" },
  "harp": { program: 46, zh: "竖琴", ja: "ハープ", en: "Harp" },
  "harpsichord": { program: 6, zh: "拨弦古钢琴", ja: "ハープシコード", en: "Harpsichord" },
  "accordion": { program: 21, zh: "手风琴", ja: "アコーディオン", en: "Accordion" },
  "bells": { program: 14, zh: "管钟", ja: "ベル", en: "Bells" },
  "vibes": { program: 11, zh: "颤音琴", ja: "ヴィブラフォン", en: "Vibes" },
  "marimba": { program: 12, zh: "马林巴琴", ja: "マリンバ", en: "Marimba" },
  "synth lead": { program: 80, zh: "合成主音", ja: "シンセリード", en: "Synth Lead" },
  "synth": { program: 80, zh: "合成音色", ja: "シンセ", en: "Synth" },
  "synth pad": { program: 88, zh: "合成铺底", ja: "シンセパッド", en: "Synth Pad" },
  "choir": { program: 52, zh: "人声合唱", ja: "クワイア", en: "Choir" },
  "backup vocals": { program: 53, zh: "和声人声", ja: "コーラス", en: "Backup Vocals" },
  "harmonica": { program: 22, zh: "口琴", ja: "ハーモニカ", en: "Harmonica" },
};

/** 查表用的规范化 key：小写、去多余空白；保留括号与连字符（GM 名里语义重要）。 */
const normalizeKey = (s: string): string => s.trim().toLowerCase().replace(/\s+/g, " ");

const BY_EN_KEY = new Map<string, GmInstrument>(GM_INSTRUMENTS.map((g) => [normalizeKey(g.en), g]));

export interface GmNameMatch {
  program: number;
  /** 翻译后的显示名（无匹配时 = 原名）。 */
  display: string;
  /** 是否命中了已知 GM/别名表。 */
  matched: boolean;
}

/**
 * 由轨道名（GM 全名或 YourMT3 口语名）解析显示名与 GM program。
 * 匹配顺序：别名精确 → GM 全名精确 → 原样返回（display=原名, program=undefined 语义用 matched 表达）。
 */
export function matchGmName(name: string, lang: string): GmNameMatch {
  const key = normalizeKey(name || "");
  if (!key) return { program: 0, display: name, matched: false };
  const alias = NAME_ALIASES[key];
  if (alias) {
    const display = lang.startsWith("zh") ? alias.zh : lang.startsWith("ja") ? alias.ja : alias.en;
    return { program: alias.program, display, matched: true };
  }
  const exact = BY_EN_KEY.get(key);
  if (exact) {
    const display = lang.startsWith("zh") ? exact.zh : lang.startsWith("ja") ? exact.ja : exact.en;
    return { program: exact.program, display, matched: true };
  }
  return { program: 0, display: name, matched: false };
}

/** 纯显示用：乐器轨道名 → 本地化名（无匹配时原样返回）。 */
export function translateGmName(name: string, lang: string): string {
  return matchGmName(name, lang).display;
}

/** 名称是否更像鼓轨（用于试听通道选择）。 */
export function isDrumLikeName(name: string): boolean {
  const key = normalizeKey(name);
  return key === "drums" || key === "drum" || key === "drum kit" || key === "percussion" || key.includes("drum");
}

/**
 * §user "演唱的干音都命名 vocal、都要有歌词"：判定一条转写轨道是否人声干音。
 * 覆盖：vocal_split / six_stem 的 "vocals" 键、YourMT3 的 "Voice"/"Singing Voice"、
 * GM 合成人声（Choir/Aahs/Oohs）、中文命名（人声/干声/主唱）。鼓轨永远不算。
 */
const VOCAL_NAME_RE =
  /vocal|voice|singer|singing|choir|aahs|oohs|人声|干声|干音|主唱|演唱|唱歌/i;

export function isVocalTrackName(name: string): boolean {
  const key = normalizeKey(name || "");
  if (!key) return false;
  if (isDrumLikeName(key)) return false; // "Drum Voice" 之类的打击乐名不算干音
  return VOCAL_NAME_RE.test(key);
}

/** 干音轨统一显示名（§user：演唱干音一律叫 Vocal）。 */
export const VOCAL_STEM_NAME = "Vocal";

/* ── GM 标准鼓组映射（channel 9）：音高 → 鼓件名，三语言。 ── */
export interface GmDrum {
  en: string;
  zh: string;
  ja: string;
  /** 分组标签（用于速读：底鼓 / 军鼓 / 镲 / 打击）。 */
  group: "kick" | "snare" | "hihat" | "cymbal" | "tom" | "perc";
}

const GM_DRUMS: Record<number, GmDrum> = {
  35: { en: "Acoustic Bass Drum", zh: "底鼓（原声）", ja: "バスドラム", group: "kick" },
  36: { en: "Bass Drum", zh: "底鼓", ja: "バスドラム", group: "kick" },
  37: { en: "Side Stick", zh: "边击", ja: "サイドスティック", group: "snare" },
  38: { en: "Acoustic Snare", zh: "军鼓", ja: "スネア", group: "snare" },
  39: { en: "Hand Clap", zh: "拍手", ja: "ハンドクラップ", group: "snare" },
  40: { en: "Electric Snare", zh: "军鼓（电子）", ja: "エレクトリック・スネア", group: "snare" },
  41: { en: "Low Floor Tom", zh: "低音落地嗵", ja: "ローフロアタム", group: "tom" },
  42: { en: "Closed Hi-Hat", zh: "闭镲", ja: "クローズハイハット", group: "hihat" },
  43: { en: "High Floor Tom", zh: "高音落地嗵", ja: "ハイフロアタム", group: "tom" },
  44: { en: "Pedal Hi-Hat", zh: "踩镲", ja: "ペダルハイハット", group: "hihat" },
  45: { en: "Low Tom", zh: "低音嗵", ja: "ロータム", group: "tom" },
  46: { en: "Open Hi-Hat", zh: "开镲", ja: "オープンハイハット", group: "hihat" },
  47: { en: "Low-Mid Tom", zh: "中低音嗵", ja: "ローミッドタム", group: "tom" },
  48: { en: "Hi-Mid Tom", zh: "中高音嗵", ja: "ハイミッドタム", group: "tom" },
  49: { en: "Crash Cymbal 1", zh: "吊镲 1", ja: "クラッシュシンバル1", group: "cymbal" },
  50: { en: "High Tom", zh: "高音嗵", ja: "ハイタム", group: "tom" },
  51: { en: "Ride Cymbal 1", zh: "叮镲 1", ja: "ライドシンバル1", group: "cymbal" },
  52: { en: "Chinese Cymbal", zh: "中国镲", ja: "チャイナシンバル", group: "cymbal" },
  53: { en: "Ride Bell", zh: "镲帽", ja: "ライドベル", group: "cymbal" },
  54: { en: "Tambourine", zh: "铃鼓", ja: "タンバリン", group: "perc" },
  55: { en: "Splash Cymbal", zh: "溅镲", ja: "スプラッシュシンバル", group: "cymbal" },
  56: { en: "Cowbell", zh: "牛铃", ja: "カウベル", group: "perc" },
  57: { en: "Crash Cymbal 2", zh: "吊镲 2", ja: "クラッシュシンバル2", group: "cymbal" },
  58: { en: "Vibraslap", zh: "颤击筒", ja: "ビブラスラップ", group: "perc" },
  59: { en: "Ride Cymbal 2", zh: "叮镲 2", ja: "ライドシンバル2", group: "cymbal" },
  60: { en: "Hi Bongo", zh: "高邦戈鼓", ja: "ハイボンゴ", group: "perc" },
  61: { en: "Low Bongo", zh: "低邦戈鼓", ja: "ローボンゴ", group: "perc" },
  62: { en: "Mute Hi Conga", zh: "康加鼓（闷）", ja: "ハイコンガ（ミュート）", group: "perc" },
  63: { en: "Open Hi Conga", zh: "康加鼓（开）", ja: "ハイコンガ（オープン）", group: "perc" },
  64: { en: "Low Conga", zh: "低康加鼓", ja: "ローコンガ", group: "perc" },
  65: { en: "High Timbale", zh: "高天巴鼓", ja: "ハイティンバル", group: "perc" },
  66: { en: "Low Timbale", zh: "低天巴鼓", ja: "ローティンバル", group: "perc" },
  67: { en: "High Agogo", zh: "高阿哥哥铃", ja: "ハイアゴゴ", group: "perc" },
  68: { en: "Low Agogo", zh: "低阿哥哥铃", ja: "ローアゴゴ", group: "perc" },
  69: { en: "Cabasa", zh: "卡巴沙", ja: "カバサ", group: "perc" },
  70: { en: "Maracas", zh: "沙锤", ja: "マラカス", group: "perc" },
  71: { en: "Short Whistle", zh: "短口哨", ja: "ショートホイッスル", group: "perc" },
  72: { en: "Long Whistle", zh: "长口哨", ja: "ロングホイッスル", group: "perc" },
  73: { en: "Short Guiro", zh: "短刮瓜", ja: "ショートギロ", group: "perc" },
  74: { en: "Long Guiro", zh: "长刮瓜", ja: "ロングギロ", group: "perc" },
  75: { en: "Claves", zh: "响棒", ja: "クラベス", group: "perc" },
  76: { en: "Hi Wood Block", zh: "高木鱼", ja: "ハイウッドブロック", group: "perc" },
  77: { en: "Low Wood Block", zh: "低木鱼", ja: "ローウッドブロック", group: "perc" },
  78: { en: "Mute Cuica", zh: "库加鼓（闷）", ja: "クイーカ（ミュート）", group: "perc" },
  79: { en: "Open Cuica", zh: "库加鼓（开）", ja: "クイーカ（オープン）", group: "perc" },
  80: { en: "Mute Triangle", zh: "三角铁（闷）", ja: "トライアングル（ミュート）", group: "perc" },
  81: { en: "Open Triangle", zh: "三角铁", ja: "トライアングル（オープン）", group: "perc" },
};

/** 鼓件名（无映射的音高回退为空串，调用方自行决定显示音名）。 */
export function gmDrumName(pitch: number, lang: string): string {
  const d = GM_DRUMS[pitch];
  if (!d) return "";
  return lang.startsWith("zh") ? d.zh : lang.startsWith("ja") ? d.ja : d.en;
}

/** 鼓件短标签（键列窄空间用）：底/军/闭镲/开镲/吊镲… */
export function gmDrumShort(pitch: number, lang: string): string {
  const d = GM_DRUMS[pitch];
  if (!d) return "";
  if (lang.startsWith("zh")) {
    switch (d.group) {
      case "kick": return "底鼓";
      case "snare": return pitch === 37 ? "边击" : pitch === 39 ? "拍手" : "军鼓";
      case "hihat": return pitch === 44 ? "踩镲" : pitch === 42 ? "闭镲" : "开镲";
      case "cymbal": return pitch === 51 || pitch === 59 ? "叮镲" : pitch === 53 ? "镲帽" : "吊镲";
      case "tom": return "嗵鼓";
      default: return d.zh;
    }
  }
  if (lang.startsWith("ja")) {
    switch (d.group) {
      case "kick": return "バスドラ";
      case "snare": return "スネア";
      case "hihat": return pitch === 44 ? "ペダルHH" : pitch === 42 ? "クローズHH" : "オープンHH";
      case "cymbal": return "シンバル";
      case "tom": return "タム";
      default: return d.ja;
    }
  }
  switch (d.group) {
    case "kick": return "Kick";
    case "snare": return pitch === 37 ? "Stick" : "Snare";
    case "hihat": return pitch === 44 ? "Pedal HH" : pitch === 42 ? "Closed HH" : "Open HH";
    case "cymbal": return pitch === 51 || pitch === 59 ? "Ride" : pitch === 53 ? "Ride Bell" : "Crash";
    case "tom": return "Tom";
    default: return d.en;
  }
}

/** 鼓轨排序权重：按组聚集（底鼓→军鼓→嗵→闭/开镲→吊/叮镲→打击），组内按音高。 */
const DRUM_GROUP_ORDER: Record<GmDrum["group"], number> = {
  kick: 0, snare: 1, tom: 2, hihat: 3, cymbal: 4, perc: 5,
};
export function gmDrumOrder(pitch: number): number {
  const d = GM_DRUMS[pitch];
  return d ? DRUM_GROUP_ORDER[d.group] * 1000 + pitch : pitch;
}
