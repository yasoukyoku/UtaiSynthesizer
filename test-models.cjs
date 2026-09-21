// 简单的 Node.js 脚本来测试模型检测逻辑
const fs = require('fs');
const path = require('path');

const modelsDir = 'e:\\软件开发\\UtaiSynthesizer-main\\UtaiSynthesizer-main\\src-tauri\\song_models';

// 模拟 Rust 的 list_song_models 函数
function listSongModels(dir) {
  const models = [];
  const exts = ['safetensors', 'ckpt', 'pt', 'pth', 'onnx', 'whl', 'bin', 'yaml', 'json', 'sf2', 'mid', 'wav'];
  
  function scan(currentPath) {
    const entries = fs.readdirSync(currentPath, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(currentPath, entry.name);
      if (entry.isDirectory()) {
        scan(fullPath);
      } else {
        const ext = path.extname(entry.name).slice(1).toLowerCase();
        if (exts.includes(ext)) {
          const rel = path.relative(dir, fullPath);
          const filename = rel.replace(/\\/g, '/');
          const size = fs.statSync(fullPath).size;
          models.push({ filename, size });
        }
      }
    }
  }
  
  scan(dir);
  return models.sort((a, b) => a.filename.localeCompare(b.filename));
}

// 模型定义
const SONG_MODEL_CATALOG = [
  {
    id: "yue2-3b",
    files: [
      { path: "model.safetensors" },
      { path: "config.json" },
      { path: "generation_config.json" },
      { path: "yue2_generation_config.json" },
      { path: "weights_manifest.json" },
      { path: "yue2_infer-0.1.5-py3-none-any.whl" },
      { path: "vae/model.safetensors" },
      { path: "vae/config.json" },
      { path: "vae/weights_manifest.json" },
    ]
  },
  {
    id: "acestep-v1-3.5b",
    files: [
      { path: "ace_step_v1_3.5b.safetensors" }
    ]
  },
  {
    id: "acestep-v1.5",
    files: [
      { path: "unet/acestep_v1.5_xl_turbo_bf16.safetensors" },
      { path: "vae/ace_1.5_vae.safetensors" },
      { path: "text_encoders/qwen_0.6b_ace15.safetensors" },
      { path: "text_encoders/qwen_1.7b_ace15.safetensors" },
    ]
  },
  {
    id: "heartmula-3b",
    files: [
      { path: "HeartMuLa-oss-3B/config.json" },
      { path: "HeartMuLa-oss-3B/model.safetensors.index.json" },
      { path: "HeartMuLa-oss-3B/model-00001-of-00004.safetensors" },
      { path: "HeartMuLa-oss-3B/model-00002-of-00004.safetensors" },
      { path: "HeartMuLa-oss-3B/model-00003-of-00004.safetensors" },
      { path: "HeartMuLa-oss-3B/model-00004-of-00004.safetensors" },
      { path: "HeartCodec-oss/config.json" },
      { path: "HeartCodec-oss/model.safetensors.index.json" },
      { path: "HeartCodec-oss/model-00001-of-00002.safetensors" },
      { path: "HeartCodec-oss/model-00002-of-00002.safetensors" },
      { path: "tokenizer.json" },
      { path: "gen_config.json" },
    ]
  }
];

const LISTED_EXTS = ['safetensors', 'ckpt', 'pt', 'pth', 'onnx', 'whl', 'bin', 'yaml', 'json', 'sf2', 'mid', 'wav'];

function getSongInstallStatus(installedFiles, model) {
  const listed = new Set(installedFiles.map(f => f.filename.replace(/\\/g, '/')));
  const prefix = `${model.id}/`;
  const required = [];
  const missing = [];
  let ready = 0;
  
  for (const f of model.files) {
    const ext = f.path.split('.').pop()?.toLowerCase() ?? '';
    if (!LISTED_EXTS.includes(ext)) continue;
    required.push(f.path);
    const fullPath = prefix + f.path;
    if (listed.has(fullPath)) {
      ready += 1;
    } else {
      missing.push(f.path);
    }
  }
  
  return { 
    installed: required.length > 0 && missing.length === 0, 
    ready, 
    required: required.length, 
    missing 
  };
}

// 执行测试
console.log('=== 开始检测模型文件 ===\n');

const files = listSongModels(modelsDir);
console.log(`检测到 ${files.length} 个文件:\n`);
files.forEach(f => console.log(`  ${f.filename}`));

console.log('\n=== 模型检测结果 ===\n');

for (const model of SONG_MODEL_CATALOG) {
  const status = getSongInstallStatus(files, model);
  console.log(`${model.id}:`);
  console.log(`  状态: ${status.installed ? '🟢 已安装' : '🔴 未安装'}`);
  console.log(`  进度: ${status.ready}/${status.required}`);
  if (status.missing.length > 0) {
    console.log(`  缺失文件:`);
    status.missing.forEach(f => console.log(`    - ${f}`));
  }
  console.log('');
}
