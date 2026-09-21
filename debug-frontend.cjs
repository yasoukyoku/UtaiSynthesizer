// 临时调试脚本：模拟前端检测逻辑
const fs = require("fs");
const path = require("path");

const BASE = path.join(__dirname, "src-tauri", "models", "song");

// 从实际的 catalog 文件读取模型定义
const catalogPath = path.join(__dirname, "src", "lib", "models", "song-catalog.ts");
const catalogContent = fs.readFileSync(catalogPath, "utf-8");

console.log("=== 检查 song-catalog.ts 文件内容 ===\n");

// 检查 ACE-Step v1 的定义
const aceV1Match = catalogContent.match(/id:\s*"acestep-v1-3\.5b"[\s\S]{0,500}files:\s*\[([\s\S]*?)\]/);
if (aceV1Match) {
  console.log("ACE-Step v1 定义:");
  console.log(aceV1Match[0]);
  console.log("\n");
}

// 检查 ACE-Step v1.5 的定义
const aceV15Match = catalogContent.match(/id:\s*"acestep-v1\.5"[\s\S]{0,800}files:\s*\[([\s\S]*?)\]/);
if (aceV15Match) {
  console.log("ACE-Step v1.5 定义:");
  console.log(aceV15Match[0]);
  console.log("\n");
}

// 模拟后端返回的文件列表（与 Rust 代码一致）
function listSongModels() {
  const result = [];
  const dirs = fs.readdirSync(BASE);
  
  for (const modelId of dirs) {
    const modelPath = path.join(BASE, modelId);
    if (!fs.statSync(modelPath).isDirectory()) continue;
    
    function walk(dir, prefix = "") {
      const items = fs.readdirSync(dir);
      for (const item of items) {
        const fullPath = path.join(dir, item);
        const stat = fs.statSync(fullPath);
        const relPath = prefix ? `${prefix}/${item}` : item;
        
        if (stat.isDirectory()) {
          walk(fullPath, relPath);
        } else {
          result.push({
            filename: `${modelId}/${relPath}`.replace(/\\/g, "/"),
            size: stat.size,
          });
        }
      }
    }
    
    walk(modelPath);
  }
  
  return result;
}

const files = listSongModels();

console.log("=== 后端返回的文件列表 ===\n");
console.log(`共 ${files.length} 个文件:\n`);
files.forEach((f) => {
  if (f.filename.includes("acestep")) {
    console.log(`  ${f.filename} (${f.size} bytes)`);
  }
});

console.log("\n=== 前端检测逻辑模拟 ===\n");

// ACE-Step v1 手动检测
const aceV1Files = files.filter((f) => f.filename.startsWith("acestep-v1-3.5b/"));
console.log(`ACE-Step v1 找到 ${aceV1Files.length} 个文件:`);
aceV1Files.forEach((f) => console.log(`  ${f.filename}`));

// ACE-Step v1.5 手动检测
const aceV15Files = files.filter((f) => f.filename.startsWith("acestep-v1.5/"));
console.log(`\nACE-Step v1.5 找到 ${aceV15Files.length} 个文件:`);
aceV15Files.forEach((f) => console.log(`  ${f.filename}`));

console.log("\n=== 关键检查 ===");
console.log("\n1. ACE-Step v1 期望文件: ace_step_v1_3.5b.safetensors");
console.log("   实际存在:", aceV1Files.some((f) => f.filename === "acestep-v1-3.5b/ace_step_v1_3.5b.safetensors") ? "✅" : "❌");

console.log("\n2. ACE-Step v1.5 期望文件:");
const v15Expected = [
  "unet/acestep_v1.5_xl_turbo_bf16.safetensors",
  "vae/ace_1.5_vae.safetensors",
  "text_encoders/qwen_0.6b_ace15.safetensors",
  "text_encoders/qwen_1.7b_ace15.safetensors",
];
v15Expected.forEach((expected) => {
  const fullPath = `acestep-v1.5/${expected}`;
  const exists = aceV15Files.some((f) => f.filename === fullPath);
  console.log(`   ${expected}: ${exists ? "✅" : "❌"}`);
});
