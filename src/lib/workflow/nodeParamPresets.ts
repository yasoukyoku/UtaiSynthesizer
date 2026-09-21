/**
 * 节点参数预设（规划 Opt 7）：把单个节点当前的 params 存成命名预设，之后在任意同类型节点上一键套用。
 *
 * 和"工作流预设"（整图，走后端 save_workflow_preset / load_workflow_presets）是两回事——这里只管
 * 一个节点的参数，存在 localStorage 里 **按节点类型分桶**，所以在 RVC 节点上存的预设只会出现在
 * RVC 节点的右键菜单里，不会污染别的类型。
 */
import { loadSetting, saveSetting } from "../settings";

const STORAGE_KEY = "utai.nodeParamPresets";

/** 每种节点类型最多存多少条——localStorage 容量有限，而且菜单再长就不好用了。 */
export const MAX_PRESETS_PER_TYPE = 30;

export interface NodeParamPreset {
  name: string;
  params: Record<string, unknown>;
  updatedAt: number;
}

/** key = ReactFlow 节点 type（rvc / soVits / …）。 */
type PresetStore = Record<string, NodeParamPreset[]>;

/** localStorage 里的内容可能被手改、或来自旧版本——形状不对就当空处理，别让右键菜单炸掉。 */
function readStore(): PresetStore {
  const raw = loadSetting<PresetStore>(STORAGE_KEY, {});
  return raw !== null && typeof raw === "object" && !Array.isArray(raw) ? raw : {};
}

/** 只保留能 JSON 往返的值：params 正常都是原始值，但万一混进函数 / undefined，
 *  存进去再读出来就会变形，不如在入口处一次性过滤干净。 */
function sanitizeParams(params: Record<string, unknown>): Record<string, unknown> | null {
  try {
    const round = JSON.parse(JSON.stringify(params)) as unknown;
    return round !== null && typeof round === "object" && !Array.isArray(round)
      ? (round as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
}

/** 某节点类型下已存的预设，按名称排序（菜单顺序稳定，不随保存先后跳动）。 */
export function listNodeParamPresets(nodeType: string): NodeParamPreset[] {
  const list = readStore()[nodeType];
  if (!Array.isArray(list)) return [];
  return list
    .filter((p) => p !== null && typeof p === "object" && typeof p.name === "string")
    .map((p) => ({ name: p.name, params: p.params ?? {}, updatedAt: p.updatedAt ?? 0 }))
    .sort((a, b) => a.name.localeCompare(b.name));
}

/** 存 / 覆盖一条预设。同名视为覆盖；到达上限且是个新名字时返回 false（由调用方提示用户）。 */
export function saveNodeParamPreset(
  nodeType: string,
  name: string,
  params: Record<string, unknown>,
): boolean {
  const trimmed = name.trim();
  if (!trimmed) return false;
  const clean = sanitizeParams(params);
  if (!clean) return false;
  const store = readStore();
  const list = Array.isArray(store[nodeType]) ? store[nodeType] : [];
  const at = list.findIndex((p) => p?.name === trimmed);
  if (at < 0 && list.length >= MAX_PRESETS_PER_TYPE) return false;
  const entry: NodeParamPreset = { name: trimmed, params: clean, updatedAt: Date.now() };
  const next = at >= 0 ? list.map((p, i) => (i === at ? entry : p)) : [...list, entry];
  saveSetting(STORAGE_KEY, { ...store, [nodeType]: next });
  return true;
}

/** 删一条预设；删空之后把该类型的桶一并移除，别在存储里留一堆空数组。 */
export function deleteNodeParamPreset(nodeType: string, name: string): void {
  const store = readStore();
  const list = Array.isArray(store[nodeType]) ? store[nodeType] : [];
  const next = list.filter((p) => p?.name !== name);
  if (next.length === list.length) return;
  const nextStore = { ...store };
  if (next.length === 0) delete nextStore[nodeType];
  else nextStore[nodeType] = next;
  saveSetting(STORAGE_KEY, nextStore);
}
