/** 最近打开的工程(localStorage 持久化,启动欢迎页卡片展示)。
 *  只在前端记录 路径+名字+时间;文件是否仍然存在由欢迎页在打开前用 path_exists 探测,
 *  失效条目自动从列表移除(不在这里做异步探测 —— 列表读取必须同步,供 useState 初始化)。 */
export interface RecentProject {
  path: string;
  name: string;
  at: number;
}

const KEY = "utai.recentProjects";
const CAP = 10;

export function getRecentProjects(): RecentProject[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const list = JSON.parse(raw);
    if (!Array.isArray(list)) return [];
    return list.filter(
      (p): p is RecentProject =>
        !!p && typeof p.path === "string" && typeof p.name === "string" && typeof p.at === "number",
    );
  } catch {
    return []; // 损坏的 JSON / 隐私模式 — 当作无历史
  }
}

function write(list: RecentProject[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(list));
  } catch {
    /* 配额满/隐私模式 — 静默降级(本次会话内列表仍可用) */
  }
}

/** 记录一次"工程被保存/打开":同路径去重后置顶,最多保留 CAP 条。 */
export function rememberRecentProject(path: string, name: string): void {
  const list = getRecentProjects().filter((p) => p.path !== path);
  list.unshift({ path, name, at: Date.now() });
  write(list.slice(0, CAP));
}

export function removeRecentProject(path: string): void {
  write(getRecentProjects().filter((p) => p.path !== path));
}

/** 更新历史记录中某个工程的显示名称（不影响文件系统）*/
export function updateRecentProjectName(path: string, newName: string): void {
  const list = getRecentProjects();
  const index = list.findIndex((p) => p.path === path);
  if (index >= 0) {
    const item = list[index];
    if (item) {
      list[index] = { path: item.path, name: newName, at: item.at };
    }
    write(list);
  }
}
