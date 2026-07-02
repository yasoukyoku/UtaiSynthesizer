/**
 * Tiny typed localStorage helpers for persisted UI / software settings (snap toggles, download
 * source, …). Safe to call when storage is unavailable (returns the fallback / silently no-ops).
 * Keys are namespaced with the `utai.` prefix. (The interface language is persisted separately under
 * the bare `lang` key by i18n; the inference device is persisted backend-side in config.json.)
 */
export function loadSetting<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(key);
    return v === null ? fallback : (JSON.parse(v) as T);
  } catch {
    return fallback;
  }
}

export function saveSetting(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable / quota exceeded — ignore */
  }
}
