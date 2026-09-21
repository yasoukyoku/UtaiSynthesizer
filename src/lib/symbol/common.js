export const SINGABLE_LOW = 36; // C2
export const SINGABLE_HIGH = 84; // C6
export function clamp01(v) {
    return Math.min(1, Math.max(0, v));
}
export function clampVel(v) {
    return Math.min(127, Math.max(1, Math.round(v)));
}
export function mod12p(v) {
    return ((v % 12) + 12) % 12;
}
export function byTick(a, b) {
    return a.tick - b.tick || a.pitch - b.pitch;
}
