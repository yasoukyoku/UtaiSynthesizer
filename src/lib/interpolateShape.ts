// ② Vocal pitch — easing shapes (S51 SynthV-aligned). PURE sine/linear easings, reused by the note-to-note
// transition curve in lib/f0eval.ts. These 4 are standard easings (not tied to any upstream model). We do
// NOT port OpenUTAU's per-point pitch-point polyline model: SynthV shapes a note connection with the
// Transition PARAMETERS (offset / durLeft / durRight / depthLeft / depthRight), evaluated in f0eval (§10.3).

export type PitchShape = "linear" | "sineIn" | "sineOut" | "sineInOut";

/**
 * Easing f(t) ∈ [0,1] for t ∈ [0,1], clamped outside:
 *   linear     f = t
 *   sineIn     f = 1 − cos(t·π/2)     (slow start / ease-in)
 *   sineOut    f = sin(t·π/2)         (slow end / ease-out)
 *   sineInOut  f = (1 − cos(t·π))/2   (S-curve — the natural default for a note-to-note glide)
 */
export function interpShape(t: number, shape: PitchShape): number {
  const x = t <= 0 ? 0 : t >= 1 ? 1 : t;
  switch (shape) {
    case "sineIn": return 1 - Math.cos((x * Math.PI) / 2);
    case "sineOut": return Math.sin((x * Math.PI) / 2);
    case "sineInOut": return (1 - Math.cos(x * Math.PI)) / 2;
    case "linear":
    default: return x;
  }
}
