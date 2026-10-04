/**
 * The loop contract.
 *
 * A living wallpaper runs for hours. Two of the project's hard requirements
 * collapse into one rule:
 *
 *   §6  seamless infinite loop        →  state must return to its start
 *   §14 no drift, no memory growth    →  state must not accumulate
 *
 * Both are satisfied if and only if every animated quantity is a *pure function
 * of the loop phase* φ ∈ [0,1), and no scene ever mutates state across frames.
 *
 * So: a scene receives φ, and rebuilds everything from φ. Nothing else.
 * Consequences:
 *
 *   • the loop is seamless by construction, including velocity — every helper
 *     here is C¹-continuous across the wrap, so there is no visible jolt
 *   • there is no accumulated error, so frame 10 000 is identical to frame 10
 *   • nothing to leak, so uptime is unbounded
 *   • any φ can be rendered in isolation, which makes the whole thing testable
 *     and recordable
 *
 * The rule for scene authors, stated once:
 *
 *   ❌  this.angle += dt * this.speed
 *   ✅  const angle = phase * TAU * this.turns;
 */

export const TAU = Math.PI * 2;

export const clamp = (v: number, lo: number, hi: number): number =>
  v < lo ? lo : v > hi ? hi : v;

export const clamp01 = (v: number): number => clamp(v, 0, 1);

export const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;

export const smoothstep = (e0: number, e1: number, v: number): number => {
  const x = clamp01((v - e0) / (e1 - e0 || 1e-9));
  return x * x * (3 - 2 * x);
};

export const smootherstep = (e0: number, e1: number, v: number): number => {
  const x = clamp01((v - e0) / (e1 - e0 || 1e-9));
  return x * x * x * (x * (x * 6 - 15) + 10);
};

/* ------------------------------------------------------------------ *
 * Phase → time
 * ------------------------------------------------------------------ */

/** Wrap any elapsed time into [0, period). */
export const wrapPhase = (elapsed: number, period: number): number => {
  const p = elapsed / period;
  return p - Math.floor(p);
};

/**
 * Integer harmonics. The workhorse: `sin(TAU * n * φ)` for integer n is exactly
 * periodic in φ, and so is its derivative, so motion is seamless in value *and*
 * velocity. Use a small set of low harmonics and the result never looks like a
 * loop — it looks like weather.
 */
export const harmonic = (phase: number, n: number, offset = 0): number =>
  Math.sin(TAU * n * phase + offset);

export const harmonic2 = (phase: number, nx: number, ny: number): number =>
  Math.sin(TAU * nx * phase) * Math.cos(TAU * ny * phase);

/**
 * A smooth 0→1→0 excursion with zero derivative at both ends. This is what
 * "breathing" should be built from — not a linear ramp, which snaps back.
 */
export const breathe = (phase: number, n = 1, sharp = 1): number => {
  const x = wrapPhase(phase * n, 1);
  return Math.pow(Math.sin(Math.PI * x), sharp);
};

/**
 * Triangle wave with smoothed corners — useful for slow drifts that must not
 * have a velocity discontinuity at the turn.
 */
export const pingPong = (phase: number, n = 1, soft = 0.12): number => {
  const x = wrapPhase(phase * n, 1);
  const tri = x < 0.5 ? x * 2 : 2 - x * 2;
  const edge = smoothstep(0, soft, tri) * smoothstep(0, soft, 1 - tri);
  return lerp(tri, edge, 0.65);
};

/**
 * A drifting offset that is itself periodic in φ. Use this for anything that
 * needs to travel — fog, currents, light drift — without ever teleporting.
 * Returns a signed offset in roughly -1..1.
 */
export const drift = (phase: number, n: number, offset = 0): number =>
  harmonic(phase, n, offset);

/* ------------------------------------------------------------------ *
 * Cyclic value noise
 * ------------------------------------------------------------------ */

const hash1i = (i: number, seed: number): number => {
  let t = (Math.imul(i + seed * 1013, 0x27d4eb2d) ^ 0x165667b1) >>> 0;
  t = Math.imul(t ^ (t >>> 15), 0x2c1b3c6d);
  t = Math.imul(t ^ (t >>> 13), 0x297a2d39);
  return ((t ^ (t >>> 16)) >>> 0) / 4294967296;
};

/**
 * 1D value noise on an integer lattice of period `n`, so the result is exactly
 * periodic in φ. Continuous in φ with zero derivative at the wrap.
 */
export const cyclicNoise1 = (phase: number, n: number, seed = 0): number => {
  const x = wrapPhase(phase, 1) * n;
  const i = Math.floor(x);
  const f = smootherstep(0, 1, x - i);
  const wrap = (k: number): number => ((k % n) + n) % n;
  const a = hash1i(wrap(i), seed);
  const b = hash1i(wrap(i + 1), seed);
  return lerp(a, b, f);
};

/** Fractal cyclic noise. Every octave shares the period, so the sum wraps too. */
export const cyclicFbm1 = (
  phase: number,
  seed = 0,
  octaves = 4,
  baseCycles = 1,
  gain = 0.5,
): number => {
  let sum = 0;
  let amp = 1;
  let norm = 0;
  for (let o = 0; o < octaves; o++) {
    sum += cyclicNoise1(phase, baseCycles * 2 ** o, seed + o * 71) * amp;
    norm += amp;
    amp *= gain;
  }
  return sum / norm;
};

/** Signed variant in -1..1, the form most scene parameters want. */
export const cyclicSigned = (
  phase: number,
  seed = 0,
  octaves = 3,
  baseCycles = 1,
): number => cyclicFbm1(phase, seed, octaves, baseCycles) * 2 - 1;

/* ------------------------------------------------------------------ *
 * Framing helpers
 * ------------------------------------------------------------------ */

/**
 * Settle rate for values that *should* respond to change (mouse parallax,
 * quality scaling) but must never accumulate error. `approach` is
 * frame-rate independent and decays exactly toward the target, so it cannot
 * drift — it is the one sanctioned exception to the purity rule, and it only
 * ever touches camera offsets and render quality, never scene state.
 */
export const approach = (
  current: number,
  target: number,
  halfLife: number,
  dt: number,
): number => target + (current - target) * Math.pow(2, -dt / Math.max(1e-4, halfLife));