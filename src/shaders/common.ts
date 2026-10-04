/**
 * GLSL shared chunks.
 *
 * Kept as TS template strings rather than `.glsl` files + a loader plugin so
 * there is no extra build step and no fetch at runtime — the shader is part of
 * the bundle, which matters for a wallpaper that must start instantly.
 */

export const GLSL_HASH = /* glsl */ `
float hash12(vec2 p){
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

vec2 hash22(vec2 p){
  vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz) * p3.zy);
}

vec3 hash33(vec3 p){
  p = fract(p * vec3(0.1031, 0.1030, 0.0973));
  p += dot(p, p.yxz + 33.33);
  return fract((p.xxy + p.yxx) * p.zyx);
}
`;

/**
 * Dither.
 *
 * Two rules, both learned the hard way:
 *
 *  1. The offset must be *static in screen space*. A wallpaper's noise that
 *     moves relative to the image reads as a smudge on the lens, not as air.
 *     It also means the offset cannot be a function of `uTime`, or frame 0 and
 *     frame N stop being comparable and §6 becomes untestable.
 *  2. It should be interleaved gradient noise, not a hash. IGN is a structured
 *     dither with an even spatial spectrum; hash noise clumps, and clumping in
 *     a volumetric integrator shows up as blotches in the fog.
 */
export const GLSL_DITHER = /* glsl */ `
// Jimenez's interleaved gradient noise.
float ign(vec2 p){
  return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}
`;

export const GLSL_NOISE = /* glsl */ `
float vnoise2(vec2 p){
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float a = hash12(i);
  float b = hash12(i + vec2(1.0, 0.0));
  float c = hash12(i + vec2(0.0, 1.0));
  float d = hash12(i + vec2(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float vnoise3(vec3 p){
  vec3 i = floor(p), f = fract(p);
  vec3 u = f * f * (3.0 - 2.0 * f);
  float n000 = hash12(i.xy + i.z * 37.0);
  float n100 = hash12(i.xy + vec2(1.0, 0.0) + i.z * 37.0);
  float n010 = hash12(i.xy + vec2(0.0, 1.0) + i.z * 37.0);
  float n110 = hash12(i.xy + vec2(1.0, 1.0) + i.z * 37.0);
  float n001 = hash12(i.xy + (i.z + 1.0) * 37.0);
  float n101 = hash12(i.xy + vec2(1.0, 0.0) + (i.z + 1.0) * 37.0);
  float n011 = hash12(i.xy + vec2(0.0, 1.0) + (i.z + 1.0) * 37.0);
  float n111 = hash12(i.xy + vec2(1.0, 1.0) + (i.z + 1.0) * 37.0);
  return mix(
    mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
    mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
    u.z);
}

float fbm2(vec2 p, int octaves){
  float s = 0.0, a = 0.5, n = 0.0;
  for (int i = 0; i < 8; i++){
    if (i >= octaves) break;
    s += vnoise2(p) * a;
    n += a;
    a *= 0.5;
    p = p * 2.03 + vec2(17.3, 9.1);
  }
  return s / max(n, 1e-4);
}

float fbm3(vec3 p, int octaves){
  float s = 0.0, a = 0.5, n = 0.0;
  for (int i = 0; i < 8; i++){
    if (i >= octaves) break;
    s += vnoise3(p) * a;
    n += a;
    a *= 0.5;
    p = p * 2.02 + vec3(11.7, 5.3, 19.1);
  }
  return s / max(n, 1e-4);
}

// Ridged multifractal. The crease is what reads as a rock face rather than a
// dune, so scene authors reach for this whenever they need geology.
float ridged2(vec2 p, int octaves){
  float s = 0.0, a = 0.5, n = 0.0;
  for (int i = 0; i < 8; i++){
    if (i >= octaves) break;
    float r = 1.0 - abs(vnoise2(p) * 2.0 - 1.0);
    s += r * r * a;
    n += a;
    a *= 0.5;
    p = p * 2.07 + vec2(5.2, 1.3);
  }
  return s / max(n, 1e-4);
}
`;

/**
 * Cyclic noise in the fragment shader.
 *
 * The wallpaper loop has to hold in GLSL exactly as it does in JS. These sample
 * a lattice indexed by an integer *phase* rather than by time, so the signal is
 * periodic by construction — pass `uPhase` (0..1) and the result wraps cleanly.
 * Time-based fbm cannot do this: it either drifts or has a visible seam.
 */
export const GLSL_CYCLIC = /* glsl */ `
const float TAU = 6.283185307179586;

float cycHash(float i, float seed){
  return hash12(vec2(mod(i, 1024.0), seed));
}

// 1D noise on a lattice of period n, continuous in phase with zero slope at wrap.
float cycNoise1(float phase, float n, float seed){
  float x = fract(phase) * n;
  float i = floor(x);
  float f = x - i;
  f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  float a = cycHash(mod(i, n), seed);
  float b = cycHash(mod(i + 1.0, n), seed);
  return mix(a, b, f);
}

float cycFbm1(float phase, float seed, float baseCycles, int octaves){
  float s = 0.0, a = 0.5, n = 0.0;
  for (int i = 0; i < 8; i++){
    if (i >= octaves) break;
    s += cycNoise1(phase, baseCycles * exp2(float(i)), seed + float(i) * 71.0) * a;
    n += a;
    a *= 0.5;
  }
  return s / max(n, 1e-4);
}

/**
 * A closed drift for a 2D noise domain, in noise-space units.
 *
 * The obvious way to make a field drift across the loop is a linear ramp,
 * \`p += phase * k\`. It is wrong, and wrong in a way that is invisible in a
 * still frame: at the wrap the entire field teleports by k. One cell of value
 * noise is one unit here, so a k of 0.4 is not a nudge, it is a different
 * landscape — every ridge ends up somewhere else, and the loop restarts visibly
 * once per cycle.
 *
 * Sampling along a circle instead returns to the start with a matching
 * derivative, so the field keeps genuinely changing for the whole loop and
 * closes without a seam. \`squash\` turns the orbit into an ellipse when the two
 * axes should travel at different rates.
 */
vec2 driftCircle(float phase, float radius, float squash){
  float a = phase * TAU;
  return vec2(cos(a) * radius, sin(a) * radius * squash);
}
`;

/**
 * Scattering.
 *
 * Every volumetric in this project is in-scattering, so the one function that
 * decides *where* the light comes out is shared rather than copied: a phase
 * function is easy to get subtly wrong and impossible to tune once it is
 * duplicated into four shaders.
 */
export const GLSL_SCATTER = /* glsl */ `
/** Henyey-Greenstein. g > 0 is forward-scattering — light piles up when you look
 *  toward the source, which is the entire reason god rays exist. */
float hgPhase(float cosT, float g){
  float g2 = g * g;
  return (1.0 - g2) / (4.0 * 3.14159265 * pow(1.0 + g2 - 2.0 * g * cosT, 1.5));
}
`;

export const GLSL_COLOR = /* glsl */ `
vec3 acesToneMap(vec3 x){
  const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

float luma(vec3 c){ return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

vec3 saturateColor(vec3 c, float amount){
  return mix(vec3(luma(c)), c, amount);
}

// Lift / gamma / gain. The cheapest way to make a render look graded rather
// than computed.
vec3 grade(vec3 c, vec3 lift, vec3 gain, float gamma, float sat, float contrast){
  c = pow(max(c + lift, 0.0), vec3(gamma)) * gain;
  c = (c - 0.5) * contrast + 0.5;
  return saturateColor(c, sat);
}

vec3 srgbToLinear(vec3 c){
  return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}

vec3 linearToSrgb(vec3 c){
  return mix(c * 12.92, 1.055 * pow(max(c, 1e-5), vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}
`;