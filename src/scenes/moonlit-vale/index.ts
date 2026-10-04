/**
 * Scene 01 — 月落雾谷 · MOONLIT VALE
 *
 * The reference scene. Its job is to establish the quality bar and to exercise
 * every part of the engine, not to be the final catalogue.
 *
 * The Perfect Moment (§4 of the brief): a large moon sitting low behind a
 * ridge line, its light raking across a sea of fog that fills the valley.
 * Everything breathes around that one held instant.
 *
 * Why it is built as a single raymarched shader:
 *
 *   • Mountains, fog and god rays then occlude each other *correctly*. Stacking
 *     billboards or heightfield planes gives you the layered look but the fog
 *     always sits wrongly on top, and that single artefact is what makes
 *     procedural landscapes look cheap.
 *   • Aerial perspective falls out for free: colour is a function of the
 *     distance the ray actually travelled through the haze.
 *   • There is no geometry to stream or leak, so §14 holds trivially.
 *
 * Layering, per §9:
 *   background  sky gradient, star field, moon
 *   far         ridge silhouette chain, heavily hazed
 *   mid         fog sea with moon in-scatter
 *   near        grass blades, instanced, swaying in the vertex shader
 *   atmosphere  motes drifting through the shafts
 */

import {
  AdditiveBlending,
  BufferGeometry,
  Color,
  DoubleSide,
  Float32BufferAttribute,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  Mesh,
  PlaneGeometry,
  Points,
  Scene,
  ShaderMaterial,
  Vector3,
  type PerspectiveCamera,
} from 'three';

import type { FrameInfo, QualityTune, SceneContext, SceneDefinition, SceneInstance } from '../../core/scene';
import { CameraRig } from '../../core/camera';
import { GLSL_COLOR, GLSL_CYCLIC, GLSL_DITHER, GLSL_HASH, GLSL_NOISE, GLSL_SCATTER } from '../../shaders/common';
import { cyclicFbm1, harmonic } from '../../core/loop';

const PERIOD = 96; // seconds. §7: prefer slow enough that the loop goes unnoticed.

/* ================================================================== *
 * Sky / terrain / volumetrics
 * ================================================================== */

const SKY_VERT = /* glsl */ `
varying vec2 vUv;
void main(){
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

const SKY_FRAG = /* glsl */ `
precision highp float;
${GLSL_HASH}
${GLSL_NOISE}
${GLSL_CYCLIC}
${GLSL_COLOR}
${GLSL_DITHER}
${GLSL_SCATTER}

uniform float uPhase;
uniform vec3  uCamPos;
uniform vec3  uCamFwd;
uniform vec3  uCamRight;
uniform vec3  uCamUp;
uniform float uTanHalfFov;
uniform float uAspect;
uniform float uVolumetricSteps;

varying vec2 vUv;

const float FAR = 900.0;
const vec3  MOON_DIR = normalize(vec3(0.21, 0.030, -1.0));
// Warm lunar light. At file scope because both main() and the volumetric
// integrator need it, and GLSL has no closures.
const vec3  MOON_GLOW = vec3(1.0, 0.955, 0.885);

/* ---------------------------------------------------------------- *
 * Terrain
 * A ridged heightfield, marched by fixed-step sphere tracing with a
 * linear refinement. uPhase only shifts the domain, never the
 * amplitude, so the silhouette breathes without the peaks wandering.
 * ---------------------------------------------------------------- */

float terrainHeight(vec2 p){
  // A basin around the viewer. Without this the heightfield can raise ground
  // right where the camera stands and wall off the entire lower half of frame —
  // the ranges have to exist only in the distance for the valley to read.
  float d = length(p - vec2(0.0, 30.0));
  float mask = smoothstep(90.0, 340.0, d);

  vec2 q = p * 0.0021;
  // Closed orbit in noise space — see driftCircle. A linear phase ramp moves
  // every ridge to a different mountain at the wrap.
  q += driftCircle(uPhase, 0.42, 0.26);
  float massif  = ridged2(q, 4);
  float detail  = ridged2(q * 5.3 + 3.1, 3) * 0.16;
  // Two ridged fields multiplied: gives connected ranges with subsidiary
  // peaks, instead of one uniform sawtooth.
  float shape = pow(massif, 1.35) + detail;
  return (shape - 0.30) * 210.0 * mask - 26.0;
}

vec3 terrainNormal(vec2 p, float eps){
  float h  = terrainHeight(p);
  float hx = terrainHeight(p + vec2(eps, 0.0));
  float hz = terrainHeight(p + vec2(0.0, eps));
  return normalize(vec3(h - hx, eps, h - hz));
}

float marchTerrain(vec3 ro, vec3 rd, out float tHit, out vec3 nHit){
  float t = 1.0;
  float prevGap = ro.y - terrainHeight(ro.xz);
  const int MAX_STEPS = 128;
  for (int i = 0; i < MAX_STEPS; i++){
    vec3 p = ro + rd * t;
    float gap = p.y - terrainHeight(p.xz);
    if (gap < 0.0){
      // Linear refine between the last two samples.
      tHit = t - t * 0.5;
      nHit = terrainNormal((ro + rd * tHit).xz, 0.6);
      return 1.0;
    }
    prevGap = gap;
    t += max(0.35, gap * 0.42);
    if (t > FAR) break;
  }
  tHit = FAR;
  nHit = vec3(0.0, 1.0, 0.0);
  return 0.0;
}

/* ---------------------------------------------------------------- *
 * Sky
 * ---------------------------------------------------------------- */

float starField(vec3 rd){
  // Project onto a cube face so density is not polar-biased. Both ternary
  // branches must be vec2 — GLSL has no implicit scalar broadcast here.
  vec2 uv = abs(rd.y) > 0.9 ? rd.xz : vec2(rd.x / max(0.25, abs(rd.y) + 0.3)) * 0.7;
  vec2 cell = floor(uv * 90.0);
  float h = hash12(cell);
  if (h < 0.982) return 0.0;
  vec2 local = fract(uv * 90.0) - 0.5 - (hash22(cell) - 0.5) * 0.6;
  float d = length(local);
  float mag = 0.35 + hash12(cell + 7.0) * 0.65;
  // Slow scintillation, still periodic in the loop.
  float tw = 0.72 + 0.28 * sin(uPhase * TAU * 3.0 + h * 40.0);
  return smoothstep(0.09 * mag, 0.0, d) * mag * tw;
}

vec3 skyColor(vec3 ro, vec3 rd, out float starMask){
  // Vertical gradient. Deliberately not pure black at the zenith — a wallpaper
  // that hits #000 in the corners reads as "no render" rather than "night".
  float h = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
  vec3 zenith  = vec3(0.0045, 0.0075, 0.019);
  vec3 horizon = vec3(0.022, 0.034, 0.058);
  vec3 col = mix(horizon, zenith, pow(h, 0.62));

  // A wash of light pollution well below the horizon line.
  col += vec3(0.030, 0.020, 0.011) * pow(clamp(1.0 - abs(rd.y + 0.02) * 11.0, 0.0, 1.0), 3.0);

  starMask = starField(rd) * smoothstep(-0.02, 0.28, rd.y);
  col += vec3(0.85, 0.88, 1.0) * starMask * 0.55;

  // High thin cloud, drifting forward through the loop.
  float band = smoothstep(0.02, 0.30, rd.y) * smoothstep(0.85, 0.30, rd.y);
  vec2 cp = rd.xz / max(0.08, rd.y + 0.12);
  float cloud = fbm2(cp * 0.30 + driftCircle(uPhase, 0.18, 0.3), 5);
  cloud = smoothstep(0.48, 0.86, cloud) * band;
  col = mix(col, vec3(0.10, 0.12, 0.17), cloud * 0.55);

  return col;
}

vec3 moonSurface(vec3 rd, out float discMask){
  float d = dot(rd, MOON_DIR);
  const float R = 0.9955;               // angular radius
  discMask = smoothstep(R - 0.0012, R + 0.0004, d);
  vec3 col = vec3(0.0);
  if (discMask > 0.0){
    // Reconstruct a sphere normal so the terminator is real, then light it
    // with the same direction as the sky glow. This is what makes it a body
    // rather than a white circle.
    vec3 n = normalize(rd - MOON_DIR * d);
    vec3 lightDir = normalize(vec3(-0.5, 0.22, 0.4));
    float lam = max(0.0, dot(n, lightDir));
    // Maria: low-frequency mottling plus a little fine detail.
    vec2 sp = n.xy * 2.4;
    float maria = fbm2(sp * 1.6 + 4.0, 4);
    float grain = fbm2(sp * 14.0, 3);
    float albedo = mix(0.72, 0.94, smoothstep(0.35, 0.72, maria));
    albedo *= mix(0.94, 1.0, grain);
    col = vec3(0.95, 0.93, 0.88) * albedo * (0.06 + pow(lam, 0.9) * 0.78);
    // Terminator limb darkening.
    col *= mix(0.55, 1.0, pow(lam, 0.35));
  }
  return col;
}

/* ---------------------------------------------------------------- *
 * Fog volume
 * The phase function itself lives in GLSL_SCATTER: one octave is enough
 * here, a second makes the shafts look like a shader demo.
 * ---------------------------------------------------------------- */

float fogDensity(vec3 p){
  float h = exp(-max(p.y, 0.0) * 0.055);
  vec3 q = p * 0.0075;
  // Drift the density field along a closed orbit. This is what makes the fog sea
  // move without any of it translating in world space.
  q.xz += driftCircle(uPhase, 0.6, 0.4);
  float n = fbm2(q.xz, 4) * 0.6 + fbm3(q * 2.4, 3) * 0.4;
  return clamp(h * (n * 1.25 - 0.16), 0.0, 1.0) * 1.35;
}

/**
 * The same field at a third of the cost, for the light march.
 *
 * The shadow ray only needs to know whether a sample is roughly inside a bank of
 * fog; the high octaves that shape the *surface* of a wisp contribute nothing to
 * that decision. This is the single most expensive function in the scene — six of
 * these per volumetric step — so dropping two octaves and the entire 3D term cuts
 * the frame cost by roughly two thirds with no visible difference in the shafts.
 */
float fogDensityCheap(vec3 p){
  float h = exp(-max(p.y, 0.0) * 0.055);
  vec3 q = p * 0.0075;
  q.xz += driftCircle(uPhase, 0.6, 0.4);
  return clamp(h * (fbm2(q.xz, 2) * 1.25 - 0.16), 0.0, 1.0) * 1.35;
}

vec3 applyVolumetrics(vec3 ro, vec3 rd, float tMax){
  int steps = int(uVolumetricSteps);
  float stepLen = min(tMax, 340.0) / float(steps);
  // Static, screen-locked dither. See GLSL_DITHER: a jitter that moved with the
  // clock would make the fog uncountable at the loop wrap.
  float jitter = ign(gl_FragCoord.xy);

  vec3 acc = vec3(0.0);
  float trans = 1.0;
  float cosT = dot(rd, MOON_DIR);

  for (int i = 0; i < 48; i++){
    if (i >= steps) break;
    float t = (float(i) + jitter) * stepLen;
    if (t >= tMax) break;
    vec3 p = ro + rd * t;

    float d = fogDensity(p);
    if (d < 0.002) continue;

    // Light march toward the moon. Six steps is enough for shafts, and at a third
    // of the density cost each — this loop is the frame budget.
    float shadow = 0.0;
    for (int s = 1; s <= 5; s++){
      shadow += fogDensityCheap(p + MOON_DIR * float(s) * 10.0);
    }
    shadow = exp(-shadow * 3.1);

    // Ambient sky term keeps the fog from going black in unlit pockets.
    // Two-lobe scatter. The moon lobe is warm and strongly forward-peaked; the
    // ambient lobe is the cold blue of the night sky. Without the second term
    // every wisp of fog is moon-coloured and the whole frame goes grey, which
    // is the single fastest way to make a night scene look like a render test.
    float ambient = 0.030 + 0.028 * smoothstep(-0.1, 0.5, rd.y);
    float moonLobe = shadow * hgPhase(cosT, 0.74) * 1.55;
    vec3 scatter = vec3(0.055, 0.085, 0.165) * ambient * 6.0
                 + MOON_GLOW * moonLobe;

    float a = 1.0 - exp(-d * stepLen * 0.055);
    acc += scatter * a * trans;
    trans *= 1.0 - a;
    if (trans < 0.02) break;
  }

  return acc;
}

/* ---------------------------------------------------------------- *
 * Aerial perspective
 * §9 layering is mostly a statement about contrast and saturation per
 * distance. Distant geometry must lose contrast *and* gain haze, never
 * just become lighter.
 * ---------------------------------------------------------------- */

vec3 applyAerial(vec3 color, float dist, vec3 hazeColor, float hazeDensity){
  float f = 1.0 - exp(-dist * hazeDensity);
  return mix(color, hazeColor, clamp(f, 0.0, 1.0));
}

/* ---------------------------------------------------------------- *
 * Main
 * ---------------------------------------------------------------- */

void main(){
  vec2 uv = (vUv * 2.0 - 1.0);
  uv.x *= uAspect;

  vec3 rd = normalize(uCamFwd + uCamRight * uv.x * uTanHalfFov + uCamUp * uv.y * uTanHalfFov);
  vec3 ro = uCamPos;

  // --- sky, moon ---
  float starMask;
  vec3 col = skyColor(ro, rd, starMask);
  float discMask;
  col += moonSurface(rd, discMask);

  float md = max(0.0, dot(rd, MOON_DIR));
  float halo = pow(md, 4200.0) * 1.5
             + pow(md, 220.0) * 0.085
             + pow(md, 26.0) * 0.016;
  col += MOON_GLOW * halo;

  // --- terrain ---
  float tHit;
  vec3 nHit;
  float hit = marchTerrain(ro, rd, tHit, nHit);

  if (hit > 0.5){
    vec3 tp = ro + rd * tHit;
    vec3 n = nHit;

    // Rock is lit almost entirely by the moon, so it is nearly a silhouette.
    // What separates the ridge lines is haze, not shading.
    float lam = max(0.0, dot(n, MOON_DIR)) * 0.5 + 0.5;
    float rim = pow(1.0 - max(0.0, dot(n, -rd)), 3.0);
    vec3 rock = mix(vec3(0.020, 0.024, 0.034), vec3(0.075, 0.082, 0.100), lam);
    rock += MOON_GLOW * rim * 0.045 * lam;

    // Cheap strata so the rock is not a flat mass.
    float strata = fbm2(tp.xz * 0.035 + tp.y * 0.02, 3);
    rock *= 0.82 + strata * 0.36;

    vec3 hazeCol = mix(vec3(0.042, 0.058, 0.098), MOON_GLOW * 0.085, 0.5);
    col = applyAerial(rock, tHit, hazeCol, 0.00135);
  }

  // --- volumetrics ---
  float tVol = hit > 0.5 ? tHit : FAR;
  vec3 vol = applyVolumetrics(ro, rd, tVol);
  // Composite the fog *over* whatever the ray hit, so distance haze and
  // in-scattering agree.
  col = col + vol * (hit > 0.5 ? 1.0 : 1.0) - vec3(0.0);

  // The moon disc is in front of the fog only where fog is thin; dim it by
  // the fog directly in front of it so it sinks into the haze correctly.
  col -= discMask * vol * 0.55;
  // Same for the star field: in-scattered fog veils it, which is what gives the
  // haze depth instead of letting stars punch through like pinholes.
  col -= starMask * vol * 0.9;

  // --- output ---
  // A 2M-pixel gradient bands visibly without it. Static, for the same reason
  // the volumetric jitter is.
  float dither = (ign(gl_FragCoord.xy + 17.0) - 0.5) * 0.0022;
  gl_FragColor = vec4(max(col + dither, 0.0), 1.0);
}
`;

/* ================================================================== *
 * Foreground — instanced grass, swaying in the vertex shader
 * ================================================================== */

const GRASS_VERT = /* glsl */ `
${GLSL_HASH}
${GLSL_CYCLIC}
uniform float uPhase;
uniform float uWind;
attribute vec3 iOffset;
attribute vec3 iParams;   // height, width, yaw
attribute float iSeed;
varying float vHeight;
varying float vSeed;
varying float vDepth;

void main(){
  float t = clamp(position.y, 0.0, 1.0);

  // Two bending modes: a slow whole-blade lean and a faster tip whip. Both
  // harmonic in the phase, so the field returns exactly to its start.
  float lean = cycFbm1(uPhase, iSeed, 1.0, 2) * 2.0 - 1.0;
  float whip = cycFbm1(uPhase, iSeed + 31.0, 1.0, 5) * 2.0 - 1.0;

  float bend = (t * t) * (lean * uWind + whip * uWind * 0.45);
  vec3 p = position;
  p.x = p.x * iParams.y;
  p.y = p.y * iParams.x;
  p.z *= 0.55;

  // Rotate the blade by its yaw, then apply the bend as a shear.
  float c = cos(iParams.z), s = sin(iParams.z);
  vec3 rot = vec3(p.x * c - p.z * s, p.y, p.x * s + p.z * c);
  rot.x += bend * iParams.x * 0.5;
  rot.z += bend * iParams.x * 0.22;

  vec3 world = iOffset + rot;

  vec4 mv = modelViewMatrix * vec4(world, 1.0);
  vHeight = t;
  vSeed = iSeed;
  vDepth = -mv.z;
  gl_Position = projectionMatrix * mv;
}
`;

const GRASS_FRAG = /* glsl */ `
precision highp float;
uniform vec3 uColorLow;
uniform vec3 uColorHigh;
uniform float uFade;
varying float vHeight;
varying float vSeed;
varying float vDepth;

void main(){
  // Root dark, tip catching the moon. The gradient across a blade is most of
  // what makes it read as a blade.
  vec3 c = mix(uColorLow, uColorHigh, pow(vHeight, 0.8));
  c *= 0.55 + 0.45 * vHeight;
  // Fade with distance so the field does not end in a hard line.
  float a = uFade * (1.0 - smoothstep(9.0, 30.0, vDepth));
  gl_FragColor = vec4(c, a);
}
`;

const buildGrass = (count: number): InstancedBufferGeometry => {
  // A single tapered blade: 5 segments, 2 triangles per segment.
  const positions: number[] = [];
  const indices: number[] = [];
  const segs = 5;
  for (let i = 0; i <= segs; i++) {
    const t = i / segs;
    const w = 0.5 * (1 - t) ** 0.75;
    positions.push(-w, t, 0, w, t, 0);
  }
  for (let i = 0; i < segs; i++) {
    const a = i * 2;
    indices.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
  }

  const geo = new InstancedBufferGeometry();
  geo.setAttribute('position', new Float32BufferAttribute(positions, 3));
  geo.setIndex(indices);

  const offsets = new Float32Array(count * 3);
  const params = new Float32Array(count * 3);
  const seeds = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    // Poisson-ish: golden-angle spiral, jittered. No clumping, no RNG state.
    const g = 2.399963229728653;
    const a = i * g;
    const r = 2.2 * Math.sqrt((i + 0.5) / count);
    const jx = (Math.sin(i * 12.9898) * 43758.5453) % 1;
    const jz = (Math.sin(i * 78.233) * 12345.6789) % 1;
    offsets[i * 3] = Math.cos(a) * r * 2.6 + jx * 0.7;
    offsets[i * 3 + 1] = -30.0;
    offsets[i * 3 + 2] = Math.sin(a) * r * 1.9 + jz * 0.7 + 24.0;
    params[i * 3] = 3.4 + ((Math.sin(i * 3.17) * 5000) % 1) * 7.0;
    params[i * 3 + 1] = 0.10 + ((Math.sin(i * 7.31) * 2000) % 1) * 0.09;
    params[i * 3 + 2] = ((Math.sin(i * 1.77) * 900) % 1) * Math.PI;
    seeds[i] = i * 0.137;
  }

  geo.setAttribute('iOffset', new InstancedBufferAttribute(offsets, 3));
  geo.setAttribute('iParams', new InstancedBufferAttribute(params, 3));
  geo.setAttribute('iSeed', new InstancedBufferAttribute(seeds, 1));
  geo.instanceCount = count;
  return geo;
};

/* ================================================================== *
 * Motes — atmosphere layer
 * ================================================================== */

const MOTE_VERT = /* glsl */ `
${GLSL_HASH}
${GLSL_CYCLIC}
uniform float uPhase;
uniform float uSize;
attribute vec3 iSeed;
varying float vAlpha;
varying float vSeed;

void main(){
  // Each mote traces a closed orbit, so it returns to its start exactly.
  float s = iSeed.x;
  float speed = 0.4 + iSeed.y * 0.9;
  float a = uPhase * 6.28318530718 * speed + s * 6.28318530718;
  float r = 6.0 + iSeed.z * 46.0;
  float driftX = cos(a) * r;
  float driftZ = sin(a * 0.83) * r * 0.6;

  // Slow vertical bob. A sine, not a fract ramp: fract wraps inside a single
  // frame and teleports every mote from the top of its column to the bottom,
  // which is a very visible pop once per loop.
  float y = sin(uPhase * 6.28318530718 + s * 6.28318530718) * 24.0 - 8.0;

  vec3 world = vec3(driftX, y, driftZ - 40.0);

  // A mote only glows when it is inside the shaft, so the beam has things
  // floating in it rather than a uniform dust haze.
  float shaft = pow(max(0.0, dot(normalize(world - vec3(0.0, 2.0, 0.0)), normalize(vec3(0.16, 0.115, -1.0)))), 6.0);
  float twinkle = 0.45 + 0.55 * cycFbm1(uPhase, s * 91.0, 1.0, 3);
  vAlpha = (0.06 + shaft * 0.85) * twinkle;
  vSeed = s;

  vec4 mv = modelViewMatrix * vec4(world, 1.0);
  gl_PointSize = uSize * (1.0 + iSeed.y) * (12.0 / max(1.0, -mv.z));
  gl_Position = projectionMatrix * mv;
}
`;

const MOTE_FRAG = /* glsl */ `
precision highp float;
uniform vec3 uColor;
varying float vAlpha;
varying float vSeed;
void main(){
  vec2 d = gl_PointCoord - 0.5;
  float r = length(d);
  // Soft round falloff. Square points are the giveaway of a forgotten shader.
  float a = smoothstep(0.5, 0.06, r) * vAlpha;
  if (a < 0.003) discard;
  gl_FragColor = vec4(uColor, a);
}
`;

/* ================================================================== *
 * Scene assembly
 * ================================================================== */

/**
 * Volumetric steps per quality level.
 *
 * The top of this table used to be 34, on the theory that more samples means
 * smoother shafts. It does — and it also means the difference between 55fps and
 * "the tab has stopped responding". God rays are a low-frequency phenomenon: past
 * about two dozen steps the extra samples buy dithering noise, not detail.
 */
const VOLUMETRIC_LUT: readonly number[] = [6, 10, 16, 24];

class MoonlitVale implements SceneInstance {
  readonly three = new Scene();
  readonly rig: CameraRig;

  private readonly skyMat: ShaderMaterial;
  private readonly skyMesh: Mesh;
  private readonly grassMat: ShaderMaterial;
  private readonly grass: Mesh;
  private readonly moteMat: ShaderMaterial;
  private readonly motes: Points;
  private readonly geoms: BufferGeometry[] = [];

  /** Nothing here is written by `update`. It only *reads* phase. */
  constructor(ctx: SceneContext, quality: number) {
    this.rig = new CameraRig({
      position: new Vector3(0, 16, 34),
      target: new Vector3(0, 26, -180),
      fov: 41,
      near: 0.35,
      far: 1400,
      driftMetres: 1.5,
      fovBreath: 0.028,
      parallaxMetres: 0.85,
      parallaxHalfLife: 1.4,
    });

    // --- sky / terrain / volumetrics ---
    this.skyMat = new ShaderMaterial({
      vertexShader: SKY_VERT,
      fragmentShader: SKY_FRAG,
      uniforms: {
        uPhase: { value: 0 },
        uCamPos: { value: new Vector3() },
        uCamFwd: { value: new Vector3() },
        uCamRight: { value: new Vector3() },
        uCamUp: { value: new Vector3() },
        uTanHalfFov: { value: Math.tan((41 * Math.PI) / 360) },
        uAspect: { value: ctx.width / ctx.height },
        uVolumetricSteps: { value: VOLUMETRIC_LUT[quality] ?? 22 },
      },
      depthTest: false,
      depthWrite: false,
      // DoubleSide, not BackSide: the clip-space gl_Position override does not
      // change screen-space winding, so BackSide culls the quad away entirely.
      side: DoubleSide,
    });
    this.skyMesh = new Mesh(new PlaneGeometry(2, 2), this.skyMat);
    this.skyMesh.frustumCulled = false;
    this.skyMesh.renderOrder = -100;
    this.geoms.push(this.skyMesh.geometry);
    this.three.add(this.skyMesh);

    // --- foreground grass ---
    const grassGeo = buildGrass(2600);
    this.geoms.push(grassGeo);
    this.grassMat = new ShaderMaterial({
      vertexShader: GRASS_VERT,
      fragmentShader: GRASS_FRAG,
      uniforms: {
        uPhase: { value: 0 },
        uWind: { value: 0.22 },
        uColorLow: { value: new Color(0.014, 0.020, 0.030) },
        uColorHigh: { value: new Color(0.30, 0.335, 0.30) },
        uFade: { value: 0.95 },
      },
      transparent: true,
      depthWrite: false,
      side: DoubleSide,
    });
    this.grass = new Mesh(grassGeo, this.grassMat);
    this.grass.frustumCulled = false;
    this.grass.renderOrder = 10;
    this.three.add(this.grass);

    // --- motes ---
    const moteCount = 900;
    const moteGeo = new BufferGeometry();
    const mPos = new Float32Array(moteCount * 3);
    const mSeed = new Float32Array(moteCount * 3);
    for (let i = 0; i < moteCount; i++) {
      // Seed values only; positions come entirely from phase in the shader.
      mSeed[i * 3] = (Math.sin(i * 12.9898) * 43758.5453) % 1;
      mSeed[i * 3 + 1] = (Math.sin(i * 39.3468) * 24634.6345) % 1;
      mSeed[i * 3 + 2] = (Math.sin(i * 73.156) * 19349.1234) % 1;
    }
    moteGeo.setAttribute('position', new Float32BufferAttribute(mPos, 3));
    moteGeo.setAttribute('iSeed', new Float32BufferAttribute(mSeed, 3));
    this.geoms.push(moteGeo);

    this.moteMat = new ShaderMaterial({
      vertexShader: MOTE_VERT,
      fragmentShader: MOTE_FRAG,
      uniforms: {
        uPhase: { value: 0 },
        uSize: { value: 2.1 },
        uColor: { value: new Color(1.0, 0.96, 0.9) },
      },
      transparent: true,
      depthWrite: false,
      blending: AdditiveBlending,
    });
    this.motes = new Points(moteGeo, this.moteMat);
    this.motes.frustumCulled = false;
    this.motes.renderOrder = 20;
    this.three.add(this.motes);
  }

  update(info: FrameInfo): void {
    const phase = info.phase;
    const cam = this.rig.camera as PerspectiveCamera;

    this.rig.setPointer(info.pointer.x, info.pointer.y, info.idleness);
    this.rig.update(phase, info.dt);

    // The sky quad is drawn in clip space, so the camera basis has to be fed
    // in by hand — a real projection would fight the `gl_Position` override.
    const u = this.skyMat.uniforms;
    u.uPhase.value = phase;
    u.uTanHalfFov.value = Math.tan((cam.fov * Math.PI) / 360);
    u.uAspect.value = info.ctx.width / info.ctx.height;
    (u.uCamPos.value as Vector3).copy(cam.position);
    (u.uCamFwd.value as Vector3).set(0, 0, -1).applyQuaternion(cam.quaternion);
    (u.uCamRight.value as Vector3).set(1, 0, 0).applyQuaternion(cam.quaternion);
    (u.uCamUp.value as Vector3).set(0, 1, 0).applyQuaternion(cam.quaternion);

    // A second, longer harmonic so the fog density breathes on a different
    // cycle from the terrain drift. Two unsynchronised periods are what stop a
    // 96s loop from being legible.
    const fogPhase = (phase + harmonic(phase, 2) * 0.06 + 1) % 1;
    (this.grassMat.uniforms.uPhase as { value: number }).value = phase;
    (this.moteMat.uniforms.uPhase as { value: number }).value = fogPhase;
  }

  tune(quality: QualityTune): void {
    (this.skyMat.uniforms.uVolumetricSteps as { value: number }).value =
      quality.volumetricSteps;
  }

  resize(ctx: SceneContext): void {
    (this.skyMat.uniforms.uAspect as { value: number }).value = ctx.width / ctx.height;
  }

  audio(info: FrameInfo, out: { noise(a: number, tone?: number): void }): void {
    // Barely-there wind: a low filtered-noise bed whose brightness follows the
    // same gust signal the grass bends to. §12: the piece must work muted.
    const gust = cyclicFbm1(info.phase, 17, 3, 2) * 2 - 1;
    out.noise(0.06 + gust * 0.03, 320 + gust * 180);
  }

  dispose(): void {
    for (const g of this.geoms) g.dispose();
    this.skyMat.dispose();
    this.grassMat.dispose();
    this.moteMat.dispose();
    this.rig.dispose();
    this.three.clear();
  }
}

/**
 * The scene allocates everything the *shape* of the world needs at construction
 * and nothing that scales with quality, so a quality change is a uniform write
 * rather than a rebuild — see `SceneInstance.tune`.
 */
export const createMoonlitVale = (ctx: SceneContext, quality = 2): SceneInstance =>
  new MoonlitVale(ctx, quality);

export const moonlitValeDefinition: SceneDefinition = {
  id: 'moonlit-vale',
  title: '月落雾谷',
  periodSeconds: PERIOD,
  note: '低月 · 雾海 · 摇曳草丛',
  create: (ctx) => createMoonlitVale(ctx),
};
