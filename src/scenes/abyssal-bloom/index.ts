/**
 * Scene 02 — 深海流光 · ABYSSAL BLOOM
 *
 * Deep water, and light that only arrives from above.
 *
 * The whole scene is organised around one physical fact: at depth, the sun is
 * the only light source, so every subject is lit from the top and falls into
 * blue-black underneath. Get that backwards — light the bells from below, or
 * fill the water uniformly — and the result reads as a swimming pool, not an
 * ocean. That single decision is most of the difference between the two.
 *
 * Layering, per §9:
 *   background  the water column itself, darkening downward, plus the broad
 *               brightening the shafts come from
 *   far         god rays, and the small jellies that are mostly silhouette
 *   mid         the two hero jellies, where the bell structure is readable
 *   near        tentacles and oral arms, out of focus and crossing frame
 *   atmosphere  marine snow, drifting through the shafts
 *
 * Why the water and the creatures are separate draw calls:
 *
 *   The volumetric pass writes no depth and everything else draws after it, so a
 *   jellyfish in front of a light shaft occludes it for free. The price is that
 *   the shafts are *not* occluded by the animals' own shadows, which at this
 *   turbidity is invisible — and the alternative, marching the volume after the
 *   geometry, needs the depth buffer bound as a texture while it is still the
 *   render target.
 *
 *   The creatures then fog themselves against the same tint and extinction the
 *   volume uses, so aerial perspective stays consistent across the seam between
 *   the two halves of the image.
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
  PerspectiveCamera,
  PlaneGeometry,
  Points,
  Scene,
  ShaderMaterial,
  Vector3,
} from 'three';

import type {
  AudioBus,
  FrameInfo,
  QualityTune,
  SceneContext,
  SceneDefinition,
  SceneInstance,
} from '../../core/scene';
import { CameraRig } from '../../core/camera';
import {
  GLSL_COLOR,
  GLSL_CYCLIC,
  GLSL_DITHER,
  GLSL_HASH,
  GLSL_NOISE,
  GLSL_SCATTER,
} from '../../shaders/common';
import { cyclicFbm1 } from '../../core/loop';

const PERIOD = 84;

/* ================================================================== *
 * Palette and constants
 * ================================================================== */

/** Height of the surface, in metres above the origin. Nothing else is lit. */
const SURFACE_Y = 46;
/**
 * Unit vector pointing *toward* the light, i.e. up. Callers that want the
 * direction light travels negate it; keeping one convention stops the sign from
 * being guessed wrong in a shader that is already hard to read.
 */
const TO_LIGHT = new Vector3(0.19, 1, -0.16).normalize();
/** What the shafts are made of. Cold, and nowhere near white. */
const LIGHT_COLOR = new Color(0.42, 0.72, 0.68);
/** Water's own colour: what the volume scatters when no shaft hits it. */
const WATER_TINT = new Color(0.016, 0.052, 0.072);

/**
 * Per-jellyfish bioluminescence.
 *
 * Warm accents are a minority on purpose. A frame where everything is cyan has
 * no accents left, and the eye has nothing to land on.
 */
const HUES: readonly [number, number, number][] = [
  [0.34, 0.86, 1.0],
  [0.52, 0.62, 1.0],
  [1.0, 0.58, 0.42],
  [0.62, 1.0, 0.78],
];

/**
 * Volumetric steps per quality level.
 *
 * God rays are a low-frequency phenomenon. Past about two dozen samples the extra
 * steps buy dithering noise rather than detail, and each one costs a full noise
 * evaluation per pixel — so this table is the frame budget, and its top is not
 * negotiable on the grounds of "more looks better".
 */
const VOLUMETRIC_LUT: readonly number[] = [8, 12, 18, 26];

const vec3 = (c: { r: number; g: number; b: number }): Vector3 =>
  new Vector3(c.r, c.g, c.b);

/* ================================================================== *
 * Water — fullscreen pass, drawn first, writes no depth
 * ================================================================== */

const FULLSCREEN_VERT = /* glsl */ `
varying vec2 vUv;
void main(){
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

const WATER_FRAG = /* glsl */ `
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
uniform float uSurfaceY;
uniform vec3  uToLight;
uniform vec3  uLightColor;
uniform vec3  uWaterTint;

varying vec2 vUv;

const float FAR = 320.0;

/**
 * Shaft mask.
 *
 * Each sample is traced back along the light to the surface plane, so the
 * pattern lives in a 2D coordinate and therefore stays parallel to the light
 * instead of smearing across the screen. This projection is the entire
 * difference between god rays and a foggy screen with bright blobs in it.
 */
float shaftMask(vec3 p){
  vec2 q = p.xz + (uSurfaceY - p.y) * uToLight.xz / uToLight.y;
  q += driftCircle(uPhase, 0.55, 0.5) * 0.6;
  // The frequency here is the whole composition of the upper frame. Too low and
  // two or three beams span the image, the pattern converges to a vanishing
  // point like a gobo, and the animals stop being the subject.
  float n = fbm2(q * 0.13, 4);
  // A second, finer scale. One octave of shafts looks like a gobo; two read as
  // water.
  float fine = fbm2(q * 0.42 + 11.0, 3);
  float m = smoothstep(0.44, 0.68, n * 0.72 + fine * 0.28);
  // Squared, so the beam has a core and a falloff instead of a soft edge. A shaft
  // with a soft edge reads as haze; the harder core is what reads as a beam.
  return m * m;
}

void main(){
  vec2 uv = vUv * 2.0 - 1.0;
  uv.x *= uAspect;
  vec3 rd = normalize(uCamFwd + uCamRight * uv.x * uTanHalfFov + uCamUp * uv.y * uTanHalfFov);
  vec3 ro = uCamPos;

  float toSurface = max(0.0, dot(rd, uToLight));

  // --- the water column ---
  // Never pure black: a wallpaper that hits #000 reads as "no render". The floor
  // of this gradient keeps the bottom corners from looking like a hole, and the
  // contrast between the lit upper half and the dark lower half is most of the
  // sense of depth.
  float up = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
  vec3 col = mix(vec3(0.0022, 0.0062, 0.0098), vec3(0.010, 0.030, 0.040), pow(up, 1.15));
  // The surface, seen from 46m down: a broad brightening, not a sun. Anything
  // with a hot core and the whole frame turns grey.
  col += uLightColor * pow(toSurface, 2.4) * 0.055;
  col += uLightColor * pow(toSurface, 14.0) * 0.045;

  // --- in-scattering along the ray ---
  int steps = int(uVolumetricSteps);
  float stepLen = FAR / float(steps);
  float jitter = ign(gl_FragCoord.xy);
  // A gentler lobe than a cloud would use: at 0.66 the beams only light up when
  // you look almost straight at the surface, and the camera is tilted 12° off
  // horizontal, so the whole upper frame would stay dark.
  float phase = hgPhase(dot(rd, uToLight), 0.42);

  vec3 acc = vec3(0.0);
  float trans = 1.0;
  for (int i = 0; i < 48; i++){
    if (i >= steps) break;
    vec3 p = ro + rd * ((float(i) + jitter) * stepLen);

    float below = max(0.0, uSurfaceY - p.y);
    // Light is absorbed on the way down, so the shafts weaken with depth. This
    // gradient is the reason the top of frame feels like an opening.
    float att = exp(-below * 0.0165);
    // Turbidity, raised inside a shaft rather than only its emission raised: a
    // shaft has to look like it is made of something.
    float d = 0.0055 + shaftMask(p) * 0.070 * att;
    vec3 emit = uLightColor * phase * att * 2.7 + uWaterTint * 0.6;

    float a = 1.0 - exp(-d * stepLen);
    acc += emit * a * trans;
    trans *= 1.0 - a;
    if (trans < 0.015) break;
  }

  col += acc;

  float dither = (ign(gl_FragCoord.xy + 17.0) - 0.5) * 0.0022;
  gl_FragColor = vec4(max(col + dither, 0.0), 1.0);
}
`;

/* ================================================================== *
 * The bell
 *
 * Geometry is a parametric grid — `position` carries (a, theta) and nothing
 * else. Every shape decision, including the pulse and the scalloped margin,
 * happens in the vertex shader, so the mesh is allocated once and never touched
 * again. That is also what lets one InstancedBufferGeometry carry the colony.
 * ================================================================== */

const BELL_VERT = /* glsl */ `
${GLSL_HASH}
${GLSL_CYCLIC}

uniform float uPhase;

attribute vec4 iJelly;  // world position + per-instance seed
attribute vec4 iBell;   // radius, height, lobes, yaw
attribute vec2 iLife;   // contractions per loop, phase offset
attribute vec3 iHue;
attribute float iGlow;  // bioluminescence strength, per animal

varying vec3  vNormal;
varying vec3  vWorld;
varying vec2  vUv;
varying vec3  vHue;
varying float vPulse;
varying float vCrest;
varying float vGlow;

const float PI = 3.14159265359;

/**
 * One pulse, as a function of phase.
 *
 * Contraction is fast and relaxation is slow, which is the opposite of a sine and
 * the reason a jellyfish looks alive: the pow() puts the energy into the squeeze
 * and leaves the recovery soft. Integer rate, so it closes across the wrap like
 * everything else here.
 */
float pulseAt(float seed, float rate, float offset){
  float a = sin(TAU * (uPhase * rate + offset + seed));
  return pow(a * 0.5 + 0.5, 3.5);
}

/** Bell surface at parameter t: 0 apex → 1 margin tip. */
vec3 bellPoint(float t, float theta, float R, float H, float lobes, float spin, out float scallop){
  // The margin is where the scalloping lives; a lobed apex is not a thing.
  float rimward = smoothstep(0.30, 1.0, t);
  scallop = 0.105 * rimward * cos(lobes * theta + spin);

  // Two profiles, cross-faded rather than switched. Branching on t puts a crease
  // right around the widest part of the animal, and the crease catches a specular
  // highlight and reads as a polygon edge.
  float a = clamp(t / 0.76, 0.0, 1.0) * (PI * 0.5);
  float rDome = pow(sin(a), 0.78);
  float yDome = H * pow(cos(a), 1.28);

  // The margin rolls *under* the bell and pulls inward. This is the silhouette
  // that separates a jellyfish from a dome: without it the animal has no
  // underside to look at and reads as a mushroom cap.
  float k = clamp((t - 0.76) / 0.24, 0.0, 1.0);
  float s = k * k * (3.0 - 2.0 * k);
  float rEdge = mix(1.0, 0.66, s);
  float yEdge = -R * 0.36 * s;

  float blend = smoothstep(0.68, 0.88, t);
  float r = mix(rDome, rEdge, blend) * R * (1.0 + scallop);
  // The lobes dip as well as bulge, so the margin is not a flat disc.
  float y = mix(yDome, yEdge, blend) + scallop * R * 0.5 * rimward;

  return vec3(cos(theta) * r, y, sin(theta) * r);
}

void main(){
  float t = position.x;
  float theta = position.y;

  float seed = iJelly.w;
  float pulse = pulseAt(seed, iLife.x, iLife.y);
  float R = iBell.x * (1.0 - 0.17 * pulse);
  float H = iBell.y * (1.0 + 0.30 * pulse);
  // The margin trails the contraction, so the lobes lag the squeeze slightly.
  float spin = pulseAt(seed, iLife.x, iLife.y - 0.035) * 0.9 + seed * TAU;

  float sc;
  vec3 p = bellPoint(t, theta, R, H, iBell.z, spin, sc);

  // Normals from the parametric surface: three evaluations is cheaper and far
  // more reliable than pushing a normal attribute through a deformation.
  float e = 0.008;
  float sc2;
  vec3 dt = bellPoint(min(t + e, 1.0), theta, R, H, iBell.z, spin, sc2);
  vec3 dq = bellPoint(t, theta + e, R, H, iBell.z, spin, sc2);
  vec3 n = normalize(cross(dq - p, dt - p));

  // Place: yaw about Y, then translate.
  float c = cos(iBell.w), s = sin(iBell.w);
  vec3 world = iJelly.xyz + vec3(p.x * c - p.z * s, p.y, p.x * s + p.z * c);

  vNormal = normalize(vec3(n.x * c - n.z * s, n.y, n.x * s + n.z * c));
  vWorld = world;
  vUv = vec2(theta / (2.0 * PI), t);
  vHue = iHue;
  vPulse = pulse;
  vCrest = sc;
  vGlow = iGlow;

  gl_Position = projectionMatrix * viewMatrix * vec4(world, 1.0);
}
`;

const BELL_FRAG = /* glsl */ `
precision highp float;
${GLSL_COLOR}

uniform vec3  uToLight;
uniform vec3  uLightColor;
uniform vec3  uWaterTint;
uniform float uSurfaceY;

varying vec3  vNormal;
varying vec3  vWorld;
varying vec2  vUv;
varying vec3  vHue;
varying float vPulse;
varying float vCrest;
varying float vGlow;

const float PI = 3.14159265359;

void main(){
  // DoubleSide, so the normal has to be flipped for back faces by hand — three
  // does not do it, and without it the inside of the bell is lit from below.
  vec3 N = normalize(vNormal);
  if (!gl_FrontFacing) N = -N;
  vec3 toEye = cameraPosition - vWorld;
  float dist = length(toEye);
  vec3 V = toEye / max(dist, 0.001);

  float lam = dot(N, uToLight);

  // The bell is a thin membrane, so it is mostly *not* Lambertian: what you see
  // is light that went through it. A wrapped diffuse for the flesh, plus a
  // forward transmission term where the membrane faces away from you.
  float wrap = pow(clamp(lam * 0.5 + 0.5, 0.0, 1.0), 1.6);
  float through = pow(clamp(-lam * 0.5 + 0.5, 0.0, 1.0), 2.2);
  float top = clamp(lam * 0.5 + 0.5, 0.0, 1.0);

  // --- structure ---
  // Eight radial canals. Each is a *groove* with a bright ridge beside it, not a
  // single painted line: a flat line drawn on a smooth dome reads as a decal, and
  // the pairing is what makes the surface look like it has thickness.
  float meridian = abs(sin(vUv.x * 8.0 * PI));
  float groove = smoothstep(0.50, 0.94, meridian);
  float ridge = smoothstep(0.93, 1.0, meridian);
  // The gastric ring, just inside the margin.
  float ring = smoothstep(0.10, 0.0, abs(vUv.y - 0.70));
  // The margin itself. Wide, because the rolled edge is most of the silhouette.
  float margin = smoothstep(0.70, 0.99, vUv.y);
  // A hard bright line right at the edge. This one detail does more for the read
  // than all the shading above it.
  float edge = smoothstep(0.94, 1.0, vUv.y);
  float crest = smoothstep(0.0, 0.06, vCrest);

  float fres = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 2.4);

  vec3 flesh = vec3(0.075, 0.100, 0.135);
  vec3 col = flesh * (0.06 + 0.50 * wrap) * uLightColor * 2.6;
  col += vHue * through * 0.60;
  col += vHue * fres * 0.95;

  // Bioluminescence. Concentrated at the margin, pulsing with the contraction —
  // this is what makes the colony read as alive rather than as drifting debris.
  float glow = margin * (0.85 + 0.75 * vPulse) + ring * 0.22 + ridge * 0.30;
  col += vHue * glow * (0.80 + 0.55 * top) * vGlow;

  // The grooves are where the flesh folds, so they are darker, not brighter.
  col *= 1.0 - 0.34 * groove * (1.0 - margin);

  float spec = pow(max(0.0, dot(reflect(-uToLight, N), V)), 26.0);
  col += uLightColor * spec * 0.40 * top;
  col += mix(vHue, uLightColor, 0.45) * edge * 0.85 * vGlow;
  col += vHue * crest * margin * 0.55;

  // The apex sits in the animal's own shadow, and the underside is darker still
  // because the light never reaches it.
  col *= mix(1.0, 0.34, smoothstep(0.34, 0.0, vUv.y));
  col *= mix(1.0, 0.55, smoothstep(0.0, -0.3, dot(N, uToLight)));

  // --- the water between here and the eye ---
  // Red goes first. This is why a distant jelly reads as blue-green even when its
  // own bioluminescence is orange.
  vec3 ext = exp(-dist * vec3(0.052, 0.026, 0.018));
  float att = exp(-max(0.0, uSurfaceY - vWorld.y) * 0.0165);
  col = col * ext + uWaterTint * (1.0 - luma(ext)) * 0.55 * att;

  gl_FragColor = vec4(max(col, 0.0), 1.0);
}
`;

/* ================================================================== *
 * Tentacles and oral arms
 *
 * One instanced ribbon strip for both. The ribbon is billboarded against the
 * view vector in the vertex shader, which is what keeps a flat strip from turning
 * into an invisible sheet every time the camera drifts.
 * ================================================================== */

const ARM_VERT = /* glsl */ `
${GLSL_HASH}
${GLSL_CYCLIC}

uniform float uPhase;

attribute vec4 iJelly;  // position + seed
attribute vec4 iBell;   // radius, height, lobes, yaw
attribute vec2 iLife;   // rate, offset
attribute vec3 iHue;
attribute vec4 iArm;    // theta on the margin, length, width, kind (0 fine, 1 oral)

varying float vT;
varying float vSide;
varying vec3  vHue;
varying float vKind;

const float PI = 3.14159265359;

float pulseAt(float seed, float rate, float offset){
  float a = sin(TAU * (uPhase * rate + offset + seed));
  return pow(a * 0.5 + 0.5, 3.5);
}

/** Where the arm attaches: on the rolled margin, spun by the bell's yaw. */
vec3 anchor(out float rimR){
  float pulse = pulseAt(iJelly.w, iLife.x, iLife.y);
  float R = iBell.x * (1.0 - 0.17 * pulse);
  // The margin ends at 0.66R and has already dropped below the bell's equator, so
  // attaching at 0.66R is what puts the arms under the animal rather than
  // sprouting them from its equator.
  rimR = R;
  float c = cos(iBell.w), s = sin(iBell.w);
  vec2 radial = vec2(cos(iArm.x), sin(iArm.x));
  vec2 spun = vec2(radial.x * c - radial.y * s, radial.x * s + radial.y * c);
  return iJelly.xyz + vec3(spun * R * 0.66, -R * 0.34);
}

/**
 * The arm's spine at parameter s.
 *
 * Declared at file scope, not inside main: GLSL ES 1.0 has no nested functions,
 * and a closure would be the obvious way to write this.
 */
vec3 armCurve(float s, float amp, float len, out float rimR){
  float w1 = sin(uPhase * TAU * iLife.x + iJelly.w * 41.0 + s * 3.4 + iArm.x);
  float w2 = sin(uPhase * TAU * iLife.x * 2.0 + iJelly.w * 17.0 + s * 6.3);
  vec3 q = anchor(rimR);
  // Two superposed waves at unrelated frequencies: a single one gives a comb, and
  // a comb reads as a curtain hanging off the animal rather than as tentacles.
  // The 0.35/0.65 split keeps the base pinned and lets the tip travel, which is
  // the difference between an arm and a rope.
  float bend = 0.35 * s + 0.65 * s * s;
  q.x += w1 * amp * bend;
  q.z += w2 * amp * 0.55 * bend;
  q.y -= s * len;
  // A slow sink-and-rise, so the arms are not plumb lines.
  q.y += sin(uPhase * TAU + iJelly.w * 23.0 + s * 1.4) * 0.05 * len * s;
  return q;
}

void main(){
  float t = position.y;     // 0 at the margin, 1 at the tip
  float side = position.x;  // -1 / +1 across the ribbon
  float kind = iArm.w;

  // Sway lags the contraction: the bell finishes its squeeze before the arms
  // have finished following it, and that lag is most of what makes the motion
  // read as a body rather than as a scaling.
  float lag = pulseAt(iJelly.w, iLife.x, iLife.y - 0.055);
  float len = iArm.y * (1.0 - 0.16 * lag);
  // Oral arms are stiffer and shorter, so they get a smaller swing per unit
  // length. Quoting amplitude as a fraction of the bell radius is what keeps a
  // distant animal's arms proportionate instead of spaghetti.
  float amp = mix(0.62, 0.20, kind) * iBell.x;

  // Tangent, by finite difference along the curve.
  float e = 0.02;
  float rimR, rimR2;
  vec3 p = armCurve(t, amp, len, rimR);
  vec3 tangent = normalize(armCurve(min(t + e, 1.0), amp, len, rimR2) - p);

  vec3 toEye = cameraPosition - p;
  float dist = length(toEye);
  vec3 V = toEye / max(dist, 0.001);
  vec3 side3 = normalize(cross(tangent, V));

  // Width tapers to nothing at the tip, and the oral arms ripple along their
  // length so their silhouette is never a clean rectangle.
  float w = iArm.z * rimR * mix(1.0, 0.18, pow(t, 0.7));
  w *= 1.0 + kind * 0.55 * sin(t * 9.0 + uPhase * TAU * iLife.x * 2.0);
  // A floor on the *apparent* width. A tentacle thinner than a pixel does not
  // render, it aliases into a dotted line that flickers — which is why the first
  // pass had jellies with no tentacles at all rather than with thin ones.
  w = max(w, dist * 0.0042);

  vec3 world = p + side3 * side * w;

  vT = t;
  vSide = side;
  vHue = iHue;
  vKind = kind;

  gl_Position = projectionMatrix * viewMatrix * vec4(world, 1.0);
}
`;

const ARM_FRAG = /* glsl */ `
precision highp float;
${GLSL_COLOR}
${GLSL_HASH}
${GLSL_CYCLIC}

varying float vT;
varying float vSide;
varying vec3  vHue;
varying float vKind;

void main(){
  // Bright at the attachment, dimming to the tip: an arm is lit by the animal it
  // hangs from. The falloff is deliberately gentle — a steeper one eats the outer
  // half of every tentacle and the animal ends up looking like it has a 30cm
  // brush instead of ten metres of trailing tentacle.
  float along = pow(1.0 - vT, 0.8);
  // Across the ribbon: a soft round section. A flat strip with a hard edge is the
  // single most common giveaway of a hand-built tentacle. The floor is high
  // because a tentacle is only a few pixels wide, and a profile that goes to zero
  // at the edges means the whole thing goes to zero.
  float across = 1.0 - vSide * vSide;
  float body = along * (0.45 + 0.55 * across);

  // A slow luminous pulse travelling down the arm.
  float pulse = 0.55 + 0.45 * cycFbm1(vT * 0.5, vSide + 0.5, 1.0, 3);

  float alpha = body * (0.34 + 0.50 * vKind) * pulse;
  if (alpha < 0.004) discard;

  vec3 col = vHue * (1.05 + 1.05 * vKind) * body * pulse;
  gl_FragColor = vec4(col * alpha, alpha);
}
`;

/* ================================================================== *
 * Marine snow
 * ================================================================== */

const SNOW_VERT = /* glsl */ `
${GLSL_HASH}
${GLSL_CYCLIC}

uniform float uPhase;
uniform float uSize;
uniform vec3  uToLight;

attribute vec3 iSeed;

varying float vAlpha;
varying float vLayer;

void main(){
  // Three depth layers, because one layer of particles always reads as a texture
  // of dots rather than as water with things in it.
  float layer = floor(iSeed.z * 3.0);
  vLayer = layer;

  float speed = 0.5 + iSeed.y * 0.8;
  float a = uPhase * TAU * speed + iSeed.x * TAU;

  // A closed orbit per grain, scaled by layer: near grains sweep across frame,
  // far ones barely move.
  float near = 1.0 - layer * 0.34;
  vec3 world = vec3(
    cos(a) * (7.0 + iSeed.y * 26.0) * near,
    sin(uPhase * TAU + iSeed.x * TAU) * (5.0 + iSeed.y * 9.0) - 4.0,
    sin(a * 0.79) * (22.0 + iSeed.z * 40.0) * near - 34.0
  );

  vec4 mv = viewMatrix * vec4(world, 1.0);
  float dist = -mv.z;
  gl_PointSize = uSize * near * (1.0 + iSeed.y) * (26.0 / max(1.0, dist));
  gl_Position = projectionMatrix * mv;

  // Snow only exists where light reaches it: a grain in an unlit pocket should be
  // invisible, and the shafts are what make it visible.
  float toLight = clamp(dot(normalize(world), uToLight), -1.0, 1.0);
  float shaft = pow(max(0.0, toLight), 3.0);
  float twinkle = 0.4 + 0.6 * cycFbm1(uPhase, iSeed.x * 57.0, 1.0, 2);
  vAlpha = (0.05 + shaft * 0.55) * twinkle * (0.4 + 0.6 * near);
}
`;

const SNOW_FRAG = /* glsl */ `
precision highp float;
uniform vec3 uColor;
varying float vAlpha;
varying float vLayer;
void main(){
  vec2 d = gl_PointCoord - 0.5;
  float r = length(d);
  // The nearest layer is deliberately soft and wide: that is a defocused grain of
  // marine snow, and treating it like a sharp dot destroys the depth cue.
  float edge = mix(0.06, 0.30, vLayer / 2.0);
  float a = smoothstep(0.5, edge, r) * vAlpha;
  if (a < 0.003) discard;
  gl_FragColor = vec4(uColor * a, a);
}
`;

/* ================================================================== *
 * The colony
 * ================================================================== */

interface Placement {
  pos: [number, number, number];
  /** Bell radius in metres. Arm length scales with it. */
  size: number;
  /** 1 = hemispherical, taller = more column. */
  height: number;
  lobes: number;
  /** Contractions per loop. Integer, or the colony never settles. */
  rate: number;
  offset: number;
  hue: number;
  /** Bioluminescence strength. Varying it stops the colony reading as ten copies
   *  of the same animal stamped across the frame. */
  glow: number;
  yaw: number;
  arms: number;
}

/**
 * Hand-placed, because a colony scattered by a hash is a colony with no
 * composition: one subject, one counterweight, and the rest placed to lead the eye
 * between them. The first two are the picture. The tail is filler, and filler
 * still has to survive being looked at for an hour.
 */
const COLONY: readonly Placement[] = [
  // The subject. Right of centre, close, big enough that the scalloped margin and
  // the radial grooves are both readable — at any smaller the animal is an idea
  // rather than a creature. The brightest thing in frame, because it is the one
  // the eye is meant to land on first.
  { pos: [6.5, 13, -14], size: 4.0, height: 1.32, lobes: 8, rate: 5, offset: 0.0, hue: 0, glow: 1.35, yaw: 0.4, arms: 16 },
  // The counterweight. Smaller, further, on the other side of the light, so its
  // margin glows against dark water instead of against a beam.
  { pos: [-13, 10.5, -34], size: 2.5, height: 1.18, lobes: 6, rate: 4, offset: 0.42, hue: 1, glow: 1.0, yaw: 1.9, arms: 12 },
  // One mid, deliberately high and well behind: it breaks the horizontal band the
  // other two would otherwise form. Tall and columnar, so it does not match the
  // hero's silhouette.
  { pos: [-6, 24, -44], size: 1.9, height: 1.62, lobes: 7, rate: 6, offset: 0.7, hue: 3, glow: 0.8, yaw: 2.7, arms: 10 },
  // The warm accent, far right and deep. Nearly a silhouette, but unmistakably a
  // different animal, and it stops the frame being monochrome.
  { pos: [21, 16, -56], size: 2.2, height: 1.1, lobes: 6, rate: 3, offset: 0.15, hue: 2, glow: 1.1, yaw: 5.1, arms: 11 },
  // Distant silhouettes. Their only job is to make the water feel like it has a
  // scale, so they are all small, all high, and all hazed out.
  { pos: [-22, 30, -72], size: 1.7, height: 1.3, lobes: 8, rate: 4, offset: 0.55, hue: 1, glow: 0.7, yaw: 1.1, arms: 8 },
  { pos: [10, 34, -82], size: 2.1, height: 1.2, lobes: 7, rate: 5, offset: 0.85, hue: 0, glow: 0.65, yaw: 3.3, arms: 9 },
  { pos: [30, 27, -66], size: 1.5, height: 1.45, lobes: 6, rate: 6, offset: 0.28, hue: 3, glow: 0.75, yaw: 4.4, arms: 8 },
  { pos: [-30, 21, -88], size: 1.8, height: 1.1, lobes: 8, rate: 3, offset: 0.62, hue: 2, glow: 0.6, yaw: 2.2, arms: 8 },
  { pos: [1, 38, -98], size: 2.4, height: 1.3, lobes: 7, rate: 4, offset: 0.05, hue: 1, glow: 0.7, yaw: 0.9, arms: 9 },
  // One low and near, mostly below the horizon line. It gives the bottom of the
  // composition something to occlude, which is what stops the lower third from
  // reading as empty.
  { pos: [-11, 2.5, -22], size: 1.5, height: 1.5, lobes: 6, rate: 7, offset: 0.9, hue: 0, glow: 1.15, yaw: 5.6, arms: 9 },
];

/**
 * Fan the arms around each bell's margin.
 *
 * Every third arm is an oral arm: short, wide, frilly. Real animals carry both,
 * and the contrast between the two textures is most of what makes a jellyfish
 * read as a jellyfish rather than as a dome with strings hanging off it.
 */
const ARM_COUNT = COLONY.reduce((n, p) => n + p.arms, 0);

/** Per-instance data for the colony, and the same data expanded per arm. */
interface ColonyBuffers {
  count: number;
  jelly: Float32Array;
  bell: Float32Array;
  life: Float32Array;
  hue: Float32Array;
  glow: Float32Array;
  armJelly: Float32Array;
  armBell: Float32Array;
  armLife: Float32Array;
  armHue: Float32Array;
}

const buildColony = (): ColonyBuffers => {
  const count = COLONY.length;
  const jelly = new Float32Array(count * 4);
  const bell = new Float32Array(count * 4);
  const life = new Float32Array(count * 2);
  const hue = new Float32Array(count * 3);
  const glow = new Float32Array(count);

  const armJelly = new Float32Array(ARM_COUNT * 4);
  const armBell = new Float32Array(ARM_COUNT * 4);
  const armLife = new Float32Array(ARM_COUNT * 2);
  const armHue = new Float32Array(ARM_COUNT * 3);

  let arm = 0;
  COLONY.forEach((p, i) => {
    const j = [p.pos[0], p.pos[1], p.pos[2], i * 0.137];
    const b = [p.size, p.size * p.height, p.lobes, p.yaw];
    const l = [p.rate, p.offset];
    const h = HUES[p.hue % HUES.length];

    jelly.set(j, i * 4);
    bell.set(b, i * 4);
    life.set(l, i * 2);
    hue.set(h, i * 3);
    glow[i] = p.glow;

    // Every arm needs its *own* copy of the animal it belongs to. Sharing the
    // bell's instanced arrays across a differently-sized arm buffer means the
    // tail of the arm buffer reads past the end of the attribute — silently, and
    // as a clump of geometry at the origin.
    for (let k = 0; k < p.arms; k++) {
      armJelly.set(j, arm * 4);
      armBell.set(b, arm * 4);
      armLife.set(l, arm * 2);
      armHue.set(h, arm * 3);
      arm++;
    }
  });

  return { count, jelly, bell, life, hue, glow, armJelly, armBell, armLife, armHue };
};

/**
 * Fan the arms around each bell's margin.
 *
 * Every third arm is an oral arm: short, wide, frilly. Real animals carry both,
 * and the contrast between the two textures is most of what makes a jellyfish
 * read as a jellyfish rather than as a dome with strings hanging off it.
 */
const buildArms = (segs: number): InstancedBufferGeometry => {
  const geo = new InstancedBufferGeometry();
  const params: number[] = [];
  const index: number[] = [];
  for (let i = 0; i <= segs; i++) {
    const t = i / segs;
    params.push(-1, t, 0, 1, t, 0);
  }
  for (let i = 0; i < segs; i++) {
    const a = i * 2;
    index.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
  }
  geo.setAttribute('position', new Float32BufferAttribute(params, 3));
  geo.setIndex(index);

  const arms = new Float32Array(ARM_COUNT * 4);
  let w = 0;
  COLONY.forEach((p, i) => {
    for (let k = 0; k < p.arms; k++) {
      const oral = k % 4 === 0;
      // Golden-angle spacing: no clumping, no visible gaps, no RNG state.
      const f = (k * 2.399963229728653) % (Math.PI * 2);
      arms.set(
        [
          f,
          // Oral arms are short. Tentacles run ~2.6 bell radii, which is about
          // right for the genus this is borrowing from.
          oral ? p.size * 1.6 : p.size * (1.9 + ((i * 7 + k * 13) % 11) * 0.09),
          // Widths are quoted as a fraction of the bell radius. The tentacles are
          // genuinely thin; the shader's apparent-width floor is what keeps them
          // from disappearing entirely at distance.
          oral ? 0.095 : 0.040,
          oral ? 1 : 0,
        ],
        w * 4,
      );
      w++;
    }
  });
  geo.setAttribute('iArm', new InstancedBufferAttribute(arms, 4));
  geo.instanceCount = ARM_COUNT;
  return geo;
};

/**
 * Parametric grid for the bell: `position` carries (t, theta), with t already
 * normalised to 0..1. Storing the raw angle instead and dividing in the shader
 * works right up until the profile's smoothstep is evaluated outside its domain,
 * at which point the margin ring explodes to fifteen times its radius and one
 * stray triangle covers the screen.
 */
const buildBell = (rings: number, segs: number): InstancedBufferGeometry => {
  const geo = new InstancedBufferGeometry();
  const params: number[] = [];
  const index: number[] = [];
  for (let i = 0; i <= rings; i++) {
    const t = i / rings;
    for (let j = 0; j <= segs; j++) {
      params.push(t, (j / segs) * Math.PI * 2, 0);
    }
  }
  const row = segs + 1;
  for (let i = 0; i < rings; i++) {
    for (let j = 0; j < segs; j++) {
      const a = i * row + j;
      index.push(a, a + row, a + 1, a + 1, a + row, a + row + 1);
    }
  }
  geo.setAttribute('position', new Float32BufferAttribute(params, 3));
  geo.setIndex(index);
  return geo;
};

class AbyssalBloom implements SceneInstance {
  readonly three = new Scene();
  readonly rig: CameraRig;

  private readonly waterMat: ShaderMaterial;
  private readonly waterMesh: Mesh;
  private readonly bellMat: ShaderMaterial;
  private readonly armMat: ShaderMaterial;
  private readonly snowMat: ShaderMaterial;
  private readonly geoms: BufferGeometry[] = [];

  constructor(ctx: SceneContext, quality: number) {
    this.rig = new CameraRig({
      position: new Vector3(0, 7.5, 18),
      // Aimed up and forward. Looking *down* would give an aquarium; the shafts
      // only read when the eye travels toward their source.
      target: new Vector3(-2, 26, -70),
      fov: 47,
      near: 0.4,
      far: 900,
      driftMetres: 2.1,
      fovBreath: 0.032,
      parallaxMetres: 0.7,
      parallaxHalfLife: 1.8,
    });

    const colony = buildColony();

    // --- water ---
    this.waterMat = new ShaderMaterial({
      vertexShader: FULLSCREEN_VERT,
      fragmentShader: WATER_FRAG,
      uniforms: {
        uPhase: { value: 0 },
        uCamPos: { value: new Vector3() },
        uCamFwd: { value: new Vector3() },
        uCamRight: { value: new Vector3() },
        uCamUp: { value: new Vector3() },
        uTanHalfFov: { value: Math.tan((47 * Math.PI) / 360) },
        uAspect: { value: ctx.width / ctx.height },
        uVolumetricSteps: { value: VOLUMETRIC_LUT[quality] ?? 26 },
        uSurfaceY: { value: SURFACE_Y },
        uToLight: { value: TO_LIGHT.clone() },
        uLightColor: { value: vec3(LIGHT_COLOR) },
        uWaterTint: { value: vec3(WATER_TINT) },
      },
      depthTest: false,
      depthWrite: false,
      // DoubleSide, not BackSide: the clip-space gl_Position override does not
      // change screen-space winding, so BackSide culls the quad away entirely.
      side: DoubleSide,
    });
    this.waterMesh = new Mesh(new PlaneGeometry(2, 2), this.waterMat);
    this.waterMesh.frustumCulled = false;
    this.waterMesh.renderOrder = -100;
    this.geoms.push(this.waterMesh.geometry);
    this.three.add(this.waterMesh);

    const fogUniforms = () => ({
      uToLight: { value: TO_LIGHT.clone() },
      uLightColor: { value: vec3(LIGHT_COLOR) },
      uWaterTint: { value: vec3(WATER_TINT) },
      uSurfaceY: { value: SURFACE_Y },
    });

    // --- bells ---
    const bellGeo = buildBell(26, 40);
    bellGeo.setAttribute('iJelly', new InstancedBufferAttribute(colony.jelly, 4));
    bellGeo.setAttribute('iBell', new InstancedBufferAttribute(colony.bell, 4));
    bellGeo.setAttribute('iLife', new InstancedBufferAttribute(colony.life, 2));
    bellGeo.setAttribute('iHue', new InstancedBufferAttribute(colony.hue, 3));
    bellGeo.setAttribute('iGlow', new InstancedBufferAttribute(colony.glow, 1));
    bellGeo.instanceCount = colony.count;
    this.geoms.push(bellGeo);

    this.bellMat = new ShaderMaterial({
      vertexShader: BELL_VERT,
      fragmentShader: BELL_FRAG,
      uniforms: { uPhase: { value: 0 }, ...fogUniforms() },
      side: DoubleSide,
    });
    const bells = new Mesh(bellGeo, this.bellMat);
    bells.frustumCulled = false;
    bells.renderOrder = 10;
    // DEBUG_BISECT
    this.three.add(bells);

    // --- arms ---
    const armGeo = buildArms(22);
    armGeo.setAttribute('iJelly', new InstancedBufferAttribute(colony.armJelly, 4));
    armGeo.setAttribute('iBell', new InstancedBufferAttribute(colony.armBell, 4));
    armGeo.setAttribute('iLife', new InstancedBufferAttribute(colony.armLife, 2));
    armGeo.setAttribute('iHue', new InstancedBufferAttribute(colony.armHue, 3));
    this.geoms.push(armGeo);

    this.armMat = new ShaderMaterial({
      vertexShader: ARM_VERT,
      fragmentShader: ARM_FRAG,
      uniforms: { uPhase: { value: 0 }, ...fogUniforms() },
      transparent: true,
      depthWrite: false,
      blending: AdditiveBlending,
      side: DoubleSide,
    });
    const armMesh = new Mesh(armGeo, this.armMat);
    armMesh.frustumCulled = false;
    armMesh.renderOrder = 20;
    this.three.add(armMesh);

    // --- marine snow ---
    const snowCount = 1500;
    const snowGeo = new BufferGeometry();
    const snowSeed = new Float32Array(snowCount * 3);
    for (let i = 0; i < snowCount; i++) {
      snowSeed[i * 3] = (Math.sin(i * 12.9898) * 43758.5453) % 1;
      snowSeed[i * 3 + 1] = (Math.sin(i * 39.3468) * 24634.6345) % 1;
      snowSeed[i * 3 + 2] = (Math.sin(i * 73.156) * 19349.1234) % 1;
    }
    snowGeo.setAttribute('position', new Float32BufferAttribute(new Float32Array(snowCount * 3), 3));
    snowGeo.setAttribute('iSeed', new Float32BufferAttribute(snowSeed, 3));
    this.geoms.push(snowGeo);

    this.snowMat = new ShaderMaterial({
      vertexShader: SNOW_VERT,
      fragmentShader: SNOW_FRAG,
      uniforms: {
        uPhase: { value: 0 },
        uSize: { value: 2.8 },
        uColor: { value: new Vector3(0.62, 0.82, 0.86) },
        uToLight: { value: TO_LIGHT.clone() },
      },
      transparent: true,
      depthWrite: false,
      blending: AdditiveBlending,
    });
    const snow = new Points(snowGeo, this.snowMat);
    snow.frustumCulled = false;
    snow.renderOrder = 30;
    this.three.add(snow);
  }

  update(info: FrameInfo): void {
    const phase = info.phase;
    const cam = this.rig.camera as PerspectiveCamera;

    this.rig.setPointer(info.pointer.x, info.pointer.y, info.idleness);
    this.rig.update(phase, info.dt);

    // The water quad is drawn in clip space, so the camera basis has to be fed in
    // by hand — a real projection would fight the gl_Position override.
    const w = this.waterMat.uniforms;
    w.uPhase.value = phase;
    w.uTanHalfFov.value = Math.tan((cam.fov * Math.PI) / 360);
    w.uAspect.value = info.ctx.width / info.ctx.height;
    (w.uCamPos.value as Vector3).copy(cam.position);
    (w.uCamFwd.value as Vector3).set(0, 0, -1).applyQuaternion(cam.quaternion);
    (w.uCamRight.value as Vector3).set(1, 0, 0).applyQuaternion(cam.quaternion);
    (w.uCamUp.value as Vector3).set(0, 1, 0).applyQuaternion(cam.quaternion);

    this.bellMat.uniforms.uPhase.value = phase;
    this.armMat.uniforms.uPhase.value = phase;
    this.snowMat.uniforms.uPhase.value = phase;
  }

  tune(quality: QualityTune): void {
    (this.waterMat.uniforms.uVolumetricSteps as { value: number }).value =
      quality.volumetricSteps;
  }

  resize(ctx: SceneContext): void {
    (this.waterMat.uniforms.uAspect as { value: number }).value = ctx.width / ctx.height;
  }

  audio(info: FrameInfo, out: AudioBus): void {
    // The abyss is not silent, it is *low*: a broadband bed with almost no top,
    // plus a slow swell on the same period as the surge. §12: the piece has to be
    // complete muted, so this only ever adds to the picture.
    const swell = cyclicFbm1(info.phase, 91, 3, 1) * 2 - 1;
    out.noise(0.05 + swell * 0.025, 180 + swell * 90);
    out.tone(52 + swell * 6, 0.012 + swell * 0.008);
  }

  dispose(): void {
    for (const g of this.geoms) g.dispose();
    this.waterMat.dispose();
    this.bellMat.dispose();
    this.armMat.dispose();
    this.snowMat.dispose();
    this.rig.dispose();
    this.three.clear();
  }
}

export const abyssalBloomDefinition: SceneDefinition = {
  id: 'abyssal-bloom',
  title: '深海流光',
  periodSeconds: PERIOD,
  note: '46 米深 · 体积光柱 · 磷光水母群',
  create: (ctx) => new AbyssalBloom(ctx, 2),
};