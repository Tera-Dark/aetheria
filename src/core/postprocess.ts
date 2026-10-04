/**
 * Post-processing.
 *
 * A deliberately small chain built from `WebGLRenderTarget` passes driven by a
 * single fullscreen quad. No `EffectComposer` — the per-frame overhead it
 * adds to a scene that must still look correct on frame 10 000 is not worth the
 * flexibility.
 *
 *   scene ─► HDR target (+ DepthTexture)
 *         ─► bright pass (soft-knee threshold) at half res
 *         ─► 5-level blur pyramid, separable Gaussian
 *         ─► composite: DOF, bloom, ACES, lift/gamma/gain, vignette, grain
 *
 * Every intermediate target is allocated once in `setSize`. Per-frame
 * allocation would be a §14 violation, so the materials and targets are reused
 * verbatim for the life of the chain.
 *
 * The one exception is the second ("B") side of a scene cross-fade: it is
 * allocated when a transition starts and released the moment it ends, so a
 * wallpaper that is just sitting there is not holding two full HDR buffers for
 * a fade that happens once every few minutes.
 */

import {
  DepthTexture,
  HalfFloatType,
  LinearFilter,
  Mesh,
  AdditiveBlending,
  NoBlending,
  OrthographicCamera,
  PlaneGeometry,
  RGBAFormat,
  Scene,
  ShaderMaterial,
  Vector2,
  Vector3,
  WebGLRenderTarget,
  type Camera,
  type Texture,
  type WebGLRenderer,
} from 'three';
import { GLSL_COLOR, GLSL_HASH } from '../shaders/common';

const VERT = /* glsl */ `
varying vec2 vUv;
void main(){
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

const BRIGHT_FRAG = /* glsl */ `
precision highp float;
uniform sampler2D tSrc;
uniform float uThreshold;
uniform float uKnee;
varying vec2 vUv;
void main(){
  vec3 c = texture2D(tSrc, vUv).rgb;
  float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
  // Soft knee. A hard threshold makes bloom snap on and off as values cross it,
  // which is very visible on a slow-moving scene.
  float k = smoothstep(uThreshold - uKnee, uThreshold + uKnee, l);
  gl_FragColor = vec4(c * k, 1.0);
}
`;

const BLUR_FRAG = /* glsl */ `
precision highp float;
uniform sampler2D tSrc;
uniform vec2 uTexel;
uniform vec2 uDir;
varying vec2 vUv;
void main(){
  // 9-tap Gaussian collapsed to 5 linear-filtered fetches.
  vec2 o1 = uTexel * uDir * 1.3846153846;
  vec2 o2 = uTexel * uDir * 3.2307692308;
  vec3 c  = texture2D(tSrc, vUv).rgb * 0.2270270270;
  c += texture2D(tSrc, vUv + o1).rgb * 0.3162162162;
  c += texture2D(tSrc, vUv - o1).rgb * 0.3162162162;
  c += texture2D(tSrc, vUv + o2).rgb * 0.0702702703;
  c += texture2D(tSrc, vUv - o2).rgb * 0.0702702703;
  gl_FragColor = vec4(c, 1.0);
}
`;

const ADD_FRAG = /* glsl */ `
precision highp float;
uniform sampler2D tSrc;
uniform float uWeight;
varying vec2 vUv;
void main(){
  gl_FragColor = vec4(texture2D(tSrc, vUv).rgb * uWeight, 1.0);
}
`;

const COMPOSITE_FRAG = /* glsl */ `
precision highp float;
${GLSL_HASH}
${GLSL_COLOR}
uniform sampler2D tScene;
uniform sampler2D tSceneB;
uniform sampler2D tBloom;
uniform sampler2D tBloomB;
uniform sampler2D tDepth;
uniform sampler2D tDepthB;
/** 0 = base scene only, 1 = incoming scene only. */
uniform float uMix;
uniform float uBloom;
uniform float uGrainSeed;
uniform float uExposure;
uniform float uFocusDistance;
uniform float uAperture;
uniform float uMaxCoC;
uniform vec2 uTexel;
uniform vec3 uLift;
uniform vec3 uGain;
uniform float uGamma;
uniform float uSaturation;
uniform float uContrast;
uniform float uVignette;
uniform float uGrain;
uniform float uNear;
uniform float uFar;
varying vec2 vUv;

float viewDepth(float d){
  float z = d * 2.0 - 1.0;
  return (2.0 * uNear * uFar) / (uFar + uNear - z * (uFar - uNear));
}

/**
 * Thin-lens circle of confusion. A wallpaper wants just enough defocus to read
 * as a lens; strong bokeh would compete with the subject.
 */
vec3 sampleDof(sampler2D tex, sampler2D dep){
  float coc = 0.0;
  if (uAperture > 0.0001){
    float d = viewDepth(texture2D(dep, vUv).r);
    float c = abs(d - uFocusDistance) / max(d, 0.001) * uAperture;
    coc = clamp(c, 0.0, 1.0);
    coc = coc * coc * uMaxCoC;
  }
  vec3 col = texture2D(tex, vUv).rgb;
  if (coc > 0.75){
    vec3 sum = col;
    for (int i = 0; i < 8; i++){
      float a = float(i) * 0.7853981634;
      vec2 o = vec2(cos(a), sin(a)) * coc * uTexel;
      sum += texture2D(tex, vUv + o).rgb;
    }
    col = sum / 9.0;
  }
  return col;
}

void main(){
  // ---- depth of field + cross-fade -------------------------------------
  vec3 color = sampleDof(tScene, tDepth);
  if (uMix > 0.0){
    color = mix(color, sampleDof(tSceneB, tDepthB), uMix);
    color += mix(texture2D(tBloom, vUv).rgb, texture2D(tBloomB, vUv).rgb, uMix) * uBloom;
  } else {
    color += texture2D(tBloom, vUv).rgb * uBloom;
  }

  // ---- tone map + grade -----------------------------------------------
  color *= uExposure;
  color = acesToneMap(color);
  color = grade(color, uLift, uGain, uGamma, uSaturation, uContrast);

  // ---- vignette --------------------------------------------------------
  vec2 v = vUv - 0.5;
  color *= mix(1.0, smoothstep(0.84, 0.26, length(v)), uVignette);

  // ---- grain -----------------------------------------------------------
  // Weighted into the shadows, where real film grain lives. Grain spread evenly
  // over the frame reads as a dirty lens instead of as texture.
  //
  // The seed comes from the loop *phase*, not from a running clock. Grain is the
  // one thing in the frame that is legitimately allowed to differ between two
  // phases — it is noise — but it must still be a function of phase, or the
  // seam at φ=1 is a jump cut in the grain pattern rather than a wrap.
  if (uGrain > 0.0001){
    float g = hash12(vUv * 2048.0 + uGrainSeed);
    float shadowWeight = 1.0 - smoothstep(0.0, 0.65, luma(color));
    color += (g - 0.5) * uGrain * shadowWeight;
  }

  gl_FragColor = vec4(max(color, 0.0), 1.0);
}
`;

export interface GradeSpec {
  exposure: number;
  bloom: number;
  bloomThreshold: number;
  /** Focus distance in metres. */
  focusDistance: number;
  aperture: number;
  vignette: number;
  grain: number;
  lift: [number, number, number];
  gain: [number, number, number];
  gamma: number;
  saturation: number;
  contrast: number;
}

export const NEUTRAL_GRADE: GradeSpec = {
  exposure: 1,
  bloom: 0.7,
  bloomThreshold: 0.9,
  focusDistance: 60,
  aperture: 0.06,
  vignette: 0.5,
  grain: 0.018,
  lift: [0, 0, 0],
  gain: [1, 1, 1],
  gamma: 1,
  saturation: 1,
  contrast: 1,
};

/** The scene being faded in over the base scene. `mix` runs 0 → 1. */
export interface FadeIn {
  scene: Scene;
  camera: Camera;
  mix: number;
}

const BLOOM_LEVELS = 5;

const makeHdrTarget = (
  width: number,
  height: number,
  depth: boolean,
): WebGLRenderTarget =>
  new WebGLRenderTarget(width, height, {
    type: HalfFloatType,
    format: RGBAFormat,
    minFilter: LinearFilter,
    magFilter: LinearFilter,
    depthBuffer: depth,
    stencilBuffer: false,
  });

/** Half-res bright target plus a 5-level pyramid of ping-pong pairs. */
interface BloomRack {
  bright: WebGLRenderTarget;
  a: WebGLRenderTarget[];
  b: WebGLRenderTarget[];
}

const makeRack = (width: number, height: number): BloomRack => {
  const rack: BloomRack = {
    bright: makeHdrTarget(Math.max(2, width >> 1), Math.max(2, height >> 1), false),
    a: [],
    b: [],
  };
  let w = Math.max(2, width >> 1);
  let h = Math.max(2, height >> 1);
  for (let i = 0; i < BLOOM_LEVELS; i++) {
    rack.a.push(makeHdrTarget(w, h, false));
    rack.b.push(makeHdrTarget(w, h, false));
    w = Math.max(2, w >> 1);
    h = Math.max(2, h >> 1);
  }
  return rack;
};

const disposeRack = (rack: BloomRack | null): void => {
  if (!rack) return;
  rack.bright.dispose();
  rack.a.forEach((t) => t.dispose());
  rack.b.forEach((t) => t.dispose());
};

export class PostChain {
  private sceneTarget: WebGLRenderTarget | null = null;
  private depth: DepthTexture | null = null;
  private rackA: BloomRack | null = null;

  private sceneTargetB: WebGLRenderTarget | null = null;
  private depthB: DepthTexture | null = null;
  private rackB: BloomRack | null = null;

  /** Fullscreen quad. One mesh, reassigned between materials. */
  private readonly quadScene = new Scene();
  private readonly quadCamera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private readonly quadMesh: Mesh;
  private readonly quadGeometry = new PlaneGeometry(2, 2);

  private readonly brightMat: ShaderMaterial;
  private readonly blurMat: ShaderMaterial;
  private readonly addMat: ShaderMaterial;
  private readonly compositeMat: ShaderMaterial;

  private grade: GradeSpec = { ...NEUTRAL_GRADE };
  private width = 2;
  private height = 2;

  constructor(private readonly renderer: WebGLRenderer) {
    this.quadMesh = new Mesh(this.quadGeometry);
    this.quadMesh.frustumCulled = false;
    this.quadScene.add(this.quadMesh);

    this.brightMat = new ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: BRIGHT_FRAG,
      uniforms: {
        tSrc: { value: null },
        uThreshold: { value: 0.9 },
        uKnee: { value: 0.3 },
      },
      depthTest: false,
      depthWrite: false,
      blending: NoBlending,
    });

    this.blurMat = new ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: BLUR_FRAG,
      uniforms: {
        tSrc: { value: null },
        uTexel: { value: new Vector2() },
        uDir: { value: new Vector2(1, 0) },
      },
      depthTest: false,
      depthWrite: false,
      blending: NoBlending,
    });

    this.addMat = new ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: ADD_FRAG,
      uniforms: {
        tSrc: { value: null },
        uWeight: { value: 1 },
      },
      depthTest: false,
      depthWrite: false,
      blending: AdditiveBlending,
    });

    this.compositeMat = new ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: COMPOSITE_FRAG,
      uniforms: {
        tScene: { value: null },
        tSceneB: { value: null },
        tBloom: { value: null },
        tBloomB: { value: null },
        tDepth: { value: null },
        tDepthB: { value: null },
        uMix: { value: 0 },
        uBloom: { value: 0.7 },
        uGrainSeed: { value: 0 },
        uExposure: { value: 1 },
        uFocusDistance: { value: 60 },
        uAperture: { value: 0.06 },
        uMaxCoC: { value: 12 },
        uTexel: { value: new Vector2() },
        uLift: { value: new Vector3() },
        uGain: { value: new Vector3(1, 1, 1) },
        uGamma: { value: 1 },
        uSaturation: { value: 1 },
        uContrast: { value: 1 },
        uVignette: { value: 0.5 },
        uGrain: { value: 0.018 },
        uNear: { value: 0.1 },
        uFar: { value: 1000 },
      },
      depthTest: false,
      depthWrite: false,
      blending: NoBlending,
    });
  }

  setSize(width: number, height: number): void {
    const w = Math.max(2, Math.floor(width));
    const h = Math.max(2, Math.floor(height));
    if (w === this.width && h === this.height && this.sceneTarget) return;
    this.width = w;
    this.height = h;

    this.sceneTarget?.dispose();
    this.depth?.dispose();
    disposeRack(this.rackA);
    this.releaseSecond();

    // A DepthTexture on the scene target means DOF costs nothing extra — no
    // second scene render just to obtain depth. It has to be assigned after
    // construction: passing it through the options object reaches
    // `setDepthTexture` before the render target has initialised.
    this.depth = new DepthTexture(w, h);
    this.sceneTarget = makeHdrTarget(w, h, true);
    this.sceneTarget.depthTexture = this.depth;
    this.rackA = makeRack(w, h);

    (this.compositeMat.uniforms.uTexel.value as Vector2).set(1 / w, 1 / h);
  }

  setGrade(patch: Partial<GradeSpec>): void {
    this.grade = { ...this.grade, ...patch };
  }

  get currentGrade(): GradeSpec {
    return this.grade;
  }

  private blit(material: ShaderMaterial, target: WebGLRenderTarget | null, clear = true): void {
    this.quadMesh.material = material;
    this.renderer.setRenderTarget(target);
    if (clear) this.renderer.clear();
    this.renderer.render(this.quadScene, this.quadCamera);
  }

  /**
   * Bright pass → blur pyramid → fold the coarse levels back down.
   * Returns the texture holding the finished bloom.
   */
  private runBloom(src: Texture, rack: BloomRack): Texture {
    const bu = this.brightMat.uniforms;
    bu.tSrc.value = src;
    bu.uThreshold.value = this.grade.bloomThreshold;
    this.blit(this.brightMat, rack.bright);

    // Separable blur. Each level needs two targets: the horizontal pass writes
    // A, the vertical pass reads A and writes B. Reusing one target for both is
    // the classic mistake — the pass ends up reading the texture it is writing,
    // which drivers reject as a feedback loop and silently drop.
    const blur = this.blurMat.uniforms;
    let cursor: WebGLRenderTarget = rack.bright;
    for (let i = 0; i < rack.a.length; i++) {
      const a = rack.a[i];
      const b = rack.b[i];

      blur.tSrc.value = cursor.texture as Texture;
      (blur.uTexel.value as Vector2).set(1 / a.width, 1 / a.height);
      (blur.uDir.value as Vector2).set(1, 0);
      this.blit(this.blurMat, a);

      blur.tSrc.value = a.texture as Texture;
      (blur.uDir.value as Vector2).set(0, 1);
      this.blit(this.blurMat, b);

      cursor = b;
    }

    // Widen: fold each coarse level back into the one above it. This is what
    // turns a tight glow into a halo. Two details matter:
    //   • the fold must *accumulate*, so the blit must not clear the target —
    //     clearing here replaces level i with level i+1 and leaves only the
    //     coarsest level in the sum
    //   • additive blending sidesteps the read/write feedback constraint,
    //     because the source is always a different target
    for (let i = rack.a.length - 2; i >= 0; i--) {
      this.addMat.uniforms.tSrc.value = rack.b[i + 1].texture as Texture;
      this.addMat.uniforms.uWeight.value = 0.72;
      this.blit(this.addMat, rack.b[i], false);
    }

    return rack.b[0].texture as Texture;
  }

  private ensureSecond(): boolean {
    if (this.sceneTargetB && this.rackB) return true;
    if (!this.sceneTarget) return false;
    const w = this.width;
    const h = this.height;
    this.depthB = new DepthTexture(w, h);
    this.sceneTargetB = makeHdrTarget(w, h, true);
    this.sceneTargetB.depthTexture = this.depthB;
    this.rackB = makeRack(w, h);
    return true;
  }

  private releaseSecond(): void {
    this.sceneTargetB?.dispose();
    this.depthB?.dispose();
    disposeRack(this.rackB);
    this.sceneTargetB = null;
    this.depthB = null;
    this.rackB = null;
  }

  /**
   * @param phase loop phase, used only to seed the film grain
   */
  render(scene: Scene, camera: Camera, phase: number, incoming?: FadeIn | null): void {
    const targetA = this.sceneTarget;
    const rackA = this.rackA;
    const depthA = this.depth;
    if (!targetA || !rackA || !depthA) return;
    const r = this.renderer;

    // 1. Scene → HDR, with depth attached.
    r.setRenderTarget(targetA);
    r.clear();
    r.render(scene, camera);

    const bloomA = this.runBloom(targetA.texture as Texture, rackA);

    const u = this.compositeMat.uniforms;
    u.tScene.value = targetA.texture as Texture;
    u.tDepth.value = depthA as unknown as Texture;
    u.tBloom.value = bloomA;

    if (incoming && incoming.mix > 0 && this.ensureSecond()) {
      const targetB = this.sceneTargetB as WebGLRenderTarget;
      const rackB = this.rackB as BloomRack;
      r.setRenderTarget(targetB);
      r.clear();
      r.render(incoming.scene, incoming.camera);

      u.tSceneB.value = targetB.texture as Texture;
      u.tDepthB.value = this.depthB as unknown as Texture;
      u.tBloomB.value = this.runBloom(targetB.texture as Texture, rackB);
      u.uMix.value = incoming.mix;
    } else {
      // Nothing is fading in, so the second set of HDR targets has no user. Free
      // them here rather than in the caller: this is the only place that knows
      // whether a fade is actually being drawn, and holding a spare full-res
      // buffer for the hours between transitions is exactly the kind of quiet
      // waste a wallpaper should not carry.
      this.releaseSecond();
      // Point the unused samplers at valid textures. Leaving them null makes
      // three bind a default texture, which is harmless here but produces
      // warnings that mask real ones.
      u.tSceneB.value = targetA.texture as Texture;
      u.tDepthB.value = depthA as unknown as Texture;
      u.tBloomB.value = bloomA;
      u.uMix.value = 0;
    }

    // 4. Composite to the default framebuffer.
    const g = this.grade;
    u.uBloom.value = g.bloom;
    u.uGrainSeed.value = phase * 997.0;
    u.uExposure.value = g.exposure;
    u.uFocusDistance.value = g.focusDistance;
    u.uAperture.value = g.aperture;
    u.uGamma.value = g.gamma;
    u.uSaturation.value = g.saturation;
    u.uContrast.value = g.contrast;
    u.uVignette.value = g.vignette;
    u.uGrain.value = g.grain;
    (u.uLift.value as Vector3).fromArray(g.lift);
    (u.uGain.value as Vector3).fromArray(g.gain);
    u.uNear.value = (camera as { near?: number }).near ?? 0.1;
    u.uFar.value = (camera as { far?: number }).far ?? 1000;

    this.blit(this.compositeMat, null);
  }

  dispose(): void {
    this.sceneTarget?.dispose();
    this.depth?.dispose();
    disposeRack(this.rackA);
    this.releaseSecond();
    this.sceneTarget = null;
    this.depth = null;
    this.rackA = null;
    this.quadGeometry.dispose();
    this.brightMat.dispose();
    this.blurMat.dispose();
    this.addMat.dispose();
    this.compositeMat.dispose();
  }
}