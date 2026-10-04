/**
 * The scene contract.
 *
 * A scene is a pure function of loop phase. It is handed φ every frame and must
 * rebuild its entire visual state from that number alone. It must not keep
 * mutable state between frames — see `core/loop.ts` for why.
 *
 * The consequence worth internalising: a scene is not an animation, it is a
 * *function*. `build(0.42)` always produces the same world. That is what makes
 * the loop seamless, the runtime drift-free, and any frame reproducible.
 */

import type { Camera, Scene as ThreeScene } from 'three';

export interface SceneContext {
  /** Absolute pixel size of the drawing buffer. */
  width: number;
  height: number;
  /** 0..1, where 1 == the render target's full resolution. */
  quality: number;
  /** Muted per the user's preference; scenes must stay fully legible without audio. */
  audioOn: boolean;
}

export interface FrameInfo {
  /**
   * Loop phase in [0,1). Every animation in the scene derives from this and
   * nothing else. Scene-local phase is preferred so scenes can pick their own
   * period independent of the global one.
   */
  phase: number;
  /** Seconds since the loop wrapped. For effects that need absolute-ish time. */
  loopElapsed: number;
  /** Seconds since start. Use only for things that need not loop (perf counters). */
  elapsed: number;
  /** Wall-clock delta in seconds, already clamped. */
  dt: number;
  /**
   * Pointer position in normalised device coords (-1..1), smoothed. The ONLY
   * sanctioned non-phase input, and it must influence framing only — never
   * scene state — so that leaving the pointer still produces a perfect loop.
   */
  pointer: { x: number; y: number };
  /** 0 while the pointer is active, easing to 1 after the idle timeout. */
  idleness: number;
  ctx: SceneContext;
}

export interface SceneInstance {
  readonly three: ThreeScene;
  /**
   * Rebuild the world for `info.phase`. Called exactly once per rendered frame.
   * Must be a pure function of `info` — no `+=`, no `Math.random()`, no timers.
   */
  update(info: FrameInfo): void;
  /** Called on viewport change. Recompute resolution-dependent uniforms only. */
  resize(ctx: SceneContext): void;
  /**
   * Re-tune for a new quality level *without* being rebuilt. Optional: a scene
   * whose cost is baked into its allocations simply omits this and lets the
   * shell recreate it instead.
   *
   * The hard rule is that this must not allocate. Recreating the scene on every
   * quality change would churn GPU memory on a governor that settles after a few
   * seconds but can re-trigger whenever the machine gets busy.
   */
  tune?(quality: QualityTune): void;
  /**
   * Per-frame audio contribution. Optional. Receives the same phase so the mix
   * can loop with the picture.
   */
  audio?(info: FrameInfo, out: AudioBus): void;
  /** Release GPU resources. Called on scene change. Must be complete. */
  dispose(): void;
}

/** The subset of a quality level a scene is allowed to see. */
export interface QualityTune {
  /** Steps of volumetric integration. */
  volumetricSteps: number;
  /** Multiplier on the drawing-buffer resolution. */
  scale: number;
}

export interface AudioBus {
  /** Broadband noise level, 0..1. */
  noise(amount: number, tone?: number): void;
  /** Sine partial. */
  tone(freq: number, amount: number): void;
}

export interface SceneDefinition {
  id: string;
  title: string;
  /**
   * Loop period in seconds. §7 of the brief: prefer slow. The user's real test
   * is "did I notice it restart", and the answer should be no for minutes.
   */
  periodSeconds: number;
  /** One line, shown in the launcher. */
  note: string;
  create(ctx: SceneContext): SceneInstance;
}

export type SceneFactory = () => SceneDefinition;

/** Registry — adding a scene must not require touching anything else. */
const registry = new Map<string, SceneFactory>();

export const registerScene = (factory: SceneFactory): void => {
  const def = factory();
  registry.set(def.id, factory);
};

export const listScenes = (): SceneDefinition[] =>
  [...registry.values()].map((f) => f());

/** Helper: an `AudioBus` that discards everything. */
export const nullAudioBus: AudioBus = {
  noise: () => undefined,
  tone: () => undefined,
};

export type { Camera };