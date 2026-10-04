/**
 * The application shell.
 *
 * §17: the deliverable is a running wallpaper, not a rendered file. So the
 * launcher's job is to get out of the way completely — a fullscreen canvas, a
 * clock, and controls that appear on input and then fade.
 *
 * §13: interaction must be restrained, and §14 says it must survive days. So:
 *   • the HUD fades after a short idle and the pointer hides with it
 *   • scene switching cross-fades rather than cutting
 *   • scene transitions fully dispose the outgoing scene (§14: no leaks)
 *   • quality changes re-tune a uniform, never rebuild the scene
 */

import {
  Color,
  NoToneMapping,
  SRGBColorSpace,
  WebGLRenderer,
  type PerspectiveCamera,
} from 'three';

import { CameraRig } from './camera';
import { nullAudioBus } from './scene';
import type { AudioBus, FrameInfo, SceneDefinition, SceneInstance } from './scene';
import { Stage, measureViewport } from './renderer';
import type { Viewport } from './renderer';
import { PostChain } from './postprocess';
import type { FadeIn, GradeSpec } from './postprocess';
import { QualityGovernor, QUALITY } from './performance';
import type { QualityLevel } from './performance';
import { approach, clamp01, wrapPhase } from './loop';

export interface AppStats {
  fps: number;
  quality: string;
  /** True when the governor is pinned rather than adapting. */
  locked: boolean;
  scene: string;
  /** Loop phase of the last rendered frame, in [0,1). */
  phase: number;
  /** Current quality level index. */
  level: number;
  p50: number;
  p95: number;
}

export interface AppOptions {
  canvas: HTMLCanvasElement;
  scenes: readonly SceneDefinition[];
  /** Seconds without input before the interface disappears. */
  idleSeconds?: number;
  /** Seconds of cross-fade when switching scenes. */
  transitionSeconds?: number;
  audio?: AudioBus;
  /**
   * Keep the drawing buffer readable after present. Costs a copy per frame, so
   * it is off unless something needs to read pixels back — which is exactly
   * what `scripts/shoot.mjs` does to measure the loop seam.
   */
  capture?: boolean;
}

export class WallpaperApp {
  private readonly renderer: WebGLRenderer;
  private readonly stage: Stage;
  private readonly post: PostChain;
  private readonly governor = new QualityGovernor();
  private readonly bus: AudioBus;

  private readonly scenes: readonly SceneDefinition[];
  private instance: SceneInstance | null = null;
  private camera: PerspectiveCamera | null = null;

  private sceneIndex = 0;
  private loopElapsed = 0;
  /** Quality level the current scene has been tuned for. */
  private tuned: QualityLevel | null = null;

  private transitionFrom: SceneInstance | null = null;
  private transitionT = 1;
  /**
   * The outgoing scene keeps its *own* elapsed time. Resetting the shared clock
   * on a scene change and feeding the new value to the old scene would teleport
   * it to an arbitrary phase mid-fade, which reads as a glitch rather than a
   * cross-fade.
   */
  private fromElapsed = 0;
  private fromPeriod = 60;

  private viewport: Viewport;
  private ctx = {
    width: 1920,
    height: 1080,
    quality: 1,
    audioOn: false,
  };

  private gradeOverride: Partial<GradeSpec> | null = null;

  // Pointer state. The only genuinely persistent values in the app, and both
  // are eased toward their target so nothing snaps.
  private pointerTarget = { x: 0, y: 0 };
  private pointer = { x: 0, y: 0 };
  private lastInput = 0;
  private currentIdleness = 1;

  /** The phase actually used for the last rendered frame, pinned or not. */
  private renderedPhase = 0;
  private pinnedPhase: number | null = null;
  private pinnedFade: number | null = null;

  constructor(private readonly options: AppOptions) {
    this.scenes = options.scenes;
    this.bus = options.audio ?? nullAudioBus;

    this.renderer = new WebGLRenderer({
      canvas: options.canvas,
      antialias: false, // the post chain resolves edges; MSAA on an HDR target is not worth it
      alpha: false,
      powerPreference: 'high-performance',
      stencil: false,
      depth: true,
      preserveDrawingBuffer: options.capture ?? false,
    });
    this.renderer.outputColorSpace = SRGBColorSpace;
    this.renderer.toneMapping = NoToneMapping; // tone mapping happens in the composite pass
    this.renderer.setClearColor(new Color(0, 0, 0), 1);

    this.post = new PostChain(this.renderer);

    this.stage = new Stage({
      canvas: options.canvas,
      onResize: () => this.handleResize(),
    });

    this.viewport = measureViewport(this.renderer, this.stage.budget, 1);
    this.applyViewport();

    this.bindInput(options.canvas);
  }

  start(): void {
    this.load(0, true);
    this.stage.start((elapsed, dt, rawDt) => this.frame(elapsed, dt, rawDt));
  }

  stop(): void {
    this.stage.dispose();
    this.disposeScene(this.instance);
    this.disposeScene(this.transitionFrom);
    this.instance = null;
    this.transitionFrom = null;
    this.post.dispose();
    this.renderer.dispose();
  }

  /* ---------------------------------------------------------------- *
   * Scene management
   * ---------------------------------------------------------------- */

  private applyViewport(): void {
    this.ctx = {
      ...this.ctx,
      width: this.viewport.bufferWidth,
      height: this.viewport.bufferHeight,
      quality: this.viewport.dpr,
    };
    this.post.setSize(this.viewport.bufferWidth, this.viewport.bufferHeight);
  }

  private handleResize(): void {
    this.viewport = measureViewport(
      this.renderer,
      this.stage.budget,
      this.governor.current.scale,
    );
    this.applyViewport();
    this.instance?.resize(this.ctx);
    this.transitionFrom?.resize(this.ctx);
  }

  private rigOf(scene: SceneInstance | null): CameraRig | null {
    return (scene as unknown as { rig?: CameraRig } | null)?.rig ?? null;
  }

  private cameraFor(scene: SceneInstance): PerspectiveCamera | null {
    return this.rigOf(scene)?.camera ?? null;
  }

  /**
   * Focus distance for the DOF pass. Taken from the rig so the focal plane
   * breathes with the dolly — a fixed focus distance reads as a filter.
   */
  private focusTarget(phase: number): number {
    const rig = this.rigOf(this.instance);
    if (!rig || !this.camera) return 60;
    return rig.focusDistance(phase);
  }

  private load(index: number, immediate = false): void {
    const definition = this.scenes[index];
    if (!definition) return;

    const previous = this.instance;
    const previousPeriod = this.scenes[this.sceneIndex]?.periodSeconds ?? 60;
    this.sceneIndex = index;

    const next = definition.create(this.ctx);
    next.tune?.(this.governor.current);
    this.tuned = this.governor.current;

    if (immediate || !previous) {
      this.disposeScene(previous);
      this.instance = next;
      this.camera = this.cameraFor(next);
      this.transitionT = 1;
      this.loopElapsed = 0;
      return;
    }

    // Keep the outgoing scene alive for the length of the fade, then release it.
    this.disposeScene(this.transitionFrom);
    this.transitionFrom = previous;
    this.fromElapsed = this.loopElapsed;
    this.fromPeriod = previousPeriod;
    this.instance = next;
    this.camera = this.cameraFor(next);
    this.transitionT = 0;
    this.loopElapsed = 0;
  }

  next(): void {
    this.load((this.sceneIndex + 1) % this.scenes.length);
  }

  select(index: number): void {
    if (index === this.sceneIndex || index < 0 || index >= this.scenes.length) return;
    this.load(index);
  }

  private disposeScene(scene: SceneInstance | null): void {
    scene?.dispose();
  }

  /* ---------------------------------------------------------------- *
   * Input
   * ---------------------------------------------------------------- */

  private bindInput(canvas: HTMLCanvasElement): void {
    const wake = (): void => {
      this.lastInput = performance.now() / 1000;
    };

    window.addEventListener('pointermove', (e) => {
      this.pointerTarget.x = (e.clientX / window.innerWidth) * 2 - 1;
      this.pointerTarget.y = -((e.clientY / window.innerHeight) * 2 - 1);
      wake();
    });

    window.addEventListener('pointerdown', wake);
    window.addEventListener('wheel', wake, { passive: true });
    window.addEventListener('keydown', (e) => {
      wake();
      if (e.code === 'Space' || e.code === 'ArrowRight') {
        e.preventDefault();
        this.next();
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        this.select((this.sceneIndex - 1 + this.scenes.length) % this.scenes.length);
      } else if (e.code === 'KeyQ') {
        this.cycleQuality();
      }
    });
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  /** 0 while recently active, easing to 1 after the idle timeout. */
  private idleness(dt: number): number {
    const now = performance.now() / 1000;
    const since = now - this.lastInput;
    const timeout = this.options.idleSeconds ?? 2.4;
    const target = since > timeout ? 1 : 0;
    return approach(this.currentIdleness, target, target === 1 ? 0.9 : 0.08, dt);
  }

  /* ---------------------------------------------------------------- *
   * Frame
   * ---------------------------------------------------------------- */

  private frame(elapsed: number, dt: number, rawDt: number): void {
    this.governor.sample(dt, elapsed);

    // Quality level changed → re-measure the drawing buffer and re-tune the
    // scene. Both are one-off costs, so a governor that settles after a few
    // seconds is fine; what matters is that they are not per-frame.
    const level = this.governor.current;
    if (level !== this.tuned) {
      this.tuned = level;
      this.viewport = measureViewport(this.renderer, this.stage.budget, level.scale);
      this.applyViewport();
      this.instance?.tune?.(level);
    }

    const definition = this.scenes[this.sceneIndex];
    const period = definition?.periodSeconds ?? 60;

    // The loop phase is the whole contract: one number, from one origin,
    // recomputed from the wall clock every frame. Nothing accumulates.
    if (this.pinnedPhase === null) {
      this.loopElapsed += dt;
      if (this.loopElapsed >= period) this.loopElapsed -= period;
    }
    const phase =
      this.pinnedPhase !== null ? this.pinnedPhase : wrapPhase(this.loopElapsed, period);

    this.renderedPhase = phase;

    this.pointer.x = approach(this.pointer.x, this.pointerTarget.x, 0.35, dt);
    this.pointer.y = approach(this.pointer.y, this.pointerTarget.y, 0.35, dt);
    this.currentIdleness = this.idleness(dt);

    const info: FrameInfo = {
      phase,
      loopElapsed: this.loopElapsed,
      elapsed,
      dt,
      pointer: this.pointer,
      idleness: this.currentIdleness,
      ctx: { ...this.ctx, quality: this.governor.smoothScale(dt) },
    };

    if (this.transitionFrom) {
      if (this.pinnedFade !== null) {
        this.transitionT = 1 - this.pinnedFade;
      } else {
        // rawDt, not dt: a fade is interface timing, so it has to take 1.8
        // seconds of wall clock whether the machine is running at 120fps or 4.
        this.transitionT = Math.min(
          1,
          this.transitionT + rawDt / (this.options.transitionSeconds ?? 1.8),
        );
        // The outgoing scene advances on its own clock too — but with rawDt as
        // well, or it would crawl through the fade on a slow machine and the two
        // scenes would visibly desynchronise.
        this.fromElapsed += rawDt;
        if (this.transitionT >= 1) {
          this.disposeScene(this.transitionFrom);
          this.transitionFrom = null;
        }
      }

      if (this.transitionFrom) {
        const fromInfo: FrameInfo = {
          ...info,
          phase: wrapPhase(this.fromElapsed, this.fromPeriod),
          loopElapsed: this.fromElapsed,
        };
        this.transitionFrom.update(fromInfo);
        this.transitionFrom.audio?.(fromInfo, this.bus);
      }
    }

    if (!this.instance || !this.camera) return;

    this.instance.update(info);
    this.instance.audio?.(info, this.bus);

    this.post.setGrade({
      focusDistance: this.focusTarget(phase),
      aperture: level.dof * 0.06,
      bloom: level.bloom * 0.34,
      exposure: 0.92,
      // A hair above 1 so only genuine highlights bloom. The default of 0.9 let
      // the whole fog sea qualify.
      bloomThreshold: 1.02,
      vignette: 0.62,
      grain: 0.014,
      ...this.gradeOverride,
    });

    // Cross-fade: the outgoing scene is the base layer and the incoming one is
    // mixed over it. Both get their own HDR target, bloom pyramid and DOF, so
    // neither is graded through the other's depth.
    let base: SceneInstance = this.instance;
    let baseCamera: PerspectiveCamera = this.camera;
    let fade: FadeIn | null = null;

    const outgoingCamera = this.transitionFrom ? this.cameraFor(this.transitionFrom) : null;
    if (this.transitionFrom && outgoingCamera) {
      base = this.transitionFrom;
      baseCamera = outgoingCamera;
      fade = { scene: this.instance.three, camera: this.camera, mix: this.transitionT };
    }

    this.post.render(base.three, baseCamera, phase, fade);
  }

  /* ---------------------------------------------------------------- *
   * Inspection surface
   *
   * Everything here exists for `scripts/shoot.mjs`. A running clock cannot be
   * diffed against itself, so proving §6 needs a way to pin the phase, freeze
   * the adaptive quality, and read the pixels back.
   * ---------------------------------------------------------------- */

  /**
   * Pin the loop to an absolute phase, or release with `null`.
   *
   * Pinning overrides the clock only — every other code path is untouched, so
   * what gets captured is exactly what plays.
   */
  setPhase(phase: number | null): void {
    this.pinnedPhase = phase === null ? null : ((phase % 1) + 1) % 1;
  }

get isPinned(): boolean {
    return this.pinnedPhase !== null;
  }

  /**
   * Hold a cross-fade at a fixed mix, or release with `null`.
   *
   * Same reasoning as `setPhase`: a frame captured either side of a scene change
   * looks identical whether the cross-fade works or not, and on a machine where a
   * frame takes most of a second the middle of the fade is not reachable by
   * sleeping for a while. If no fade is running, one is started — so a caller can
   * just ask for the middle of a transition without also having to time the
   * change.
   */
  setFade(mix: number | null): void {
    if (mix === null) {
      this.pinnedFade = null;
      return;
    }
    this.pinnedFade = clamp01(mix);
    if (!this.transitionFrom) this.next();
  }

  get sceneTitles(): readonly string[] {
    return this.scenes.map((s) => s.title);
  }

  /** Freeze the quality governor at a level, or pass `null` to resume adapting. */
  lockQuality(level: number | null): void {
    this.governor.lock(level);
  }

  /**
   * Step up through the quality levels, then hand control back to the governor.
   *
   * A wallpaper needs an override — the machine may be busy with something else,
   * or the user may simply want the picture to stop changing under them — but a
   * toggle that can only be turned on and not off is worse than none.
   */
  cycleQuality(): void {
    const current = this.governor.lockedLevel;
    this.governor.lock(current === null ? 0 : current + 1 >= QUALITY.length ? null : current + 1);
  }

  /** Patch the post grade every frame, or `null` to drop the patch. */
  setGradeOverride(patch: Partial<GradeSpec> | null): void {
    this.gradeOverride = patch;
  }

  get stats(): AppStats {
    const g = this.governor.stats;
    return {
      fps: this.governor.fps,
      quality: this.governor.current.name,
      locked: this.governor.lockedLevel !== null,
      scene: this.scenes[this.sceneIndex]?.title ?? '',
      phase: this.renderedPhase,
      level: QUALITY.indexOf(this.governor.current),
      p50: g.p50,
      p95: g.p95,
    };
  }
}