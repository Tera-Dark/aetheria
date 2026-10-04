/**
 * Renderer and frame loop.
 *
 * §14 of the brief: this must run for days. So:
 *   • one clock, one loop, no per-scene timers
 *   • `dt` is clamped, so a tab that was backgrounded for an hour does not
 *     produce a single enormous step
 *   • resize is debounced through rAF, and pixel ratio is *budgeted* rather than
 *     taken from `devicePixelRatio` (a 4K wallpaper on a retina display would
 *     otherwise ask for 33M pixels and melt)
 *   • the loop phase is derived from a single monotonic origin, so it cannot
 *     drift relative to wall-clock time over hours
 */

import type { WebGLRenderer } from 'three';

export interface FrameBudget {
  /** Soft ceiling on drawing-buffer pixels. */
  maxPixels: number;
  /** Ceiling on devicePixelRatio. */
  maxDpr: number;
}

export const DEFAULT_BUDGET: FrameBudget = {
  maxPixels: 2560 * 1440,
  maxDpr: 2,
};

/**
 * Choose a device pixel ratio that fits inside a pixel budget.
 *
 * Naively using `devicePixelRatio` is the single most common way a wallpaper
 * app dies on a good monitor: 3840×2160 at DPR 2 is 33 megapixels, and every
 * one of them goes through a fullscreen fragment shader.
 */
export const resolveDpr = (
  cssWidth: number,
  cssHeight: number,
  budget: FrameBudget,
  quality = 1,
): number => {
  const cssPixels = Math.max(1, cssWidth * cssHeight);
  const affordable = Math.sqrt(budget.maxPixels / cssPixels);
  const dpr = Math.min(window.devicePixelRatio || 1, budget.maxDpr, affordable);
  // Never below 0.6 — below that the image stops being worth looking at.
  return Math.max(0.6, dpr * clampQuality(quality));
};

const clampQuality = (q: number): number => Math.min(1, Math.max(0.4, q));

export interface Viewport {
  cssWidth: number;
  cssHeight: number;
  bufferWidth: number;
  bufferHeight: number;
  dpr: number;
  aspect: number;
}

export const measureViewport = (
  renderer: WebGLRenderer,
  budget: FrameBudget,
  quality: number,
): Viewport => {
  const cssWidth = Math.max(1, window.innerWidth);
  const cssHeight = Math.max(1, window.innerHeight);
  const dpr = resolveDpr(cssWidth, cssHeight, budget, quality);
  renderer.setPixelRatio(dpr);
  renderer.setSize(cssWidth, cssHeight, true);
  return {
    cssWidth,
    cssHeight,
    bufferWidth: Math.round(cssWidth * dpr),
    bufferHeight: Math.round(cssHeight * dpr),
    dpr,
    aspect: cssWidth / cssHeight,
  };
};

export interface StageOptions {
  canvas: HTMLCanvasElement;
  budget?: Partial<FrameBudget>;
  /** Refuse any single frame step longer than this. */
  maxDt?: number;
  /**
   * Fired after a resize event, already debounced through rAF. No arguments on
   * purpose: measuring the viewport needs the renderer, which lives in the app,
   * and handing out a stale copy of the numbers is worse than handing out none.
   */
  onResize?: () => void;
}

/**
 * Owns rAF, the clock, and resize. Deliberately knows nothing about scenes —
 * it hands out a monotonic elapsed time and the derived loop phase.
 */
export class Stage {
  readonly canvas: HTMLCanvasElement;
  readonly budget: FrameBudget;
  readonly maxDt: number;

  /** Seconds since start. Monotonic, from a single origin. */
  elapsed = 0;
  /** Clamped delta for the current frame. */
  dt = 1 / 60;
  /**
   * Unclamped delta for the current frame.
   *
   * Simulation uses `dt`, because a tab that was backgrounded for an hour must
   * not hand the scene one enormous step. Interface timing uses this instead: a
   * cross-fade that advances by `dt` takes 1.8 seconds of *frames*, which on a
   * machine running at 4fps is most of a minute of wall clock — long enough that
   * the transition is still running when the next one is asked for.
   */
  rawDt = 1 / 60;

  private origin = performance.now() / 1000;
  private last = this.origin;
  private rafId = 0;
  private running = false;
  private resizeQueued = false;
  private readonly onResize?: () => void;

  constructor(options: StageOptions) {
    this.canvas = options.canvas;
    this.budget = { ...DEFAULT_BUDGET, ...options.budget };
    this.maxDt = options.maxDt ?? 1 / 15;
    this.onResize = options.onResize;

    window.addEventListener('resize', this.handleResize);
    // The tab can come back after any amount of time. Re-anchor instead of
    // letting a huge dt through.
    document.addEventListener('visibilitychange', this.handleVisibility);
  }

  private handleVisibility = (): void => {
    if (!document.hidden) this.last = performance.now() / 1000;
  };

  private handleResize = (): void => {
    if (this.resizeQueued) return;
    this.resizeQueued = true;
    requestAnimationFrame(() => {
      this.resizeQueued = false;
      this.onResize?.();
    });
  };

  start(tick: (elapsed: number, dt: number, rawDt: number) => void): void {
    if (this.running) return;
    this.running = true;
    this.last = performance.now() / 1000;

    const frame = (): void => {
      if (!this.running) return;
      const now = performance.now() / 1000;
      const raw = now - this.last;
      this.last = now;
      this.rawDt = Math.max(1 / 480, raw);
      this.dt = Math.min(this.maxDt, this.rawDt);
      this.elapsed = now - this.origin;
      tick(this.elapsed, this.dt, this.rawDt);
      this.rafId = requestAnimationFrame(frame);
    };

    this.rafId = requestAnimationFrame(frame);
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.rafId);
  }

  dispose(): void {
    this.stop();
    window.removeEventListener('resize', this.handleResize);
    document.removeEventListener('visibilitychange', this.handleVisibility);
  }
}