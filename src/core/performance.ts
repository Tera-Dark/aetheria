/**
 * Frame pacing and the adaptive quality governor.
 *
 * §14: stable framerate, reasonable GPU use, no degradation after hours.
 *
 * Two mechanisms:
 *
 *  1. A p50/p95 frame-time window rather than an instant reading. A governor
 *     that reacts to single spikes oscillates, and an oscillating wallpaper is
 *     far more distracting than one running at 45 instead of 60.
 *
 *  2. Asymmetric step sizes and a dead band, so quality settles instead of
 *     hunting. It also never raises quality faster than it lowers it, because
 *     over-provisioning then under-provisioning is what causes pumping.
 */

import { approach } from './loop';
import type { QualityTune } from './scene';

export interface QualityLevel extends QualityTune {
  name: string;
  /** 0..1 master bloom strength. */
  bloom: number;
  /** 0..1 depth-of-field strength. */
  dof: number;
}

export const QUALITY: readonly QualityLevel[] = [
  { name: 'minimal', scale: 0.62, volumetricSteps: 8, bloom: 0.7, dof: 0 },
  { name: 'low', scale: 0.78, volumetricSteps: 14, bloom: 0.85, dof: 0.4 },
  { name: 'medium', scale: 0.9, volumetricSteps: 22, bloom: 1, dof: 0.8 },
  { name: 'high', scale: 1, volumetricSteps: 34, bloom: 1, dof: 1 },
];

export interface GovernorOptions {
  /** Never exceed this. Index into QUALITY. */
  maxLevel?: number;
  /** Never go below this. */
  minLevel?: number;
  /**
   * Where to start. Defaults to the cheapest level, and that default is the whole
   * point: a wallpaper that opens at its best settings renders its first frame
   * with every volumetric sample at full resolution, which on a modest GPU is
   * several seconds of a frozen tab before anything appears. Starting at the floor
   * and climbing costs a second of slightly softer shafts and removes the stall.
   */
  startLevel?: number;
  /** Frame time to stay under, ms. */
  targetMs?: number;
  /** Sample window. */
  windowMs?: number;
}

export class QualityGovernor {
  private samples: number[] = [];
  private lastSwitch = 0;
  private level: number;
  private readonly maxLevel: number;
  private readonly minLevel: number;
  private readonly targetMs: number;
  private readonly windowMs: number;

  /** Smoothed FPS for display. */
  fps = 60;

  constructor(options: GovernorOptions = {}) {
    this.maxLevel = Math.min(options.maxLevel ?? QUALITY.length - 1, QUALITY.length - 1);
    this.minLevel = Math.max(0, options.minLevel ?? 0);
    this.targetMs = options.targetMs ?? 1000 / 55;
    // Short enough that a fast machine reaches full quality in a few seconds
    // rather than a few tens of seconds. Nobody wants to wait a minute for the
    // picture to finish improving.
    this.windowMs = options.windowMs ?? 1600;
    this.level = Math.min(this.maxLevel, Math.max(this.minLevel, options.startLevel ?? 0));
  }

  get current(): QualityLevel {
    return QUALITY[this.level];
  }

  /** Locked level, e.g. when the user overrides. Pass null to resume adapting. */
  private locked: number | null = null;

  get lockedLevel(): number | null {
    return this.locked;
  }

  lock(level: number | null): void {
    this.locked = level === null ? null : Math.min(this.maxLevel, Math.max(this.minLevel, level));
    if (this.locked !== null) this.level = this.locked;
  }

  sample(dt: number, now: number): void {
    const ms = dt * 1000;
    this.samples.push(ms);
    if (this.samples.length > 240) this.samples.shift();

    if (this.samples.length < 30) return;

    // p50 ignores hitches; p95 is what the user actually perceives.
    const sorted = [...this.samples].sort((a, b) => a - b);
    const p50 = sorted[Math.floor(sorted.length * 0.5)];
    const p95 = sorted[Math.floor(sorted.length * 0.95)];
    this.fps = Math.round(1000 / Math.max(p50, 1e-3));

    if (this.locked !== null) return;

    const span = now - this.lastSwitch;
    // Shorten the window while there is headroom left to climb, so a fast machine
    // reaches full quality in a couple of seconds instead of six. The wait before
    // *dropping* stays long, because dropping too eagerly is the mistake that
    // actually shows up as pumping.
    const headroom = this.level / Math.max(1, this.maxLevel - this.minLevel);
    const settle = (this.windowMs * (1 - 0.5 * headroom)) / 1000;
    if (span < settle) return;

    // Dead band: only act on a clear miss, so quality does not hunt.
    if (p95 > this.targetMs * 1.18) {
      // Two steps at once when badly over budget, one when marginal.
      const overshoot = p95 / this.targetMs;
      const drop = overshoot > 1.7 ? 2 : 1;
      this.level = Math.max(this.minLevel, this.level - drop);
      this.samples.length = 0;
      this.lastSwitch = now;
    } else if (p95 < this.targetMs * 0.82 && this.level < this.maxLevel) {
      // Raising is deliberately slower than lowering: it is the more visible
      // direction, and it must not outrun the measurement.
      this.level = Math.min(this.maxLevel, this.level + 1);
      this.samples.length = 0;
      this.lastSwitch = now;
    }
  }

  /** Smoothed internal scale for continuous, non-stepping resolution changes. */
  smoothScale(dt: number): number {
    const target = this.current.scale;
    const eased = approach(this.smoothedScale ?? target, target, 0.6, dt);
    this.smoothedScale = eased;
    return eased;
  }

  private smoothedScale?: number;

  reset(): void {
    this.samples.length = 0;
    this.lastSwitch = 0;
    this.smoothedScale = undefined;
  }

  get stats(): { p50: number; p95: number; level: string } {
    const sorted = [...this.samples].sort((a, b) => a - b);
    return {
      p50: sorted[Math.floor(sorted.length * 0.5)] ?? 0,
      p95: sorted[Math.floor(sorted.length * 0.95)] ?? 0,
      level: this.current.name,
    };
  }
}