/**
 * The camera rig.
 *
 * §8 asks for cinematic language with a different purpose: not storytelling,
 * but choosing where the viewer stands. And it is explicit that the camera
 * must never become the subject — the user should feel the *world* moving.
 *
 * So the rig has exactly three degrees of freedom, all tiny, all periodic in
 * the loop phase, plus a pointer-driven parallax offset that decays to zero:
 *
 *   • lateral drift   — a slow figure, a couple of low harmonics
 *   • breathing dolly — sub-percent FOV change, the "the air moved" cue
 *   • focal breathing — focus distance eased by the same signal as the dolly
 *
 * There is deliberately no roll, no orbit, no handheld shake, and no cut.
 */

import { PerspectiveCamera, Vector3 } from 'three';
import { approach, harmonic, harmonic2, lerp } from './loop';

export interface CameraSpec {
  /** Where the viewer stands. */
  position: Vector3;
  /** What the viewer looks at. */
  target: Vector3;
  /** Vertical FOV in degrees. */
  fov: number;
  near: number;
  far: number;
  /** Metres of lateral travel across the whole loop. */
  driftMetres: number;
  /** Fraction of fov the breathing may span. 0.03 == ±3%. */
  fovBreath: number;
  /** Peak parallax offset from the pointer, in metres. */
  parallaxMetres: number;
  /** Metres/second for the parallax to halve. Small == heavy, damped. */
  parallaxHalfLife: number;
}

export const makeCamera = (spec: CameraSpec): PerspectiveCamera => {
  const cam = new PerspectiveCamera(spec.fov, 1, spec.near, spec.far);
  cam.position.copy(spec.position);
  cam.lookAt(spec.target);
  return cam;
};

export class CameraRig {
  private readonly spec: CameraSpec;
  readonly camera: PerspectiveCamera;

  /** Smoothed pointer offset in metres. The only persistent state in the rig. */
  private offset = new Vector3();
  private targetOffset = new Vector3();

  constructor(spec: CameraSpec) {
    this.spec = spec;
    this.camera = makeCamera(spec);
  }

  setPointer(ndcX: number, ndcY: number, idleness: number): void {
    const gain = (1 - idleness) * this.spec.parallaxMetres;
    this.targetOffset.set(-ndcX * gain, ndcY * gain * 0.55, 0);
  }

  /**
   * @param phase loop phase in [0,1)
   * @param dt    seconds, for the parallax easing only
   */
  update(phase: number, dt: number): void {
    const s = this.spec;

    // Parallax: exponential approach, never accumulated. When the pointer
    // leaves, idleness rises to 1 and the target collapses to the origin, so
    // the world settles back to exactly its authored state.
    this.offset.set(
      approach(this.offset.x, this.targetOffset.x, s.parallaxHalfLife, dt),
      approach(this.offset.y, this.targetOffset.y, s.parallaxHalfLife, dt),
      approach(this.offset.z, this.targetOffset.z, s.parallaxHalfLife, dt),
    );

    // Lateral drift. Two incommensurate-ish harmonics make the path read as
    // unhurried rather than as a single sine the eye can lock onto.
    const drift =
      harmonic(phase, 1) * 0.68 + harmonic2(phase, 2, 1) * 0.32;

    // Breathing dolly: push in and release, never reversing visibly.
    const breath = harmonic(phase, 1, Math.PI * 0.5) * 0.5 + harmonic(phase, 2, -Math.PI * 0.25) * 0.5;

    this.camera.position.set(
      s.position.x + drift * s.driftMetres + this.offset.x,
      s.position.y + this.offset.y,
      s.position.z,
    );

    this.camera.fov = s.fov * (1 + breath * s.fovBreath);
    this.camera.updateProjectionMatrix();

    // Look at a target that drifts a fraction as far, which is what sells the
    // parallax between near and far geometry.
    this.camera.lookAt(
      s.target.x + drift * s.driftMetres * 0.12,
      s.target.y + breath * s.driftMetres * 0.05,
      s.target.z,
    );
  }

  /** Focus distance in metres for the DOF pass. Eased by the breath signal. */
  focusDistance(phase: number): number {
    return lerp(
      this.spec.position.distanceTo(this.spec.target),
      this.spec.position.distanceTo(this.spec.target) * 0.94,
      harmonic(phase, 1, Math.PI * 0.5) * 0.5 + 0.5,
    );
  }

  get target(): Vector3 {
    return this.spec.target;
  }

  dispose(): void {
    this.offset.set(0, 0, 0);
    this.targetOffset.set(0, 0, 0);
  }
}