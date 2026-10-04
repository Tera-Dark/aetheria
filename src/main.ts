/**
 * Entry point.
 *
 * The launcher's only jobs: assemble the scene list, wire the fading HUD, and
 * fail loudly. Everything else lives in core.
 */

import { WallpaperApp } from './core/app';
import { WebGLUnavailableError } from './core/gl';
import { listScenes, registerScene } from './core/scene';
import type { SceneDefinition } from './core/scene';
import { moonlitValeDefinition } from './scenes/moonlit-vale';
import { abyssalBloomDefinition } from './scenes/abyssal-bloom';

/**
 * Failure panel.
 *
 * A minified stack trace is the worst possible thing to show someone whose browser
 * would not open a graphics context: it identifies neither the machine, nor the
 * reason, nor what to try next. When the cause is known it gets said in words, with
 * the driver's own message, and the trace is kept behind a disclosure for the bug
 * report.
 */
const fatal = (error: unknown): void => {
  document.body.dataset.fatal = 'true';
  document.body.dataset.loading = 'false';

  const panel = document.getElementById('fatalmsg');
  if (panel) {
    const gl = error instanceof WebGLUnavailableError ? error.detail : null;
    const title = document.getElementById('fataltitle');
    if (title) {
      title.textContent = gl ? '此浏览器未能提供 WebGL 2' : '无法启动';
    }

    if (gl) {
      const advice =
        '这与作品无关，是浏览器拒绝了图形上下文。最常见的三种原因：' +
        '<ol>' +
        '<li>浏览器设置里「使用硬件加速」被关闭（chrome://settings/system）</li>' +
        '<li>显卡驱动在浏览器的阻止名单里（chrome://gpu 会写明）</li>' +
        '<li>显卡驱动过旧，或独显处于被禁用 / 切换中的状态</li>' +
        '</ol>' +
        '本页只需要 WebGL 2，不下载任何素材，也不调用网络。';
      panel.innerHTML = [
        `<p class="fatal-reason">${gl.reason}</p>`,
        gl.statusMessage
          ? `<p class="fatal-driver">驱动返回：<code>${escapeHtml(gl.statusMessage)}</code></p>`
          : '',
        `<p class="fatal-advice">${advice}</p>`,
        `<details><summary>技术细节</summary><pre>${escapeHtml(
          `尝试的属性：${gl.tried}\n驱动消息：${gl.statusMessage || '(无)'}`,
        )}</pre></details>`,
      ].join('');
      console.error('[aetheria] WebGL unavailable', gl);
      return;
    }

    panel.textContent = error instanceof Error ? `${error.message}\n\n${error.stack ?? ''}` : String(error);
  }
  console.error(error);
};

const escapeHtml = (s: string): string =>
  s.replace(/[&<>"']/g, (c) =>
    c === '&' ? '&amp;' : c === '<' ? '&lt;' : c === '>' ? '&gt;' : c === '"' ? '&quot;' : '&#39;',
  );

/** `?capture=1` turns on the pixel readback the seam test needs. */
const CAPTURE = new URLSearchParams(location.search).has('capture');

/**
 * Wall-clock time from module execution to the loading overlay being dismissed.
 * Measured rather than estimated, because "it feels slow" is not a number anyone
 * can argue with or improve on.
 */
const BOOT_STARTED = performance.now();
let bootMs = 0;

/** Resolve after the browser has had a chance to paint. */
const nextFrame = (): Promise<void> =>
  new Promise((resolve) => requestAnimationFrame(() => resolve()));

/**
 * Advance the loading readout.
 *
 * The steps are the real ones, in the order they happen. A percentage that does
 * not correspond to work being done is worse than no percentage, because it
 * teaches the reader to distrust the one time the number stalls.
 */
const makeProgress = () => {
  const step = document.getElementById('bootstep');
  const fill = document.getElementById('bootfill');
  const set = (fraction: number, label: string): void => {
    if (fill) fill.style.width = `${Math.round(fraction * 100)}%`;
    if (step) step.textContent = label;
  };
  return {
    set,
    done: () => {
      document.body.dataset.loading = 'false';
    },
  };
};

const boot = async (): Promise<void> => {
  const progress = makeProgress();
  // Two frames, not one. The first rAF callback runs before the first paint, so
  // yielding only once still leaves the overlay unpainted when the blocking work
  // starts — which is the exact failure this is meant to prevent.
  progress.set(0.08, '正在启动引擎');
  await nextFrame();
  await nextFrame();

  const canvas = document.getElementById('stage');
  if (!(canvas instanceof HTMLCanvasElement)) throw new Error('#stage canvas missing');

  // §16: adding a scene is a one-line registration. The registry is the whole
  // extension point.
  registerScene(() => moonlitValeDefinition);
  registerScene(() => abyssalBloomDefinition);
  const scenes: SceneDefinition[] = listScenes();

  const rail = document.getElementById('rail');
  const title = document.getElementById('title');
  const note = document.getElementById('note');
  const fps = document.getElementById('fps');
  const quality = document.getElementById('quality');
  const period = document.getElementById('period');
  const loopFill = document.getElementById('loopfill');

  progress.set(0.2, '正在分配渲染管线');
  await nextFrame();

  const app = new WallpaperApp({
    canvas,
    scenes,
    idleSeconds: 2.4,
    transitionSeconds: 1.8,
    capture: CAPTURE,
  });

  const buttons = scenes.map((scene, i) => {
    const b = document.createElement('button');
    b.innerHTML = `<span class="dot"></span><span>${scene.title}</span>`;
    b.addEventListener('click', () => {
      app.select(i);
      wake();
    });
    rail?.appendChild(b);
    return b;
  });

  // Before the first frame, deliberately: the capture tool has to be able to see
  // the app *during* boot, because how long boot takes is now a measurement
  // rather than an impression.
  installCaptureHook(app, canvas);

  progress.set(0.32, '正在编译着色器');
  app.start();

  // The first frame compiles the first scene's shaders and draws it. This is the
  // long synchronous stall, and it is the reason the overlay exists.
  await app.ready;
  progress.set(0.6, '正在预热其余作品');

  // Compile the rest now, behind the overlay, rather than as a freeze in the
  // middle of a transition later.
  await app.prewarm((done, total) => {
    progress.set(0.6 + 0.35 * (done / total), `正在预热其余作品 ${done}/${total}`);
  });

  progress.set(1, '准备就绪');
  // One frame with the bar at 100% on screen, then the fade.
  await nextFrame();
  progress.done();
  document.getElementById('boot')?.setAttribute('hidden', '');
  bootMs = performance.now() - BOOT_STARTED;

  // The HUD reflects the *current* scene, so track which one is on screen rather
  // than assuming index 0.
  const indexOfScene = (name: string): number =>
    scenes.findIndex((s) => s.title === name);

  let shown = -1;
  const syncScene = (): void => {
    const s = app.stats;
    const index = indexOfScene(s.scene);
    if (title && index >= 0 && index !== shown) {
      shown = index;
      title.textContent = s.scene;
      if (note) note.textContent = scenes[shown]?.note ?? '';
      if (period) period.textContent = `${scenes[shown]?.periodSeconds ?? 0}s loop`;
      buttons.forEach((b, i) => b.setAttribute('aria-current', String(i === shown)));
    }
    if (loopFill) loopFill.style.width = `${(s.phase * 100).toFixed(2)}%`;
    // Readouts live in the same branch as everything else, so the HUD costs
    // exactly one style write per frame while the user is present and nothing at
    // all once it has faded.
    if (fps) fps.textContent = String(s.fps);
    if (quality) quality.textContent = s.locked ? `${s.quality} ·锁` : s.quality;
  };

  // One rAF for the HUD only, and it stops touching the DOM once idle so the
  // page is genuinely still. §14: a wallpaper should not busy-poll layout.
  let hudFrames = 0;
  const tickHud = (): void => {
    requestAnimationFrame(tickHud);
    const idle = document.body.dataset.idle === 'true';
    if (idle && hudFrames > 4) return;
    hudFrames = idle ? 0 : hudFrames + 1;
    syncScene();
  };
  requestAnimationFrame(tickHud);

  // Idle is driven by real input, not a timer that runs forever.
  let wakeTimer = window.setTimeout(() => {
    document.body.dataset.idle = 'true';
  }, 2600);

  const wake = (): void => {
    document.body.dataset.idle = 'false';
    hudFrames = 0;
    window.clearTimeout(wakeTimer);
    wakeTimer = window.setTimeout(() => {
      document.body.dataset.idle = 'true';
    }, 2600);
  };

  window.addEventListener('pointermove', wake, { passive: true });
  window.addEventListener('pointerdown', wake);
  window.addEventListener('keydown', wake);
  window.addEventListener('wheel', wake, { passive: true });

  document.querySelectorAll<HTMLElement>('.key').forEach((el) => {
    el.addEventListener('click', () => {
      if (el.dataset.act === 'next') app.next();
      else if (el.dataset.act === 'quality') app.cycleQuality();
      wake();
    });
  });

  // Clicking the artwork also advances — a wallpaper should not need the
  // keyboard to be usable.
  canvas.addEventListener('click', () => {
    app.next();
    wake();
  });
};
/**
 * The inspection surface `scripts/shoot.mjs` drives.
 *
 * §6 is the one requirement that cannot be checked by looking: "does frame 0
 * match frame N" needs both frames to be reachable deterministically, and a
 * running clock cannot be diffed against itself. So the page has to be able to
 * pin the phase, freeze the adaptive quality, turn off the film grain, and hand
 * back pixels. `snap`/`diff` do the last part in-page — a 2-megapixel PNG handed
 * back through the debugger as base64 would be both slow and enormous.
 */
const installCaptureHook = (app: WallpaperApp, canvas: HTMLCanvasElement): void => {
  let reference: ImageData | null = null;

  const sample = (): ImageData => {
    const off = document.createElement('canvas');
    off.width = canvas.width;
    off.height = canvas.height;
    const ctx = off.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('2d context unavailable');
    ctx.drawImage(canvas, 0, 0);
    return ctx.getImageData(0, 0, off.width, off.height);
  };

  (window as unknown as { __aetheria: unknown }).__aetheria = {
    setPhase: (p: number | null) => app.setPhase(p),
    setFade: (m: number | null) => app.setFade(m),
    stats: () => app.stats,
    select: (i: number) => app.select(i),
    lockQuality: (level: number | null) => app.lockQuality(level),
    grade: (patch: Record<string, number> | null) =>
      app.setGradeOverride(patch as never),
    scenes: () => app.sceneTitles,
    /**
     * Boot timing and state. `loading` is the honest answer to "is it stuck?" —
     * the alternative is the user deciding, from a black rectangle, whether to
     * close the tab.
     */
    boot: () => ({
      loading: document.body.dataset.loading === 'true',
      step: document.getElementById('bootstep')?.textContent ?? '',
      ms: Math.round(bootMs),
      frames: app.stats.frames,
    }),
    /** Store the current framebuffer as the reference for `diff`. */
    snap: () => {
      reference = sample();
      return { w: reference.width, h: reference.height };
    },
    /**
     * Compare the current framebuffer against the stored one.
     *
     * Reports the mean absolute difference in 8-bit units, plus the 99.5th
     * percentile so a handful of hot pixels cannot hide behind a small mean. A
     * correct loop lands well under 1.0; a visible seam lands in the tens.
     */
    diff: () => {
      if (!reference) throw new Error('snap() first');
      const now = sample();
      if (now.width !== reference.width || now.height !== reference.height) {
        throw new Error('size changed between snapshots');
      }
      const a = reference.data;
      const b = now.data;
      let sum = 0;
      let peak = 0;
      const hist = new Uint32Array(512);
      const n = a.length;
      for (let i = 0; i < n; i += 4) {
        const d =
          (Math.abs(a[i] - b[i]) +
            Math.abs(a[i + 1] - b[i + 1]) +
            Math.abs(a[i + 2] - b[i + 2])) /
          3;
        sum += d;
        if (d > peak) peak = d;
        hist[Math.min(511, Math.round(d))]++;
      }
      const pixels = n / 4;
      let acc = 0;
      let p995 = 0;
      for (let v = 0; v < hist.length; v++) {
        acc += hist[v];
        if (acc >= pixels * 0.995) {
          p995 = v;
          break;
        }
      }
      return {
        mean: Number((sum / pixels).toFixed(4)),
        p995,
        peak,
      };
    },
  };
};

void boot().catch(fatal);