/**
 * Capture frames from the running wallpaper with headless Chrome.
 *
 *   node scripts/shoot.mjs                        # every scene at phase 0
 *   node scripts/shoot.mjs --scene 1 --phases 0,0.5
 *   node scripts/shoot.mjs --seam                 # phase 0 vs 0.999, numeric
 *
 * `--seam` is the important one. §6 says the loop must close seamlessly, and
 * that is the one requirement you cannot check by eye — a wrapped noise field
 * looks fine in isolation and pops once per cycle. So the page is pinned to a
 * phase, the quality governor is frozen, the grain is switched off, and the two
 * frames are differenced numerically. A correct loop lands well under 1.0/255.
 *
 * Requires the dev server (`npm run dev`) to be running.
 */

import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { setTimeout as sleep } from 'node:timers/promises';

const argv = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = argv.indexOf(`--${name}`);
  return i === -1 ? fallback : argv[i + 1];
};
const has = (name) => argv.includes(`--${name}`);

const URL_BASE = flag('url', 'http://localhost:5180/?capture=1');
const OUT = path.resolve(flag('out', has('seam') ? 'out/seam' : 'out/shots'));
const WIDTH = Number(flag('width', '1280'));
const HEIGHT = Number(flag('height', '720'));
const CHROME = flag(
  'chrome',
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
);
/** Freeze the quality governor so two runs are comparable. */
const QUALITY = Number(flag('quality', '3'));
const SCENE = flag('scene', null);
const PHASES = (flag('phases', '0,0.5') ?? '')
  .split(',')
  .map(Number)
  .filter((n) => !Number.isNaN(n));

/** Fail the run if the seam is worse than this, in 8-bit units. */
const SEAM_TOLERANCE = Number(flag('tolerance', '1.0'));
/**
 * Every CDP command gets a deadline. Software rendering of a volumetric shader
 * can take seconds per frame, and `Runtime.evaluate` has to wait for the page's
 * main thread — so without this a slow machine hangs the script silently instead
 * of saying which step gave up.
 */
const COMMAND_TIMEOUT_MS = Number(flag('timeout', '60000'));
/** Seconds into a scene change to grab the frame. The fade is 1.8s. */
const TRANSITION_AT = Number(flag('at', '0.5'));

await mkdir(OUT, { recursive: true });

try {
  const probe = await fetch(URL_BASE, { method: 'HEAD' });
  if (!probe.ok && probe.status !== 404) throw new Error(String(probe.status));
} catch (error) {
  console.error(`Cannot reach ${URL_BASE} — is \`npm run dev\` running? (${error})`);
  process.exit(2);
}

const PORT = 9222 + (process.pid % 400);
const chrome = spawn(
  CHROME,
  [
    '--headless=new',
    `--remote-debugging-port=${PORT}`,
    `--window-size=${WIDTH},${HEIGHT}`,
    '--hide-scrollbars',
    '--use-angle=default',
    '--enable-unsafe-swiftshader',
    '--disable-gpu-sandbox',
    '--no-first-run',
    `--user-data-dir=${path.join(process.env.TEMP ?? '.', `aetheria-shoot-${PORT}`)}`,
    'about:blank',
  ],
  { stdio: 'ignore' },
);

const cleanup = () => {
  if (chrome.exitCode !== null) return;
  // `kill()` only reaps the browser process; its renderer and GPU children keep
  // running and hold the profile lock, so the next run cannot start. Kill the
  // tree.
  if (process.platform === 'win32' && chrome.pid) {
    spawn('taskkill', ['/pid', String(chrome.pid), '/T', '/F'], { stdio: 'ignore' });
  } else {
    try {
      chrome.kill('SIGKILL');
    } catch {
      /* already gone */
    }
  }
};
process.on('exit', cleanup);
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    cleanup();
    process.exit(130);
  });
}

/** Wait for the DevTools endpoint rather than sleeping a fixed amount. */
const waitForDevtools = async () => {
  for (let i = 0; i < 80; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${PORT}/json/version`);
      if (r.ok) return (await r.json()).webSocketDebuggerUrl;
    } catch {
      /* not up yet */
    }
    await sleep(250);
  }
  throw new Error('Chrome DevTools endpoint never came up');
};

const ws = new WebSocket(await waitForDevtools());
await new Promise((resolve, reject) => {
  ws.onopen = resolve;
  ws.onerror = reject;
});

let nextId = 1;
const pending = new Map();
const logs = [];

ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) {
    pending.get(msg.id)(msg);
    pending.delete(msg.id);
    return;
  }
  // Surface everything the page complains about. A GLSL compile failure is silent
  // in Three.js apart from a console.error, and a scene that draws nothing still
  // reports a healthy frame rate — so this is not optional.
  if (msg.method === 'Runtime.consoleAPICalled') {
    const text = (msg.params.args ?? [])
      .map((a) => a.value ?? a.description ?? a.unserializableValue ?? '')
      .join(' ');
    logs.push(`[${msg.params.type}] ${text}`);
  } else if (msg.method === 'Runtime.exceptionThrown') {
    const d = msg.params.exceptionDetails;
    logs.push(`[exception] ${d.exception?.description ?? d.text}`);
  } else if (msg.method === 'Log.entryAdded') {
    logs.push(`[${msg.params.entry.level}] ${msg.params.entry.text}`);
  }
};

const withTimeout = (promise, label) =>
  Promise.race([
    promise,
    new Promise((_, reject) =>
      setTimeout(
        () => reject(new Error(`${label} timed out after ${COMMAND_TIMEOUT_MS}ms`)),
        COMMAND_TIMEOUT_MS,
      ).unref?.(),
    ),
  ]);

const send = (method, params = {}, sessionId) =>
  withTimeout(
    new Promise((resolve, reject) => {
      const id = nextId++;
      pending.set(
        id,
        (msg) =>
          msg.error
            ? reject(new Error(JSON.stringify(msg.error)))
            : resolve(msg.result),
      );
      ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
    }),
    method,
  );

/** Log a step and how long it took, so a slow machine is legible. */
const started = Date.now();
const step = async (label, fn) => {
  const t0 = Date.now();
  const value = await fn();
  const ms = Date.now() - t0;
  console.log(`  ${label} ${ms >= 400 ? `(${ms}ms)` : ''}`);
  return value;
};

const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });

await send('Page.enable', {}, sessionId);
await send('Runtime.enable', {}, sessionId);
await send('Log.enable', {}, sessionId);

// WebGL in headless Chrome needs a real backend; SwiftShader is the safe one.
await send(
  'Emulation.setDeviceMetricsOverride',
  { width: WIDTH, height: HEIGHT, deviceScaleFactor: 1, mobile: false },
  sessionId,
);

const evaluate = async (expression) => {
  const { result, exceptionDetails } = await send(
    'Runtime.evaluate',
    { expression, returnByValue: true, awaitPromise: false },
    sessionId,
  );
  if (exceptionDetails) {
    throw new Error(exceptionDetails.exception?.description ?? exceptionDetails.text);
  }
  return result.value;
};

console.log(`Loading ${URL_BASE} …`);
await send('Page.navigate', { url: URL_BASE }, sessionId);

// Give the scene time to compile shaders and settle into the loop. SwiftShader
// compiles and runs a volumetric shader slowly, so this is not generous enough
// to be called a guess — it is just what it takes.
await sleep(9000);

const fatal = await evaluate(
  "document.body.dataset.fatal === 'true' ? document.getElementById('fatalmsg').textContent : ''",
);
if (fatal) {
  console.error('The app failed to start:\n' + fatal);
  dumpLogs();
  cleanup();
  process.exit(3);
}

const hookReady = await evaluate('typeof window.__aetheria === "object"');
if (!hookReady) {
  console.error('window.__aetheria missing — load the page with ?capture=1');
  cleanup();
  process.exit(4);
}

// Freeze everything that would otherwise make two runs incomparable.
await step(`lock quality ${QUALITY}`, () =>
  evaluate(`window.__aetheria.lockQuality(${QUALITY})`),
);
if (has('seam')) await evaluate('window.__aetheria.grade({ grain: 0 })');
await sleep(2500);

const titles = await evaluate('window.__aetheria.scenes()');
const targets = SCENE === null ? titles.map((_, i) => i) : [Number(SCENE)];

const screenshot = async (file) => {
  const { data } = await send(
    'Page.captureScreenshot',
    { format: 'png', captureBeyondViewport: false },
    sessionId,
  );
  await writeFile(path.join(OUT, file), Buffer.from(data, 'base64'));
  console.log(`    ${path.relative(process.cwd(), path.join(OUT, file))}`);
};

/**
 * Mid-cross-fade. The shell's fade is a real two-source composite now, and this
 * is the only way to see it — a frame captured either side of the change looks
 * identical to a working one, and the mix is pinned rather than timed because a
 * software-rendered frame can outlast the whole fade.
 */
if (has('transition')) {
  if (titles.length < 2) throw new Error('need at least two scenes');
  for (const from of [0, 1]) {
    await evaluate(`window.__aetheria.select(${from})`);
    await sleep(3400);
    console.log(`${titles[from]} → ${titles[1 - from]} @ mix ${TRANSITION_AT}`);
    await evaluate(`window.__aetheria.setFade(${TRANSITION_AT})`);
    await sleep(1600);
    await screenshot(`x${from}-to-${1 - from}.png`);
    await evaluate('window.__aetheria.setFade(null)');
    await sleep(2600);
  }
  ws.close();
  cleanup();
  process.exit(0);
}

let failed = false;

for (const index of targets) {
  console.log(`${titles[index] ?? `#${index}`}`);
  // A scene switch cross-fades over 1.8s; wait it out or the shot catches two
  // scenes at once.
  await step('select', () => evaluate(`window.__aetheria.select(${index})`));
  await sleep(3200);

  for (const phase of PHASES) {
    await step(`φ=${phase}`, async () => {
      await evaluate(`window.__aetheria.setPhase(${phase})`);
      await sleep(1600);
    });
    await screenshot(`s${index}-${String(phase).replace('.', '_')}.png`);
  }

  if (has('seam')) {
    // 0.999 rather than 1.0: 1.0 wraps to 0 through the same `((p % 1) + 1) % 1`
    // the runtime uses, which would compare a frame against itself.
    await evaluate('window.__aetheria.setPhase(0)');
    await sleep(1600);
    await evaluate('window.__aetheria.snap()');
    await evaluate('window.__aetheria.setPhase(0.999)');
    await sleep(1600);
    const { mean, p995, peak } = JSON.parse(
      await evaluate('JSON.stringify(window.__aetheria.diff())'),
    );
    const verdict = mean <= SEAM_TOLERANCE ? 'OK' : 'SEAM';
    if (mean > SEAM_TOLERANCE) failed = true;
    console.log(
      `  seam φ=0 vs φ=0.999 → mean ${mean} · p99.5 ${p995} · peak ${peak}  [${verdict}]`,
    );
    await screenshot(`seam-s${index}.png`);
  }

  await evaluate('window.__aetheria.setPhase(null)');
}

dumpLogs();
console.log('stats:', JSON.stringify(await evaluate('window.__aetheria.stats()')));
console.log(`total ${((Date.now() - started) / 1000).toFixed(1)}s`);

ws.close();
cleanup();
process.exit(failed ? 1 : 0);

function dumpLogs() {
  if (!logs.length) {
    console.log('--- page log: clean ---');
    return;
  }
  console.log('--- page log ---');
  for (const line of logs.slice(0, 40)) console.log('  ' + line);
}