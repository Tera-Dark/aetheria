/**
 * WebGL context acquisition.
 *
 * This exists because "the page is blank" and "your browser would not give us a
 * WebGL context" are the same symptom from the outside, and only one of them is
 * the developer's fault. A failed `new WebGLRenderer()` produces a stack trace
 * pointing into a minified bundle, which tells the reader nothing and asks them to
 * do nothing useful with it.
 *
 * So: create the context ourselves, before three sees it, with a ladder of
 * progressively plainer attribute sets; capture the driver's own explanation when
 * every rung fails; and hand three a context we already know works.
 *
 * The rungs, in order:
 *
 *  1. The attribute set the post chain actually needs.
 *  2. The same, with antialias and stencil dropped.
 *
 * Note that once a context exists, `getContext` returns it regardless of the
 * attributes passed, so the ladder has to be walked by asking for contexts, not by
 * re-calling the renderer.
 */

export type GLAttributes = WebGLContextAttributes;

export interface GLCreated {
  ok: true;
  context: WebGL2RenderingContext;
  /** Human-readable note when a rung below the first had to be used. */
  relaxed: string | null;
}

export interface GLFailed {
  ok: false;
  /** What we concluded, in plain language. */
  reason: string;
  /** Whatever the driver said, verbatim. Often the single most useful line. */
  statusMessage: string;
  /** The attribute sets that were tried, for the bug report. */
  tried: string;
}

export type GLResult = GLCreated | GLFailed;

/**
 * The attribute sets that are tried, in order.
 *
 * There is deliberately no `powerPreference: 'high-performance'` rung. That hint
 * asks the browser for the discrete adapter, and on a dual-GPU laptop it is a
 * known cause of outright context-creation failure rather than a silent fallback —
 * when the discrete GPU is blocklisted, in a low-power state, or mid-transition,
 * the request fails where the default would have succeeded. There is nothing to
 * gain from asking: the quality governor decides how hard to push, so a "faster"
 * adapter buys nothing that a smaller frame time would not.
 *
 * The two rungs that remain differ in how much the driver is asked for, so the
 * second one still catches the case where a depth or antialias combination is
 * refused.
 *
 * A rung that fails does not poison the canvas for the next one: `getContext`
 * returns null and the canvas can be asked again for the same context type.
 */
const LADDER: { label: string; attributes: GLAttributes }[] = [
  {
    label: 'default',
    attributes: {
      alpha: false,
      depth: true,
      // The post chain resolves its own edges from the HDR buffer; MSAA on a
      // half-float target is not something drivers honour anyway.
      antialias: false,
      stencil: false,
      preserveDrawingBuffer: false,
    },
  },
  {
    label: 'minimal',
    attributes: { alpha: false, depth: true, antialias: false },
  },
];

/**
 * Walk the ladder and return the first context that materialises.
 *
 * `capture` is threaded through rather than read from the URL here: the caller
 * knows, because `preserveDrawingBuffer` costs a copy per frame and must not be on
 * in normal use.
 */
export const acquireGL = (
  canvas: HTMLCanvasElement,
  capture: boolean,
): GLResult => {
  // The driver reports *why* through this event, and only through it. A bare null
  // from getContext carries no explanation at all.
  let statusMessage = '';
  canvas.addEventListener('webglcontextcreationerror', (event) => {
    statusMessage = (event as WebGLContextEvent).statusMessage || statusMessage;
  });

  for (const rung of LADDER) {
    const attributes = { ...rung.attributes, preserveDrawingBuffer: capture };
    let context: WebGL2RenderingContext | null = null;
    try {
      context = canvas.getContext('webgl2', attributes);
    } catch (error) {
      statusMessage ||= error instanceof Error ? error.message : String(error);
    }
    if (context) {
      return {
        ok: true,
        context,
        relaxed: rung.label === 'default' ? null : rung.label,
      };
    }
  }

  return {
    ok: false,
    reason: classify(statusMessage),
    statusMessage,
    tried: LADDER.map((r) => r.label).join(' → '),
  };
};

/**
 * Turn the driver's complaint into something a reader can act on.
 *
 * The mapping is deliberately narrow. Guessing broadly — "maybe try Chrome?" —
 * is worse than useless, because it sends people down a browser-support rabbit
 * hole when the actual answer is a hardware acceleration switch.
 */
const classify = (statusMessage: string): string => {
  const s = statusMessage.toLowerCase();
  if (!s) {
    return '浏览器拒绝创建 WebGL 上下文，且没有给出原因。';
  }
  if (s.includes('software') || s.includes('swiftshader') || s.includes('fallback')) {
    return (
      '浏览器拒绝创建 WebGL 上下文，因为没有可用的硬件加速。' +
      '这通常意味着显卡驱动被列入了阻止名单，或浏览器设置里关闭了硬件加速。'
    );
  }
  if (s.includes('gpu') || s.includes('adapter') || s.includes('display')) {
    return (
      '浏览器在创建 GPU 上下文时失败。这通常是显卡驱动的问题，' +
      '或者独立显卡正处于被禁用 / 切换中的状态。'
    );
  }
  return '浏览器拒绝创建 WebGL 上下文。';
};

/** The GPU string, for the bug report. Only available once a context exists. */
export const describeGPU = (gl: WebGL2RenderingContext): string => {
  const ext = gl.getExtension('WEBGL_debug_renderer_info');
  if (ext) {
    const vendor = gl.getParameter(ext.UNMASKED_VENDOR_WEBGL);
    const renderer = gl.getParameter(ext.UNMASKED_RENDERER_WEBGL);
    if (renderer) return `${vendor} · ${renderer}`;
  }
  const renderer = gl.getParameter(gl.RENDERER);
  return typeof renderer === 'string' ? renderer : 'unknown';
};

/**
 * Thrown instead of three's `Error creating WebGL context.`
 *
 * The point is that this carries a diagnosis. Three's own error reaches the reader
 * as a stack trace through a minified bundle, which identifies neither the machine
 * nor the reason nor the next step.
 */
export class WebGLUnavailableError extends Error {
  constructor(readonly detail: GLFailed) {
    super(detail.reason);
    this.name = 'WebGLUnavailableError';
  }
}
