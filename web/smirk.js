// onnxruntime-web wrapper around web/models/smirk.onnx.
//
// The ONNX file outputs (vertices [1,5023,3], cam [1,3]) for a [1,3,224,224]
// RGB image in [0,1]. The web renderer applies the orthographic projection in JS.

import * as ort from "onnxruntime-web";

// Using the `bundle` build of onnxruntime-web, which inlines the wasm assets —
// no wasmPaths needed.
// SharedArrayBuffer requires the page to be cross-origin isolated (COOP/COEP
// headers from serve.py). Without it, multi-threaded WASM silently falls back
// to single-threaded.
const _crossOriginIsolated = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;
const _hwThreads = (typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 4;
ort.env.wasm.numThreads = _crossOriginIsolated ? Math.min(_hwThreads, 8) : 1;
console.log(
  `[smirk] crossOriginIsolated=${_crossOriginIsolated} hwThreads=${_hwThreads} wasm.numThreads=${ort.env.wasm.numThreads}`,
);

let session = null;
let backend = null;

export async function init() {
  if (session) return session;
  const url = "./models/smirk.onnx";
  // Log which GPU adapter onnxruntime-web will actually use, so we can rule out
  // the wrong adapter (e.g. integrated vs discrete) being picked.
  try {
    const adapter = await navigator.gpu?.requestAdapter?.();
    if (adapter) {
      const info = adapter.info ?? {};
      console.log(
        `[smirk] WebGPU adapter: vendor=${info.vendor || "?"} arch=${info.architecture || "?"} device=${info.device || "?"} desc=${info.description || "?"}`,
      );
    } else {
      console.warn("[smirk] navigator.gpu.requestAdapter() returned null");
    }
  } catch (e) {
    console.warn("[smirk] adapter probe failed:", e);
  }
  // Default to WASM: empirically ~20x faster than WebGPU for this model on
  // Apple Silicon (WebGPU EP overhead dominates SMIRK's small-op graph).
  // Override via URL hash, e.g. http://localhost:8080/#ep=webgpu
  const hashParams = new URLSearchParams(location.hash.slice(1));
  const wanted = (hashParams.get("ep") || "wasm").toLowerCase();
  const tryOrder = wanted === "webgpu" ? ["webgpu", "wasm"] : ["wasm"];
  let lastErr = null;
  for (const ep of tryOrder) {
    try {
      const t0 = performance.now();
      session = await ort.InferenceSession.create(url, {
        executionProviders: [ep],
        graphOptimizationLevel: "all",
      });
      const dt = performance.now() - t0;
      backend = ep;
      console.log(`[smirk] session created on ${ep} in ${dt.toFixed(0)}ms`);
      break;
    } catch (err) {
      console.warn(`[smirk] EP ${ep} failed:`, err);
      lastErr = err;
    }
  }
  if (!session) throw lastErr ?? new Error("no EP available");
  console.log(`[smirk] inference backend: ${backend}`);
  return session;
}

export function getBackend() {
  return backend;
}

/**
 * @param {Float32Array} input - length 1*3*224*224, RGB planar in [0,1]
 * @returns {Promise<{vertices: Float32Array, cam: Float32Array}>}
 */
export async function run(input) {
  if (!session) throw new Error("smirk.js: call init() first");
  const tensor = new ort.Tensor("float32", input, [1, 3, 224, 224]);
  const tRunStart = performance.now();
  const out = await session.run({ image: tensor });
  const tRunEnd = performance.now();
  // Force the readback synchronously here so we can attribute its cost
  // separately from session.run() itself.
  const tReadStart = performance.now();
  const vertices = out.vertices.data;
  const cam = out.cam.data;
  const tReadEnd = performance.now();
  _record(tRunEnd - tRunStart, tReadEnd - tReadStart);
  return { vertices, cam };
}

// Rolling timing stats. Logged once per second.
const _timing = { runs: 0, runMs: 0, readMs: 0, lastLog: 0 };
function _record(runMs, readMs) {
  _timing.runs += 1;
  _timing.runMs += runMs;
  _timing.readMs += readMs;
  const now = performance.now();
  if (now - _timing.lastLog > 1000) {
    const elapsed = (now - _timing.lastLog) / 1000;
    const ips = _timing.runs / elapsed;
    const avgRun = _timing.runMs / _timing.runs;
    const avgRead = _timing.readMs / _timing.runs;
    console.log(
      `[smirk] ${ips.toFixed(1)} inferences/s · run=${avgRun.toFixed(1)}ms · readback=${avgRead.toFixed(2)}ms (${backend})`,
    );
    _timing.runs = 0;
    _timing.runMs = 0;
    _timing.readMs = 0;
    _timing.lastLog = now;
  }
}
