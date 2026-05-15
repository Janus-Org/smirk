// onnxruntime-web wrapper around web/models/smirk.onnx.
//
// The ONNX file outputs (vertices [1,5023,3], cam [1,3]) for a [1,3,224,224]
// RGB image in [0,1]. The web renderer applies the orthographic projection in JS.

import * as ort from "onnxruntime-web";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
ort.env.wasm.numThreads = 1;

let session = null;
let backend = null;

export async function init() {
  if (session) return session;
  const url = "./models/smirk.onnx";
  try {
    session = await ort.InferenceSession.create(url, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "all",
    });
    backend = "webgpu";
  } catch (err) {
    console.warn("[smirk] WebGPU EP unavailable, falling back to WASM:", err);
    session = await ort.InferenceSession.create(url, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    backend = "wasm";
  }
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
  const out = await session.run({ image: tensor });
  return {
    vertices: out.vertices.data,
    cam: out.cam.data,
  };
}
