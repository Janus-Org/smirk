// Orchestrator: webcam -> MediaPipe FaceLandmarker -> crop -> ONNX (SMIRK) -> three.js mesh.

import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { computeBbox, cropAt, getCropCanvas } from "./crop.js";
import * as smirk from "./smirk.js";
import * as renderer3d from "./renderer3d.js";
import { OneEuroFilter, VectorOneEuroFilter } from "./oneEuroFilter.js";

const statusEl = document.getElementById("status");
const videoEl = document.getElementById("cam");
const canvasEl = document.getElementById("three");
const debugCanvas = document.getElementById("debug-crop");
const debugCtx = debugCanvas.getContext("2d");
const overlayEl = document.getElementById("video-overlay");
const overlayCtx = overlayEl.getContext("2d");

// Decouple from the <video> element: every frame we copy the current video pixels
// onto this canvas at the stream's native size. Both mediapipe and the SMIRK crop
// then operate against an unambiguous source whose width/height match the
// landmark coordinate space. Avoids browser/CSS-related coordinate mismatches.
const frameCanvas = document.createElement("canvas");
const frameCtx = frameCanvas.getContext("2d");

// One Euro filters. Two layers:
//   (1) bbox center/size — stabilizes the *input* crop against mediapipe jitter.
//   (2) cam + vertices  — stabilizes the *output* against per-frame encoder noise
//                          (SMIRK is a static-image model with no temporal prior).
const filterParams = { mincutoff: 1.0, beta: 0.05 };
const cxFilter = new OneEuroFilter(filterParams);
const cyFilter = new OneEuroFilter(filterParams);
const sizeFilter = new OneEuroFilter(filterParams);
const camFilter = new VectorOneEuroFilter(3, filterParams);
// 5023 FLAME verts × 3 components.
const vertFilter = new VectorOneEuroFilter(5023 * 3, filterParams);
const camOut = new Float32Array(3);
const vertOut = new Float32Array(5023 * 3);

function applyFilterParams() {
  cxFilter.setParams(filterParams);
  cyFilter.setParams(filterParams);
  sizeFilter.setParams(filterParams);
  camFilter.setParams(filterParams);
  vertFilter.setParams(filterParams);
}

function setupFilterUI() {
  const mc = document.getElementById("filter-mincutoff");
  const mcVal = document.getElementById("filter-mincutoff-value");
  const beta = document.getElementById("filter-beta");
  const betaVal = document.getElementById("filter-beta-value");
  mc.addEventListener("input", () => {
    filterParams.mincutoff = parseFloat(mc.value);
    mcVal.textContent = filterParams.mincutoff.toFixed(1);
    applyFilterParams();
  });
  beta.addEventListener("input", () => {
    filterParams.beta = parseFloat(beta.value);
    betaVal.textContent = filterParams.beta.toFixed(3);
    applyFilterParams();
  });
  // Sync initial UI display to the JS defaults.
  mc.value = filterParams.mincutoff;
  mcVal.textContent = filterParams.mincutoff.toFixed(1);
  beta.value = filterParams.beta;
  betaVal.textContent = filterParams.beta.toFixed(3);
}

function drawLandmarkOverlay(landmarks) {
  const vw = frameCanvas.width;
  const vh = frameCanvas.height;
  if (overlayEl.width !== vw || overlayEl.height !== vh) {
    overlayEl.width = vw;
    overlayEl.height = vh;
  }
  overlayCtx.clearRect(0, 0, vw, vh);
  // Draw bbox in native video pixel coords.
  let left = Infinity, right = -Infinity, top = Infinity, bottom = -Infinity;
  for (const p of landmarks) {
    const x = p.x * vw;
    const y = p.y * vh;
    if (x < left) left = x;
    if (x > right) right = x;
    if (y < top) top = y;
    if (y > bottom) bottom = y;
  }
  overlayCtx.strokeStyle = "lime";
  overlayCtx.lineWidth = 2;
  overlayCtx.strokeRect(left, top, right - left, bottom - top);
  // Outline image rect for reference.
  overlayCtx.strokeStyle = "red";
  overlayCtx.lineWidth = 1;
  overlayCtx.strokeRect(0, 0, vw, vh);
  // Plot landmarks.
  overlayCtx.fillStyle = "rgba(0,255,0,0.7)";
  for (const p of landmarks) {
    overlayCtx.fillRect(p.x * vw - 1, p.y * vh - 1, 2, 2);
  }
}

function setStatus(msg) {
  statusEl.textContent = msg;
  console.log("[smirk]", msg);
}

async function startCamera() {
  setStatus("requesting webcam…");
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error(
      `webcam API unavailable — open this page from http://localhost:8080 or http://127.0.0.1:8080 ` +
      `(secure-context-only API; LAN IPs and file:// won't work without HTTPS). ` +
      `Current origin: ${location.origin}`,
    );
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  videoEl.srcObject = stream;
  await new Promise((resolve) => {
    if (videoEl.readyState >= 2) return resolve();
    videoEl.addEventListener("loadeddata", () => resolve(), { once: true });
  });
  await videoEl.play();
}

async function loadLandmarker() {
  setStatus("loading mediapipe…");
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.10/wasm",
  );
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: "./assets/face_landmarker.task" },
    runningMode: "VIDEO",
    numFaces: 1,
    minFaceDetectionConfidence: 0.1,
    minFacePresenceConfidence: 0.1,
    minTrackingConfidence: 0.1,
  });
}

async function main() {
  try {
    setupFilterUI();
    await startCamera();
    const [landmarker] = await Promise.all([
      loadLandmarker(),
      renderer3d.init(canvasEl),
      smirk.init(),
    ]);
    setStatus("running");

    let pending = false;
    let lastTs = 0;
    let frameCount = 0;
    let lastFpsTs = performance.now();
    let lastCamLogTs = 0;

    const loop = async () => {
      if (videoEl.videoWidth > 0) {
        const ts = performance.now();
        // Mediapipe requires strictly increasing timestamps in VIDEO mode.
        const t = ts > lastTs ? ts : lastTs + 1;
        lastTs = t;
        // Snapshot the current frame at native pixel dimensions.
        if (frameCanvas.width !== videoEl.videoWidth || frameCanvas.height !== videoEl.videoHeight) {
          frameCanvas.width = videoEl.videoWidth;
          frameCanvas.height = videoEl.videoHeight;
        }
        frameCtx.drawImage(videoEl, 0, 0);
        const result = landmarker.detectForVideo(frameCanvas, t);
        const landmarks = result.faceLandmarks && result.faceLandmarks[0];
        if (landmarks && landmarks.length) {
          drawLandmarkOverlay(landmarks);
        }
        if (landmarks && landmarks.length && !pending) {
          const bbox = computeBbox(landmarks, frameCanvas.width, frameCanvas.height);
          let input = null;
          if (bbox) {
            const cx = cxFilter.filter(bbox.cx, ts);
            const cy = cyFilter.filter(bbox.cy, ts);
            const size = sizeFilter.filter(bbox.size, ts);
            input = cropAt(frameCanvas, cx, cy, size);
          }
          if (input) {
            debugCtx.drawImage(getCropCanvas(), 0, 0);
            pending = true;
            smirk
              .run(input)
              .then(({ vertices, cam }) => {
                const tNow = performance.now();
                const camF = camFilter.filter(cam, tNow, camOut);
                const vertF = vertFilter.filter(vertices, tNow, vertOut);
                renderer3d.update(vertF, camF);
                if (tNow - lastCamLogTs > 1000) {
                  console.log(
                    `[smirk] cam s=${camF[0].toFixed(3)} tx=${camF[1].toFixed(3)} ty=${camF[2].toFixed(3)} ` +
                    `vert[0]=(${vertF[0].toFixed(3)},${vertF[1].toFixed(3)},${vertF[2].toFixed(3)})`,
                  );
                  lastCamLogTs = tNow;
                }
              })
              .catch((err) => {
                console.error("smirk.run failed", err);
              })
              .finally(() => {
                pending = false;
              });
          }
        }
        frameCount++;
        if (ts - lastFpsTs > 1000) {
          const fps = (frameCount * 1000) / (ts - lastFpsTs);
          setStatus(`running · ${fps.toFixed(1)} fps · ${smirk.getBackend()}`);
          frameCount = 0;
          lastFpsTs = ts;
        }
      }
      renderer3d.render();
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    setStatus(`error: ${err.message}`);
  }
}

main();
