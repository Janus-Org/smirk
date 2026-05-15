// 224x224 similarity-transform face crop, matching demo.py:crop_face.
//
// The source and destination corner sets are both axis-aligned squares, so
// skimage.estimate_transform('similarity', ...) reduces to a uniform scale +
// axis-aligned translation. We mirror that math here and render through an
// offscreen 2D canvas to produce the float32 [1,3,224,224] tensor SMIRK expects.

const IMAGE_SIZE = 224;
const CROP_SCALE = 1.4;

const offscreen = document.createElement("canvas");
offscreen.width = IMAGE_SIZE;
offscreen.height = IMAGE_SIZE;
const ctx = offscreen.getContext("2d", { willReadFrequently: true });

// Pre-allocated buffer reused every frame (planar float32 RGB in [0,1]).
const inputBuffer = new Float32Array(1 * 3 * IMAGE_SIZE * IMAGE_SIZE);

let diagLogged = false;

/**
 * Compute the SMIRK-style crop bbox (cx, cy, size) from mediapipe landmarks.
 * Same math as demo.py:crop_face — bbox of all landmarks, expanded by CROP_SCALE.
 *
 * @param {Array<{x:number,y:number}>} landmarks - normalized [0,1] coords
 * @param {number} vw - source width in pixels
 * @param {number} vh - source height in pixels
 * @returns {{cx:number, cy:number, size:number} | null}
 */
export function computeBbox(landmarks, vw, vh) {
  let left = Infinity, right = -Infinity, top = Infinity, bottom = -Infinity;
  for (const p of landmarks) {
    const x = p.x * vw;
    const y = p.y * vh;
    if (x < left) left = x;
    if (x > right) right = x;
    if (y < top) top = y;
    if (y > bottom) bottom = y;
  }
  const oldSize = (right - left + bottom - top) / 2;
  const cx = right - (right - left) / 2;
  const cy = bottom - (bottom - top) / 2;
  const size = oldSize * CROP_SCALE;
  if (size <= 0) return null;

  if (!diagLogged) {
    diagLogged = true;
    const p0 = landmarks[0];
    console.log(
      `[crop] source=${vw}x${vh} ` +
      `landmark[0]=(${p0.x.toFixed(3)}, ${p0.y.toFixed(3)}) ` +
      `bbox=[${left.toFixed(0)},${top.toFixed(0)}]→[${right.toFixed(0)},${bottom.toFixed(0)}] ` +
      `center=(${cx.toFixed(0)},${cy.toFixed(0)}) size=${size.toFixed(0)} ` +
      `landmarks.length=${landmarks.length}`,
    );
  }
  return { cx, cy, size };
}

/**
 * Crop a square region around (cx, cy) of given size from the source, resize to
 * 224x224, and pack as planar float32 RGB [1,3,224,224] in [0,1].
 *
 * @param {HTMLCanvasElement | HTMLVideoElement} source
 * @param {number} cx
 * @param {number} cy
 * @param {number} size - crop side length in source pixels
 * @returns {Float32Array} the inputBuffer (re-used across calls)
 */
export function cropAt(source, cx, cy, size) {
  if (size <= 0) return null;
  const sx = cx - size / 2;
  const sy = cy - size / 2;
  ctx.clearRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
  ctx.drawImage(source, sx, sy, size, size, 0, 0, IMAGE_SIZE, IMAGE_SIZE);

  const { data } = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
  const plane = IMAGE_SIZE * IMAGE_SIZE;
  for (let i = 0; i < plane; i++) {
    const j = i * 4;
    inputBuffer[i] = data[j] / 255.0;
    inputBuffer[plane + i] = data[j + 1] / 255.0;
    inputBuffer[2 * plane + i] = data[j + 2] / 255.0;
  }
  return inputBuffer;
}

/** Returns the offscreen 224x224 canvas holding the last computed crop. */
export function getCropCanvas() {
  return offscreen;
}

export { IMAGE_SIZE };
