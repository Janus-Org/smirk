// Three.js renderer for the SMIRK FLAME mesh.
//
// Mirrors src/renderer/renderer.py:
//   - Uses the face-only filtered triangles (from flame_topology.json).
//   - Replicates batch_orth_proj (util.py) + the Y/Z flip (renderer.py:102)
//     per-vertex on the CPU each frame.
//   - 5 directional lights at the same positions as the reference renderer.

import * as THREE from "three";

const TOPOLOGY_URL = "./models/flame_topology.json";

let topology = null;
let renderer = null;
let scene = null;
let camera = null;
let geom = null;
let posAttr = null;

export async function init(canvas) {
  topology = await fetch(TOPOLOGY_URL).then((r) => {
    if (!r.ok) throw new Error(`failed to fetch ${TOPOLOGY_URL}: ${r.status}`);
    return r.json();
  });

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);

  scene = new THREE.Scene();
  scene.background = null;

  // FLAME verts are ~±0.1 in object space; cam[0] ≈ 7 from PoseEncoder bias init,
  // so post-projection coords land in roughly [-0.7, 0.7]. Frustum [-1.2, 1.2]
  // leaves a small margin around the head.
  const frustum = 1.2;
  camera = new THREE.OrthographicCamera(-frustum, frustum, frustum, -frustum, 0.1, 100);
  camera.position.set(0, 0, 5);
  camera.lookAt(0, 0, 0);

  // Lights mirror renderer.py:128 (5 directional lights, intensity 1.7 there).
  for (const p of [[-1, 1, 1], [1, 1, 1], [-1, -1, 1], [1, -1, 1], [0, 0, 1]]) {
    const light = new THREE.DirectionalLight(0xffffff, 0.55);
    light.position.set(p[0], p[1], p[2]);
    scene.add(light);
  }
  scene.add(new THREE.AmbientLight(0xffffff, 0.18));

  // Build BufferGeometry from topology JSON.
  geom = new THREE.BufferGeometry();
  const positions = new Float32Array(topology.n_render_verts * 3);
  posAttr = new THREE.BufferAttribute(positions, 3);
  posAttr.setUsage(THREE.DynamicDrawUsage);
  geom.setAttribute("position", posAttr);

  // Flatten faces_render into a typed index buffer.
  const facesFlat = new Uint32Array(topology.faces_render.length * 3);
  for (let i = 0; i < topology.faces_render.length; i++) {
    facesFlat[i * 3 + 0] = topology.faces_render[i][0];
    facesFlat[i * 3 + 1] = topology.faces_render[i][1];
    facesFlat[i * 3 + 2] = topology.faces_render[i][2];
  }
  geom.setIndex(new THREE.BufferAttribute(facesFlat, 1));

  const material = new THREE.MeshPhongMaterial({
    color: 0xb4b4b4,
    side: THREE.DoubleSide,
    shininess: 8,
    specular: 0x222222,
  });
  const mesh = new THREE.Mesh(geom, material);
  scene.add(mesh);

  _resize();
  window.addEventListener("resize", _resize);
}

function _resize() {
  if (!renderer) return;
  const canvas = renderer.domElement;
  const { clientWidth, clientHeight } = canvas;
  if (clientWidth === 0 || clientHeight === 0) return;
  renderer.setSize(clientWidth, clientHeight, false);
}

/**
 * Update the mesh from a fresh SMIRK inference.
 * @param {Float32Array} vertices  length 5023*3 (FLAME object space)
 * @param {Float32Array} cam       length 3 (scale, tx, ty)
 */
export function update(vertices, cam) {
  if (!topology || !posAttr) return;
  const mask = topology.face_mask;
  const dst = posAttr.array;
  const s = cam[0], tx = cam[1], ty = cam[2];
  // batch_orth_proj from util.py:64. The Y/Z negation in renderer.py:102 is a
  // pytorch3d-image-space convention (Y-down); three.js is already Y-up, so we
  // do NOT negate here.
  for (let i = 0; i < mask.length; i++) {
    const src = mask[i] * 3;
    dst[i * 3 + 0] = s * (vertices[src + 0] + tx);
    dst[i * 3 + 1] = s * (vertices[src + 1] + ty);
    dst[i * 3 + 2] = s *  vertices[src + 2];
  }
  posAttr.needsUpdate = true;
  geom.computeVertexNormals();
}

export function render() {
  if (!renderer) return;
  renderer.render(scene, camera);
}
