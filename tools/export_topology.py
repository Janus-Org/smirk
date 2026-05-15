"""Dump FLAME mesh topology (faces + face_mask + face-only filtered faces) to JSON
so the browser renderer can build its three.js BufferGeometry without re-loading the
FLAME assets in the client.

Run from repo root:
    .venv/bin/python tools/export_topology.py
"""
import json
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.FLAME.FLAME import FLAME


def keep_vertices_and_update_faces(faces: torch.Tensor, vertices_to_keep: torch.Tensor) -> torch.Tensor:
    """Pure-torch copy of src/renderer/renderer.py:11 (which can't be imported because
    its module pulls in pytorch3d). Filters faces to those whose vertices are all in
    `vertices_to_keep`, then remaps indices to 0..len(unique(vertices_to_keep))-1."""
    vertices_to_keep = torch.unique(vertices_to_keep)
    max_vertex_index = faces.max().long().item() + 1
    mask = torch.zeros(max_vertex_index, dtype=torch.bool)
    mask[vertices_to_keep] = True
    new_vertex_indices = torch.full((max_vertex_index,), -1, dtype=torch.long)
    new_vertex_indices[mask] = torch.arange(len(vertices_to_keep))
    valid_faces_mask = (new_vertex_indices[faces] != -1).all(dim=1)
    return new_vertex_indices[faces[valid_faces_mask]]


def main():
    out_path = REPO_ROOT / "web" / "models" / "flame_topology.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    flame = FLAME(flame_model_path="assets/FLAME2020/FLAME2020/generic_model.pkl")
    faces_full = flame.faces_tensor.cpu().numpy().astype(np.int64)
    n_total_verts = int(flame.v_template.shape[0])

    with open("assets/FLAME_masks/FLAME_masks.pkl", "rb") as f:
        masks = pickle.load(f, encoding="latin1")
    face_mask = np.asarray(masks["face"], dtype=np.int64)

    faces_render = keep_vertices_and_update_faces(
        torch.from_numpy(faces_full), torch.from_numpy(face_mask)
    ).cpu().numpy()

    # keep_vertices_and_update_faces sorts/uniques the kept vertex list internally;
    # use the same sorted unique list so face_mask order lines up with remapped indices
    face_mask_sorted = np.unique(face_mask).astype(np.int64)

    n_render_verts = int(face_mask_sorted.shape[0])
    assert int(faces_render.max()) < n_render_verts, "faces_render indices out of range"

    payload = {
        "n_total_verts": n_total_verts,
        "n_render_verts": n_render_verts,
        "face_mask": face_mask_sorted.tolist(),
        "faces_render": faces_render.tolist(),
        "faces_full": faces_full.tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)

    print(f"wrote {out_path}")
    print(f"  n_total_verts={n_total_verts}")
    print(f"  n_render_verts={n_render_verts}")
    print(f"  faces_render={len(faces_render)}")
    print(f"  faces_full={len(faces_full)}")


if __name__ == "__main__":
    main()
