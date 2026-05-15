"""Export SmirkEncoder + FLAME(vertices only) to a single ONNX file.

Input:  image  [B, 3, 224, 224] float32 in [0, 1] (RGB)
Output: vertices [B, 5023, 3] float32 (FLAME object-space vertices, pre-projection)
        cam      [B, 3]        float32 (scale, tx, ty)

The web renderer applies the orthographic projection + Y/Z flip in JS so that the
ONNX graph stays free of view-dependent transforms.

Run from repo root:
    .venv/bin/python tools/export_onnx.py
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn

# Monkey-patch create_backbone BEFORE importing the encoder so timm doesn't try to
# fetch ImageNet weights at construction time — the checkpoint will overwrite weights.
import timm
import src.smirk_encoder as _enc_module

_original_create_model = timm.create_model


def _create_model_no_pretrained(name, *args, **kwargs):
    kwargs["pretrained"] = False
    return _original_create_model(name, *args, **kwargs)


timm.create_model = _create_model_no_pretrained

from src.smirk_encoder import SmirkEncoder  # noqa: E402
from src.FLAME.FLAME import FLAME  # noqa: E402
from src.FLAME.lbs import lbs  # noqa: E402


class SmirkOnnxWrapper(nn.Module):
    """Combined SmirkEncoder + FLAME forward returning ONLY vertices and cam.

    Strips landmark computation from FLAME.forward to keep the ONNX graph minimal
    and avoid dynamic-landmark Python loops.
    """

    def __init__(self, encoder: SmirkEncoder, flame: FLAME):
        super().__init__()
        self.encoder = encoder
        self.flame = flame

    def forward(self, image: torch.Tensor):
        out = self.encoder(image)

        shape_params = out["shape_params"]
        expression_params = out["expression_params"]
        pose_params = out["pose_params"]
        jaw_params = out["jaw_params"]
        eyelid_params = out["eyelid_params"]
        cam = out["cam"]

        batch_size = shape_params.shape[0]
        flame = self.flame

        eye_pose = flame.eye_pose.expand(batch_size, -1)
        neck_pose = flame.neck_pose.expand(batch_size, -1)

        betas = torch.cat([shape_params, expression_params], dim=1)
        full_pose = torch.cat([pose_params, neck_pose, jaw_params, eye_pose], dim=1)

        template_vertices = flame.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        vertices, _ = lbs(
            betas,
            full_pose,
            template_vertices,
            flame.shapedirs,
            flame.posedirs,
            flame.J_regressor,
            flame.parents,
            flame.lbs_weights,
            dtype=flame.dtype,
        )

        vertices = vertices + flame.r_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 1:2, None]
        vertices = vertices + flame.l_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 0:1, None]

        return vertices, cam


def _load_smirk_encoder(checkpoint_path: str) -> SmirkEncoder:
    encoder = SmirkEncoder()
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    enc_state = {
        k.replace("smirk_encoder.", ""): v for k, v in ckpt.items() if "smirk_encoder" in k
    }
    encoder.load_state_dict(enc_state)
    encoder.eval()
    return encoder


def _crop_image_like_demo(image_path: str) -> torch.Tensor:
    """Replicates demo.py's crop_face for the parity self-test."""
    import cv2
    from skimage.transform import estimate_transform, warp

    from utils.mediapipe_utils import run_mediapipe

    image = cv2.imread(image_path)
    kpt = run_mediapipe(image)
    if kpt is None:
        raise RuntimeError(f"no face detected in {image_path}")
    kpt = kpt[..., :2]

    left, right = kpt[:, 0].min(), kpt[:, 0].max()
    top, bottom = kpt[:, 1].min(), kpt[:, 1].max()
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * 1.4)
    image_size = 224

    src_pts = np.array(
        [
            [center[0] - size / 2, center[1] - size / 2],
            [center[0] - size / 2, center[1] + size / 2],
            [center[0] + size / 2, center[1] - size / 2],
        ]
    )
    dst_pts = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform("similarity", src_pts, dst_pts)
    cropped = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    cropped = cv2.resize(cropped, (224, 224))
    t = torch.from_numpy(cropped).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return t


def _self_test(wrapper: SmirkOnnxWrapper, encoder: SmirkEncoder, flame: FLAME):
    sample_path = "samples/test_image2.png"
    if not Path(sample_path).exists():
        print(f"[self-test] skipping (sample not found at {sample_path})")
        return

    with torch.no_grad():
        x = _crop_image_like_demo(sample_path)
        wrap_verts, wrap_cam = wrapper(x)

        ref_enc = encoder(x)
        ref_flame = flame.forward(ref_enc)
        ref_verts = ref_flame["vertices"]
        ref_cam = ref_enc["cam"]

    diff_v = (wrap_verts - ref_verts).abs().max().item()
    diff_c = (wrap_cam - ref_cam).abs().max().item()
    print(f"[self-test] wrapper vs reference vertices max abs diff: {diff_v:.3e}")
    print(f"[self-test] wrapper vs reference cam max abs diff:      {diff_c:.3e}")
    assert diff_v < 1e-5, f"vertices parity failed: {diff_v}"
    assert diff_c < 1e-5, f"cam parity failed: {diff_c}"
    print("[self-test] OK")


def main():
    checkpoint = "pretrained_models/SMIRK_em1.pt"
    out_path = REPO_ROOT / "web" / "models" / "smirk.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = _load_smirk_encoder(checkpoint)
    flame = FLAME(flame_model_path="assets/FLAME2020/FLAME2020/generic_model.pkl")
    flame.eval()

    wrapper = SmirkOnnxWrapper(encoder, flame).eval()

    dummy = torch.zeros(1, 3, 224, 224)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(out_path),
            input_names=["image"],
            output_names=["vertices", "cam"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes={
                "image": {0: "B"},
                "vertices": {0: "B"},
                "cam": {0: "B"},
            },
        )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"wrote {out_path} ({size_mb:.1f} MiB)")

    _self_test(wrapper, encoder, flame)


if __name__ == "__main__":
    main()
