"""Export SmirkEncoder + FLAME-2020 expression-displacement evaluator to ONNX.

This is the "approach C" exporter: we emit a per-vertex expression+eyelid
displacement on the bare 5023-vertex FLAME-2020 topology. Identity (shape) and
pose are stripped. The displacement is what janus needs as input to the LAM
subdivision matrix; v_template cancels out so we never need to ship it.

Outputs:
    expression_displacement [B, 5023, 3]  B_expr·expr + L/R eyelid offsets
    pose_params             [B, 3]         global head rotation (axis-angle)
    jaw_params              [B, 3]         jaw rotation (axis-angle)
    eyelid_params           [B, 2]         L/R eyelid weights (for iris occlusion)
    cam                     [B, 3]         (scale, tx, ty)

Why pose-stripped: pose is applied client-side by FLAME-2023's joint chain (the
kinematic tree is identical between FLAME 2020 and 2023). Sending pre-posed
vertices would force janus to undo and re-apply pose, which is both wasteful
and lossy.

Why identity-stripped: the avatar's identity is baked into LAM-generated
gaussian xyz (in FLAME-2023's canonical frame). SMIRK's shape coefficients fit
FLAME 2020 and would double-count if applied to a FLAME-2023-shaped avatar.

Why displacement (not vertices): the client computes Δv = expr_verts − v_template
to feed the upsampler; emitting the delta directly skips the subtraction and
makes v_template_2020 unnecessary client-side.

Run from repo root:
    .venv/bin/python tools/export_smirk_encoder_onnx.py
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

import timm
import src.smirk_encoder as _enc_module  # noqa: F401  (registers module before patch)

_original_create_model = timm.create_model


def _create_model_no_pretrained(name, *args, **kwargs):
    kwargs["pretrained"] = False
    return _original_create_model(name, *args, **kwargs)


timm.create_model = _create_model_no_pretrained

from src.smirk_encoder import SmirkEncoder  # noqa: E402
from src.FLAME.FLAME import FLAME  # noqa: E402


class SmirkExpressionOnnxWrapper(nn.Module):
    """SmirkEncoder + identity/pose-stripped FLAME-2020 displacement evaluator.

    Emits a per-vertex displacement on the 5023-vertex FLAME 2020 mesh carrying
    only expression + eyelid deformation. Pose and jaw axis-angle are returned
    alongside for client-side application against FLAME 2023.
    """

    def __init__(self, encoder: SmirkEncoder, flame: FLAME):
        super().__init__()
        self.encoder = encoder
        self.flame = flame
        # FLAME's stored shapedirs is (V, 3, n_shape + n_exp) after concat in FLAME.__init__.
        # The expression portion lives at columns n_shape: onwards.
        self._n_shape = int(flame.n_shape)

    def forward(self, image: torch.Tensor):
        out = self.encoder(image)
        expression_params = out["expression_params"]   # (B, n_exp)
        pose_params = out["pose_params"]               # (B, 3)
        jaw_params = out["jaw_params"]                 # (B, 3)
        eyelid_params = out["eyelid_params"]           # (B, 2)
        cam = out["cam"]                               # (B, 3)

        batch_size = expression_params.shape[0]

        # Expression displacement: blend_shapes restricted to the expression basis.
        # flame.shapedirs is (V, 3, n_shape + n_exp); expression part is columns n_shape:.
        expr_basis = self.flame.shapedirs[:, :, self._n_shape:]  # (V, 3, n_exp)
        # einsum: for each batch b, sum over expression PCs k → (V, 3) displacement.
        expr_disp = torch.einsum("vck,bk->bvc", expr_basis, expression_params)

        # Eyelid offsets — same math as FLAME.forward and the original onnx wrapper.
        l_disp = self.flame.l_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 0:1, None]
        r_disp = self.flame.r_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 1:2, None]

        # Δv directly — no v_template added; janus consumes this as a displacement.
        expression_displacement = expr_disp + l_disp + r_disp

        return expression_displacement, pose_params, jaw_params, eyelid_params, cam


def _load_smirk_encoder(checkpoint_path: str) -> SmirkEncoder:
    encoder = SmirkEncoder()
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    enc_state = {
        k.replace("smirk_encoder.", ""): v for k, v in ckpt.items() if "smirk_encoder" in k
    }
    encoder.load_state_dict(enc_state)
    encoder.eval()
    return encoder


def _self_test(wrapper: SmirkExpressionOnnxWrapper, encoder: SmirkEncoder, flame: FLAME):
    """Verify wrapper matches FLAME.forward with shape/pose explicitly zeroed."""
    sample_path = "samples/test_image2.png"
    if not Path(sample_path).exists():
        print(f"[self-test] skipping (sample not found at {sample_path})")
        return

    import cv2
    import numpy as np
    from skimage.transform import estimate_transform, warp

    from utils.mediapipe_utils import run_mediapipe

    image = cv2.imread(sample_path)
    kpt = run_mediapipe(image)
    if kpt is None:
        raise RuntimeError(f"no face detected in {sample_path}")
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
    cropped = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(
        np.uint8
    )
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    cropped = cv2.resize(cropped, (224, 224))
    x = torch.from_numpy(cropped).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        wrap_disp, wrap_pose, wrap_jaw, wrap_eyelid, wrap_cam = wrapper(x)
        enc = encoder(x)

        # Reference: full FLAME forward with shape and all pose components zeroed,
        # then subtract v_template to get the displacement the wrapper emits.
        B = enc["expression_params"].shape[0]
        ref_params = {
            "shape_params": torch.zeros(B, flame.n_shape),
            "expression_params": enc["expression_params"],
            "pose_params": torch.zeros(B, 3),
            "jaw_params": torch.zeros(B, 3),
            "eye_pose_params": torch.zeros(B, 6),
            "neck_pose_params": torch.zeros(B, 3),
            "eyelid_params": enc["eyelid_params"],
        }
        ref = flame.forward(ref_params)
        ref_disp = ref["vertices"] - flame.v_template.unsqueeze(0)

    pairs = [
        ("expression_displacement vs FLAME.forward(zero shape, zero pose) − v_template",
         wrap_disp, ref_disp),
        ("pose_params",   wrap_pose,   enc["pose_params"]),
        ("jaw_params",    wrap_jaw,    enc["jaw_params"]),
        ("eyelid_params", wrap_eyelid, enc["eyelid_params"]),
        ("cam",           wrap_cam,    enc["cam"]),
    ]
    for name, a, b in pairs:
        diff = (a - b).abs().max().item()
        print(f"[self-test] {name:70s} max abs diff: {diff:.3e}")
        assert diff < 1e-5, f"{name} parity failed: {diff}"
    print("[self-test] OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="pretrained_models/SMIRK_em1.pt",
        help="path to SMIRK checkpoint .pt",
    )
    parser.add_argument(
        "--flame2020",
        default="assets/FLAME2020/FLAME2020/generic_model.pkl",
        help="path to FLAME 2020 generic_model.pkl",
    )
    parser.add_argument(
        "--out",
        default="web/models/smirk_encoder.onnx",
        help="output ONNX path",
    )
    parser.add_argument(
        "--no-self-test",
        action="store_true",
        help="skip the parity self-test",
    )
    args = parser.parse_args()

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = _load_smirk_encoder(args.checkpoint)
    flame = FLAME(flame_model_path=args.flame2020)
    flame.eval()

    wrapper = SmirkExpressionOnnxWrapper(encoder, flame).eval()

    dummy = torch.zeros(1, 3, 224, 224)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(out_path),
            input_names=["image"],
            output_names=[
                "expression_displacement",
                "pose_params",
                "jaw_params",
                "eyelid_params",
                "cam",
            ],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes={
                "image": {0: "B"},
                "expression_displacement": {0: "B"},
                "pose_params": {0: "B"},
                "jaw_params": {0: "B"},
                "eyelid_params": {0: "B"},
                "cam": {0: "B"},
            },
        )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"wrote {out_path} ({size_mb:.1f} MiB)")

    if not args.no_self_test:
        _self_test(wrapper, encoder, flame)


if __name__ == "__main__":
    main()
