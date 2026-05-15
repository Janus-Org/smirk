"""Run web/models/smirk.onnx through onnxruntime and compare against the PyTorch
wrapper to confirm numerical parity.

Run from repo root:
    .venv/bin/python tools/verify_onnx.py
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import onnxruntime as ort
import torch

# Reuse the wrapper-building helpers from export_onnx.py
sys.path.insert(0, str(REPO_ROOT / "tools"))
from export_onnx import (  # noqa: E402
    SmirkOnnxWrapper,
    _crop_image_like_demo,
    _load_smirk_encoder,
)
from src.FLAME.FLAME import FLAME  # noqa: E402


def main():
    onnx_path = "web/models/smirk.onnx"
    assert Path(onnx_path).exists(), f"missing {onnx_path}; run tools/export_onnx.py first"

    encoder = _load_smirk_encoder("pretrained_models/SMIRK_em1.pt")
    flame = FLAME(flame_model_path="assets/FLAME2020/FLAME2020/generic_model.pkl")
    flame.eval()
    wrapper = SmirkOnnxWrapper(encoder, flame).eval()

    x = _crop_image_like_demo("samples/test_image2.png")

    with torch.no_grad():
        torch_v, torch_c = wrapper(x)
    torch_v = torch_v.cpu().numpy()
    torch_c = torch_c.cpu().numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_v, ort_c = sess.run(["vertices", "cam"], {"image": x.numpy()})

    diff_v = np.max(np.abs(ort_v - torch_v))
    diff_c = np.max(np.abs(ort_c - torch_c))
    print(f"ORT vs PyTorch vertices max abs diff: {diff_v:.3e}")
    print(f"ORT vs PyTorch cam      max abs diff: {diff_c:.3e}")
    assert diff_v < 1e-4, f"vertices parity failed: {diff_v}"
    assert diff_c < 1e-4, f"cam parity failed: {diff_c}"
    print("OK")


if __name__ == "__main__":
    main()
