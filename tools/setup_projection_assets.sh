#!/usr/bin/env bash
# Focused asset setup for the FLAME-2020 → FLAME-2023 projection pipeline
# (tools/export_smirk_encoder_onnx.py + tools/build_flame_expr_projection.py).
#
# Downloads only what those two scripts need:
#   1. SMIRK checkpoint  (Google Drive, no auth)
#   2. FLAME 2020 pkl    (MPI portal, requires flame.is.tue.mpg.de account)
#   3. FLAME 2023 pkl    (HuggingFace LAM-assets bundle, no auth)
#
# Skips anything already present. Re-runnable.
#
# Lighter than quick_install.sh (which also pulls EMOCA/MICA/training templates).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CHECKPOINT="pretrained_models/SMIRK_em1.pt"
FLAME2020_DIR="assets/FLAME2020"
FLAME2020_PKL="${FLAME2020_DIR}/FLAME2020/generic_model.pkl"
FLAME2023_DIR="assets/FLAME2023"
FLAME2023_PKL="${FLAME2023_DIR}/flame2023.pkl"

urle () {
  [[ "${1}" ]] || return 1
  local LANG=C i x
  for (( i = 0; i < ${#1}; i++ )); do
    x="${1:i:1}"
    [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"
  done
  echo
}

# ── 1. SMIRK checkpoint ─────────────────────────────────────────────────────
if [[ -f "${CHECKPOINT}" ]]; then
  echo "[skip] SMIRK checkpoint already present: ${CHECKPOINT}"
else
  echo "[1/3] Downloading SMIRK checkpoint (Google Drive)..."
  mkdir -p "$(dirname "${CHECKPOINT}")"
  if ! command -v gdown >/dev/null; then
    echo "  installing gdown..."
    pip install --quiet gdown
  fi
  gdown --id 1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE -O "${CHECKPOINT}"
fi

# ── 2. FLAME 2020 (MPI portal, registration required) ───────────────────────
if [[ -f "${FLAME2020_PKL}" ]]; then
  echo "[skip] FLAME 2020 already present: ${FLAME2020_PKL}"
else
  echo "[2/3] FLAME 2020 needs your MPI portal credentials."
  echo "      Register at https://flame.is.tue.mpg.de/ if you don't have an account."
  read -p "      Username (FLAME): " username
  read -s -p "      Password (FLAME): " password
  echo
  u=$(urle "${username}")
  p=$(urle "${password}")

  mkdir -p "${FLAME2020_DIR}"
  zip="FLAME2020.zip"
  wget --post-data "username=${u}&password=${p}" \
       "https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1" \
       -O "${zip}" --no-check-certificate --continue
  unzip -q "${zip}" -d "${FLAME2020_DIR}/"
  rm -f "${zip}"
  if [[ ! -f "${FLAME2020_PKL}" ]]; then
    echo "ERROR: expected ${FLAME2020_PKL} after unzip — check credentials/portal." >&2
    exit 1
  fi
fi

# ── 3. FLAME 2023 (HuggingFace LAM-assets bundle) ───────────────────────────
if [[ -f "${FLAME2023_PKL}" ]]; then
  echo "[skip] FLAME 2023 already present: ${FLAME2023_PKL}"
else
  echo "[3/3] Downloading FLAME 2023 from HuggingFace (3DAIGC/LAM-assets)..."
  if ! command -v huggingface-cli >/dev/null; then
    echo "  installing huggingface_hub..."
    pip install --quiet 'huggingface_hub[cli]'
  fi
  mkdir -p "${FLAME2023_DIR}"
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "${tmp_dir}"' EXIT
  # Pull just the tar that contains flame_assets — the bundle is 969 MB so this is
  # heavy, but it's the only public source for flame2023.pkl that doesn't require
  # MPI registration. The tar also has DECA/MICA/etc; we discard those.
  huggingface-cli download 3DAIGC/LAM-assets thirdparty_models.tar --local-dir "${tmp_dir}"
  echo "  extracting flame2023.pkl only..."
  tar -xf "${tmp_dir}/thirdparty_models.tar" -C "${tmp_dir}" \
      --wildcards '*flame_assets/flame/flame2023.pkl'
  found_pkl="$(find "${tmp_dir}" -name flame2023.pkl -type f | head -n1)"
  if [[ -z "${found_pkl}" ]]; then
    echo "ERROR: flame2023.pkl not found inside thirdparty_models.tar" >&2
    exit 1
  fi
  cp "${found_pkl}" "${FLAME2023_PKL}"
fi

echo
echo "Done. Assets:"
echo "  SMIRK checkpoint: ${CHECKPOINT}"
echo "  FLAME 2020 pkl:   ${FLAME2020_PKL}"
echo "  FLAME 2023 pkl:   ${FLAME2023_PKL}"
echo
echo "Next steps:"
echo "  .venv/bin/python tools/export_smirk_encoder_onnx.py"
echo "  .venv/bin/python tools/build_flame_expr_projection.py"
