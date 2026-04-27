#!/usr/bin/env bash
#
# One-shot setup for the qwen36-27b-single-5090 stack.
#   - clones Sandermage/genesis-vllm-patches into ./patches/genesis
#   - downloads Lorbus/Qwen3.6-27B-int4-AutoRound into $MODEL_DIR with
#     SHA256 verification against HF x-linked-etag
#
# Env vars (optional):
#   MODEL_DIR      Where to place the model. Default: ./models
#   HF_TOKEN       HF token (public model, so usually unnecessary)
#   SKIP_MODEL     Set to 1 to skip the model download step
#   SKIP_GENESIS   Set to 1 to skip cloning Genesis patches
#
# Idempotent: safe to re-run — skips steps already done.

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models}"
MODEL_REPO="Lorbus/Qwen3.6-27B-int4-AutoRound"
MODEL_SUBDIR="qwen3.6-27b-autoround-int4"
GENESIS_DIR="${ROOT_DIR}/patches/genesis"

cd "${ROOT_DIR}"

# ---------- Tool checks ----------
need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: required tool '$1' not found in PATH." >&2
    exit 1
  }
}
need git
need curl
need sha256sum

echo "Setup root:   ${ROOT_DIR}"
echo "Model dir:    ${MODEL_DIR}"

# ---------- Genesis patches ----------
if [[ "${SKIP_GENESIS:-0}" != "1" ]]; then
  if [[ -d "${GENESIS_DIR}/.git" ]]; then
    echo "[genesis] Already cloned at ${GENESIS_DIR} — skipping."
  else
    echo "[genesis] Cloning Sandermage/genesis-vllm-patches ..."
    git clone --depth 1 https://github.com/Sandermage/genesis-vllm-patches.git "${GENESIS_DIR}"
  fi

  if [[ ! -d "${GENESIS_DIR}/vllm/_genesis" ]]; then
    echo "ERROR: genesis clone succeeded but vllm/_genesis/ missing." >&2
    exit 1
  fi
else
  echo "[genesis] SKIP_GENESIS=1 — not cloning."
fi

# ---------- Model download ----------
if [[ "${SKIP_MODEL:-0}" == "1" ]]; then
  echo "[model]   SKIP_MODEL=1 — not downloading."
  exit 0
fi

mkdir -p "${MODEL_DIR}/${MODEL_SUBDIR}"

# Prefer `hf` CLI if available (faster with hf_transfer); fall back to curl.
download_via_hf() {
  echo "[model]   Using 'hf download' (hf_transfer if available) ..."
  HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1 \
    hf download "${MODEL_REPO}" --local-dir "${MODEL_DIR}/${MODEL_SUBDIR}"
}

if command -v hf >/dev/null 2>&1; then
  download_via_hf
elif command -v huggingface-cli >/dev/null 2>&1; then
  echo "[model]   Using 'huggingface-cli download' ..."
  HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1 \
    huggingface-cli download "${MODEL_REPO}" --local-dir "${MODEL_DIR}/${MODEL_SUBDIR}"
else
  echo "ERROR: neither 'hf' nor 'huggingface-cli' found. Install with:" >&2
  echo "  pip install 'huggingface-hub[hf_transfer]'" >&2
  echo "or:" >&2
  echo "  uv tool install --with hf_transfer huggingface-hub" >&2
  exit 1
fi

# ---------- SHA verification ----------
echo "[verify]  Checking SHA256 of every *.safetensors against HF x-linked-etag ..."
cd "${MODEL_DIR}/${MODEL_SUBDIR}"

fail=0
count=0
for f in *.safetensors; do
  [[ -f "$f" ]] || continue
  count=$((count + 1))
  expected="$(curl -sfI "https://huggingface.co/${MODEL_REPO}/resolve/main/$f" \
    | grep -i '^x-linked-etag:' | tr -d '"\r' | awk '{print $NF}' || true)"
  actual="$(sha256sum "$f" | awk '{print $1}')"
  if [[ -z "$expected" ]]; then
    printf "  %-50s SKIP (no etag)\n" "$f"
  elif [[ "$expected" == "$actual" ]]; then
    printf "  %-50s OK\n" "$f"
  else
    printf "  %-50s FAIL  exp=%.12s  act=%.12s\n" "$f" "$expected" "$actual"
    fail=$((fail + 1))
  fi
done
cd "${ROOT_DIR}"

if [[ "$fail" != "0" ]]; then
  echo "[verify]  ${fail} shard(s) failed SHA check." >&2
  echo "          Delete ${MODEL_DIR}/${MODEL_SUBDIR} and re-run setup.sh." >&2
  exit 1
fi

if [[ "$count" == "0" ]]; then
  echo "[verify]  No .safetensors found in ${MODEL_DIR}/${MODEL_SUBDIR} — download may have failed." >&2
  exit 1
fi

echo ""
echo "Done. ${count} shards SHA-verified, ${GENESIS_DIR} in place."
echo ""
echo "Next:"
echo "  cd compose && docker compose up -d"
echo "  docker logs -f vllm-qwen36-27b"
echo ""
echo "Wait for 'Application startup complete', then test with:"
echo ""
echo "  curl -sf http://localhost:8020/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"qwen3.6-27b-autoround\",\"messages\":[{\"role\":\"user\",\"content\":\"Capital of France?\"}],\"max_tokens\":30}'"
