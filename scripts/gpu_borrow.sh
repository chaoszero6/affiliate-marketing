#!/usr/bin/env bash
# gpu_borrow.sh — the GPU handoff wrapper.
#
#   GPU is a single 32 GB card that cannot host the chat engine and ComfyUI at
#   once. This script owns the swap: it releases the GPU from the chat engine,
#   starts ComfyUI, runs the command it is given (any shell command), then
#   stops ComfyUI and brings the chat engine back — on success, on failure,
#   and on Ctrl-C alike (the EXIT trap).
#
#   ./scripts/gpu_borrow.sh <command> [args...]
#   GRACE=20 ./scripts/gpu_borrow.sh ./scripts/render.sh <run-id>
set -uo pipefail

LLM_UNIT=strata-orca-iq3-xxs.service
COMFY_UNIT=comfyui.service
COMFY=http://127.0.0.1:8188
GRACE=${GRACE:-20}
LOG_DIR=/root/affiliate-marketing/runs
GPU_READY=${GPU_READY:-4000}   # MiB; below this the card is considered free

restore() {
  # Always put the chat engine back, whatever happened to the render.
  systemctl stop comfyui.service 2>/dev/null || true
  sleep 2
  systemctl start "$LLM_UNIT" 2>/dev/null || true
}
trap restore EXIT

echo "[gpu] waiting ${GRACE}s for the current chat turn to finish"
sleep "$GRACE"
systemctl stop "$LLM_UNIT" 2>/dev/null || true
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" \
        -ge "$GPU_READY" ]; do sleep 2; done
echo "[gpu] engine released"

systemctl start "$COMFY_UNIT" 2>/dev/null
for _ in $(seq 1 180); do
  curl -s --max-time 5 "$COMFY/object_info" 2>/dev/null | grep -q . && break
  sleep 2
done
echo "[gpu] comfy up"

"$@"
rc=$?

systemctl stop "$COMFY_UNIT" 2>/dev/null || true
sleep 2
systemctl start "$LLM_UNIT" 2>/dev/null || true
exit $rc
