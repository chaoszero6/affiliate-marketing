#!/usr/bin/env bash
# gpu.sh — borrow the GPU from the chat engine, run a command, give it back.
#
#   ./scripts/gpu.sh <command> [args...]
#   GRACE=30 ./scripts/gpu.sh ./scripts/make_sheet.py ...
#
# The GPU is single-GPU: the chat engine (strata) and ComfyUI cannot share it.
# This wrapper performs the handoff around any command:
#
#   1. stop the chat engine (grace period lets in-flight requests drain)
#   2. start ComfyUI and wait for it to answer
#   3. run the command
#   4. stop ComfyUI, start the chat engine again — on success AND on failure
#
# The restore runs from an EXIT trap, so Ctrl-C, a crash, or a non-zero exit
# all still put the engine back.
set -euo pipefail

LLM_UNIT=strata-orca-iq3-xxs.service
COMFY_UNIT=comfyui.service
COMFY=http://127.0.0.1:8188
GRACE=${GRACE:-15}
KEEP_COMFY=${KEEP_COMFY:-0}

if [ $# -eq 0 ]; then
  echo "usage: gpu.sh <command> [args...]   (env: GRACE=seconds KEEP_COMFY=1)" >&2
  exit 1
fi

say() { echo "[gpu.sh] $*"; }
unit_of_comfy() { systemctl is-active "$COMFY_UNIT" 2>/dev/null; }

restore() {
  local rc=$?
  trap - EXIT
  if [ "$KEEP_COMFY" = "1" ]; then
    say "keeping ComfyUI up (KEEP_COMFY=1)"
    return
  fi
  say "restoring the chat engine"
  systemctl stop "$COMFY_UNIT" 2>/dev/null || true
  sleep 2
  systemctl start "$LLM_UNIT" 2>/dev/null || true
  for _ in $(seq 1 40); do
    curl -s --max-time 3 http://127.0.0.1:8322/v1/models >/dev/null 2>&1 && break
    sleep 2
  done
  say "chat engine back up (exit $rc)"
}
trap restore EXIT

# ---- take the GPU ----
if [ "$(unit_of_comfy)" != "active" ]; then
  say "stopping the chat engine (grace ${GRACE}s)"
  sleep "$GRACE"
  systemctl stop "$LLM_UNIT" 2>/dev/null || true
  # wait for VRAM to actually drain
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 4000 ] && break
    sleep 2
  done
  say "starting ComfyUI"
  systemctl start "$COMFY_UNIT" 2>/dev/null || true
fi

# ---- wait for ComfyUI ----
for _ in $(seq 1 90); do
  curl -s --max-time 3 "$COMFY/system_stats" >/dev/null 2>&1 && break
  sleep 2
done
if ! curl -s --max-time 3 "$COMFY/system_stats" >/dev/null 2>&1; then
  say "ERROR: ComfyUI never answered on 8188"
  exit 1
fi
say "ComfyUI ready"

# ---- run the payload ----
"$@"
rc=$?
exit $rc
