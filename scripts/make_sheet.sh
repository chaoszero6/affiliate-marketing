#!/usr/bin/env bash
# make_sheet.sh — build the multi-view character sheet.
#
#   ./scripts/make_sheet.sh [--angles 0,45,90,180] [--grace 20] [--keep-comfy]
#
# The sheet is a one-time asset: it turns the single portrait into a multi-angle
# reference (front, three-quarter, side, back) that the video workflow reuses
# forever. The script borrows the GPU from the chat engine, runs the image jobs,
# then hands the GPU back.
set -euo pipefail

ROOT="/root/affiliate-marketing"
COMFY="http://127.0.0.1:8188"
LLM_UNIT=strata-orca-iq3-xxs.service
COMFY_UNIT=comfyui.service
GRACE=20
ANGLES="0,45,90,180"
KEEP=0

ARGS=("$@")
i=0
while [ "$i" -lt "${#ARGS[@]}" ]; do
  a="${ARGS[$i]}"
  case "$a" in
    --angles) i=$((i + 1)); ANGLES="${ARGS[$i]}" ;;
    --grace) i=$((i + 1)); GRACE="${ARGS[$i]}" ;;
    --keep-comfy) KEEP=1 ;;
    *) echo "unknown flag $a" >&2; exit 1 ;;
  esac
  i=$((i + 1))
done

say() { echo "[$(date '+%H:%M:%S')] $*"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

say "sheet run: angles $ANGLES"
say "sleeping ${GRACE}s so the current chat turn finishes before the GPU moves"
sleep "$GRACE"

say "releasing the GPU from the chat engine"
systemctl stop "$LLM_UNIT" 2>/dev/null || true
for _ in $(seq 1 40); do
  [ "$(gpu_used)" -lt 4000 ] && break
  sleep 2
done
say "gpu free: $(gpu_used) MiB in use"

say "starting ComfyUI"
systemctl start "$COMFY_UNIT" 2>/dev/null || nohup /opt/comfyui/start_comfyui.sh >/dev/null 2>&1 &
for _ in $(seq 1 90); do
  curl -s --max-time 3 "$COMFY/system_stats" >/dev/null 2>&1 && break
  sleep 2
done
curl -s --max-time 3 "$COMFY/system_stats" >/dev/null || { echo "ComfyUI never came up"; exit 1; }
say "ComfyUI up"

# the sheet is a one-time asset; keep the run id stable per invocation
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "/opt/comfyui/ComfyUI/output/sheet/$STAMP" 2>/dev/null || true

python3 "$ROOT/scripts/make_sheet.py" \
  --base "$ROOT/assets/character/face.jpg" \
  --angles "$ANGLES" \
  --out "$ROOT/assets/character/sheet.png" || { say "sheet generation failed"; }

# hand the GPU back
if [ "$KEEP" = "0" ]; then
  say "stopping ComfyUI, restoring the chat engine"
  systemctl stop "$COMFY_UNIT" 2>/dev/null || true
  sleep 3
  systemctl start "$LLM_UNIT" 2>/dev/null || true
  for _ in $(seq 1 40); do
    curl -s --max-time 3 http://127.0.0.1:8322/v1/models >/dev/null 2>&1 && break
    sleep 3
  done
  say "chat engine back up"
fi

say "done — assets/character/sheet.png"
