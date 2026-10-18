#!/usr/bin/env bash
# rebuild_character.sh — change a fixed trait of the character and rebuild
# every identity reference from it, in one GPU borrow.
#
#   ./scripts/rebuild_character.sh --prompt "Change her skin tone to ..." [--angles 0,45,90,180]
#
# 1. backs up assets/character/ to assets/character_backup_<stamp>/
# 2. edits face.jpg with Qwen Image Edit (scripts/recolor_face.py)
# 3. regenerates the multi-view sheet from the new face (scripts/make_sheet.py)
#    and writes sheet.png (all views side by side) + sheet-N.png (one per view)
# 4. syncs the ComfyUI input copies in haul_base/
#
# The chat engine is stopped for the duration and restarted afterwards.
set -uo pipefail
ROOT=/root/affiliate-marketing
COMFY_IN=/opt/comfyui/ComfyUI/input
PROMPT=""
ANGLES="0,45,90,180"
while [ $# -gt 0 ]; do
  case "$1" in
    --prompt) PROMPT="$2"; shift 2 ;;
    --angles) ANGLES="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 1 ;;
  esac
done
[ -n "$PROMPT" ] || { echo "need --prompt" >&2; exit 1; }

STAMP=$(date +%Y%m%d_%H%M%S)
cd "$ROOT" || exit 1
cp -a assets/character "assets/character_backup_$STAMP"
echo "[rebuild] backup -> assets/character_backup_$STAMP"

inner() {
  python3 "$ROOT/scripts/recolor_face.py" --base "$ROOT/assets/character/face.jpg" \
    --out /tmp/face_new.png --prompt "$PROMPT" || return 1
  # keep JPEG name/format the rest of the pipeline expects
  convert /tmp/face_new.png -quality 95 "$ROOT/assets/character/face.jpg" || return 1
  echo "[rebuild] face.jpg replaced"
  # regenerate the multi-view sheet from the new face
  python3 "$ROOT/scripts/make_sheet.py" --base "$ROOT/assets/character/face.jpg" \
    --angles "$ANGLES" --out /tmp/sheet_new.png || return 1
  local views
  views=$(ls -1dt /opt/comfyui/ComfyUI/output/sheet/*/ | head -1)
  local i=0
  for v in $(ls -1 "$views"*.png | sort); do
    convert "$v" "$ROOT/assets/character/sheet-$i.png"; i=$((i + 1))
  done
  convert $(ls -1 "$views"*.png | sort) +append "$ROOT/assets/character/sheet.png"
  echo "[rebuild] sheet.png + $i panels rebuilt"
}
export -f inner
export ROOT PROMPT ANGLES
GRACE=5 "$ROOT/scripts/gpu.sh" bash -c inner
rc=$?

mkdir -p "$COMFY_IN/haul_base"
cp -f assets/character/face.jpg assets/character/sheet.png assets/character/sheet-0.png "$COMFY_IN/haul_base/" 2>/dev/null
echo "[rebuild] done (rc=$rc); identity references:"
identify assets/character/face.jpg assets/character/sheet.png | cut -d' ' -f1-3
exit $rc
