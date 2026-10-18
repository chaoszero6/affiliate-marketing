#!/usr/bin/env bash
# run.sh — the haul-video agent's entry point. Two phases.
#
#   ./scripts/run.sh [options] <myntra-url | image.jpg | folder> [...]
#
#   --pose NAME        pose preset from config/style.json (turn | static | walk)
#   --hint "phrase"    override the outfit description for the Nth ref (repeat)
#   --max-images N     dress photos used as references per outfit (default 4)
#   --no-mask          keep the catalogue model's face in the dress photos
#   --keep-comfy       leave ComfyUI running afterwards
#   --keyframe-only    stop after the try-on still (~1.5 min) so it can be
#                      approved before the 4-minute video render
#   --seed N           seed for the try-on still (re-roll a wrong keyframe)
#   --video-from DIR   render the video for an existing run dir whose keyframe
#                      was approved (skips scrape + keyframe); no URL needed
#
# Phase 1 (THIS process, chat engine still up): scrape the listing, download the
#   gallery, blur the catalogue model's face, write runs/<id>/manifest.json.
#   Network + CPU only.
#
# Phase 2 (detached daemon, scripts/render.sh): release the GPU from the chat
#   engine, start ComfyUI, build one 10 s clip per outfit, concat, QA-score the
#   face, hand the GPU back.
#
# Why detached? The engine this conversation runs on is strata-orca-iq3-xxs.
# Phase 2 must `systemctl stop` that very engine to free the GPU — and any
# process still in this session's process group dies with it. So Phase 2 is
# `setsid`'d so it survives the engine stop, and its progress lands in
# runs/<id>/run.log for the agent to poll.
set -uo pipefail

ROOT="/root/affiliate-marketing"
LLM_UNIT=strata-orca-iq3-xxs.service
KEEP=0
POSE=""
KEYFRAME_ONLY=0
KF_SEED=""
VIDEO_FROM=""
REFS=()
SCRAPE_ARGS=()

ARGS=("$@")
i=0
while [ "$i" -lt "${#ARGS[@]}" ]; do
  a="${ARGS[$i]}"
  case "$a" in
    --keep-comfy) KEEP=1 ;;
    --keyframe-only) KEYFRAME_ONLY=1 ;;
    --seed) i=$((i + 1)); KF_SEED="${ARGS[$i]}" ;;
    --video-from) i=$((i + 1)); VIDEO_FROM="$(realpath -m "${ARGS[$i]%/}")" ;;
    --pose) i=$((i + 1)); POSE="${ARGS[$i]}" ;;
    --hint) i=$((i + 1)); SCRAPE_ARGS+=(--hint "${ARGS[$i]}") ;;
    --max-images) i=$((i + 1)); SCRAPE_ARGS+=(--max-images "${ARGS[$i]}") ;;
    --no-mask) SCRAPE_ARGS+=(--no-mask) ;;
    --no-caption) SCRAPE_ARGS+=(--no-caption) ;;
    -h|--help) sed -n 2,20p "$0"; exit 0 ;;
    *) REFS+=("$a") ;;
  esac
  i=$((i + 1))
done

if pgrep -f "[s]cripts/render.sh --env" >/dev/null; then
  echo "a render is already running (the GPU holds one job at a time); wait for it to finish" >&2
  exit 3
fi

# Start the detached render as its own transient systemd unit so it survives
# whatever launched run.sh (the chat engine being stopped, the Discord bot
# service being restarted, a login shell closing, ...). A plain `setsid` only
# escapes the process group, not the parent systemd service's cgroup, so a
# service restart would kill the render mid-job. Falls back to setsid if
# systemd-run is unavailable.
start_detached_render() {  # start_detached_render <env-file> <unit-suffix>
  local envf="$1" suffix="$2"
  if command -v systemd-run >/dev/null 2>&1; then
    if systemd-run --unit="haul-render-$suffix" --collect --quiet \
         --setenv=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
         --working-directory="$ROOT" \
         bash "$ROOT/scripts/render.sh" --env "$envf" >/dev/null 2>&1; then
      return 0
    fi
  fi
  setsid bash "$ROOT/scripts/render.sh" --env "$envf" </dev/null >/dev/null 2>&1 &
}

# ---- resume mode: video for an already approved keyframe ----
if [ -n "$VIDEO_FROM" ]; then
  [ -f "$VIDEO_FROM/manifest.json" ] || { echo "no manifest in $VIDEO_FROM" >&2; exit 1; }
  grep -q '"keyframe"' "$VIDEO_FROM/manifest.json" || \
    echo "warn: $VIDEO_FROM has no keyframe recorded; the video will use references only" >&2
  STAMP=$(basename "$VIDEO_FROM")
  LOG="$VIDEO_FROM/run.log"
  say() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG" >&2; }
  say "video phase for approved keyframe in $VIDEO_FROM${POSE:+, pose $POSE}"
  printf '%s\n' "LLM_UNIT=$LLM_UNIT" "MANIFEST=$VIDEO_FROM/manifest.json" \
    "OUT_DIR=$VIDEO_FROM" "STAMP=$STAMP" "KEEP_COMFY=$KEEP" "POSE=$POSE" \
    "KEYFRAME_ONLY=0" "SKIP_KEYFRAME=1" "KF_SEED=" > "$VIDEO_FROM/env.sh"
  start_detached_render "$VIDEO_FROM/env.sh" "$STAMP"
  say "render detached; follow with: tail -f $LOG"
  echo "$VIDEO_FROM"
  exit 0
fi

if [ "${#REFS[@]}" -eq 0 ]; then
  echo "usage: run.sh [--pose NAME] [--hint PHRASE] [--max-images N] [--no-mask] [--keyframe-only] [--seed N] [--keep-comfy] <url|image|folder> [...]" >&2
  echo "       run.sh --video-from runs/<id>   (after approving runs/<id>/keyframe1.png)" >&2
  exit 1
fi

STAMP=$(date +%Y%m%d_%H%M%S)
RUN="$ROOT/runs/$STAMP"
mkdir -p "$RUN/out" "$ROOT/config" /root/production_pack/output

LOG="$RUN/run.log"
say() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG" >&2; }

say "run $STAMP — ${#REFS[@]} outfit(s)${POSE:+, pose $POSE}${KEYFRAME_ONLY:+}$([ "$KEYFRAME_ONLY" = 1 ] && echo ', keyframe only')${KF_SEED:+, seed $KF_SEED}"

# ---- Phase 1: scrape dress references (engine still up, network + CPU) ----
say "scraping listing(s), downloading gallery, masking catalogue faces"
if ! python3 "$ROOT/scripts/scrape.py" --run-id "$STAMP" \
      --manifest "$RUN/manifest.json" \
      ${SCRAPE_ARGS[@]+"${SCRAPE_ARGS[@]}"} "${REFS[@]}" 2>&1 | tee -a "$LOG" >&2; then
  say "scrape failed — see $LOG. If Myntra blocked the request, pass the product photos as files/folders instead of the URL."
  exit 2
fi
N=$(python3 -c "
import json
m = json.load(open('$RUN/manifest.json'))
print(sum(1 for i in m['items'] if i.get('images')))")
say "manifest: $N outfit(s) with reference photos"
[ "$N" -gt 0 ] || { say "nothing to render"; exit 2; }

# write what the detached render needs
printf '%s\n' "LLM_UNIT=$LLM_UNIT" "MANIFEST=$RUN/manifest.json" \
  "OUT_DIR=$RUN" "STAMP=$STAMP" "KEEP_COMFY=$KEEP" "POSE=$POSE" \
  "KEYFRAME_ONLY=$KEYFRAME_ONLY" "SKIP_KEYFRAME=0" "KF_SEED=$KF_SEED" > "$RUN/env.sh"

# ---- Phase 2: detach and render (own transient unit; see start_detached_render) ----
start_detached_render "$RUN/env.sh" "$STAMP"

say "render detached; follow with: tail -f $LOG"
echo "$RUN"           # callers read the run dir from stdout
exit 0
