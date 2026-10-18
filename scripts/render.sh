#!/usr/bin/env bash
# render.sh — the detached GPU render phase, launched by run.sh.
#
#   render.sh --env <env-file>
#
# Runs in its own session (setsid) so it survives the chat engine being
# stopped. It releases the GPU from the engine, starts ComfyUI fresh, waits for
# the node registry, builds one 10 s / 9:16 workflow per outfit, queues, waits,
# concatenates, then scores the face match on the CPU — and on every exit path
# restarts the chat engine.
set -uo pipefail

ROOT="/root/affiliate-marketing"
PACK="/root/production_pack"
COMFY_UNIT=comfyui.service
COMFY="http://127.0.0.1:8188"
COMFY_OUT=/opt/comfyui/ComfyUI/output
COMFY_IN=/opt/comfyui/ComfyUI/input
SCHEMA="$ROOT/config/node_schemas.json"
VENV_PY=/opt/comfyui/venv/bin/python
ENV_FILE=""

i=0
ARGS=("$@")
while [ "$i" -lt "${#ARGS[@]}" ]; do
  case "${ARGS[$i]}" in
    --env) i=$((i + 1)); ENV_FILE="${ARGS[$i]}" ;;
  esac
  i=$((i + 1))
done
[ -n "$ENV_FILE" ] && [ -f "$ENV_FILE" ] || { echo "no env file"; exit 1; }
source "$ENV_FILE"
# absolute paths: the ffmpeg concat list is resolved relative to the list file
MANIFEST="$(realpath -m "${MANIFEST:?}")"
OUT_DIR="$(realpath -m "${OUT_DIR:?}")"
STAMP="${STAMP:?}"
KEEP_COMFY="${KEEP_COMFY:-0}"
POSE="${POSE:-}"
KEYFRAME_ONLY="${KEYFRAME_ONLY:-0}"
SKIP_KEYFRAME="${SKIP_KEYFRAME:-0}"
KF_SEED="${KF_SEED:-}"
LOG="$OUT_DIR/run.log"
LLM_UNIT="${LLM_UNIT:-strata-orca-iq3-xxs.service}"

say() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" >> "$LOG"; }

# Ensure the engine comes back no matter what.
trap 'st=$?
      if [ "$KEEP_COMFY" != 1 ]; then
        systemctl stop '"$COMFY_UNIT"' 2>/dev/null
        rm -f '"$PACK"'/output/.llm_parked 2>/dev/null
      fi
      systemctl reset-failed '"$LLM_UNIT"' 2>/dev/null
      systemctl start '"$LLM_UNIT"' 2>/dev/null
      say "[detached] exit $st"
      exit $st' EXIT INT TERM

say "[detached] render phase started (stamp $STAMP)"

give_gpu_back() {  # stop ComfyUI, restart the chat engine, wait for its API
  if [ "$KEEP_COMFY" != 1 ]; then
    systemctl stop "$COMFY_UNIT" 2>/dev/null
    rm -f "$PACK/output/.llm_parked" 2>/dev/null
  fi
  systemctl reset-failed "$LLM_UNIT" 2>/dev/null
  systemctl start "$LLM_UNIT" 2>/dev/null
  for _ in $(seq 1 60); do
    curl -s --max-time 3 http://127.0.0.1:8322/v1/models 2>/dev/null | grep -q '"data"' && return 0
    sleep 5
  done
  say "[detached] chat engine slow to come back; garment check may be skipped"
}

garment_check() {  # garment_check <image> <out.json> [outfit-index]   (needs the chat engine)
  local img="$1" out="$2" idx="${3:-1}" prods
  prods=$(python3 -c "
import json
m=json.load(open('$MANIFEST'))
items=[i for i in m['items'] if i.get('images_raw')]
k=min(max(int('$idx'),1),len(items))-1
if items: print(' '.join('--product $COMFY_IN/'+p for p in items[k]['images_raw'][:2]))")
  [ -n "$prods" ] || return 0
  # shellcheck disable=SC2086
  python3 "$ROOT/scripts/garment_check.py" $prods --image "$img" --out "$out" \
    2>>"$LOG" | tail -1 >> "$LOG" && say "[detached] garment check -> $out"
}

# ---- 0. make sure the fixed references ComfyUI loads are the ones in assets/ ----
mkdir -p "$COMFY_IN/haul_base"
cp -f "$ROOT/assets/character/face.jpg"  "$COMFY_IN/haul_base/face.jpg"  2>/dev/null
cp -f "$ROOT/assets/character/sheet.png" "$COMFY_IN/haul_base/sheet.png" 2>/dev/null
cp -f "$ROOT/assets/scene/room.jpg"      "$COMFY_IN/haul_base/room.jpg"  2>/dev/null
# try-on identity panel: face (left) + the sheet's full-body view (right, the
# last of the sheet's four equal panels), so the image-edit model takes
# hairstyle and face-to-body proportion from the sheet. Only face.jpg and
# sheet.png are ever read from assets/character.
ffmpeg -v error -nostdin -y -i "$ROOT/assets/character/face.jpg" -i "$ROOT/assets/character/sheet.png" \
  -filter_complex "[1]crop=iw/4:ih:3*iw/4:0[body];[0]scale=-2:1104[a];[body]scale=-2:1104[b];[a][b]hstack" \
  "$COMFY_IN/haul_base/identity_kf.jpg" 2>> "$LOG"
chmod 644 "$COMFY_IN"/haul_base/* 2>/dev/null

take_gpu() {  # park the chat engine, start ComfyUI fresh, wait for its node registry
  touch "$PACK/output/.llm_parked" 2>/dev/null || true
  systemctl stop "$LLM_UNIT" 2>/dev/null || true
  systemctl stop "$COMFY_UNIT" 2>/dev/null || true
  sleep 3
  say "[detached] engine parked, stale services stopped"
  systemctl reset-failed "$COMFY_UNIT" 2>/dev/null || true
  systemctl start "$COMFY_UNIT" 2>/dev/null
  python3 "$ROOT/scripts/schema_cache.py" --out "$SCHEMA" --timeout 420 \
    >> "$LOG" 2>&1 || { say "[detached] Comfy never finished loading"; exit 1; }
  say "[detached] ComfyUI ready, node schemas cached"
}

# ---- 1-2. take the GPU ----
take_gpu

queue_json() {  # queue_json <file> -> prints prompt_id or empty; logs rejections
  local f="$1" resp code
  resp=$(curl -s -w '\n%{http_code}' -X POST "$COMFY/prompt" \
    -H 'Content-Type: application/json' --data-binary @"$f")
  code=${resp##*$'\n'}
  if [ "$code" = "200" ]; then
    printf '%s' "${resp%$'\n'*}" | python3 -c "import json,sys;print(json.load(sys.stdin)['prompt_id'])"
  else
    say "[detached] $(basename "$f") REJECTED (HTTP $code)"
    say "[detached] $(printf '%s' "${resp%$'\n'*}" | head -c 800)"
  fi
}

wait_job() {  # wait_job <prompt_id> -> prints success|error|timeout
  local id="$1" st=""
  for _ in $(seq 1 1800); do
    st=$(curl -s --max-time 5 "$COMFY/history/$id" \
         | python3 -c "import json,sys
try:
    print(json.load(sys.stdin)['$id']['status']['status_str'])
except Exception: print('')" 2>/dev/null)
    [ "$st" = "success" ] || [ "$st" = "error" ] && break
    sleep 5
  done
  if [ "$st" = "error" ]; then
    curl -s --max-time 5 "$COMFY/history/$id" | python3 -c "import json,sys
d=json.load(sys.stdin)['$id']
for m in d['status'].get('messages',[]):
    if m[0]=='execution_error': print(m[1].get('node_type'), '|', m[1].get('exception_message','')[:600])" >> "$LOG" 2>&1
  fi
  echo "${st:-timeout}"
}

# ---- 2b. Stage A: try-on keyframe per outfit (Qwen Image Edit) ----
KF_ENABLED=$(python3 -c "import json;print(int(bool(json.load(open('$ROOT/config/style.json')).get('keyframe',{}).get('enabled'))))")
if [ "$SKIP_KEYFRAME" = "1" ]; then
  say "[detached] stage A skipped: reusing the approved keyframe in the manifest"
elif [ "$KF_ENABLED" = "1" ]; then
  if python3 "$ROOT/scripts/build_keyframe.py" --manifest "$MANIFEST" \
        --out-dir "$OUT_DIR" ${KF_SEED:+--seed "$KF_SEED"} >> "$LOG" 2>&1; then
    K=$(ls -1 "$OUT_DIR"/keyframe*_c1.json 2>/dev/null | wc -l)
    CQ=$(ls -1 "$OUT_DIR"/keyframe1_c*.json 2>/dev/null | wc -l)   # Qwen candidates
    ENGINE=$(python3 -c "import json;print(json.load(open('$ROOT/config/style.json')).get('keyframe',{}).get('engine','qwen'))")
    CK=0
    [ "$ENGINE" = "krea2" ] || [ "$ENGINE" = "auto" ] && \
      CK=$(python3 -c "import json;print(int(json.load(open('$ROOT/config/style.json'))['keyframe'].get('krea_candidates',2)))")
    [ "$ENGINE" = "krea2" ] && CQ=0
    C=$((CQ + CK))
    say "[detached] stage A: $K outfit(s) x $C try-on candidate(s) (engine $ENGINE: $CQ qwen + $CK krea2)"
    W=$(python3 -c "import json;o=json.load(open('$ROOT/config/style.json'))['output'];print(o['width'])")
    Hh=$(python3 -c "import json;o=json.load(open('$ROOT/config/style.json'))['output'];print(o['height'])")
    # queue every candidate, then crop each finished still to the clip framing
    declare -A KF_IDS=()
    # Qwen candidates first (c = 1..CQ): queued as ComfyUI jobs
    if [ "$CQ" -gt 0 ]; then
      for ((k = 1; k <= K; k++)); do
        for ((c = 1; c <= CQ; c++)); do
          [ -f "$OUT_DIR/keyframe${k}_c${c}.json" ] || continue
          id=$(queue_json "$OUT_DIR/keyframe${k}_c${c}.json")
          [ -n "$id" ] && KF_IDS["$k:$c"]="$id"
        done
      done
    fi
    # Krea 2 candidates next (c = CQ+1..C): two passes each, run inline
    if [ "$CK" -gt 0 ]; then
      # Krea 2 Identity Edit, two passes per candidate (product photo -> our
      # woman in the garment -> placed in our room). Results are dropped where
      # the Qwen path would have saved them so the rest of the stage is shared.
      KSTEPS=$(python3 -c "import json;print(json.load(open('$ROOT/config/style.json'))['keyframe'].get('krea_steps',10))")
      KBOOST=$(python3 -c "import json;print(json.load(open('$ROOT/config/style.json'))['keyframe'].get('krea_ref_boost',1.5))")
      CH=$(python3 -c "import json;print(json.load(open('$ROOT/config/style.json'))['character']['description'])")
      BASESEED=$(python3 -c "import json;print(json.load(open('$ROOT/config/style.json'))['keyframe'].get('seed',20261020))")
      mkdir -p "$COMFY_OUT/affiliate/$STAMP"
      say "[detached] stage A engine: Krea 2 Identity Edit ($KSTEPS steps, ref_boost $KBOOST)"
      for ((k = 1; k <= K; k++)); do
        read -r PROD CAP COL <<< "$(python3 - "$MANIFEST" "$k" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); k = int(sys.argv[2])
it = [i for i in m["items"] if i.get("images")][k - 1]
print(it["images"][0], json.dumps(it.get("prompt_hint", "")), json.dumps((it.get("colour") or "").lower()))
PY
)"
        CAP=$(python3 -c "import json,sys;print(json.loads(sys.argv[1]))" "$CAP")
        COL=$(python3 -c "import json,sys;print(json.loads(sys.argv[1]))" "$COL")
        for ((c = CQ + 1; c <= C; c++)); do
          if [ -n "$KF_SEED" ]; then SEED=$((KF_SEED + c - 1)); else SEED=$((BASESEED + k * 100 + c - 1)); fi
          python3 "$ROOT/scripts/krea_tryon.py" --product "$COMFY_IN/$PROD" \
            --identity "$COMFY_IN/haul_base/face.jpg" --room "$COMFY_IN/haul_base/room.jpg" \
            --outfit "$CAP" --colour "$COL" --character "$CH" --seed "$SEED" \
            --steps "$KSTEPS" --ref-boost "$KBOOST" --width "$(python3 -c "import json;print(json.load(open('$ROOT/config/style.json'))['keyframe']['width'])")" \
            --height "$(python3 -c "import json;print(json.load(open('$ROOT/config/style.json'))['keyframe']['height'])")" \
            --out-dir "$OUT_DIR/krea_k${k}_c${c}" >> "$LOG" 2>&1
          if [ -f "$OUT_DIR/krea_k${k}_c${c}/pass2.png" ]; then
            cp -f "$OUT_DIR/krea_k${k}_c${c}/pass2.png" "$COMFY_OUT/affiliate/$STAMP/keyframe_seg${k}_c${c}_00001_.png"
            KF_IDS["$k:$c"]="krea"
          else
            say "[detached] keyframe${k} candidate ${c}: Krea pass failed"
          fi
        done
      done
    fi
    SWAP=$(python3 -c "import json;print(int(bool(json.load(open('$ROOT/config/style.json')).get('keyframe',{}).get('face_swap',True))))")
    for key in "${!KF_IDS[@]}"; do
      k=${key%%:*}; c=${key##*:}
      if [ "${KF_IDS[$key]}" = "krea" ]; then st="success"; else st=$(wait_job "${KF_IDS[$key]}"); fi
      src=$(ls -1 "$COMFY_OUT/affiliate/$STAMP"/keyframe_seg${k}_c${c}_*.png 2>/dev/null | tail -1)
      if [ "$st" = "success" ] && [ -n "$src" ] && [ "$SWAP" = "1" ]; then
        # put her real face on the still (ReActor, ArcFace-driven); the
        # garments, pose and room are untouched
        mkdir -p "$COMFY_IN/haul_$STAMP/swap"
        cp -f "$src" "$COMFY_IN/haul_$STAMP/swap/cand_${k}_${c}.png"
        python3 "$ROOT/scripts/build_swap.py" --image "haul_$STAMP/swap/cand_${k}_${c}.png" \
          --face "haul_base/face.jpg" --prefix "affiliate/$STAMP/swap_seg${k}_c${c}" \
          --out "$OUT_DIR/swap${k}_c${c}.json" >/dev/null 2>>"$LOG"
        sid=$(queue_json "$OUT_DIR/swap${k}_c${c}.json")
        if [ -n "$sid" ]; then
          sst=$(wait_job "$sid")
          swapped=$(ls -1 "$COMFY_OUT/affiliate/$STAMP"/swap_seg${k}_c${c}_*.png 2>/dev/null | tail -1)
          if [ "$sst" = "success" ] && [ -n "$swapped" ]; then
            cp -f "$src" "$OUT_DIR/keyframe${k}_c${c}_noswap.png"
            src="$swapped"
            say "[detached] keyframe${k} candidate ${c}: face swapped"
          else
            say "[detached] keyframe${k} candidate ${c}: face swap $sst, keeping original"
          fi
        fi
      fi
      if [ "$st" = "success" ] && [ -n "$src" ]; then
        cp -f "$src" "$OUT_DIR/keyframe${k}_c${c}_full.png"
        if [ -x "$VENV_PY" ]; then
          "$VENV_PY" "$ROOT/scripts/face_tools.py" frame --input "$src" \
            --output "$OUT_DIR/keyframe${k}_c${c}_cropped.png" --width "$W" --height "$Hh" \
            2>/dev/null | tail -1 >> "$LOG" || cp -f "$src" "$OUT_DIR/keyframe${k}_c${c}_cropped.png"
        else
          cp -f "$src" "$OUT_DIR/keyframe${k}_c${c}_cropped.png"
        fi
      fi
      say "[detached] keyframe${k} candidate ${c} -> $st"
    done
    # pick the best candidate per outfit with the garment check (needs the
    # chat engine, so the GPU goes back for a minute when there is a choice)
    if [ "$C" -gt 1 ]; then
      give_gpu_back
      mkdir -p "$OUT_DIR/qa"
      for ((k = 1; k <= K; k++)); do
        for ((c = 1; c <= C; c++)); do
          [ -f "$OUT_DIR/keyframe${k}_c${c}_cropped.png" ] || continue
          garment_check "$OUT_DIR/keyframe${k}_c${c}_cropped.png" "$OUT_DIR/qa/keyframe${k}_c${c}_garment_check.json" "$k"
        done
        best=$(python3 - "$OUT_DIR" "$k" "$C" <<'PY'
import json, os, sys
out, k, C = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = []
for c in range(1, C + 1):
    p = os.path.join(out, "qa", f"keyframe{k}_c{c}_garment_check.json")
    if not os.path.exists(os.path.join(out, f"keyframe{k}_c{c}_cropped.png")):
        continue
    try:
        d = json.load(open(p)); score = float(d.get("score") or 0); match = bool(d.get("match"))
    except Exception:
        score, match = 0.0, False
    rows.append((match, score, -c, c))
rows.sort(reverse=True)
print(rows[0][3] if rows else 1)
PY
)
        say "[detached] keyframe${k}: best candidate = ${best} ($(python3 -c "import json;d=json.load(open('$OUT_DIR/qa/keyframe${k}_c${best}_garment_check.json'));print('match' if d.get('match') else 'no-match', d.get('score'))" 2>/dev/null))"
        cp -f "$OUT_DIR/keyframe${k}_c${best}_cropped.png" "$OUT_DIR/keyframe${k}_cropped.png"
        cp -f "$OUT_DIR/keyframe${k}_c${best}_full.png" "$OUT_DIR/keyframe${k}_full.png" 2>/dev/null
        cp -f "$OUT_DIR/qa/keyframe${k}_c${best}_garment_check.json" "$OUT_DIR/qa/keyframe${k}_garment_check.json" 2>/dev/null
      done
      [ "$KEYFRAME_ONLY" = "1" ] || take_gpu
    else
      for ((k = 1; k <= K; k++)); do
        cp -f "$OUT_DIR/keyframe${k}_c1_cropped.png" "$OUT_DIR/keyframe${k}_cropped.png" 2>/dev/null
        cp -f "$OUT_DIR/keyframe${k}_c1_full.png" "$OUT_DIR/keyframe${k}_full.png" 2>/dev/null
      done
    fi
    for ((k = 1; k <= K; k++)); do
      src="$OUT_DIR/keyframe${k}_cropped.png"; st="success"
      [ -f "$src" ] || st="missing"
      if [ "$st" = "success" ] && [ -n "$src" ]; then
        # register the still in the manifest item it belongs to
        python3 - "$MANIFEST" "$k" "$src" "$COMFY_IN" "$OUT_DIR" <<'PY' >> "$LOG" 2>&1
import json, os, shutil, sys
man, k, src, cin, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
m = json.load(open(man))
items = [i for i in m["items"] if i.get("images")]
it = items[k - 1]
d = os.path.join(cin, os.path.dirname(it["images"][0]))
dst = os.path.join(d, "keyframe.png")
shutil.copy(src, dst)
shutil.copy(src, os.path.join(out, f"keyframe{k}.png"))
it["keyframe"] = os.path.relpath(dst, cin)
json.dump(m, open(man, "w"), indent=2)
print(f"keyframe{k} -> {it['keyframe']}")
PY
      else
        say "[detached] keyframe$k failed; seg$k will render from references only"
      fi
    done
  else
    say "[detached] build_keyframe.py failed; rendering from references only"
  fi
fi

if [ "$KEYFRAME_ONLY" = "1" ]; then
  # hand the GPU back first: the face score is CPU-only but the garment check
  # needs the chat engine, which cannot share the GPU with ComfyUI
  mkdir -p "$OUT_DIR/qa"
  give_gpu_back
  for ((n = 1; n <= K; n++)); do
    kfp="$OUT_DIR/keyframe${n}_cropped.png"
    [ -f "$kfp" ] || continue
    [ -x "$VENV_PY" ] && "$VENV_PY" "$ROOT/scripts/face_tools.py" score \
      --ref "$ROOT/assets/character/face.jpg" --image "$kfp" \
      --out "$OUT_DIR/qa/keyframe${n}_face_score.json" 2>/dev/null | tail -1 >> "$LOG"
    [ -f "$OUT_DIR/qa/keyframe${n}_garment_check.json" ] || \
      garment_check "$kfp" "$OUT_DIR/qa/keyframe${n}_garment_check.json" "$n"
  done
  say "[detached] keyframe-only run done; approve $OUT_DIR/keyframe{1..$K}_cropped.png then: ./scripts/run.sh --video-from $OUT_DIR"
  exit 0
fi

# ---- 3. build + queue (one segment per outfit) ----
if ! python3 "$ROOT/scripts/build_workflow.py" --manifest "$MANIFEST" \
      --out-dir "$OUT_DIR" ${POSE:+--pose "$POSE"} >> "$LOG" 2>&1; then
  say "[detached] build_workflow.py failed"; exit 1
fi
N=$(ls -1 "$OUT_DIR"/seg*.json 2>/dev/null | wc -l)
say "[detached] built $N workflow(s)"
[ "$N" -gt 0 ] || { say "[detached] nothing to queue"; exit 1; }

IDS=()
for ((n = 1; n <= N; n++)); do
  id=$(queue_json "$OUT_DIR/seg$n.json")
  [ -n "$id" ] || exit 1
  say "[detached] seg$n queued -> $id"
  IDS+=("$id")
done

say "[detached] waiting for ${#IDS[@]} job(s)"
for id in "${IDS[@]}"; do
  st=$(wait_job "$id")
  say "[detached] $id -> $st"
done

# ---- 4. collect + concat ----
OUT="$COMFY_OUT/affiliate/$STAMP"
if [ -d "$OUT" ]; then
  cp "$OUT"/*.mp4 "$OUT_DIR/out/" 2>/dev/null || true
fi
LIST="$OUT_DIR/out/list.txt"
: > "$LIST"
for f in $(ls -1 "$OUT_DIR/out"/*.mp4 2>/dev/null | sort); do
  echo "file '$f'" >> "$LIST"
done
if [ -s "$LIST" ]; then
  if ffmpeg -v error -nostdin -y -f concat -safe 0 -i "$LIST" -c copy \
       "$OUT_DIR/final.mp4" 2>> "$LOG"; then
    say "[detached] concatenated -> $OUT_DIR/final.mp4"
  else
    say "[detached] concat failed"
  fi
else
  say "[detached] no segments produced; skipping concat"
fi

# ---- 5. QA (CPU only): contact strip + face-identity score ----
if [ -f "$OUT_DIR/final.mp4" ]; then
  mkdir -p "$OUT_DIR/qa"
  ffmpeg -v error -nostdin -y -i "$OUT_DIR/final.mp4" -vf "fps=1/2,scale=320:-1,tile=5x1" \
    -frames:v 1 "$OUT_DIR/qa/strip.jpg" 2>> "$LOG" && say "[detached] QA strip -> $OUT_DIR/qa/strip.jpg"
  LEAK=$(python3 -c "
import json
m=json.load(open('$MANIFEST'))
for i in m['items']:
    if i.get('images_raw'): print('$COMFY_IN/'+i['images_raw'][0]); break")
  if [ -x "$VENV_PY" ]; then
    "$VENV_PY" "$ROOT/scripts/face_tools.py" score \
      --ref "$ROOT/assets/character/face.jpg" ${LEAK:+--leak "$LEAK"} \
      --video "$OUT_DIR/final.mp4" --out "$OUT_DIR/qa/face_score.json" \
      2>/dev/null | tail -1 >> "$LOG" && say "[detached] face score -> $OUT_DIR/qa/face_score.json"
    for kfp in "$OUT_DIR"/keyframe*_cropped.png; do
      [ -f "$kfp" ] || continue
      "$VENV_PY" "$ROOT/scripts/face_tools.py" score \
        --ref "$ROOT/assets/character/face.jpg" ${LEAK:+--leak "$LEAK"} \
        --image "$kfp" --out "$OUT_DIR/qa/$(basename "${kfp%.png}")_face_score.json" \
        2>/dev/null | tail -1 >> "$LOG"
    done
  fi
fi

# ---- 6. hand the GPU back, then the garment check (needs the chat engine) ----
give_gpu_back
if [ -f "$OUT_DIR/final.mp4" ]; then
  mkdir -p "$OUT_DIR/qa"
  # one mid-segment frame per outfit (segments are 10 s each, in manifest order)
  SEGLEN=$(python3 -c "import json;o=json.load(open('$ROOT/config/style.json'))['output'];print(o['frames_per_segment']/o['fps'])")
  NSEG=$(ls -1 "$OUT_DIR"/out/*.mp4 2>/dev/null | wc -l)
  for ((n = 1; n <= NSEG; n++)); do
    T=$(python3 -c "print(($n-1)*$SEGLEN + $SEGLEN/2)")
    ffmpeg -v error -nostdin -y -ss "$T" -i "$OUT_DIR/final.mp4" -frames:v 1 "$OUT_DIR/qa/mid_frame_seg$n.png" 2>> "$LOG"
    if [ -f "$OUT_DIR/keyframe${n}_cropped.png" ] && [ ! -f "$OUT_DIR/qa/keyframe${n}_garment_check.json" ]; then
      garment_check "$OUT_DIR/keyframe${n}_cropped.png" "$OUT_DIR/qa/keyframe${n}_garment_check.json" "$n"
    fi
    [ -f "$OUT_DIR/qa/mid_frame_seg$n.png" ] && garment_check "$OUT_DIR/qa/mid_frame_seg$n.png" "$OUT_DIR/qa/video_seg${n}_garment_check.json" "$n"
  done
fi
if [ "$KEEP_COMFY" = "1" ]; then
  say "[detached] leaving ComfyUI running (--keep-comfy)"
fi
