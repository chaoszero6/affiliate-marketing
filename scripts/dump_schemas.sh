#!/usr/bin/env bash
# dump_schemas.sh — borrow the GPU just long enough to read Comfy's
# /object_info for the classes this pipeline uses, then hand the GPU back.
set -uo pipefail
LLM=strata-orca-iq3-xxs.service
COMFY_UNIT=comfyui.service
COMFY=http://127.0.0.1:8188
OUT=/root/affiliate-marketing/config/node_schemas.json
GRACE=${GRACE:-12}

trap 'systemctl stop "$COMFY_UNIT" 2>/dev/null; sleep 2; systemctl start "$LLM" 2>/dev/null' EXIT

sleep "$GRACE"
systemctl stop "$LLM" 2>/dev/null
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -ge 4000 ]; do sleep 2; done
systemctl start "$COMFY_UNIT" 2>/dev/null
for _ in $(seq 1 180); do
  curl -s --max-time 5 "$COMFY/object_info" 2>/dev/null | grep -q SaveVideo && break
  sleep 2
done

curl -s --max-time 30 "$COMFY/object_info" > /tmp/object_info_full.json
python3 - "$@" <<'PY'
import json
d = json.load(open('/tmp/object_info_full.json'))
want = ["UNETLoader","MiniMaxH3TurboLoRA","MiniMaxH3MemoryEfficientSageAttentionPatch",
        "CLIPLoader","VAELoader","LoadImage","MiniMaxH3ReferenceToVideo",
        "BasicGuider","RandomNoise","MiniMaxH3TurboSampler","BasicScheduler",
        "SamplerCustomAdvanced","VAEDecodeTiled","VAEDecodeAudio","CreateVideo",
        "SaveVideo"]
out = {}
for c in want:
    v = d.get(c)
    if not v:
        print("MISSING CLASS:", c); continue
    out[c] = {"required": v.get("required", {}), "optional": v.get("optional", {})}
    req = v.get("required", {})
    line = ", ".join(f"{k}={_fmt if isinstance(_fmt, list) else _fmt}" for k, _fmt in req.items())
    print(f"{c:42s} req: {list(req.keys())}")
json.dump(out, open('/root/affiliate-marketing/config/node_schemas.json','w'), indent=1, default=str)
PY
