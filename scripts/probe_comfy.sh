#!/usr/bin/env bash
# probe.sh — validate the most recent built workflow against a live ComfyUI and
# print the raw /prompt response, so a 400 validation error is visible instead of
# being swallowed by the orchestrator.
GRACE=${GRACE:-20}
COMFY="http://127.0.0.1:8188"
RUN_DIR=$(ls -1dt /root/affiliate-marketing/runs/*/ 2>/dev/null | head -1)

trap 'systemctl start strata-orca-iq3-xxs.service 2>/dev/null' EXIT

echo "grace ${GRACE}s"
sleep "$GRACE"
systemctl stop strata-orca-iq3-xxs.service 2>/dev/null || true
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -ge 4000 ]; do sleep 2; done
systemctl start comfyui.service 2>/dev/null

# wait for the server to finish loading nodes, not just to open the port
for i in $(seq 1 120); do
  if curl -s --max-time 4 "$COMFY/object_info/MiniMaxH3ReferenceToVideo" 2>/dev/null \
       | grep -q 'MiniMaxH3ReferenceToVideo'; then
    echo "comfy fully loaded after $((i * 2))s"
    break
  fi
  sleep 2
done

echo "--- validating ${RUN_DIR}seg1.json ---"
curl -s -o /tmp/probe_resp.json -w 'HTTP %{http_code}\n' -X POST "$COMFY/prompt" \
  -H 'Content-Type: application/json' -d @"${RUN_DIR}seg1.json"
python3 - <<'PY'
import json
d = json.load(open('/tmp/probe_resp.json'))
if 'prompt_id' in d:
    print('OK prompt_id =', d['prompt_id'])
else:
    print(json.dumps(d, indent=2)[:4000])
PY
