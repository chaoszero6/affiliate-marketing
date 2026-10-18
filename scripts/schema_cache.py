#!/usr/bin/env python3
"""schema_cache.py — the GPU-handoff readiness gate + schema cache.

ComfyUI opens its port long before it finishes importing custom nodes. A client
that treats "port open" as "ready" queues against a half-built server and gets
HTTP 400 / required_input_missing. This script polls /object_info until the
node classes report *populated* required maps, then caches them.

Schema layout (this ComfyUI version): each class dict has
    {"input": {"required": {...}, "optional": {...}, "hidden": {...}},
     "output": [...], ...}
We read input.required / input.optional — NOT the top-level "required" key.

    schema_cache.py [--timeout 420] [--out FILE]

Exit 0 = Comfy finished loading and the cache is written.
Exit 1 = it never finished loading within the timeout.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

COMFY = "http://127.0.0.1:8188"
# If these classes report non-empty required maps, the registry is done loading.
PROBES = ["SaveVideo", "VAEDecodeTiled", "CreateVideo",
          "MiniMaxH3ReferenceToVideo", "BasicScheduler"]
UA = {"User-Agent": "haul-pipeline/1.0"}


def fetch(timeout=15):
    req = urllib.request.Request(f"{COMFY}/object_info", headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def required_inputs(spec):
    """return {input: default} for slots the graph may not set itself.

    Raw /object_info gives each input as [token_or_default, {default, ...}].
    We keep:
      - inputs with a real scalar default (int/float/bool/str): steps, denoise,
        low_vram, width, ...
      - combo (dropdown) inputs: their first option is the sensible default
        (e.g. weight_dtype -> "default", scheduler -> "simple").
    Connection inputs (links: MODEL/LATENT/IMAGE/VAE/CLIP/AUDIO/VIDEO/CONDITIONING)
    carry no scalar default — they are wired explicitly by the graph, so skip.
    """
    inp = spec.get("input") if isinstance(spec, dict) else None
    if not isinstance(inp, dict):
        return {}
    req = inp.get("required") or {}
    out = {}
    for k, v in req.items():
        if not isinstance(v, list) or not v:
            continue
        meta = v[1] if len(v) > 1 and isinstance(v[1], dict) else {}
        d = meta.get("default")
        first = v[0]
        if isinstance(d, (int, float, bool, str)):
            out[k] = d
        elif isinstance(first, list) and first:
            # combo / dropdown: first option is the default choice
            out[k] = first[0] if isinstance(first[0], (str, int, float, bool)) \
                else first[0]
    return out


def ready(d):
    """True when every probe class reports a non-empty required input map."""
    try:
        for c in PROBES:
            if not required_inputs(d[c]):
                return False
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--out",
                    default="/root/affiliate-marketing/config/node_schemas.json")
    a = ap.parse_args()

    deadline = time.time() + a.timeout
    while time.time() < deadline:
        try:
            d = fetch()
        except Exception:
            time.sleep(3)
            continue
        if ready(d):
            out = {}
            for cls, spec in d.items():
                if not isinstance(spec, dict):
                    continue
                m = required_inputs(spec)
                if m:
                    out[cls] = m
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump(out, open(a.out, "w"), indent=1)
            print(f"comfy ready — cached {len(out)} node schemas")
            return 0
        time.sleep(3)
    print("comfy did not finish loading in time", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())