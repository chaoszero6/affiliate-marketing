#!/usr/bin/env python3
"""garment_check.py — ask the local vision model whether a generated image shows
the *same garment* as the product photo.

The face score tells us whether the woman is ours; this tells us whether the
dress is the product. It runs on the CPU/network side (the chat engine on
:8322), so render.sh calls it after the engine has been handed the GPU back.

    garment_check.py --product 00.jpg [--product 01.jpg] --image keyframe.png \
                     [--out check.json]

Prints one JSON line: {"match": bool, "score": 0-10, "differences": [...]}
Exit 0 always (it is a report, not a gate) unless the engine is unreachable.
"""
import argparse
import base64
import json
import re
import sys
import time
import urllib.request

VLM_URL = "http://127.0.0.1:8322/v1/chat/completions"
VLM_MODEL = "flash-next-uncensored-iq3_xxs"

PROMPT = (
    "You are a strict quality checker for an online fashion store. The FIRST "
    "image(s) are the official product photos of an outfit. The LAST image is a "
    "generated picture of a different woman that is supposed to be wearing EXACTLY "
    "that outfit. Compare only the garments, not the person, pose, lighting or "
    "background. Check: garment type; length on the body; colour; neckline shape "
    "and trim; sleeve length, shape and cuffs; where embroidery/print/trim is and "
    "where the fabric is plain; the bottoms; any added or missing piece (dupatta, "
    "belt, border). Answer with ONLY a JSON object, no prose, in this form: "
    '{"match": true or false, "score": 0-10, "differences": ["...", "..."]}. '
    "match=true only if a shopper would recognise it as the same product with no "
    "misleading difference; score 10 = identical, 7-9 = minor texture/density "
    "differences, below 7 = a visible design difference. List each difference in "
    "one short sentence, or an empty list if none."
)


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def wait_for_engine(max_wait=300):
    deadline = time.time() + max_wait
    url = VLM_URL.replace("/chat/completions", "/models")
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if (json.loads(r.read()).get("data") or []):
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", action="append", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()

    if not wait_for_engine():
        print(json.dumps({"error": "chat engine unreachable"}))
        sys.exit(1)

    content = [{"type": "text", "text": PROMPT}]
    for p in a.product[:2]:
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + b64(p)}})
    content.append({"type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + b64(a.image)}})
    body = {"model": VLM_MODEL, "max_tokens": 800, "temperature": 0.0, "seed": 7,
            "messages": [{"role": "user", "content": content}],
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "low"}
    req = urllib.request.Request(VLM_URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    text = (r["choices"][0]["message"].get("content") or "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    try:
        verdict = json.loads(m.group(0)) if m else {"raw": text}
    except Exception:
        verdict = {"raw": text}
    verdict.update({"image": a.image, "products": a.product[:2]})
    if a.out:
        with open(a.out, "w") as f:
            json.dump(verdict, f, indent=1)
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
