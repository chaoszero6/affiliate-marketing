#!/usr/bin/env python3
"""fetch_dresses.py — turn product URLs into local dress reference images.

Myntra renders its gallery client-side, but the server HTML still exposes the
gallery image URLs, and its CDN accepts transform prefixes (h_1152,q_90,w_640)
that let us request the exact pixel canvas we generate at.

Images land in /opt/comfyui/ComfyUI/input/haul_<run-id>/<slug>/NN.jpg. The
manifest written on disk is what build_workflow.py consumes.

    fetch_dresses.py --run-id ID --manifest FILE <url|id> [...]
"""
import argparse
import json
import os
import re
import sys
import urllib.request

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")

IMG_RE = re.compile(
    r"(?:https?:)?//assets\.myntassets\.com/"
    r"(?:h_\d+,q_\d+,w_\d+/)?v1/assets/images/[A-Za-z0-9/_.-]+\.(?:jpg|jpeg|png|webp)",
    re.I,
)
TITLE_RE = re.compile(r'<meta property="og:title" content="([^"]+)"')
COMFY_INPUT = "/opt/comfyui/ComfyUI/input"


def fetch(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def slugify(title: str) -> str:
    t = re.sub(r"^buy\s+", "", title.lower())
    t = re.sub(r"[^a-z0-9]+", "_", t).strip("_")
    return t[:48] or "dress"


def base_url(u: str) -> str:
    """Strip any CDN transform prefix so we can re-apply our own."""
    u = re.sub(r"^https?:", "https:", u)
    return re.sub(r"assets\.myntassets\.com/h_\d+,q_\d+,w_\d+/",
                  "assets.myntassets.com/", u)


def transformed(u: str, width: int, height: int) -> str:
    return re.sub(r"assets\.myntassets\.com/",
                  f"assets.myntassets.com/h_{height},q_90,w_{width}/",
                  base_url(u))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("refs", nargs="+", help="product URLs or numeric ids")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--manifest", required=True, help="manifest json to write")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=1152)
    ap.add_argument("--max-images", type=int, default=2,
                    help="gallery images kept per dress (front, then detail)")
    a = ap.parse_args()

    out_root = os.path.join(COMFY_INPUT, f"haul_{a.run_id}")
    os.makedirs(out_root, exist_ok=True)
    items = []

    for ref in a.refs:
        m = re.search(r"(\d{6,})", ref)
        pid = m.group(1) if m else re.sub(r"\W", "", ref)[-8:]
        url = ref if ref.startswith("http") else \
            f"https://www.myntra.com/co-ords/x/x/{pid}/buy"
        try:
            html = fetch(url).decode("utf-8", "ignore")
        except Exception as e:
            print(f"warn: fetch {url}: {e}", file=sys.stderr)
            continue

        t = TITLE_RE.search(html)
        title = t.group(1) if t else "unknown outfit"
        seen, urls = set(), []
        for u in IMG_RE.findall(html):
            b = base_url(u)
            if b not in seen:
                seen.add(b)
                urls.append(b)

        slug = f"{slugify(title)}_{pid}"
        d = os.path.join(out_root, slug)
        os.makedirs(d, exist_ok=True)
        saved = []
        for i, u in enumerate(urls[: a.max_images]):
            try:
                blob = fetch(transformed(u, a.width, a.height))
                with open(os.path.join(d, f"{i:02d}.jpg"), "wb") as f:
                    f.write(blob)
                saved.append(f"haul_{a.run_id}/{slug}/{i:02d}.jpg")
            except Exception as e:
                print(f"warn: image {u}: {e}", file=sys.stderr)
        items.append({"id": pid, "title": title, "slug": slug, "images": saved})

    os.makedirs(os.path.dirname(a.manifest) or ".", exist_ok=True)
    json.dump({"run_id": a.run_id, "items": items}, open(a.manifest, "w"), indent=2)
    print(a.manifest)


if __name__ == "__main__":
    main()
