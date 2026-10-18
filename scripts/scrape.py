#!/usr/bin/env python3
"""scrape.py — turn product URLs (or local photos) into dress reference images
plus a factual outfit description.

Pure network + CPU; runs while the chat engine still owns the GPU.

Myntra renders the gallery client-side, but the server HTML embeds the whole
product record as `window.__myx = {...}`. We decode that JSON and read:

  pdpData.media.albums[0].images[].imageURL   every gallery photo, in order
  pdpData.name / brand / baseColour            what the outfit is
  pdpData.articleAttributes                    neck, sleeves, hemline, fabric...
  pdpData.productDetails[0].description        the human-readable detail list

The attributes become `outfit_phrase`: a description of the garment that is
*read from the listing*, never invented. Pass --hint to override it per URL.

Every gallery photo shows Myntra's model wearing the outfit. Her face would
compete with our character's identity, so each photo is also saved with the
face blurred (scripts/face_tools.py, CPU). The manifest lists the masked copies
under `images` (what the video model sees) and the originals under `images_raw`
(for QA leakage scoring).

Refs may also be local image files or directories (when Myntra blocks the
server or the user pastes photos): they are copied and masked the same way.

    scrape.py --run-id ID --manifest FILE [--hint PHRASE ...] [--max-images 4]
              <url | product-id | image.jpg | folder> [...]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = "/root/affiliate-marketing"
COMFY_INPUT = "/opt/comfyui/ComfyUI/input"
VENV_PY = "/opt/comfyui/venv/bin/python"
FACE_TOOLS = os.path.join(ROOT, "scripts", "face_tools.py")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")

# fallback when the JSON blob is missing: any gallery asset URL in the HTML
IMG_RE = re.compile(
    r"(?:https?:)?//assets\.myntassets\.com/"
    r"(?:h_\d+,q_\d+,w_\d+/)?v1/assets/images/[A-Za-z0-9/_.-]+\.(?:jpg|jpeg|png|webp)",
    re.I,
)
TITLE_RE = re.compile(r'<meta property="og:title" content="([^"]+)"')
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


# --------------------------------------------------------------------------- net
def fetch(url, timeout=30):
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def base_url(u):
    """Strip any CDN transform prefix so we can apply our own."""
    u = re.sub(r"^(https?:)?//", "https://", u)
    return re.sub(r"assets\.myntassets\.com/(h_\d+,q_\d+,w_\d+/)+(v1/)?",
                  "assets.myntassets.com/", u)


def cdn(u, width, height):
    """Myntra's CDN resizes on request: h_<H>,q_<Q>,w_<W>/ before the path."""
    return re.sub(r"assets\.myntassets\.com/",
                  f"assets.myntassets.com/h_{height},q_95,w_{width}/",
                  base_url(u))


# --------------------------------------------------------------------- product
def myx_blob(html):
    """The embedded product record, or None."""
    m = re.search(r"window\.__myx\s*=\s*", html)
    if not m:
        return None
    try:
        d, _ = json.JSONDecoder().raw_decode(html[m.end():])
        return d.get("pdpData") or None
    except Exception:
        return None


def _clean(s):
    s = re.sub(r"<[^>]+>", " ", s or "")
    return re.sub(r"\s+", " ", s).strip()


def _val(attrs, *keys):
    for k in keys:
        v = (attrs.get(k) or "").strip()
        if v and v.upper() not in ("NA", "N/A", "NONE", "-", ""):
            return v
    return ""


def outfit_phrase(pdp):
    """Build the garment description from the listing's own attributes.

    Works across Myntra categories: co-ord sets, kurta sets, kurtis, dresses.
    Every clause is dropped when its attribute is absent, so nothing is made up.
    """
    attrs = pdp.get("articleAttributes") or {}
    name = _clean(pdp.get("name") or "")
    colour = (pdp.get("baseColour") or _val(attrs, "Colour", "Color")).lower()
    article = ((pdp.get("analytics") or {}).get("articleType") or "").lower()

    lo = lambda s: s.lower()
    parts = []

    # ----- top / main garment
    top_type = _val(attrs, "Top Type", "Type", "Kurta Type", "Dress Type")
    if not top_type:
        # fall back to the product name's noun ("... Tunic with Trousers")
        m = re.search(r"(kurta|kurti|tunic|dress|top|shirt|gown|jumpsuit|"
                      r"blouse|shrug|jacket|co-?ord)", name, re.I)
        top_type = m.group(1) if m else (article or "outfit")
    pattern = _val(attrs, "Top Pattern", "Pattern", "Print or Pattern Type")
    desc = f"a {colour} {lo(pattern)} {lo(top_type)}".replace("  ", " ")
    feats = []
    neck = _val(attrs, "Neck")
    if neck:
        feats.append(f"a {lo(neck)} neckline")
    sl = _val(attrs, "Sleeve Length")
    if sl:
        st = _val(attrs, "Sleeve Styling")
        feats.append(f"{lo(sl)}" + (f" ({lo(st)})" if st and st.lower() != "regular sleeves" else ""))
    hem = _val(attrs, "Top Hemline", "Hemline")
    if hem:
        feats.append(f"a {lo(hem)} hemline")
    length = _val(attrs, "Top Length", "Length")
    if length:
        feats.append(f"{lo(length)} length")
    shape = _val(attrs, "Top Shape", "Shape")
    if shape:
        feats.append(f"{lo(shape)} shape")
    slit = _val(attrs, "Slit Detail")
    if slit:
        feats.append(lo(slit))
    fab = _val(attrs, "Top Fabric", "Fabric")
    if fab:
        feats.append(f"in {lo(fab)}")
    if feats:
        desc += " with " + ", ".join(feats)
    parts.append(desc)

    # ----- bottom
    bt = _val(attrs, "Bottom Type")
    if bt:
        bp = _val(attrs, "Bottom Pattern")
        bd = f"matching {colour} {lo(bp)} {lo(bt)}".replace("  ", " ")
        bf = []
        bc = _val(attrs, "Bottom Closure")
        if bc:
            bf.append(f"a {lo(bc)} waist")
        bh = _val(attrs, "Bottom Hemline")
        if bh:
            bf.append(f"a {lo(bh)} hem")
        bfab = _val(attrs, "Bottom Fabric")
        if bfab and bfab != fab:
            bf.append(f"in {lo(bfab)}")
        if bf:
            bd += " with " + " and ".join(bf)
        parts.append(bd)

    # ----- dupatta: only when the listing says there is one
    dup = _val(attrs, "Dupatta", "Dupatta Fabric", "With Dupatta")
    contains = _val(attrs, "Package Contains")
    has_dupatta = (dup and dup.lower() not in ("without dupatta", "no")) or \
                  ("dupatta" in contains.lower())
    if has_dupatta:
        dfab = _val(attrs, "Dupatta Fabric")
        parts.append("a " + (lo(dfab) + " " if dfab else "") + "dupatta")

    phrase = "; ".join(parts)
    if contains:
        phrase += f". The set contains exactly: {contains.strip(' ;')}"
        if not has_dupatta:
            phrase += " — no dupatta, no scarf, no extra layers"
    return phrase


def product_details(pdp):
    for d in pdp.get("productDetails") or []:
        if (d.get("title") or "").lower().startswith("product details"):
            return _clean((d.get("description") or "").replace("<br>", "; "))
    return ""


# ------------------------------------------------------------------ caption
VLM_URL = "http://127.0.0.1:8322/v1/chat/completions"
VLM_MODEL = "flash-next-uncensored-iq3_xxs"
CAPTION_PROMPT = (
    "You are writing a factual garment spec for an image-generation model. Look at "
    "all the photos (same outfit, different angles). Describe ONLY the clothing in "
    "100-150 words of plain prose, in this order: (1) garment type and exact length "
    "on the body (hip / upper thigh / mid-thigh / knee); (2) colour and fabric look; "
    "(3) neckline shape and its trim, if any; (4) sleeve length, sleeve shape and cuff "
    "style; (5) DECORATION PLACEMENT: name every place that has embroidery, print or "
    "trim and what it looks like, then name the places that are PLAIN fabric with no "
    "decoration (for example: 'the tunic body, hem and neckline are plain black'); "
    "(6) the bottoms: type, shape, length and any hem detail; (7) CONSTRUCTION: say "
    "whether the top is a single layer of fabric or has a jacket, shrug, overlay, "
    "wrap front, open front or visible inner garment, and whether it is straight, "
    "A-line or flared; (8) what the set does NOT include (dupatta, belt, jacket, "
    "second layer) if a viewer might assume it. Use plain, "
    "literal words. No adjectives like elegant, intricate, heavily, delicate. No "
    "bullet points, no headings. Do not describe the model, her pose, her face, the "
    "background or the mirror. Do not guess hidden details."
)


def wait_for_engine(max_wait=240):
    """The chat engine is restarted after every render and takes a minute or
    two to load its weights. Poll /v1/models until it reports a loaded model
    (or the port at least answers) so a run started right after another one
    does not lose the visual caption to a 'connection refused'."""
    import time
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(VLM_URL.replace("/chat/completions", "/models"),
                                        timeout=5) as r:
                d = json.loads(r.read())
            models = d.get("data") or []
            if any((m.get("status") or {}).get("value", "loaded") == "loaded"
                   for m in models) or models:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def caption_garment(paths, listing_phrase="", timeout=240):
    """Ask the local vision-capable chat engine to describe the garments in the
    (unmasked) product photos. Returns '' when the engine is unavailable, so the
    caller falls back to the listing attributes. Runs in Phase 1, while the
    engine still owns the GPU."""
    import base64
    if not paths:
        return ""
    if not wait_for_engine():
        print("warn: chat engine did not come up in time; captioning anyway",
              file=sys.stderr)
    content = [{"type": "text", "text": CAPTION_PROMPT + (
        f"\n\nFor reference, the listing's own attributes say: {listing_phrase}. "
        "Trust the photos over the attributes where they differ."
        if listing_phrase else "")}]
    for p in paths[:3]:
        with open(p, "rb") as f:
            b = base64.b64encode(f.read()).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + b}})
    body = {"model": VLM_MODEL, "max_tokens": 2000, "temperature": 0.0, "seed": 7,
            "messages": [{"role": "user", "content": content}],
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "low"}
    req = urllib.request.Request(VLM_URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        msg = r["choices"][0]["message"]
        text = (msg.get("content") or "").strip()
        if not text:
            # thinking could not be disabled; salvage the tail of the reasoning
            text = (msg.get("reasoning_content") or "").strip()[-1200:]
        return re.sub(r"\s+", " ", text)
    except Exception as e:
        print(f"warn: garment caption unavailable ({e}); using listing "
              f"attributes only", file=sys.stderr)
        return ""


# ---------------------------------------------------------------------- refs
def is_local(ref):
    return os.path.exists(ref)


def local_images(ref):
    if os.path.isdir(ref):
        return sorted(os.path.join(ref, f) for f in os.listdir(ref)
                      if f.lower().endswith(IMG_EXT))
    return [ref] if ref.lower().endswith(IMG_EXT) else []


def slugify(s):
    s = re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")
    return s[:48] or "dress"


def mask_faces(paths):
    """Blur faces via face_tools.py (venv python preferred). Returns
    {src: dst}; on failure maps each src to itself so the run still proceeds."""
    if not paths:
        return {}
    py = VENV_PY if os.path.exists(VENV_PY) else sys.executable
    try:
        out = subprocess.run([py, FACE_TOOLS, "mask", *paths],
                             capture_output=True, text=True, timeout=600)
        if out.stderr.strip():
            print(out.stderr.strip(), file=sys.stderr)
        rows = json.loads(out.stdout.strip().splitlines()[-1])
        return {r["src"]: (r.get("dst", r["src"]), int(r.get("faces", -1)))
                for r in rows}
    except Exception as e:
        print(f"warn: face masking failed ({e}); using unmasked references",
              file=sys.stderr)
        return {p: (p, -1) for p in paths}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("refs", nargs="+", help="product URLs, numeric ids, "
                                            "image files or folders")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1440)
    ap.add_argument("--max-images", type=int, default=4,
                    help="gallery photos kept per outfit, gallery order "
                         "(front, back, detail...)")
    ap.add_argument("--hint", action="append", default=[],
                    help="override the outfit phrase for the Nth ref")
    ap.add_argument("--no-mask", action="store_true",
                    help="keep the catalogue model's face in the references")
    ap.add_argument("--no-caption", action="store_true",
                    help="skip the vision-model garment description")
    a = ap.parse_args()

    out_root = os.path.join(COMFY_INPUT, f"haul_{a.run_id}")
    os.makedirs(out_root, exist_ok=True)
    items = []

    for j, ref in enumerate(a.refs):
        item = {"source": ref, "images": [], "images_raw": []}
        saved = []

        if is_local(ref):
            # ---- user-supplied photos
            pid = slugify(os.path.basename(ref.rstrip("/")))
            slug = f"local_{pid}"
            d = os.path.join(out_root, slug)
            os.makedirs(d, exist_ok=True)
            for i, src in enumerate(local_images(ref)[: a.max_images]):
                dst = os.path.join(d, f"{i:02d}{os.path.splitext(src)[1].lower()}")
                shutil.copy(src, dst)
                saved.append(dst)
            item.update({"id": pid, "title": os.path.basename(ref), "slug": slug,
                         "brand": "", "colour": "", "attributes": {},
                         "details": ""})
            phrase = ""
        else:
            # ---- Myntra product page
            m = re.search(r"(\d{6,})", ref)
            pid = m.group(1) if m else re.sub(r"\W", "", ref)[-8:]
            url = ref if ref.startswith("http") else \
                f"https://www.myntra.com/co-ords/x/x/{pid}/buy"
            try:
                html = fetch(url).decode("utf-8", "ignore")
            except urllib.error.HTTPError as e:
                print(f"error: Myntra answered HTTP {e.code} for {url} — "
                      f"the page may be blocked for this IP. Pass the product "
                      f"photos as local files instead.", file=sys.stderr)
                item.update({"id": pid, "error": f"HTTP {e.code}"})
                items.append(item)
                continue
            except Exception as e:
                print(f"error: fetch {url}: {e}", file=sys.stderr)
                item.update({"id": pid, "error": str(e)})
                items.append(item)
                continue

            pdp = myx_blob(html)
            if pdp:
                title = _clean(pdp.get("name") or "")
                urls = [im.get("imageURL") or im.get("secureSrc") or ""
                        for alb in (pdp.get("media") or {}).get("albums") or []
                        if alb.get("name") in (None, "default")
                        for im in alb.get("images") or []]
                urls = [base_url(u) for u in urls if u]
                phrase = outfit_phrase(pdp)
                item.update({
                    "brand": (pdp.get("brand") or {}).get("name", ""),
                    "colour": pdp.get("baseColour", ""),
                    "attributes": pdp.get("articleAttributes") or {},
                    "details": product_details(pdp),
                    "article_type": (pdp.get("analytics") or {}).get("articleType", ""),
                })
            else:
                t = TITLE_RE.search(html)
                title = _clean(t.group(1) if t else "unknown outfit")
                title = re.sub(r"(?i)^buy\s+", "", title)
                title = re.sub(r"\s*-\s*-\s*.*$", "", title)
                seen, urls = set(), []
                for u in IMG_RE.findall(html):
                    b = base_url(u)
                    if b not in seen:
                        seen.add(b)
                        urls.append(b)
                phrase = re.sub(r"[^A-Za-z ]+", " ", title).strip().lower()
                item.update({"brand": "", "colour": "", "attributes": {},
                             "details": "", "article_type": ""})
                print("warn: product JSON not found; using og:title and "
                      "HTML image scan", file=sys.stderr)

            slug = f"{slugify(title)}_{pid}"
            d = os.path.join(out_root, slug)
            os.makedirs(d, exist_ok=True)
            for i, u in enumerate(urls[: a.max_images]):
                try:
                    blob = fetch(cdn(u, a.width, a.height))
                    p = os.path.join(d, f"{i:02d}.jpg")
                    with open(p, "wb") as f:
                        f.write(blob)
                    saved.append(p)
                except Exception as e:
                    print(f"warn: image {u}: {e}", file=sys.stderr)
            item.update({"id": pid, "title": title, "slug": slug,
                         "gallery_total": len(urls)})

        # ---- mask the catalogue model's face
        masked = {p: (p, -1) for p in saved} if a.no_mask else mask_faces(saved)
        rel = lambda p: os.path.relpath(p, COMFY_INPUT)
        item["images_raw"] = [rel(p) for p in saved]
        item["images"] = [rel(masked.get(p, (p, -1))[0]) for p in saved]
        # faces per photo: 0 = a close-up / flat-lay with no model's head,
        # which build_keyframe.py uses as the detail panel
        item["face_counts"] = [masked.get(p, (p, -1))[1] for p in saved]

        # ---- describe the garment from the photos (vision model), with the
        # listing attributes as a cross-check; the user's --hint wins outright
        hint = a.hint[j].strip() if j < len(a.hint) else ""
        caption = "" if (hint or a.no_caption) else caption_garment(saved, phrase)
        item["listing_phrase"] = phrase
        item["garment_caption"] = caption
        if hint:
            item["prompt_hint"], item["prompt_hint_source"] = hint, "user"
        elif caption:
            item["prompt_hint"], item["prompt_hint_source"] = caption, "vision"
        elif phrase:
            item["prompt_hint"], item["prompt_hint_source"] = phrase, "listing"
        else:
            item["prompt_hint"] = "the outfit in the reference photos"
            item["prompt_hint_source"] = "none"
        items.append(item)
        print(f"  {item['slug']}: {len(saved)} reference photo(s)"
              + (f" of {item.get('gallery_total')}" if item.get("gallery_total") else ""))
        print(f"    outfit ({item['prompt_hint_source']}): {item['prompt_hint']!r}")

    os.makedirs(os.path.dirname(a.manifest) or ".", exist_ok=True)
    with open(a.manifest, "w") as f:
        json.dump({"run_id": a.run_id, "items": items}, f, indent=2)
    ok = [i for i in items if i.get("images")]
    if not ok:
        print("error: no reference images for any outfit", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
