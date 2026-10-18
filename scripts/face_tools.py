#!/opt/comfyui/venv/bin/python
"""face_tools.py — CPU-only face utilities for the haul pipeline.

Two jobs, both run on the CPU so they never need the GPU handoff:

  mask   Blur every face out of dress reference photos. Myntra's catalogue
         shots show *their* model wearing the outfit; if her face reaches the
         video model as a reference it competes with our character's identity
         and usually wins. Blurring the face leaves the garment, the neckline
         and the drape intact while removing the competing identity.

  score  Measure how well a rendered video keeps our character's face. Samples
         one frame per second, finds the largest face, and reports the cosine
         similarity of its ArcFace embedding to the reference portrait. Same
         person ≈ 0.45+, different person ≈ 0.0-0.2. Optionally also reports
         similarity to the catalogue model's face so identity leakage from the
         dress references is visible as a number, not a hunch.

Detection uses insightface (buffalo_l) from the ComfyUI venv on the
CPUExecutionProvider; if that import fails we fall back to OpenCV's Haar
cascades for `mask` (no embeddings, so `score` needs insightface).

    face_tools.py mask  --suffix _masked <image> [<image> ...]
    face_tools.py score --ref face.jpg [--leak dress.jpg] --video final.mp4 \
                        [--out score.json] [--every 1.0]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

INSIGHT_ROOT = "/opt/comfyui/ComfyUI/models/insightface"
_app = None


def _insight():
    """Lazy insightface loader. Returns None when the package is unavailable."""
    global _app
    if _app is not None:
        return _app or None
    try:
        import warnings
        warnings.filterwarnings("ignore")
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", root=INSIGHT_ROOT,
                           providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _app = app
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"face_tools: insightface unavailable ({e}); Haar fallback",
              file=sys.stderr)
        _app = False
    return _app or None


def detect(img):
    """[(x1, y1, x2, y2, score, embedding_or_None)] largest first."""
    app = _insight()
    out = []
    if app is not None:
        for f in app.get(img):
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            emb = getattr(f, "normed_embedding", None)
            out.append((x1, y1, x2, y2, float(f.det_score), emb))
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for name in ("haarcascade_frontalface_default.xml",
                     "haarcascade_profileface.xml"):
            cas = cv2.CascadeClassifier(cv2.data.haarcascades + name)
            for (x, y, w, h) in cas.detectMultiScale(gray, 1.1, 5,
                                                     minSize=(24, 24)):
                out.append((int(x), int(y), int(x + w), int(y + h), 1.0, None))
            # mirrored pass catches left-facing profiles
            if name.startswith("haarcascade_profile"):
                W = gray.shape[1]
                for (x, y, w, h) in cas.detectMultiScale(cv2.flip(gray, 1),
                                                         1.1, 5, minSize=(24, 24)):
                    out.append((int(W - x - w), int(y), int(W - x), int(y + h),
                                1.0, None))
    out.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return out


def blur_box(img, box, grow=0.45):
    """Pixelate + blur an expanded face box in place. `grow` widens the box so
    hairline, ears and jaw go with it; the top grows a little more to catch the
    forehead/hair without eating the neckline below."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box[:4]
    bw, bh = x2 - x1, y2 - y1
    X1 = max(0, int(x1 - bw * grow))
    X2 = min(w, int(x2 + bw * grow))
    Y1 = max(0, int(y1 - bh * (grow + 0.25)))
    Y2 = min(h, int(y2 + bh * grow * 0.6))
    if X2 - X1 < 4 or Y2 - Y1 < 4:
        return
    roi = img[Y1:Y2, X1:X2]
    small = cv2.resize(roi, (max(2, (X2 - X1) // 14), max(2, (Y2 - Y1) // 14)),
                       interpolation=cv2.INTER_LINEAR)
    pix = cv2.resize(small, (X2 - X1, Y2 - Y1), interpolation=cv2.INTER_LINEAR)
    k = max(9, ((X2 - X1) // 6) | 1)
    img[Y1:Y2, X1:X2] = cv2.GaussianBlur(pix, (k, k), 0)
    # soften the seam so the edit does not read as a hard rectangle
    pad = max(6, k // 2)
    PX1, PY1 = max(0, X1 - pad), max(0, Y1 - pad)
    PX2, PY2 = min(w, X2 + pad), min(h, Y2 + pad)
    edge = img[PY1:PY2, PX1:PX2]
    img[PY1:PY2, PX1:PX2] = cv2.GaussianBlur(edge, (pad | 1, pad | 1), 0)


def cmd_mask(a):
    results = []
    for p in a.images:
        img = cv2.imread(p)
        if img is None:
            results.append({"src": p, "error": "unreadable"})
            continue
        faces = detect(img)
        for f in faces:
            blur_box(img, f, a.grow)
        root, ext = os.path.splitext(p)
        dst = root + a.suffix + (ext or ".jpg")
        cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 94])
        results.append({"src": p, "dst": dst, "faces": len(faces)})
        print(f"  masked {os.path.basename(p)}: {len(faces)} face(s)",
              file=sys.stderr)
    print(json.dumps(results))


def _frames(video, every):
    """Yield BGR frames sampled every `every` seconds via ffmpeg."""
    tmp = tempfile.mkdtemp(prefix="facescore_")
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", video,
                    "-vf", f"fps=1/{every}", os.path.join(tmp, "f_%03d.jpg")],
                   check=False)
    for n in sorted(os.listdir(tmp)):
        yield n, cv2.imread(os.path.join(tmp, n))


def _faces_upscaled(img, min_face=160):
    """Detect, then re-detect on a crop upscaled so the face is >= min_face px.
    ArcFace embeddings of a 60-80 px face (a full-body 640x1152 frame) are
    noisy; a 2-4x bicubic crop gives a usable embedding."""
    faces = [f for f in detect(img) if f[5] is not None]
    if not faces:
        return []
    x1, y1, x2, y2 = faces[0][:4]
    fw = max(1, x2 - x1)
    if fw >= min_face:
        return faces
    s = min(4.0, min_face / fw)
    h, w = img.shape[:2]
    pad = int(fw * 1.2)
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = img[cy1:cy2, cx1:cx2]
    big = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    again = [f for f in detect(big) if f[5] is not None]
    return again or faces


def _embedding(path):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"cannot read {path}")
    faces = [f for f in detect(img) if f[5] is not None]
    if not faces:
        sys.exit(f"no face found in {path}")
    return faces[0][5]


def cmd_score(a):
    if _insight() is None:
        sys.exit("score needs insightface (run with /opt/comfyui/venv/bin/python)")
    ref = _embedding(a.ref)
    leak = _embedding(a.leak) if a.leak else None
    per = []
    if a.image:
        source = [(os.path.basename(a.image), cv2.imread(a.image))]
    else:
        source = _frames(a.video, a.every)
    for name, img in source:
        if img is None:
            continue
        faces = _faces_upscaled(img)
        row = {"frame": name, "face": bool(faces)}
        if faces:
            e = faces[0][5]
            row["sim_ref"] = round(float(np.dot(ref, e)), 3)
            if leak is not None:
                row["sim_leak"] = round(float(np.dot(leak, e)), 3)
        per.append(row)
    sims = [r["sim_ref"] for r in per if "sim_ref" in r]
    leaks = [r["sim_leak"] for r in per if "sim_leak" in r]
    summary = {
        "video": a.video or a.image, "ref": a.ref, "leak_ref": a.leak,
        "frames_sampled": len(per), "frames_with_face": len(sims),
        "sim_ref_mean": round(float(np.mean(sims)), 3) if sims else None,
        "sim_ref_min": round(float(np.min(sims)), 3) if sims else None,
        "sim_ref_max": round(float(np.max(sims)), 3) if sims else None,
        "sim_leak_mean": round(float(np.mean(leaks)), 3) if leaks else None,
        "verdict": ("match" if sims and np.mean(sims) >= a.threshold else
                    "MISMATCH" if sims else "no-face"),
        "threshold": a.threshold,
        "frames": per,
    }
    if a.out:
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "frames"}))


def cmd_frame(a):
    """Crop a try-on still to the video's framing: a 9:16 window with the head
    near the top edge and the feet at the bottom, centred on the person.
    Qwen Image Edit copies the room photo's wide composition however the
    prompt is worded; H3 then 'pushes in' from that wide first frame to its
    natural full-body framing. Cropping the keyframe first removes the zoom."""
    img = cv2.imread(a.input)
    if img is None:
        sys.exit(f"cannot read {a.input}")
    H, W = img.shape[:2]
    faces = detect(img)
    if not faces:
        print(json.dumps({"input": a.input, "output": a.output, "cropped": False,
                          "reason": "no face"}))
        cv2.imwrite(a.output, img)
        return
    x1, y1, x2, y2 = faces[0][:4]
    fh = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0
    top = max(0, int(y1 - a.head_room * fh))
    bottom = H  # product shots stand on the floor at the bottom of the frame
    ch = bottom - top
    cw = int(round(ch * a.width / a.height))
    if cw > W:  # image too narrow for that height: shrink the window
        cw = W
        ch = int(round(cw * a.height / a.width))
        top = max(0, min(top, H - ch))
        bottom = top + ch
    left = int(round(cx - cw / 2.0))
    left = max(0, min(left, W - cw))
    crop = img[top:bottom, left:left + cw]
    crop = cv2.resize(crop, (a.width, a.height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(a.output, crop)
    print(json.dumps({"input": a.input, "output": a.output, "cropped": True,
                      "box": [left, top, left + cw, bottom], "src": [W, H]}))


def cmd_head(a):
    """Crop a portrait to head and hair only, cut just below the chin, so an
    image-edit model given it as the identity reference has no clothing (and
    no clothing colour) to copy from it."""
    img = cv2.imread(a.input)
    if img is None:
        sys.exit(f"cannot read {a.input}")
    H, W = img.shape[:2]
    faces = detect(img)
    if not faces:
        cv2.imwrite(a.output, img)
        print(json.dumps({"input": a.input, "output": a.output, "cropped": False}))
        return
    x1, y1, x2, y2 = faces[0][:4]
    fw, fh = x2 - x1, y2 - y1
    left = max(0, int(x1 - fw * a.side))
    right = min(W, int(x2 + fw * a.side))
    top = max(0, int(y1 - fh * a.above))
    bottom = min(H, int(y2 + fh * a.below))
    crop = img[top:bottom, left:right].copy()
    if a.mask_outside:
        # everything outside an ellipse around the head becomes the backdrop
        # colour (median of the top rows), so no clothing pixel survives
        ch, cw = crop.shape[:2]
        bg = np.median(crop[: max(2, ch // 12)].reshape(-1, 3), axis=0).astype(crop.dtype)
        mask = np.zeros((ch, cw), dtype=np.uint8)
        cx = int((x1 + x2) / 2 - left)
        cy = int((y1 + y2) / 2 - top - fh * 0.08)
        cv2.ellipse(mask, (cx, cy), (int(fw * 0.56), int(fh * 0.86)), 0, 0, 360, 255, -1)
        # nothing below the chin line survives either (collars, necklines)
        chin = int(y2 - top + fh * 0.06)
        if 0 < chin < ch:
            mask[chin:, :] = 0
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        crop = (crop.astype(np.float32) * alpha
                + np.full_like(crop, bg, dtype=np.float32) * (1 - alpha)).astype(crop.dtype)
    cv2.imwrite(a.output, crop)
    print(json.dumps({"input": a.input, "output": a.output, "cropped": True,
                      "box": [left, top, right, bottom], "src": [W, H],
                      "masked_outside": bool(a.mask_outside)}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    hd = sub.add_parser("head")
    hd.add_argument("--input", required=True)
    hd.add_argument("--output", required=True)
    hd.add_argument("--side", type=float, default=0.55, help="margin beside the face, in face widths")
    hd.add_argument("--above", type=float, default=0.75, help="margin above the face (hair), in face heights")
    hd.add_argument("--below", type=float, default=0.12, help="margin below the chin, in face heights")
    hd.add_argument("--mask-outside", type=int, default=1,
                    help="1 = fill everything outside the head ellipse with the backdrop colour")
    fr = sub.add_parser("frame")
    fr.add_argument("--input", required=True)
    fr.add_argument("--output", required=True)
    fr.add_argument("--width", type=int, default=640)
    fr.add_argument("--height", type=int, default=1152)
    fr.add_argument("--head-room", type=float, default=0.9,
                    help="space above the face, in face heights")
    m = sub.add_parser("mask")
    m.add_argument("images", nargs="+")
    m.add_argument("--suffix", default="_masked")
    m.add_argument("--grow", type=float, default=0.45)
    s = sub.add_parser("score")
    s.add_argument("--ref", required=True, help="our character's portrait")
    s.add_argument("--leak", help="catalogue model's photo (unmasked) to "
                                  "measure identity leakage")
    s.add_argument("--video")
    s.add_argument("--image", help="score a single still instead of a video")
    s.add_argument("--out")
    s.add_argument("--every", type=float, default=1.0)
    s.add_argument("--threshold", type=float, default=0.40)
    a = ap.parse_args()
    {"mask": cmd_mask, "score": cmd_score, "frame": cmd_frame,
     "head": cmd_head}[a.cmd](a)


if __name__ == "__main__":
    main()
