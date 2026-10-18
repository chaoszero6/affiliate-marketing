#!/usr/bin/env python3
"""build_keyframe.py — emit a Qwen Image Edit 2511 "try-on" workflow per outfit.

Why a keyframe at all: MiniMax H3 treats reference images as *guidance*. Given
four photos of a short embroidered tunic it still drifted to its prior for
"Indian woman, black embroidered top" — a knee-length kurta. Affiliate videos
need the exact product, so we first make one still image of our woman wearing
the exact garments in our room, then hand that still to H3 as frame 0 (a hard
pixel anchor) *and* as a reference picture. Qwen Image Edit 2511 is the model
that built the character sheet on this box; it is good at "person from image 1
wearing the clothes from image 2 in the place from image 3".

Graph (same chain scripts/make_sheet.py already proved on this server):

    UNETLoader -> LoraLoaderModelOnly(lightning) -> ModelSamplingAuraFlow -> CFGNorm
    CLIPLoader(qwen_image), VAELoader
    LoadImage(identity), LoadImage(dress, face-masked), LoadImage(room)
      -> TextEncodeQwenImageEditPlus(prompt, image1..3) -> FluxKontextMultiReferenceLatentMethod
      -> KSampler(EmptySD3LatentImage 9:16) -> VAEDecode -> SaveImage

    build_keyframe.py --manifest FILE --out-dir DIR [--style FILE]
"""
import argparse
import json
import os

import numpy as np

ROOT = "/root/affiliate-marketing"
DEFAULT_STYLE = os.path.join(ROOT, "config", "style.json")
SCHEMA_CACHE = os.path.join(ROOT, "config", "node_schemas.json")


def load_json(p):
    with open(p) as f:
        return json.load(f)


COMFY_INPUT = "/opt/comfyui/ComfyUI/input"


def tryon_prompt(style, item, panels=1):
    kf = style["keyframe"]
    outfit = (item.get("prompt_hint") or "the outfit in image 2").strip().rstrip(".")
    layout = kf.get("ref_layout_two" if panels == 2 else "ref_layout_single",
                    "a photo of the outfit")
    colour = (item.get("colour") or "").strip().lower() or "the colour shown in image 2"
    return kf["prompt_template"].format(
        outfit=outfit,
        colour=colour,
        ref_layout=layout,
        character=style["character"]["description"],
        room=style["scene"]["description"],
    )


def garment_ref(style, item):
    """The image Qwen copies the garment from. With ref_panels == 2 and at least
    two product photos, a side-by-side of the first photo (full-length front)
    and the most detailed later photo (the close-up, if the gallery has one)
    is written next to the photos. One angled catalogue shot is not enough
    evidence for construction and embroidery density; the close-up is."""
    import cv2
    kf = style["keyframe"]
    imgs = [p for p in (item.get("images") or []) if p]
    if int(kf.get("ref_panels", 1)) < 2 or len(imgs) < 2:
        return imgs[0], 1
    paths = [os.path.join(COMFY_INPUT, p) for p in imgs]
    loaded = [(p, cv2.imread(p)) for p in paths]
    loaded = [(p, im) for p, im in loaded if im is not None]
    if len(loaded) < 2:
        return imgs[0], 1
    front = loaded[0][1]
    # "detail" = the first photo in which scrape.py found no face (a close-up
    # or flat-lay, where the garment fills the frame); else the last photo.
    counts = item.get("face_counts") or []
    detail_idx = next((i for i, c in enumerate(counts) if i > 0 and c == 0), None)
    if detail_idx is None or detail_idx >= len(loaded):
        detail_idx = len(loaded) - 1
    detail = loaded[detail_idx][1]
    H = max(front.shape[0], detail.shape[0])
    def fit(im):
        s = H / im.shape[0]
        return cv2.resize(im, (int(im.shape[1] * s), H), interpolation=cv2.INTER_AREA)
    a, b = fit(front), fit(detail)
    gap = 255 * np.ones((H, 24, 3), dtype=a.dtype)
    panel = cv2.hconcat([a, gap, b])
    out = os.path.join(os.path.dirname(paths[0]), "kf_ref.jpg")
    cv2.imwrite(out, panel, [cv2.IMWRITE_JPEG_QUALITY, 94])
    return os.path.relpath(out, COMFY_INPUT), 2


def negative_for(style, item):
    """One generic negative for every product. Product-specific exclusions
    (no dupatta, single layer, ...) are stated in the positive prompt by the
    garment caption and the construction/decoration rules, so the negative
    never has to know what the garment is."""
    return style["keyframe"].get("negative", "")


def graph(style, item, dress_img, run_id, seg_no, seed, panels=1, cand=1):
    kf = style["keyframe"]
    m = kf["models"]
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": kf["identity_ref"]}},
        "2": {"class_type": "LoadImage", "inputs": {"image": dress_img}},
        "3": {"class_type": "LoadImage", "inputs": {"image": style["refs"]["scene"]}},
        "4": {"class_type": "UNETLoader",
              "inputs": {"unet_name": m["unet"], "weight_dtype": "default"}},
        "5": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["4", 0], "lora_name": m["lightning_lora"],
                         "strength_model": 1.0}},
        "6": {"class_type": "ModelSamplingAuraFlow",
              "inputs": {"model": ["5", 0], "shift": kf["shift"]}},
        "7": {"class_type": "CFGNorm",
              "inputs": {"model": ["6", 0], "strength": 1.0, "pre_cfg": False}},
        "8": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": m["text_encoder"], "type": "qwen_image"}},
        "9": {"class_type": "VAELoader", "inputs": {"vae_name": m["vae"]}},
        "10": {"class_type": "TextEncodeQwenImageEditPlus",
               "inputs": {"clip": ["8", 0], "vae": ["9", 0],
                          "image1": ["1", 0], "image2": ["2", 0], "image3": ["3", 0],
                          "prompt": tryon_prompt(style, item, panels)}},
        "11": {"class_type": "FluxKontextMultiReferenceLatentMethod",
               "inputs": {"conditioning": ["10", 0],
                          "reference_latents_method": "index_timestep_zero"}},
        "12": {"class_type": "TextEncodeQwenImageEditPlus",
               "inputs": {"clip": ["8", 0], "vae": ["9", 0],
                          "image1": ["1", 0], "image2": ["2", 0], "image3": ["3", 0],
                          "prompt": negative_for(style, item)}},
        "13": {"class_type": "FluxKontextMultiReferenceLatentMethod",
               "inputs": {"conditioning": ["12", 0],
                          "reference_latents_method": "index_timestep_zero"}},
        "14": {"class_type": "EmptySD3LatentImage",
               "inputs": {"width": kf["width"], "height": kf["height"],
                          "batch_size": 1}},
        "15": {"class_type": "KSampler",
               "inputs": {"model": ["7", 0], "positive": ["11", 0],
                          "negative": ["13", 0], "latent_image": ["14", 0],
                          "seed": seed, "steps": kf["steps"], "cfg": kf["cfg"],
                          "sampler_name": "euler", "scheduler": "simple",
                          "denoise": 1.0}},
        "16": {"class_type": "VAEDecode",
               "inputs": {"samples": ["15", 0], "vae": ["9", 0]}},
        "17": {"class_type": "SaveImage",
               "inputs": {"images": ["16", 0],
                          "filename_prefix": f"affiliate/{run_id}/keyframe_seg{seg_no}_c{cand}"}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--style", default=DEFAULT_STYLE)
    ap.add_argument("--seed", type=int, default=None,
                    help="base seed; candidates use seed, seed+1, ...")
    ap.add_argument("--candidates", type=int, default=None,
                    help="try-on stills per outfit (default style.keyframe.candidates)")
    a = ap.parse_args()

    style = load_json(a.style)
    manifest = load_json(a.manifest)
    run_id = manifest["run_id"]
    os.makedirs(a.out_dir, exist_ok=True)

    written = []
    seg = 0
    for item in manifest["items"]:
        imgs = [p for p in (item.get("images") or []) if p]
        if not imgs:
            continue
        seg += 1
        # garment reference: front photo, or front + close-up side by side
        ref_img, panels = garment_ref(style, item)
        n_cand = max(1, int(a.candidates or style["keyframe"].get("candidates", 1)))
        base = a.seed if a.seed is not None else style["keyframe"].get("seed", 20261018) + seg * 100
        for c in range(1, n_cand + 1):
            body = {"prompt": graph(style, item, ref_img, run_id, seg, base + c - 1,
                                    panels, c),
                    "client_id": f"haul-kf-{run_id}"}
            p = os.path.join(a.out_dir, f"keyframe{seg}_c{c}.json")
            with open(p, "w") as f:
                json.dump(body, f, indent=1)
            written.append(p)
        with open(os.path.join(a.out_dir, f"keyframe{seg}.prompt.txt"), "w") as f:
            f.write(tryon_prompt(style, item, panels))
            f.write("\n\n--- negative ---\n" + negative_for(style, item))
            f.write(f"\n\n--- garment ref ({panels} panel) ---\n{ref_img}\n")
            f.write(f"\n--- candidates: {n_cand}, seeds {base}..{base + n_cand - 1} ---\n")
    print(json.dumps({"run_id": run_id, "keyframes": written}))


if __name__ == "__main__":
    main()
