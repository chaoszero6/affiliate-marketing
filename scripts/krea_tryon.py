#!/usr/bin/env python3
"""krea_tryon.py — try-on still with Krea 2 Identity Edit (two-reference editor).

Krea 2 Identity Edit takes a SOURCE image (the scene being edited) and a
SUBJECT image (the person) and rewrites the source so the subject is in it.
We use it in two passes:

  pass 1  source = masked product photo, subject = our woman's portrait
          -> our woman wearing the product, in the catalogue setting.
             The garment is the preserved source, so it stays exact.
  pass 2  source = our room photo, subject = pass-1 result
          -> our woman in that outfit standing in our room.

Runs while ComfyUI owns the GPU (scripts/gpu.sh handles the handoff):

    krea_tryon.py --product /opt/comfyui/ComfyUI/input/haul_x/.../00_masked.jpg \
                  --identity /opt/comfyui/ComfyUI/input/haul_base/face.jpg \
                  --room /opt/comfyui/ComfyUI/input/haul_base/room.jpg \
                  --outfit "black tunic ..." --colour black --out-dir /tmp/krea
"""
import argparse
import json
import os
import shutil
import sys
import time
import urllib.request

COMFY = "http://127.0.0.1:8188"
INPUT_ROOT = "/opt/comfyui/ComfyUI/input"
OUTPUT_ROOT = "/opt/comfyui/ComfyUI/output"

UNET = "krea2_raw_bf16.safetensors"
TURBO_LORA = "krea2_turbo_lora_rank_64_bf16.safetensors"
EDIT_LORA = "krea2_identity_edit_v1_2.safetensors"
TEXT_ENCODER = "qwen3vl_4b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"


def get(path):
    with urllib.request.urlopen(COMFY + path, timeout=60) as r:
        return json.loads(r.read())


def post(path, payload):
    req = urllib.request.Request(COMFY + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:1500]}")


def wait_ready(timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if "Krea2EditModelPatch" in get("/object_info/Krea2EditModelPatch"):
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def graph(source_rel, subject_rel, prompt, negative, prefix, seed, width, height,
          steps, ref_boost, ref_boost_a, grounding):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["1", 0], "lora_name": TURBO_LORA, "strength_model": 1.0}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["2", 0], "lora_name": EDIT_LORA, "strength_model": 1.0}},
        "4": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": TEXT_ENCODER, "type": "krea2", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "6": {"class_type": "LoadImage", "inputs": {"image": source_rel}},
        "7": {"class_type": "LoadImage", "inputs": {"image": subject_rel}},
        "8": {"class_type": "VAEEncode", "inputs": {"pixels": ["6", 0], "vae": ["5", 0]}},
        "9": {"class_type": "VAEEncode", "inputs": {"pixels": ["7", 0], "vae": ["5", 0]}},
        "10": {"class_type": "EmptySD3LatentImage",
               "inputs": {"width": width, "height": height, "batch_size": 1}},
        "11": {"class_type": "Krea2EditModelPatch", "inputs": {
            "model": ["3", 0], "source_latent": ["8", 0], "source_latent_b": ["9", 0],
            "ref_boost": ref_boost, "ref_boost_a": ref_boost_a, "fit_mode": "fit",
            "vae": ["5", 0], "source_image": ["6", 0], "source_image_b": ["7", 0],
            "target_latent": ["10", 0]}},
        "12": {"class_type": "Krea2EditGroundedEncode", "inputs": {
            "clip": ["4", 0], "prompt": prompt, "image": ["6", 0], "image_b": ["7", 0],
            "grounding_px": grounding, "system_prompt": ""}},
        "13": {"class_type": "Krea2EditGroundedEncode", "inputs": {
            "clip": ["4", 0], "prompt": negative, "image": ["6", 0], "image_b": ["7", 0],
            "grounding_px": grounding, "system_prompt": ""}},
        "14": {"class_type": "KSampler", "inputs": {
            "model": ["11", 0], "positive": ["12", 0], "negative": ["13", 0],
            "latent_image": ["10", 0], "seed": seed, "steps": steps, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["5", 0]}},
        "16": {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": prefix}},
    }


def run_job(g, tag):
    r = post("/prompt", {"prompt": g, "client_id": "haul-krea"})
    pid = r["prompt_id"]
    print(f"{tag}: queued {pid}", flush=True)
    t0 = time.time()
    while True:
        try:
            h = get(f"/history/{pid}").get(pid, {})
            st = h.get("status", {}).get("status_str")
        except Exception:
            st = None
        if st in ("success", "error"):
            break
        time.sleep(4)
    print(f"{tag}: {st} in {time.time() - t0:.0f}s", flush=True)
    if st != "success":
        for m in h.get("status", {}).get("messages", []):
            if m[0] == "execution_error":
                print(m[1].get("node_type"), m[1].get("exception_message", "")[:800])
        sys.exit(1)
    outs = []
    for node_out in h.get("outputs", {}).values():
        for im in node_out.get("images", []):
            outs.append(os.path.join(OUTPUT_ROOT, im.get("subfolder", ""), im["filename"]))
    return outs[-1]


def rel(path):
    return os.path.relpath(path, INPUT_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", required=True, help="masked product photo (abs path inside ComfyUI input)")
    ap.add_argument("--identity", required=True, help="her portrait (abs path inside ComfyUI input)")
    ap.add_argument("--room", required=True, help="room photo (abs path inside ComfyUI input)")
    ap.add_argument("--outfit", required=True)
    ap.add_argument("--colour", default="")
    ap.add_argument("--character", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=1472)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--ref-boost", type=float, default=1.5)
    ap.add_argument("--grounding", type=int, default=1024)
    ap.add_argument("--skip-pass2", action="store_true")
    a = ap.parse_args()

    if not wait_ready():
        sys.exit("ComfyUI did not finish loading")
    os.makedirs(a.out_dir, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    stage = os.path.join(INPUT_ROOT, f"krea_{run_id}")
    os.makedirs(stage, exist_ok=True)

    colour = a.colour or "the colour shown in the first image"
    who = a.character or "the woman in the second image"
    negative = "different person, different face, changed outfit, changed garment colour, added dupatta, added necklace, blurry, deformed hands, extra limbs, text, watermark"

    # ---- pass 1: our woman into the product photo (garment preserved)
    p1 = (f"Replace the woman wearing this outfit with the woman from the second image: use her face, "
          f"her skin tone, her hairstyle ({who}). She wears exactly the same outfit as in the first "
          f"image — {a.outfit} — unchanged in colour ({colour}), cut, length, neckline, trim, sleeves, "
          f"cuffs, embroidery and bottoms, in the same standing pose. Small stud earrings, no necklace. "
          f"Keep the background of the first image. Photorealistic, natural skin texture, sharp detail.")
    out1 = run_job(graph(rel(a.product), rel(a.identity), p1, negative,
                         f"krea/{run_id}/pass1", a.seed, a.width, a.height, a.steps,
                         a.ref_boost, 1.0, a.grounding), "pass1")
    shutil.copy(out1, os.path.join(a.out_dir, "pass1.png"))
    print("pass1 ->", os.path.join(a.out_dir, "pass1.png"), flush=True)
    if a.skip_pass2:
        return

    # ---- pass 2: that woman into our room
    staged = os.path.join(stage, "pass1.png")
    shutil.copy(out1, staged)
    p2 = (f"Place the woman from the second image into this room. She stands at the centre of the "
          f"frame with both feet on the round jute rug, squarely facing the camera, feet a little apart, "
          f"arms relaxed at her sides, a calm natural expression with a small closed-lip smile. Keep her "
          f"face, hairstyle, body and her complete outfit exactly as in the second image — {a.outfit}, "
          f"{colour} — nothing added or removed. Full body visible from the top of her hair to her feet "
          f"with a little floor below her shoes; the room, its white panelled wall, rattan pendant lamp, "
          f"window and furniture stay exactly as in the first image; soft natural daylight from the left. "
          f"Photorealistic, natural skin and fabric texture, no text.")
    out2 = run_job(graph(rel(a.room), rel(staged), p2, negative,
                         f"krea/{run_id}/pass2", a.seed + 1, a.width, a.height, a.steps,
                         a.ref_boost, 1.0, a.grounding), "pass2")
    shutil.copy(out2, os.path.join(a.out_dir, "pass2.png"))
    print("pass2 ->", os.path.join(a.out_dir, "pass2.png"), flush=True)


if __name__ == "__main__":
    main()
