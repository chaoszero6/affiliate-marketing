#!/usr/bin/env python3
"""make_sheet.py — turn one portrait into a multi-view character sheet.

One Qwen Image Edit 2511 job per camera angle (the multi-angle LoRA is what the
reference workflow uses), then an ImageMagick montage into a single sheet.
The sheet becomes the fixed identity reference for every haul video.

    python3 make_sheet.py --base assets/character/face.jpg \
                          --angles 0,45,90,180 \
                          --out assets/character/sheet.png

Run it while ComfyUI owns the GPU; scripts/make_sheet.sh handles the handoff
from the chat engine.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

COMFY = "http://127.0.0.1:8188"
ROOT = "/root/affiliate-marketing"

UNET = "qwen_image_edit_2511_fp8mixed.safetensors"
LIGHTNING_LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
ANGLE_LORA = "qwen-image-edit-2511-multiple-angles-lora.safetensors"
TEXT_ENCODER = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"
AURA_FLOW_SHIFT = 3.1

# azimuth -> the phrase the multi-angle camera node emits for it
ANGLE_WORDS = {0: "front view", 45: "front-right quarter view",
               90: "right side view", 135: "back-right quarter view",
               180: "back view", 225: "back-left quarter view",
               270: "left side view", 315: "front-left quarter view"}


def get(path):
    with urllib.request.urlopen(COMFY + path, timeout=30) as r:
        return json.loads(r.read())


def post(path, payload):
    req = urllib.request.Request(
        COMFY + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")
        try:
            body = json.dumps(json.loads(body), indent=2)
        except Exception:
            pass
        sys.stderr.write(f"HTTP {e.code} from {path}:\n{body}\n")
        raise


def queue(outdir, graph):
    """POST /prompt with the documented envelope: {prompt: graph, client_id}."""
    return post("/prompt", {"prompt": graph, "client_id": "haul-sheet"})


def check_nodes(graph):
    """Fail early with a readable message if a class_type is unknown."""
    known = set(get("/object_info").keys())
    missing = [n["class_type"] for n in graph.values()
               if n["class_type"] not in known]
    if missing:
        sys.exit(f"unknown node types (missing custom nodes?): {missing}")


def graph(base_rel, run_id, index, angle):
    """Flattened ref2image graph: base photo -> camera angle -> saved image.

    Mirrors the shipped multi-angle example workflow. The camera node emits an
    angle phrase; the edit-prompt text encoder consumes it; the conditioning is
    wrapped by FluxKontextMultiReferenceLatentMethod before reaching the sampler.
    """
    return {
        # identity: the base portrait, scaled for reference latents
        "1": {"class_type": "LoadImage", "inputs": {"image": base_rel}},
        "2": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["1", 0]}},
        # model chain: unet -> lightning lora -> multi-angle lora -> aura shift -> cfgnorm
        "3": {"class_type": "UNETLoader", "inputs": {
            "unet_name": UNET, "weight_dtype": "default"}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["3", 0], "lora_name": LIGHTNING_LORA,
            "strength_model": 1.0}},
        "5": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["4", 0], "lora_name": ANGLE_LORA, "strength_model": 1.0}},
        "6": {"class_type": "ModelSamplingAuraFlow",
              "inputs": {"model": ["5", 0], "shift": AURA_FLOW_SHIFT}},
        "7": {"class_type": "CFGNorm",
              "inputs": {"model": ["6", 0], "strength": 1.0, "pre_cfg": False}},
        "8": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": TEXT_ENCODER, "type": "qwen_image"}},
        "9": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        # camera node turns the angle into a phrase for the text encoder
        "10": {"class_type": "QwenMultiangleCameraNode", "inputs": {
            "image": ["2", 0], "horizontal_angle": angle,
            "vertical_angle": 0, "zoom": 2.8,
            "default_prompts": False, "camera_view": False}},
        "11": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {
            "conditioning": ["12", 0],
            "reference_latents_method": "index_timestep_zero"}},
        "12": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
            "clip": ["8", 0], "vae": ["9", 0], "image1": ["2", 0],
            "prompt": ["10", 0]}},
        # negative conditioning: empty prompt through the same path
        "13": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
            "clip": ["8", 0], "vae": ["9", 0], "image1": ["2", 0], "prompt": ""}},
        "14": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {
            "conditioning": ["13", 0],
            "reference_latents_method": "index_timestep_zero"}},
        "15": {"class_type": "VAEEncode",
               "inputs": {"pixels": ["2", 0], "vae": ["9", 0]}},
        "16": {"class_type": "KSampler", "inputs": {
            "model": ["7", 0], "positive": ["11", 0], "negative": ["14", 0],
            "latent_image": ["15", 0],
            "seed": 20261017 + angle, "steps": 4, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "17": {"class_type": "VAEDecode",
               "inputs": {"samples": ["16", 0], "vae": ["9", 0]}},
        "18": {"class_type": "SaveImage", "inputs": {
            "images": ["17", 0],
            "filename_prefix": f"sheet/{run_id}/view{index:02d}"}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base portrait (in Comfy input dir)")
    ap.add_argument("--angles", default="0,45,90,180")
    ap.add_argument("--out", default=os.path.join(ROOT, "assets/character/sheet.png"))
    ap.add_argument("--dry-run", action="store_true",
                    help="validate node types against the running server, don't queue")
    a = ap.parse_args()

    angles = [int(x) for x in a.angles.split(",")]
    run_id = time.strftime("%Y%m%d_%H%M%S")

    # LoadImage only sees files inside the ComfyUI input root, so stage the
    # base portrait there for this run instead of referencing it by path.
    input_root = "/opt/comfyui/ComfyUI/input"
    run_input = os.path.join(input_root, f"sheet_{run_id}")
    os.makedirs(run_input, exist_ok=True)
    import shutil
    shutil.copy(os.path.abspath(a.base), os.path.join(run_input, "base.jpg"))
    base_rel = f"sheet_{run_id}/base.jpg"

    g = graph(base_rel, run_id, 0, angles[0])
    check_nodes(g)
    if a.dry_run:
        print(json.dumps({"run_id": run_id, "nodes_ok": True,
                         "angles": angles}, indent=2))
        return

    outdir = f"/opt/comfyui/ComfyUI/output/sheet/{run_id}"
    os.makedirs(outdir, exist_ok=True)

    ids = []
    for i, ang in enumerate(angles):
        r = queue(outdir, graph(base_rel, run_id, i, ang))
        ids.append(r["prompt_id"])
        print(f"queued {ANGLE_WORDS.get(ang, str(ang) + ' deg'):24s} -> {r['prompt_id']}",
              flush=True)

    print("waiting for the GPU work...", flush=True)
    for pid in ids:
        while True:
            try:
                h = get(f"/history/{pid}")
            except Exception:
                time.sleep(5)
                continue
            st = h.get(pid, {}).get("status", {}).get("status_str")
            if st in ("success", "error"):
                break
            time.sleep(8)

    if not os.path.isdir(outdir):
        sys.exit(f"expected outputs in {outdir}, not found")
    views = sorted(os.listdir(outdir))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    subprocess.run(["convert"] + [os.path.join(outdir, v) for v in views]
                   + [a.out], check=True)
    print(f"sheet written -> {a.out} ({len(views)} panels)")


if __name__ == "__main__":
    main()
