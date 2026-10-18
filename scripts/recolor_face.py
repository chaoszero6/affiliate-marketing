#!/usr/bin/env python3
"""recolor_face.py — edit the character's portrait with Qwen Image Edit 2511.

Used once to change a fixed trait of the character (skin tone, hair) while
keeping her identity, expression, hair, earrings and background. The result
replaces assets/character/face.jpg (the original is backed up by the caller)
and the sheet is regenerated from it, so every reference agrees.

Run while ComfyUI owns the GPU (scripts/gpu.sh handles the handoff):

    recolor_face.py --base assets/character/face.jpg --out /tmp/face_fair.png \
        --prompt "Change her skin tone to a fair, light complexion ..."
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

UNET = "qwen_image_edit_2511_fp8mixed.safetensors"
LIGHTNING_LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
TEXT_ENCODER = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"


def get(path):
    with urllib.request.urlopen(COMFY + path, timeout=30) as r:
        return json.loads(r.read())


def post(path, payload):
    req = urllib.request.Request(COMFY + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def wait_ready(timeout=420):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if "TextEncodeQwenImageEditPlus" in get("/object_info/TextEncodeQwenImageEditPlus"):
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def graph(base_rel, prompt, negative, prefix, seed, steps=4):
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": base_rel}},
        "2": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["1", 0]}},
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["3", 0], "lora_name": LIGHTNING_LORA, "strength_model": 1.0}},
        "6": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["4", 0], "shift": 3.1}},
        "7": {"class_type": "CFGNorm", "inputs": {"model": ["6", 0], "strength": 1.0, "pre_cfg": False}},
        "8": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "qwen_image"}},
        "9": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "12": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
            "clip": ["8", 0], "vae": ["9", 0], "image1": ["2", 0], "prompt": prompt}},
        "11": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {
            "conditioning": ["12", 0], "reference_latents_method": "index_timestep_zero"}},
        "13": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
            "clip": ["8", 0], "vae": ["9", 0], "image1": ["2", 0], "prompt": negative}},
        "14": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {
            "conditioning": ["13", 0], "reference_latents_method": "index_timestep_zero"}},
        "15": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["9", 0]}},
        "16": {"class_type": "KSampler", "inputs": {
            "model": ["7", 0], "positive": ["11", 0], "negative": ["14", 0],
            "latent_image": ["15", 0], "seed": seed, "steps": steps, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "17": {"class_type": "VAEDecode", "inputs": {"samples": ["16", 0], "vae": ["9", 0]}},
        "18": {"class_type": "SaveImage", "inputs": {"images": ["17", 0], "filename_prefix": prefix}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="different person, changed face shape, changed eyes, "
                                          "changed hair, changed expression, makeup, blur, text")
    ap.add_argument("--seed", type=int, default=20261018)
    a = ap.parse_args()

    if not wait_ready():
        sys.exit("ComfyUI did not finish loading")
    run_id = time.strftime("%Y%m%d_%H%M%S")
    stage = os.path.join(INPUT_ROOT, f"recolor_{run_id}")
    os.makedirs(stage, exist_ok=True)
    shutil.copy(a.base, os.path.join(stage, "base" + os.path.splitext(a.base)[1]))
    base_rel = f"recolor_{run_id}/base{os.path.splitext(a.base)[1]}"
    prefix = f"recolor/{run_id}/face"
    r = post("/prompt", {"prompt": graph(base_rel, a.prompt, a.negative, prefix, a.seed),
                         "client_id": "haul-recolor"})
    pid = r["prompt_id"]
    print("queued", pid, flush=True)
    while True:
        try:
            st = get(f"/history/{pid}").get(pid, {}).get("status", {}).get("status_str")
        except Exception:
            st = None
        if st in ("success", "error"):
            break
        time.sleep(4)
    if st != "success":
        sys.exit(f"recolor job ended with {st}")
    outdir = os.path.join(OUTPUT_ROOT, "recolor", run_id)
    files = sorted(f for f in os.listdir(outdir) if f.endswith(".png"))
    if not files:
        sys.exit("no output image")
    shutil.copy(os.path.join(outdir, files[-1]), a.out)
    print("written", a.out)


if __name__ == "__main__":
    main()
