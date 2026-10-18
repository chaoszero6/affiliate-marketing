#!/usr/bin/env python3
"""build_workflow.py — emit one ComfyUI API workflow per outfit for MiniMax H3.

One segment per manifest item (one product link = one 10 s clip). All of that
outfit's reference photos ride into a single ref2va call, so the model sees the
garment from the front, back and in detail at once. The graph never changes:

    UNETLoader -> TurboLoRA -> SageAttentionPatch -> BasicGuider
    CLIPLoader, video VAELoader, audio VAELoader
    LoadImage x N -> MiniMaxH3ReferenceToVideo (ref_image_0..N-1)
    SamplerCustomAdvanced -> VAEDecodeTiled (frames) + VAEDecodeAudio (sound)
    CreateVideo -> SaveVideo

Reference order (and therefore <Picture N> numbering) is fixed by
config/style.json: identity images first, then the room, then the dress photos.

The prompt follows MiniMax's official full-reference format — six sections
(subject_definitions, summary, retention_analysis, detailed_description,
overall_soundscape, non_diegetic_music) — which is how H3 is told *which*
reference carries identity, which carries the location and which carries only
the clothes. A flat sentence prompt leaves that to chance, and the catalogue
model's face wins.

Each seg file is the complete /prompt request body {"prompt": graph,
"client_id": ...}. Required inputs this script doesn't set are filled from
config/node_schemas.json (cached by schema_cache.py).

    build_workflow.py --manifest FILE --out-dir DIR [--pose NAME] [--style FILE]
"""
import argparse
import json
import os

ROOT = "/root/affiliate-marketing"
DEFAULT_STYLE = os.path.join(ROOT, "config", "style.json")
SCHEMA_CACHE = os.path.join(ROOT, "config", "node_schemas.json")
MAX_REF_IMAGES = 9  # hard limit of MiniMaxH3ReferenceToVideo


def load_json(path):
    with open(path) as f:
        return json.load(f)


def schemas():
    try:
        d = load_json(SCHEMA_CACHE)
    except Exception:
        return {}
    return {cls: dict(spec) for cls, spec in d.items() if dict(spec)}


# ------------------------------------------------------------------ prompt
def pics(nums):
    """'<Picture 4>, <Picture 5> and <Picture 6>'"""
    tags = [f"<Picture {n}>" for n in nums]
    if len(tags) == 1:
        return tags[0]
    return ", ".join(tags[:-1]) + " and " + tags[-1]


def build_prompt(style, item, pose_name, n_identity, n_dress, has_keyframe):
    """The six-section MiniMax H3 reference prompt for one outfit.

    Picture order: identity (1..n_identity), room, dress photos, and — when a
    try-on still exists — the keyframe as the last picture, which is also
    anchored at frame 0 through MiniMaxH3AddGuide."""
    ident = list(range(1, n_identity + 1))
    scene = n_identity + 1
    dress = list(range(scene + 1, scene + 1 + n_dress))
    kf = dress[-1] + 1 if has_keyframe else None

    ch = style["character"]
    outfit = (item.get("prompt_hint") or "the outfit in the reference photos").strip()
    if not outfit.endswith((".", "!", "?")):
        outfit += "."
    listing = (item.get("listing_phrase") or "").strip()
    brand = (item.get("brand") or "").strip()
    pose = style["poses"].get(pose_name) or style["poses"][style["default_pose"]]

    ident_desc = pics(ident)
    if n_identity >= 2:
        ident_desc = (f"<Picture 1> (a close-up portrait) and "
                      + pics(ident[1:]) + " (front, three-quarter, side and back views)")

    defs = [
        f"<Subject 1> is the woman whose face and identity come from {ident_desc}: "
        f"{ch['description']}. Her hairstyle and her face-to-body proportions are exactly "
        f"those shown in <Picture {ident[-1]}>. Only her identity, hairstyle and "
        f"proportions are referenced; {ch['ignore_in_refs']}.",
        f"<Subject 2> is the living room in <Picture {scene}>: {style['scene']['description']}.",
        f"<Subject 3> is the outfit shown in {pics(dress)}"
        + (f" (a {brand} set)" if brand else "") + f": {outfit}"
        + (f" The listing describes it as: {listing}." if listing and listing != outfit else "")
        + " The person modelling the outfit in those pictures is not a subject and "
          "does not appear in the target video; only the garments themselves — every "
          "piece of the set, with its colour, fabric, cut, length, neckline, sleeves, "
          "hems, trims and the exact placement of any print or embroidery — are "
          "referenced.",
    ]
    if kf:
        defs.append(
            f"<Picture {kf}> is the first frame of [Shot 1]: <Subject 1> standing on the "
            f"rug in <Subject 2>, already wearing <Subject 3> exactly as it must appear "
            f"for the whole clip.")
    subject_defs = "\n".join(defs)

    task = "[keyframe completion + reference generation]" if kf else "[reference generation]"
    summary = (
        f"{task} A single continuous 10-second vertical fashion-haul clip"
        + (f" that begins from <Picture {kf}>" if kf else "")
        + ": <Subject 1> stands in <Subject 2> wearing <Subject 3> and poses for the "
          "camera so the outfit is shown clearly from the front, the side and the back. "
          "No cuts, no camera movement, no speech."
    )

    ret = [
        "<Subject 1> (appears in [Shot 1]): fully_preserved - her exact face, skin "
        "tone, facial proportions, eyes, smile, hairstyle and jewellery are kept "
        "unchanged throughout; only her clothing is different from the reference "
        "pictures.",
        "<Subject 2> (appears in [Shot 1]): fully_preserved - the white panelled "
        "walls, glossy beige tiles, round jute rug, rattan pendant lamp, left-hand "
        "window and soft warm daylight are retained as the single location; the rug, "
        "the furniture and the lamp keep one fixed position and size for the whole "
        "clip and never move relative to her feet or to the camera.",
        "<Subject 3> (appears in [Shot 1]): attribute_transfer - every piece of the "
        "outfit, with its exact colour, fabric, cut, length, neckline, trims, sleeve "
        "shape, hems and the placement of any print or embroidery, is transferred onto "
        "<Subject 1> and worn by her unchanged for the entire clip; nothing is added "
        "to or removed from the set, and the original wearer's face, body, pose and "
        "the backdrop from the product photos are discarded.",
    ]
    if kf:
        ret.append(
            f"<Picture {kf}> ([Shot 1] first frame): fully_preserved - the clip starts "
            f"on this exact frame and keeps its camera position, framing, lighting and "
            f"background for the whole shot; the garments, the woman and the room stay "
            f"identical to it as she moves. There is no cut and no reframing after the "
            f"first frame.")
    retention = "\n".join(ret)

    detailed = (
        f"{style['style_line']}\n"
        f"[Shot 1] "
        + (f"The shot begins from <Picture {kf}> and holds that exact camera "
           f"position and framing, without any cut, zoom or reframing, until the "
           f"end. " if kf else "")
        + f"{style['framing']} <Subject 1>, {ch['description']}, stands "
        + (f"exactly where she stands in <Picture {kf}>, on the same spot of the floor, "
           if kf else "centred on the round jute rug ")
        + f"in <Subject 2>, with the white panelled wall behind her, "
        f"the rattan pendant lamp above and soft warm daylight falling across her from "
        f"the tall window on the left. She is wearing <Subject 3>: {outfit} The "
        f"garments fit her naturally, the fabric hangs and moves realistically, and the "
        f"length, neckline, sleeves, hems and decoration stay exactly as in the "
        f"reference photos from the first frame to the last. {pose} The lighting, framing and "
        f"location do not change for the entire clip; there is no text, no caption and "
        f"no watermark."
    )

    return (
        f"subject_definitions:\n{subject_defs}\n\n"
        f"summary:\n{summary}\n\n"
        f"retention_analysis:\n{retention}\n\n"
        f"detailed_description:\n{detailed}\n\n"
        f"overall_soundscape:\n{style['soundscape']}\n\n"
        f"non_diegetic_music:\n{style.get('music') or 'N/A'}"
    )


# ------------------------------------------------------------------- graph
def graph(style, defaults, ref_paths, prompt, run_id, index, keyframe=None):
    o = style["output"]
    seg_no = index + 1

    def _is_scalar(v):
        return not isinstance(v, str) or (v.isdigit() or v.lower() in
                                          ("auto", "default", "true", "false"))

    def fill(cls, explicit):
        out = dict(explicit)
        for k, v in defaults.get(cls, {}).items():
            if k not in out and _is_scalar(v):
                out[k] = v
        return out

    g = {
        "1": {"class_type": "UNETLoader",
              "inputs": fill("UNETLoader",
                             {"unet_name": style["models"]["unet"],
                              "weight_dtype": "default"})},
        "2": {"class_type": "MiniMaxH3TurboLoRA",
              "inputs": fill("MiniMaxH3TurboLoRA",
                             {"model": ["1", 0],
                              "lora_name": style["models"]["turbo_lora"],
                              "strength": 1.0})},
        "3": {"class_type": "MiniMaxH3MemoryEfficientSageAttentionPatch",
              "inputs": fill("MiniMaxH3MemoryEfficientSageAttentionPatch",
                             {"model": ["2", 0]})},
        "4": {"class_type": "CLIPLoader",
              "inputs": fill("CLIPLoader",
                             {"clip_name": style["models"]["text_encoder"],
                              "type": "minimax"})},
        "5": {"class_type": "VAELoader",
              "inputs": fill("VAELoader",
                             {"vae_name": style["models"]["video_vae"]})},
        "6": {"class_type": "VAELoader",
              "inputs": fill("VAELoader",
                             {"vae_name": style["models"]["audio_vae"]})},
    }

    # LoadImage nodes 100.. in <Picture N> order
    ref_inputs = {}
    for i, p in enumerate(ref_paths):
        nid = str(100 + i)
        g[nid] = {"class_type": "LoadImage",
                  "inputs": fill("LoadImage", {"image": p})}
        ref_inputs[f"ref_image_{i}"] = [nid, 0]

    g.update({
        "10": {"class_type": "MiniMaxH3ReferenceToVideo",
               "inputs": fill("MiniMaxH3ReferenceToVideo", {
                   "clip": ["4", 0],
                   "vae": ["5", 0],
                   "audio_vae": ["6", 0],
                   "prompt": prompt,
                   "width": o["width"],
                   "height": o["height"],
                   "length": o["frames_per_segment"],
                   "ref_image_size": style["refs"].get("ref_image_size", "max"),
                   "ref_images": ref_inputs})},
        "11": {"class_type": "BasicGuider",
               "inputs": fill("BasicGuider", {"model": ["3", 0],
                                            "conditioning": ["10", 0]})},
        "12": {"class_type": "RandomNoise",
               "inputs": fill("RandomNoise",
                              {"noise_seed": 11 + index * 7})},
        "13": {"class_type": "MiniMaxH3TurboSampler",
               "inputs": fill("MiniMaxH3TurboSampler", {})},
        "14": {"class_type": "BasicScheduler",
               "inputs": fill("BasicScheduler",
                              {"model": ["3", 0],
                               "scheduler": style["sampler"]["scheduler"],
                               "steps": style["sampler"]["steps"]})},
        "15": {"class_type": "SamplerCustomAdvanced",
               "inputs": fill("SamplerCustomAdvanced", {
                   "noise": ["12", 0], "guider": ["11", 0],
                   "sampler": ["13", 0], "sigmas": ["14", 0],
                   "latent_image": ["10", 1]})},
        "16": {"class_type": "VAEDecodeTiled",
               "inputs": fill("VAEDecodeTiled", {
                   "samples": ["15", 0], "vae": ["5", 0],
                   "tile_size": 512, "overlap": 64,
                   "temporal_size": 32, "temporal_overlap": 8})},
        "17": {"class_type": "VAEDecodeAudio",
               "inputs": fill("VAEDecodeAudio", {"samples": ["15", 0],
                                               "vae": ["6", 0]})},
        "18": {"class_type": "CreateVideo",
               "inputs": fill("CreateVideo", {"images": ["16", 0],
                                             "audio": ["17", 0],
                                             "fps": o["fps"]})},
        "19": {"class_type": "SaveVideo",
               "inputs": fill("SaveVideo", {
                   "video": ["18", 0],
                   "filename_prefix": f"affiliate/{run_id}/seg{seg_no}",
                   "format": "auto"})},
    })
    if keyframe:
        # hard anchor: the try-on still is frame 0 of the clip. The same image is
        # already the last LoadImage so the text encoder sees it as <Picture N>.
        g["20"] = {"class_type": "LoadImage",
                   "inputs": fill("LoadImage", {"image": keyframe})}
        g["21"] = {"class_type": "MiniMaxH3AddGuide",
                   "inputs": {"positive": ["10", 0], "vae": ["5", 0],
                              "latent": ["10", 1], "image": ["20", 0],
                              "frame_idx": 0}}
        g["11"]["inputs"]["conditioning"] = ["21", 0]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--style", default=DEFAULT_STYLE)
    ap.add_argument("--pose", default=None,
                    help="pose preset from style.json (default: default_pose)")
    a = ap.parse_args()

    style = load_json(a.style)
    manifest = load_json(a.manifest)
    run_id = manifest["run_id"]
    os.makedirs(a.out_dir, exist_ok=True)
    defaults = schemas()

    refs = style["refs"]
    identity = list(refs["identity"])
    scene = refs["scene"]
    max_dress = int(refs.get("max_dress_refs", 4))
    use_kf = bool(style.get("keyframe", {}).get("enabled", False))
    room_left = MAX_REF_IMAGES - len(identity) - 1 - (1 if use_kf else 0)
    if max_dress > room_left:
        print(f"warn: max_dress_refs {max_dress} exceeds the {room_left} slots "
              f"left after identity+scene; clamping", flush=True)
        max_dress = room_left
    pose_name = a.pose or style["default_pose"]
    if pose_name not in style["poses"]:
        print(f"warn: unknown pose {pose_name!r}; using {style['default_pose']!r}")
        pose_name = style["default_pose"]

    written, prompts = [], {}
    seg = 0
    for item in manifest["items"]:
        imgs = [p for p in (item.get("images") or []) if p][:max_dress]
        if not imgs:
            print(f"skip: {item.get('slug') or item.get('source')} has no images")
            continue
        keyframe = item.get("keyframe") if use_kf else None
        if use_kf and not keyframe:
            print(f"warn: {item.get('slug')} has no keyframe; rendering from "
                  f"references only", flush=True)
        ref_paths = identity + [scene] + imgs + ([keyframe] if keyframe else [])
        prompt = build_prompt(style, item, pose_name, len(identity), len(imgs),
                              bool(keyframe))
        body = {"prompt": graph(style, defaults, ref_paths, prompt, run_id, seg,
                                keyframe),
                "client_id": f"haul-{run_id}"}
        p = os.path.join(a.out_dir, f"seg{seg + 1}.json")
        with open(p, "w") as f:
            json.dump(body, f, indent=1)
        with open(os.path.join(a.out_dir, f"seg{seg + 1}.prompt.txt"), "w") as f:
            f.write(prompt)
        written.append(p)
        prompts[f"seg{seg + 1}"] = {"slug": item.get("slug"), "refs": ref_paths,
                                   "pose": pose_name, "keyframe": keyframe}
        seg += 1

    print(json.dumps({"run_id": run_id, "segments": written, "detail": prompts}))


if __name__ == "__main__":
    main()
