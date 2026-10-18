#!/usr/bin/env python3
"""build_swap.py — emit a ReActor face-swap workflow for a try-on still.

Qwen Image Edit draws a plausible, slightly idealised version of the woman in
the identity picture. ReActor (inswapper_128 + GFPGAN restore) replaces the
face in the generated still with *her* face from the reference portrait, driven
by the ArcFace embedding — the same embedding our QA scores against. The
garments, pose, room and lighting of the still are untouched.

    build_swap.py --image <rel path in ComfyUI input> --face <rel path> \
                  --prefix affiliate/<run>/swap_seg1_c1 --out file.json
"""
import argparse
import json

ROOT = "/root/affiliate-marketing"


def graph(image_rel, face_rel, prefix, restore="GFPGANv1.4.pth", visibility=1.0,
          codeformer_weight=0.5):
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_rel}},
        "2": {"class_type": "LoadImage", "inputs": {"image": face_rel}},
        "3": {"class_type": "ReActorFaceSwap", "inputs": {
            "enabled": True,
            "input_image": ["1", 0],
            "source_image": ["2", 0],
            "swap_model": "inswapper_128.onnx",
            "facedetection": "retinaface_resnet50",
            "face_restore_model": restore,
            "face_restore_visibility": visibility,
            "codeformer_weight": codeformer_weight,
            "detect_gender_input": "no",
            "detect_gender_source": "no",
            "input_faces_index": "0",
            "source_faces_index": "0",
            "console_log_level": 1}},
        "4": {"class_type": "SaveImage",
              "inputs": {"images": ["3", 0], "filename_prefix": prefix}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--face", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--restore", default="GFPGANv1.4.pth")
    ap.add_argument("--visibility", type=float, default=1.0)
    a = ap.parse_args()
    body = {"prompt": graph(a.image, a.face, a.prefix, a.restore, a.visibility),
            "client_id": "haul-swap"}
    with open(a.out, "w") as f:
        json.dump(body, f, indent=1)
    print(a.out)


if __name__ == "__main__":
    main()
