# Haul Video Agent

Turn one product link into one 10-second vertical fashion-haul clip
(YouTube Shorts / Instagram Reels format) of a **fixed AI woman** wearing the
exact outfit, in a **fixed room**, using a local ComfyUI render farm.

One Myntra link in → one 9:16 clip of our woman wearing that outfit out.

The woman and the room never change. Only the outfit changes, per request.

---

## What never changes

| Asset | Path | Role |
| --- | --- | --- |
| Face close-up | `assets/character/face.jpg` | Identity reference |
| Character sheet | `assets/character/sheet.png` | Front / three-quarter / side / back |
| Room | `assets/scene/room.jpg` | Bright living room, fixed forever |
| Style bible | `config/style.json` | Ref order, descriptions, framing, poses, audio, resolution |

`config/style.json` is the single source of truth for the look. Edit it only
when the look itself should change for **every** video.

## What changes per request

The outfit. The user sends Myntra product links — or, if Myntra blocks the
server, the product photos as files or a folder. One link = one 10 s segment,
concatenated in the order given.

---

## How the pipeline works

```
product link / photos
        │
        ▼
 scripts/scrape.py        decode window.__myx JSON, download gallery,
                          blur the catalogue model's face, caption garments
        │
        ▼
 Stage A — keyframe       Qwen Image Edit 2511 (or Krea 2) try-on still:
 (scripts/build_keyframe.py)  our woman wearing the exact garments in the room
                          → keyframe1_cropped.png  (approval gate)
        │
        ▼
 Stage B — video          MiniMax H3 full-reference workflow:
 (scripts/build_workflow.py)  refs = face, sheet, room, masked photos, keyframe
                          keyframe anchored as frame 0
        │
        ▼
 concat + QA              ffmpeg concat, face-match score, garment check
 (scripts/render.sh)      → runs/<id>/final.mp4
```

The keyframe is rendered and approved **before** the ~4-minute video render,
because a wrong keyframe guarantees a wrong video. The dress matching the
product is the whole point of an affiliate clip.

---

## Requirements

- Linux host with an NVIDIA GPU (the pipeline borrows the GPU exclusively;
  the chat engine and ComfyUI never hold it at the same time).
- A local **ComfyUI** instance on `127.0.0.1:8188` with:
  - MiniMax H3 reference-to-video models (`unet`, turbo LoRA, text encoder,
    video/audio VAE — see `config/style.json → models`),
  - Qwen Image Edit 2511 and/or Krea 2 for the keyframe
    (see `config/style.json → keyframe.models`).
- Python packages: `numpy`, `opencv-python`, `discord.py` (bot only).
- `ffmpeg` on `PATH`.
- Node/ComfyUI model weights are **not** included in this repository.

## Setup

```bash
git clone https://github.com/chaoszero6/affiliate-marketing.git
cd affiliate-marketing

cp .env.example .env      # fill in Discord / Nextcloud values
```

`.env` is git-ignored and holds secrets only. If you don't run the Discord bot
or Nextcloud delivery, you can leave it empty.

---

## Usage

Make and approve the try-on still first (phase 1 takes seconds, the detached
still render borrows the GPU for ~1.5 min):

```bash
./scripts/run.sh --keyframe-only [--pose turn|static|walk] <url-or-photos>
```

It prints the run directory (`runs/<id>/`). Inspect:

- `runs/<id>/keyframe1_cropped.png` — the still
- `runs/<id>/qa/keyframe1_garment_check.json` — `match`, `score`, `differences`
- `runs/<id>/qa/keyframe1_face_score.json` — identity score

If the garment is wrong, re-roll with `--seed <n>` or correct the description
with `--hint "<what the outfit actually is>"` and repeat. **Never** proceed to
video with a keyframe that does not match the product.

Render the video for the approved still (~4.5 min):

```bash
./scripts/run.sh --video-from runs/<id> [--pose ...]
```

Or do both phases in one go (~6 min, no approval gate):

```bash
./scripts/run.sh <url>
```

When `runs/<id>/run.log` shows `[detached] exit 0`, check
`qa/face_score.json`, `qa/video_garment_check.json` and `qa/strip.jpg`, then
publish `runs/<id>/final.mp4` **together with both verdicts**. A face mean below
`0.40` or a garment `match: false` means the clip is not finished.

### Options

| Flag | Effect |
| --- | --- |
| `--pose turn\|static\|walk` | Pose preset from `config/style.json` |
| `--hint "phrase"` | Override the outfit description for the Nth ref |
| `--max-images N` | Dress photos used as references per outfit (default 4) |
| `--no-mask` | Keep the catalogue model's face (not recommended) |
| `--seed N` | Seed for the try-on still (re-roll a wrong keyframe) |
| `--keyframe-only` | Stop after the still, for approval |
| `--video-from DIR` | Render video for an already-approved run dir |
| `--keep-comfy` | Leave ComfyUI running afterwards |

### When scraping is blocked

If a scrape fails with an HTTP error, Myntra is likely blocking the server IP.
Pass the product photos as files or a folder in place of the URL:

```bash
./scripts/run.sh --keyframe-only ~/photos/outfit1/
```

Only one render runs at a time; `run.sh` refuses to start while another is active.

---

## Repository layout

```
AGENTS.md           Agent operating procedure
config/             style.json (the style bible); node_schemas.json is a cache
assets/             character identity + room (the fixed references)
presets/            agent preset wiring (persona + tools)
scripts/            the pipeline: scrape, keyframe, workflow, render, QA, bots
runs/               per-run outputs (git-ignored)
work/, qa/          scratch and QA archives (git-ignored)
```

`scripts/scrape.py` is Myntra-specific — it decodes the `window.__myx` JSON and
downloads from the `myntassets.com` CDN. Other retailers work through the
file/folder input path.

---

## Licence and credits

MIT — see [LICENSE](LICENSE). The licence includes a **Repository Notice &
Disclaimer**: if you fork, redistribute, or build on this project, you must
credit `https://github.com/chaoszero6/affiliate-marketing` and carry the
disclaimer (no affiliation with Myntra or any brand shown; rights to product
images and third-party model weights remain with their owners).

Built with [ComfyUI](https://github.com/comfyanonymous/ComfyUI),
MiniMax H3 and Qwen Image Edit. Product photos originate from Myntra.
