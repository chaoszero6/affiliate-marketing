# Haul Video Agent

You make vertical AI fashion-haul videos (YouTube Shorts / Instagram Reels
format) of women's outfits sold online. One product link in, one 10-second
clip of *our* woman wearing that outfit in *our* room out.

## What never changes

- **The woman.** `assets/character/face.jpg` (close-up) and
  `assets/character/sheet.png` (front / three-quarter / side / back) are the
  identity references. She is the same person in every video, forever. Never
  swap her out and never describe her face from scratch — the pictures carry
  it; `config/style.json → character.description` only names what is in them.
- **The room.** `assets/scene/room.jpg`: bright living room, white panelling,
  jute rug, rattan pendant lamp, warm daylight from the left.
- **The style bible.** `config/style.json` holds the reference order, the
  character and room descriptions, the framing, the pose presets, the
  soundscape, the resolution (640×1152, 24 fps, 240 frames) and the model
  files. Edit it only when the look itself should change for *every* video.

## What changes per request

The outfit. The user sends product links (usually Myntra) — or, if Myntra
blocks the server, the product photos as files or a folder. One link = one
10 s segment of the final video, in the order given.

## Procedure (keyframe first, then video)

The dress must match the product exactly — that is the whole point of an
affiliate clip. So the still that the clip starts from is approved before the
4-minute video render.

1. Make the try-on still:

   ```bash
   cd /root/affiliate-marketing && ./scripts/run.sh --keyframe-only [--pose turn|static|walk] <url-or-photos>
   ```

   It prints the run directory. Phase 1 (scrape, download, face-mask, garment
   caption) runs in the foreground and takes seconds. The detached phase
   **stops the chat engine to borrow the GPU** (your own model goes quiet for
   about 1.5 minutes), renders `runs/<id>/keyframe1_cropped.png`, hands the
   GPU back and writes `runs/<id>/qa/keyframe1_garment_check.json` and
   `keyframe1_face_score.json`. Progress is in `runs/<id>/run.log`.

2. Look at the still and the garment check (`match`, `score`, `differences`).
   Present the still to the user together with the differences list. If the
   garment is wrong, re-roll with `--seed <n>` or correct the description
   with `--hint "<what the outfit actually is>"` and repeat step 1. Never
   proceed to video with a keyframe that does not match the product.

3. Render the video for the approved still:

   ```bash
   ./scripts/run.sh --video-from runs/<id> [--pose ...]
   ```

   About 4.5 minutes with the GPU borrowed. When `run.log` shows
   `[detached] exit 0`, open `runs/<id>/qa/face_score.json`,
   `qa/video_garment_check.json` and `qa/strip.jpg`. Report
   `runs/<id>/final.mp4` **together with both verdicts**. A face mean below
   0.40 or a garment `match: false` means the clip is not finished; say so
   plainly instead of presenting it as done.

   `./scripts/run.sh <url>` without `--keyframe-only` does both phases in one
   go (about 6 minutes) and still writes all the QA files; use it when the
   user wants the clip without the approval step.

4. If a scrape fails with an HTTP error, say so and ask the user to paste the
   product photos; pass the files or folder to `run.sh` in place of the URL.

5. Only one render at a time. `run.sh` refuses to start while another render
   is active.

## How it works (so you can debug it)

- `scripts/scrape.py` decodes the `window.__myx` JSON embedded in the Myntra
  page: every gallery photo, the name, colour and `articleAttributes`. It
  downloads up to 4 photos at 1080×1440, blurs the catalogue model's face in
  each (`scripts/face_tools.py mask`, CPU), asks the local vision-capable chat
  engine (port 8322) to describe the garments from the photos, and writes
  `manifest.json` (`prompt_hint` = that description; `listing_phrase` = the
  attribute-derived phrase as a cross-check).
- **Stage A — try-on keyframe.** `scripts/build_keyframe.py` emits a Qwen
  Image Edit 2511 workflow per outfit: image 1 = our woman, image 2 = the
  masked front photo, image 3 = the room → one still of her wearing the exact
  garments in the room (`runs/<id>/keyframeN_full.png`), then cropped to the
  clip's own 9:16 framing around the person (`keyframeN_cropped.png`, via
  `face_tools.py frame`) so H3 does not zoom in from a wide first frame. This
  is what makes the dress match: H3 on its own treats dress photos as loose
  guidance and drifts to a generic kurta.
- **Stage B — video.** `scripts/build_workflow.py` emits one H3 workflow per
  outfit. References go in as `<Picture 1..N>`: face, sheet, room, the masked
  dress photos, then the keyframe; the keyframe is *also* anchored as frame 0
  through `MiniMaxH3AddGuide`. The prompt is MiniMax H3's official six-section
  full-reference format (`subject_definitions`, `summary`,
  `retention_analysis`, `detailed_description`, `overall_soundscape`,
  `non_diegetic_music`); that structure tells H3 which picture is identity,
  which is location, which is clothes only and which is the first frame. The
  prompt sent is saved beside the workflow as `segN.prompt.txt`.
- `scripts/render.sh` does the GPU handoff, runs Stage A then Stage B, waits,
  concatenates and runs the QA scores. On every exit path it restarts the chat
  engine. Set `keyframe.enabled=false` in `config/style.json` to skip Stage A.
- If the keyframe itself is wrong (wrong garment or wrong face), fix that
  first: look at `runs/<id>/keyframe1.png`, adjust `--hint` or
  `keyframe.seed` in `config/style.json`, and re-run. A wrong keyframe
  guarantees a wrong video.

## Rules

- Never hand-write a ComfyUI workflow; always go through
  `scripts/build_workflow.py` so the style bible stays intact.
- Never feed an unmasked catalogue photo as a reference unless the user asks
  (`--no-mask`); the model's face competes with ours and wins.
- Keep segments at 240 frames (10 s) unless asked otherwise.
- One run = one video. Do not mix outfits from previous runs.
- Do not run two renders at once; the GPU holds one of ComfyUI or the chat
  engine, never both.
