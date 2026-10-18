#!/usr/bin/env python3
"""Discord front-end for the haul-video agent.

Watches for Myntra product links (or product photos) posted by an allow-list of
user IDs, runs the normal keyframe+video pipeline, copies the assembled clip to
the storage box, and replies that it is ready. One render at a time.

Config comes from /root/affiliate-marketing/.env:
    DISCORD_BOT_TOKEN=...
    DISCORD_AUTH_USERS=<id>,<id>,...
"""
from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import discord

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nc_deliver  # noqa: E402

ROOT = Path("/root/affiliate-marketing")
RUNS = ROOT / "runs"
ENV_FILE = ROOT / ".env"
RUN_SH = ROOT / "scripts" / "run.sh"
WORK = ROOT / "work" / "discord"
BOT_LOG = RUNS / "discord_bot.log"

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MYNTRA_URL = r"https?://(?:www\.)?myntra\.com/\S*"
RENDER_PAT = "[s]cripts/render.sh --env"

# how long to wait for a whole render before giving up (seconds)
RUN_TIMEOUT = 4 * 60 * 60


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        RUNS.mkdir(parents=True, exist_ok=True)
        with BOT_LOG.open("a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if not ENV_FILE.exists():
        return env
    for raw in ENV_FILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        env[key.strip()] = val.strip().strip('"').strip("'")
    return env


ENV = load_env()
TOKEN = ENV.get("DISCORD_BOT_TOKEN", "").strip()
ALLOWED = {u.strip() for u in ENV.get("DISCORD_AUTH_USERS", "").split(",") if u.strip()}


def render_active() -> bool:
    return subprocess.run(["pgrep", "-f", RENDER_PAT],
                          capture_output=True).returncode == 0


def engine_up() -> bool:
    r = subprocess.run(
        ["curl", "-s", "--max-time", "3", "http://127.0.0.1:8322/v1/models"],
        capture_output=True, text=True)
    return r.returncode == 0 and '"data"' in (r.stdout or "")


async def wait_idle_and_engine() -> None:
    """Block until no render is running and the chat engine answers again."""
    waited = 0
    while render_active():
        await asyncio.sleep(15)
        waited += 15
    for _ in range(72):  # up to 6 min for the engine to come back
        if engine_up():
            return
        await asyncio.sleep(5)
    log("engine never answered; starting anyway (caption step may be skipped)")


async def run_pipeline(refs: list[str]) -> tuple[str | None, bool, str]:
    """Run scripts/run.sh on the refs. Returns (run_dir, ok, log_tail)."""
    proc = await asyncio.create_subprocess_exec(
        str(RUN_SH), *refs,
        cwd=str(ROOT),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate()
    stdout = out.decode(errors="replace")
    run_dir = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith(str(RUNS)):
            run_dir = line
            break
    if not run_dir:
        tail = err.decode(errors="replace")[-1500:]
        return None, False, tail
    logfile = Path(run_dir) / "run.log"
    deadline = asyncio.get_event_loop().time() + RUN_TIMEOUT
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(10)
        text = logfile.read_text(errors="replace") if logfile.exists() else ""
        m = re.findall(r"\[detached\] exit (\d+)", text)
        if "scrape failed" in text:
            return run_dir, False, "scrape failed"
        if m:
            return run_dir, m[-1] == "0", f"exit {m[-1]}"
    return run_dir, False, "timeout"


def read_qa(run_dir: str) -> dict:
    def load(p: Path):
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}

    qa = Path(run_dir) / "qa"
    face = load(qa / "face_score.json")
    segments = []
    for p in sorted(qa.glob("video_seg*_garment_check.json")):
        d = load(p)
        if d:
            segments.append(d)
    if not segments:
        for p in sorted(qa.glob("keyframe*_garment_check.json")):
            d = load(p)
            if d:
                segments.append(d)
    return {
        "face_mean": face.get("sim_ref_mean"),
        "face_verdict": face.get("verdict"),
        "segments": [{"match": s.get("match"), "score": s.get("score"),
                      "differences": s.get("differences", [])} for s in segments],
    }


def extract_refs(message: discord.Message) -> tuple[list[str], list[discord.Attachment]]:
    urls, seen = [], set()
    content = message.content or ""
    # Split on every URL start first, so links pasted with no space or newline
    # between them ("...shared=truehttps://...") still register as separate
    # outfits instead of one glued-together URL.
    for chunk in re.split(r"(?=https?://)", content):
        m = re.match(rf"\s*({MYNTRA_URL})", chunk, re.I)
        if not m:
            continue
        u = m.group(1).rstrip(".,;)]}>\"'")
        if u not in seen:
            seen.add(u)
            urls.append(u)
    images = [a for a in message.attachments
              if (a.content_type or "").startswith("image/")
              or Path(a.filename).suffix.lower() in IMG_EXT]
    return urls, images


class Job:
    def __init__(self, message: discord.Message, refs: list[str], n_outfits: int):
        self.message = message
        self.refs = refs
        self.n_outfits = n_outfits


QUEUE: asyncio.Queue[Job] = asyncio.Queue()
_worker_started = False


async def handle_job(job: Job) -> None:
    msg = job.message
    try:
        await msg.reply(f"queued {job.n_outfits} outfit(s). waiting for a free GPU…",
                        mention_author=False)
        await wait_idle_and_engine()
        await msg.reply("rendering now — keyframes, then the clip. this takes a while.",
                        mention_author=False)
        run_dir, ok, detail = await run_pipeline(job.refs)
        if not run_dir:
            await msg.reply(f"could not start the render: {detail}", mention_author=False)
            return
        if not ok:
            await msg.reply(
                f"render failed ({detail}). If Myntra blocked the scrape, "
                f"post the product photos instead and I'll retry.",
                mention_author=False)
            return
        final = Path(run_dir) / "final.mp4"
        if not final.exists():
            await msg.reply(f"render finished but no final.mp4 in {run_dir}",
                            mention_author=False)
            return
        qa = read_qa(run_dir)
        stamp = Path(run_dir).name
        dest_name = f"haul_{stamp}.mp4"
        copied, where = nc_deliver.deliver(final, dest_name, log=log)

        lines = [f"**ready** — {job.n_outfits} outfit(s), {final.stat().st_size / 1e6:.0f} MB"]
        if copied:
            lines.append(where)
        else:
            lines.append(f"copy/link failed: {where}")
        if qa["face_mean"] is not None:
            verdict = qa["face_verdict"] or ("match" if qa["face_mean"] >= 0.4 else "MISMATCH")
            lines.append(f"face: {qa['face_mean']:.3f} ({verdict})")
        for i, s in enumerate(qa["segments"], 1):
            mark = "match" if s["match"] else "NO-MATCH"
            lines.append(f"seg{i} garment: {mark} {s['score']}")
        text = "\n".join(lines)
        # the clip link goes to whoever asked, as a private message
        try:
            await msg.author.send(text)
            log(f"DM'd result to {msg.author} ({msg.author.id})")
            await msg.reply("done — sent the link to you in DMs.", mention_author=False)
        except discord.Forbidden:
            log(f"DM to {msg.author.id} blocked; falling back to the channel")
            await msg.reply(text, mention_author=False)
    except Exception as exc:  # never let the worker die
        log(f"job error: {exc!r}")
        try:
            await msg.reply(f"something went wrong: {exc}", mention_author=False)
        except Exception:
            pass


async def worker() -> None:
    while True:
        job = await QUEUE.get()
        try:
            await handle_job(job)
        finally:
            QUEUE.task_done()


client_intents = discord.Intents.default()
client_intents.message_content = True
client = discord.Client(intents=client_intents)


@client.event
async def on_ready() -> None:
    global _worker_started
    if not _worker_started:
        _worker_started = True
        asyncio.create_task(worker())
    log(f"logged in as {client.user} (id {client.user.id}); "
        f"allowed users: {sorted(ALLOWED) or 'NONE'}")

    if not TOKEN:
        log("ERROR: DISCORD_BOT_TOKEN missing in .env")
    if not ALLOWED:
        log("ERROR: DISCORD_AUTH_USERS empty — the bot will answer nobody")


@client.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return
    if str(message.author.id) not in ALLOWED:
        log(f"ignored {message.author} ({message.author.id})")
        return

    content = (message.content or "").strip().lower()
    if content in ("!status", "status", "!ping", "ping"):
        state = "rendering" if render_active() else "idle"
        await message.reply(f"{state}; queue depth {QUEUE.qsize()}", mention_author=False)
        return
    if content in ("!help", "help"):
        await message.reply(
            "post one or more Myntra links (one per outfit, in order) and I'll "
            "render them into one clip. If Myntra blocks me, post the product "
            "photos instead.", mention_author=False)
        return

    urls, images = extract_refs(message)
    refs: list[str] = list(urls)
    if not urls and images:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = WORK / stamp
        folder.mkdir(parents=True, exist_ok=True)
        for i, att in enumerate(images):
            ext = Path(att.filename).suffix.lower() or ".jpg"
            await att.save(folder / f"{i:02d}{ext}")
        refs = [str(folder)]
    if not refs:
        return

    job = Job(message, refs, len(urls) if urls else 1)
    await QUEUE.put(job)
    log(f"queued from {message.author} ({message.author.id}): {refs}")


def main() -> int:
    if not TOKEN:
        log("no DISCORD_BOT_TOKEN; refusing to start")
        return 1
    client.run(TOKEN, log_handler=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
