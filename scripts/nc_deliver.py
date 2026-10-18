#!/usr/bin/env python3
"""Copy an assembled clip to the storage box and return a Nextcloud public link.

Used by scripts/discord_bot.py, and runnable on its own:

    python3 scripts/nc_deliver.py runs/<stamp>/final.mp4 [remote-name]

Steps: scp the file to /storage/ChaosWares2/haul/ on 192.168.0.245, rescan that
folder in Nextcloud, create (or reuse) a public link share, and print the link.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/affiliate-marketing")
ENV_FILE = ROOT / ".env"


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

STORAGE_DIR = ENV.get("STORAGE_DIR", "/storage/ChaosWares2/haul")
SSH = ENV.get("NEXTCLOUD_SSH", "root@192.168.0.245")
CONTAINER = ENV.get("NEXTCLOUD_CONTAINER", "nextcloud_stack-app-1")
NC_URL = ENV.get("NEXTCLOUD_URL", "https://192.168.0.245").rstrip("/")
NC_USER = ENV.get("NEXTCLOUD_USER", "admin")
NC_PW = ENV.get("NEXTCLOUD_APP_PASSWORD", "")
WEBROOT = "/" + ENV.get("NEXTCLOUD_WEBROOT", "/nextcloud").strip("/")
SHARE_DIR = "/" + ENV.get("NEXTCLOUD_SHARE_DIR", "/ChaosWares2/haul").strip("/")
PUBLIC_BASE = ENV.get("NEXTCLOUD_PUBLIC_BASE", "").rstrip("/") or NC_URL
SSH_KEY = "/root/.ssh/id_ed25519"


def _ssh(remote_cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-i", SSH_KEY, "-o", "BatchMode=yes",
         "-o", "StrictHostKeyChecking=accept-new", SSH, remote_cmd],
        capture_output=True, text=True)


def _ocs(path: str, method: str = "GET", data: dict | None = None):
    url = f"{NC_URL}{WEBROOT}/ocs/v2.php{path}"
    cmd = ["curl", "-sk", "-u", f"{NC_USER}:{NC_PW}",
           "-H", "OCS-APIRequest: true", "-X", method, url,
           "-o", "-", "-w", "\n%{http_code}"]
    sep = "&" if "?" in url else "?"
    cmd[cmd.index(url)] = f"{url}{sep}format=json"
    for k, v in (data or {}).items():
        cmd += ["--data-urlencode", f"{k}={v}"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    body, _, code = out.rpartition("\n")
    try:
        parsed = json.loads(body)
    except Exception:
        parsed = {"raw": body}
    return code.strip(), parsed


def _public_url(token: str) -> str:
    return f"{PUBLIC_BASE}{WEBROOT}/s/{token}"


def deliver(local_mp4: str | Path, remote_name: str | None = None,
            log=print) -> tuple[bool, str]:
    """Copy + share. Returns (ok, link_or_error)."""
    src = Path(local_mp4)
    if not src.exists():
        return False, f"missing {src}"
    name = remote_name or src.name

    # 1. copy to the storage box
    cp = subprocess.run(
        ["scp", "-i", SSH_KEY, "-o", "BatchMode=yes",
         "-o", "StrictHostKeyChecking=accept-new",
         str(src), f"{SSH}:{STORAGE_DIR}/{name}"],
        capture_output=True, text=True)
    if cp.returncode != 0:
        return False, f"scp failed: {(cp.stderr or '').strip()[-300:]}"
    log(f"copied -> {SSH}:{STORAGE_DIR}/{name}")

    # 2. make Nextcloud notice the new file
    nc_path = f"{SHARE_DIR}/{name}"
    _ssh(f"chmod 644 '{STORAGE_DIR}/{name}'")
    scan = _ssh(
        f"docker exec -u www-data {CONTAINER} php occ "
        f"files:scan --path=\"admin/files{SHARE_DIR}\"")
    if scan.returncode != 0:
        log(f"scan warning: {(scan.stderr or scan.stdout or '').strip()[-300:]}")

    # 3. reuse an existing public link, else create one
    token = None
    _code, existing = _ocs(f"/apps/files_sharing/api/v1/shares?path={nc_path}")
    shares = existing.get("ocs", {}).get("data") or []
    for s in shares if isinstance(shares, list) else [shares]:
        if str(s.get("share_type")) == "3" and s.get("token"):
            token = s["token"]
            break

    msg = ""
    if not token:
        code, res = _ocs("/apps/files_sharing/api/v1/shares", "POST",
                         {"path": nc_path, "shareType": "3", "permissions": "1"})
        data = res.get("ocs", {}).get("data")
        token = data.get("token") if isinstance(data, dict) else None
        msg = res.get("ocs", {}).get("meta", {}).get("message", "share failed")
        if not token:
            log(f"share POST {code}: {msg}")
    if not token:
        return False, f"could not create a public link ({msg or code})"

    link = _public_url(token)
    log(f"public link -> {link}")
    return True, link


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: nc_deliver.py <local.mp4> [remote-name]", file=sys.stderr)
        return 2
    ok, result = deliver(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(result)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
