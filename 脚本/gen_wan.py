#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a sprite-sheet segment through the local Magpie gateway (wan2.7-image-pro).

Usage:
    python gen_wan.py --prompt prompt.txt --out gemini/素材/idle/part1.jpg \
        [--ref path1.png path2.png] [--size 1024x1024]

The prompt file is read as UTF-8 text. Reference images are inlined as base64
data URLs so the gateway never has to fetch them itself.
"""
import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.request

ENDPOINT_GEN = "http://127.0.0.1:3425/v1/images/generations"
ENDPOINT_EDIT = "http://127.0.0.1:3425/v1/images/edits"
DEFAULT_MODEL = "aliyun-qw/wan2.7-image-pro"


def data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as fh:
        return f"data:{mime};base64," + base64.b64encode(fh.read()).decode()


def post(url: str, payload: dict, timeout: int = 600) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download(url: str, out: str, timeout: int = 300) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(out, "wb") as fh:
        fh.write(resp.read())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="path to a UTF-8 prompt text file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", nargs="*", default=[], help="reference image paths")
    ap.add_argument("--size", default="1280x960")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    with open(args.prompt, "r", encoding="utf-8") as fh:
        prompt = fh.read().strip()

    payload = {"model": args.model, "prompt": prompt, "n": 1, "size": args.size}
    url = ENDPOINT_GEN
    if args.ref:
        payload["images"] = [data_url(p) for p in args.ref]
        url = ENDPOINT_EDIT

    last_err = None
    for attempt in range(1, args.retries + 2):
        try:
            res = post(url, payload)
            item = (res.get("data") or [{}])[0]
            link = item.get("url")
            if not link and item.get("b64_json"):
                raw = base64.b64decode(item["b64_json"])
                os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                with open(args.out, "wb") as fh:
                    fh.write(raw)
                print(args.out)
                return 0
            if not link:
                raise RuntimeError(f"no image in response: {json.dumps(res)[:400]}")
            download(link, args.out)
            print(args.out)
            return 0
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"[retry {attempt}] {exc}", file=sys.stderr)
            if attempt <= args.retries:
                time.sleep(4 * attempt)
    print(f"FAILED: {last_err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
