#!/usr/bin/env python
"""数据集多段并发下载脚本 v2(纯标准库,按段校验断点续传)。

v2 修复: 每段下载后/复用时校验字节数,不完整段自动重下;拼接前全量校验。

用法:
    python yolo_optimization/scripts/download_dataset.py \
        --url "https://zenodo.org/records/3587843/files/TACO.zip?download=1" \
        --output data/raw/taco/TACO.zip \
        --segments 8
"""

from __future__ import annotations

import argparse
import logging
import math
import shutil
import sys
import threading
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("download_dataset")

RETRIES = 3
RESUME_RETRIES = 30  # 段内断点续传上限(每次失败从断点继续,避免整段重下)


def get_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        length = r.headers.get("Content-Length")
        if length is None:
            raise RuntimeError("server did not return Content-Length")
        return int(length)


def fetch_part(url: str, start: int, end: int, dest: Path) -> None:
    """下载 [start, end] 到 dest;支持段内断点续传(短读后从已写字节继续)。

    end 为 inclusive。短读/断连最多重试 RESUME_RETRIES 次,每次从 dest 当前
    长度对应的字节偏移续传,避免整段重下。
    """
    expected = end - start + 1
    for attempt in range(1, RESUME_RETRIES + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if have >= expected:
            return
        from_byte = start + have
        req = urllib.request.Request(
            url, headers={"Range": f"bytes={from_byte}-{end}"})
        try:
            mode = "ab" if have else "wb"
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, mode) as f:
                shutil.copyfileobj(r, f, length=4 * 1024 * 1024)
            got = dest.stat().st_size
            if got != expected:
                raise RuntimeError(f"short read: {got} != {expected}")
            return
        except Exception as e:  # noqa: BLE001
            if attempt == RESUME_RETRIES:
                raise
            log.warning("seg [%d-%d] attempt %d failed at %d B: %s; resume", start, end, attempt, have, e)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--url", required=True, help="下载 URL(支持 Range)")
    ap.add_argument("--output", required=True, type=Path, help="目标文件路径")
    ap.add_argument("--segments", type=int, default=8, help="并发分段数(默认 8)")
    ap.add_argument("--size", type=int, default=None, help="总字节数(缺省自动 HEAD 探测)")
    args = ap.parse_args()

    if args.segments < 1:
        log.error("--segments must be >= 1")
        return 1

    total = args.size or get_size(args.url)
    chunk = math.ceil(total / args.segments)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    parts = [args.output.with_name(f"{args.output.name}.part{i:03d}") for i in range(args.segments)]

    ranges = []
    for i in range(args.segments):
        start = i * chunk
        end = min((i + 1) * chunk - 1, total - 1)
        if start < total:
            ranges.append((i, start, end))

    need = 0
    for i, start, end in ranges:
        p = parts[i]
        if p.exists() and p.stat().st_size == end - start + 1:
            log.info("seg %d already complete, skip", i)
        else:
            need += 1
            log.info("seg %d need download [%d, %d] (%d bytes)", i, start, end, end - start + 1)

    lock = threading.Lock()

    def work(item: tuple[int, int, int]) -> None:
        i, start, end = item
        p = parts[i]
        if p.exists() and p.stat().st_size == end - start + 1:
            return
        fetch_part(args.url, start, end, p)
        with lock:
            log.info("seg %d done", i)

    if need:
        threads = [threading.Thread(target=work, args=(item,)) for item in ranges]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # 拼接前全量校验
    bad = [i for i, _, _ in ranges if not parts[i].exists()
           or parts[i].stat().st_size != ranges[i][2] - ranges[i][1] + 1]
    if bad:
        log.error("segments incomplete after download: %s — rerun to retry", bad)
        return 1

    with open(args.output, "wb") as out:
        for i, start, end in ranges:
            with open(parts[i], "rb") as f:
                shutil.copyfileobj(f, out, length=8 * 1024 * 1024)
    for p in parts:
        p.unlink()

    final_size = args.output.stat().st_size
    if final_size != total:
        log.error("size mismatch: got %d, expected %d", final_size, total)
        return 1
    log.info("done: %s (%d bytes)", args.output, final_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())