#!/usr/bin/env python3
"""Print real download progress for the DeepSeek-V4-Flash-0731 checkpoint download.

aria2c preallocates full file size via fallocate() immediately, so du/ls size is
useless for progress -- must parse aria2c's own periodic summary for in-flight
files and only count completed files (no .aria2 sidecar) at their full size.
"""
import glob
import os
import re
import sys

TARGET = os.path.expanduser("~/models/models/deepseek-ai-DeepSeek-V4-Flash-0731")
TOTAL_BYTES = 166_898_661_074

UNIT = {"KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "B": 1}


def to_bytes(val, unit):
    return float(val) * UNIT[unit]


def main():
    logs = sorted(glob.glob(os.path.expanduser(
        "~/CMP-170HX-PROJECT/bench-results/deepseek-v4-native-download_j1_*.log")))
    if not logs:
        print("no j1 log found")
        return
    log = logs[-1]

    # completed files: safetensors present, no .aria2 sidecar
    done_bytes = 0
    completed = 0
    for f in glob.glob(os.path.join(TARGET, "*.safetensors")):
        if not os.path.exists(f + ".aria2"):
            done_bytes += os.path.getsize(f)
            completed += 1
    # tiny metadata files (config/tokenizer etc), always small, count if present w/o sidecar
    for f in glob.glob(os.path.join(TARGET, "*.json")):
        if not f.endswith(".aria2") and not os.path.exists(f + ".aria2"):
            done_bytes += os.path.getsize(f)

    # in-flight file(s): parse the LAST summary block in the log
    with open(log, "r", errors="ignore") as fh:
        text = fh.read()
    blocks = text.split("Download Progress Summary")
    inflight_bytes = 0
    current_file = None
    if len(blocks) > 1:
        last = blocks[-1]
        # lines like: [#gid 470MiB/0.9GiB(46%) CN:1 DL:1.0MiB ETA:8m23s]
        for m in re.finditer(
            r"\[#\w+\s+([\d.]+)(KiB|MiB|GiB|B)/[\d.]+(?:KiB|MiB|GiB|B)\(\d+%\).*?\]\s*\nFILE:\s*(\S+)",
            last,
        ):
            val, unit, fname = m.groups()
            inflight_bytes += to_bytes(val, unit)
            current_file = os.path.basename(fname)

    total_done = int(done_bytes + inflight_bytes)

    # Downloaded bytes are physically monotonic; a lower reading than last time means we
    # caught the log mid-write between summary blocks, not a real regression. Clamp against
    # the last known value so a parsing race never reports a false stall/regression.
    floor_file = "/tmp/dsv4-dl-progress-floor.txt"
    prev = 0
    try:
        with open(floor_file) as fh:
            prev = int(fh.read().strip())
    except (FileNotFoundError, ValueError):
        pass
    if total_done < prev:
        total_done = prev
    else:
        with open(floor_file, "w") as fh:
            fh.write(str(total_done))

    pct = 100 * total_done / TOTAL_BYTES
    print(f"done: {total_done/1024**3:.2f} GiB / {TOTAL_BYTES/1024**3:.2f} GiB "
          f"({pct:.2f}%)  completed_shards={completed}/48  current={current_file}")
    print(total_done)  # last line: raw bytes, for rate calc by the caller


if __name__ == "__main__":
    main()
