"""Still frames of a clip for the model, each with its time.

OpenRouter and Ollama read pictures, not video files. Every stage that used to upload
footage sends a few frames spread over the stretch instead, and tells the model the
time of each one, so cut points can be placed against what it actually judged.

The ffmpeg CLI writes the frames into a directory. A frame that does not decode is
simply absent, and the others keep their times.
"""
import os
import subprocess
from typing import List, Tuple

MIN_INTERVAL_SECONDS = 2.0


def sample(video_path, start, end, out_dir, *, count, width,
           min_interval=MIN_INTERVAL_SECONDS) -> Tuple[float, List[Tuple[float, bytes]]]:
    """Up to ``count`` JPEG frames of [start, end], ``width`` px wide, one every
    ``interval`` seconds and never closer than ``min_interval``.

    Returns ``(interval, [(t, jpeg_bytes), ...])``. ``t`` is in seconds from ``start``,
    so the first frame is at 0. Raises RuntimeError when ffmpeg decodes no frame at all."""
    length = max(0.001, float(end) - float(start))
    count = max(1, int(count))
    interval = max(float(min_interval), length / count)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-ss", f"{float(start):.3f}", "-to", f"{float(end):.3f}", "-i", video_path,
           "-vf", f"fps=1/{interval:.3f},scale={int(width)}:-2",
           "-q:v", "4", os.path.join(out_dir, "frame_%05d.jpg")]
    res = subprocess.run(cmd, capture_output=True)
    names = sorted(n for n in os.listdir(out_dir) if n.startswith("frame_") and n.endswith(".jpg"))
    if not names:
        stderr = (res.stderr or b"").decode("utf-8", errors="replace").strip()[-300:]
        raise RuntimeError(f"no frames decoded from {float(start):.0f}-{float(end):.0f}s: {stderr}")
    frames: List[Tuple[float, bytes]] = []
    for index, name in enumerate(names[:count]):
        with open(os.path.join(out_dir, name), "rb") as fh:
            data = fh.read()
        if data:
            frames.append((round(index * interval, 3), data))
    return interval, frames


def labels(frames) -> str:
    """``1=0.0s, 2=9.4s, ...``: the image order and each frame's time, for the prompt."""
    return ", ".join(f"{i + 1}={t:.1f}s" for i, (t, _) in enumerate(frames))
