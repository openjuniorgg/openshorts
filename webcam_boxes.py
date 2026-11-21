"""Per-channel webcam boxes, calibrated by hand where detection is not reliable.

webcam_boxes.json maps "<platform>:<channel>" to [x, y, width, height] as fractions
of the frame (0 to 1), so one box works at any source resolution.
"""
import json
import os

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webcam_boxes.json")


def load(path=DEFAULT_PATH):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def lookup(platform, channel, path=DEFAULT_PATH):
    """"x,y,w,h" as fractions for this channel, or None when it has no calibrated box."""
    key = f"{str(platform or '').lower()}:{str(channel or '').lower()}"
    box = load(path).get(key)
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        return None
    try:
        return ",".join(f"{float(v):.4f}" for v in box)
    except (TypeError, ValueError):
        return None
