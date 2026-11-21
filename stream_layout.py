"""Live-stream layout: the streamer's webcam on top, the centre of the gameplay below.

Only live clips use it (IS_STREAM_SOURCE=1), and only when a webcam is found: main's
process_video_to_vertical tries it first and runs the normal reframe when this returns
False. reframe_v2 and camera_inset are not changed, so every other video keeps its layouts.

The vertical frame (1080x1920 for 9:16):
- top 30%: the whole webcam, fitted into the band with its aspect kept. The space left
  over is a blurred copy of the same webcam, so no gameplay shows inside the band.
- bottom 70%: the centre of the source at the band's aspect. It is a fixed crop, nothing
  tracks, so the picture does not move from frame to frame.

Finding the webcam, detection first:
1. Frames spread over the clip: faces are found on the full-resolution frame, edges on
   the frame at ANALYSIS_WIDTH.
2. MediaPipe face detection (the detector the pipeline already runs) finds the faces.
   The camera is the face that recurs most in the outer strips of the frame, where a
   webcam sits, not in the middle where game characters are.
3. The camera's sides are the outermost edges that persist across the frames around that
   face. The overlay border never moves while the person and the game do, and the outermost
   persistent edge is the border rather than the static things inside it.
A channel calibrated in webcam_boxes.json (WEBCAM_BOX) is the fallback when no webcam is found.
"""
import contextlib
import os
import subprocess
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

OUT_W = 1080
OUT_H = 1920
WEBCAM_BAND = 0.30        # share of the vertical frame given to the webcam
BLUR_SIGMA = 20           # the blurred copy behind the webcam in the band
SAMPLE_FRAMES = 40        # frames spread over the clip for the detection
ANALYSIS_WIDTH = 1280     # frames are analysed at this width: a small webcam's face needs the pixels
ANCHOR_SIDE = 0.40        # the camera's face sits in the outer 40% of the frame, left or right
ANCHOR_HITS_SHARE = 0.08  # the camera's face is seen in at least this share of the frames...
ANCHOR_HITS_MIN = 3       # ...and in at least this many (a game face is seen once or twice)
EDGE_SHARE_MIN = 0.20     # a side needs an edge in at least this share of the frames along its length
ENCODE_TIMEOUT = 3600     # seconds

Box = Tuple[int, int, int, int]                 # x, y, w, h in pixels
FaceFinder = Callable[[np.ndarray], List[Box]]  # one BGR frame -> face boxes


def _even(value: float) -> int:
    value = int(round(value))
    return max(2, value - value % 2)


def clamp_box(box: Sequence[float], src_w: int, src_h: int) -> Box:
    """The box inside the frame, with even sides and offsets (ffmpeg crops that way)."""
    x, y, w, h = (int(round(v)) for v in box)
    w = min(max(2, w), src_w)
    h = min(max(2, h), src_h)
    x = min(max(0, x), src_w - w)
    y = min(max(0, y), src_h - h)
    return x - x % 2, y - y % 2, w - w % 2, h - h % 2


def fit_size(w: int, h: int, box_w: int, box_h: int) -> Tuple[int, int]:
    """The camera at its own aspect, as large as the band allows."""
    scale = min(box_w / float(w), box_h / float(h))
    return min(box_w, _even(w * scale)), min(box_h, _even(h * scale))


def gameplay_crop(src_w: int, src_h: int, out_w: int, game_h: int) -> Box:
    """The centre of the source at the gameplay band's aspect: the full height of a wide
    source, the full width of a narrow one. Sides and offsets are even."""
    want = out_w / float(game_h)
    if src_w / float(src_h) > want:
        crop_w, crop_h = min(src_w, _even(src_h * want)), src_h
    else:
        crop_w, crop_h = src_w, min(src_h, _even(src_w / want))
    crop_w, crop_h = crop_w - crop_w % 2, crop_h - crop_h % 2
    x = (src_w - crop_w) // 2
    y = (src_h - crop_h) // 2
    return x - x % 2, y - y % 2, crop_w, crop_h


def filtergraph(src_w: int, src_h: int, box: Sequence[float],
                out_w: int = OUT_W, out_h: int = OUT_H) -> str:
    """The filter_complex for one clip: the webcam fitted into the top band over a blurred
    copy of itself, the centred gameplay below, stacked. Only crop, scale, gblur, overlay
    and vstack, with fixed numbers: nothing moves from frame to frame."""
    band_h = _even(out_h * WEBCAM_BAND)
    game_h = out_h - band_h
    x, y, w, h = clamp_box(box, src_w, src_h)
    fit_w, fit_h = fit_size(w, h, out_w, band_h)
    off_x = (out_w - fit_w) // 2
    off_y = (band_h - fit_h) // 2
    off_x, off_y = off_x - off_x % 2, off_y - off_y % 2
    gx, gy, gw, gh = gameplay_crop(src_w, src_h, out_w, game_h)
    return (
        "[0:v]split=3[bgsrc][fgsrc][gamesrc];"
        f"[bgsrc]crop={w}:{h}:{x}:{y},scale={out_w}:{band_h}:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{band_h},gblur=sigma={BLUR_SIGMA}[bandbg];"
        f"[fgsrc]crop={w}:{h}:{x}:{y},scale={fit_w}:{fit_h}[cam];"
        f"[bandbg][cam]overlay=x={off_x}:y={off_y}[top];"
        f"[gamesrc]crop={gw}:{gh}:{gx}:{gy},scale={out_w}:{game_h}[bottom];"
        "[top][bottom]vstack=inputs=2,setsar=1[v]"
    )


# --- faces -------------------------------------------------------------------------- #

@contextlib.contextmanager
def mediapipe_faces() -> Iterator[FaceFinder]:
    """The pipeline's own face detector (MediaPipe, full-range model), as a plain function."""
    import mediapipe as mp  # imported here so the layout module stays light to import

    with mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as detector:
        def find(frame_bgr: np.ndarray) -> List[Box]:
            return _faces_in(frame_bgr, detector)
        yield find


def _faces_in(frame_bgr: np.ndarray, detector) -> List[Box]:
    height, width = frame_bgr.shape[:2]
    result = detector.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    faces: List[Box] = []
    for detection in result.detections or []:
        box = detection.location_data.relative_bounding_box
        x0 = max(0, int(round(box.xmin * width)))
        y0 = max(0, int(round(box.ymin * height)))
        x1 = min(width, int(round((box.xmin + box.width) * width)))
        y1 = min(height, int(round((box.ymin + box.height) * height)))
        if x1 - x0 >= 4 and y1 - y0 >= 4:
            faces.append((x0, y0, x1 - x0, y1 - y0))
    return faces


def _same_face(seed: Box, faces: List[Box]) -> Optional[Box]:
    """The face in ``faces`` that sits where ``seed`` sits, if there is one."""
    sx, sy, sw, sh = seed
    scx, scy = sx + sw / 2.0, sy + sh / 2.0
    reach = 0.6 * max(sw, sh)
    best: Optional[Box] = None
    best_d = None
    for fx, fy, fw, fh in faces:
        distance = float(np.hypot(fx + fw / 2.0 - scx, fy + fh / 2.0 - scy))
        if distance <= reach and 0.7 <= fw / float(sw) <= 1.4 and (best_d is None or distance < best_d):
            best, best_d = (fx, fy, fw, fh), distance
    return best


def _anchor_face(per_frame: List[List[Box]], width: int) -> Optional[Tuple[float, float, float, float]]:
    """The face that recurs most among the faces in the outer strips of the frame (where a webcam
    sits), as the median of its matches. None when it recurs too rarely to be a camera."""

    def in_strip(face: Box) -> bool:
        centre = face[0] + face[2] / 2.0
        return centre < ANCHOR_SIDE * width or centre > (1.0 - ANCHOR_SIDE) * width

    strips = [[f for f in faces if in_strip(f)] for faces in per_frame]
    best: List[Box] = []
    for faces in strips:
        for seed in faces:
            matches = [m for m in (_same_face(seed, others) for others in strips) if m is not None]
            if len(matches) > len(best):
                best = matches
    if len(best) < max(ANCHOR_HITS_MIN, ANCHOR_HITS_SHARE * len(per_frame)):
        return None
    median = np.median(np.array(best, dtype=float), axis=0)
    return float(median[0]), float(median[1]), float(median[2]), float(median[3])


# --- the camera's sides ----------------------------------------------------------- #

def _persistence(frames_bgr: List[np.ndarray]) -> np.ndarray:
    """How often each pixel lies on an edge across the frames, from 0 to 1."""
    acc = None
    for frame in frames_bgr:
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        edges = (cv2.Canny(gray, 25, 75) > 0).astype(np.float32)
        acc = edges if acc is None else acc + edges
    return acc / float(len(frames_bgr))


def _smooth(profile: np.ndarray) -> np.ndarray:
    return np.convolve(profile, np.ones(3, dtype=np.float32) / 3.0, mode="same")


def _outermost(profile: np.ndarray, lo: float, hi: float, outward: int) -> Optional[int]:
    """The outermost index in [lo, hi] that is nearly as persistent as the strongest one, so the
    camera's border wins over the static things inside it (a desk, a wall). ``outward`` is -1 for
    a left or top side and +1 for a right or bottom side. None when nothing is persistent enough."""
    lo_i = max(0, int(np.floor(lo)))
    hi_i = min(len(profile) - 1, int(np.ceil(hi)))
    if hi_i <= lo_i:
        return None
    segment = profile[lo_i:hi_i + 1]
    peak = float(segment.max())
    if peak < EDGE_SHARE_MIN:
        return None
    strong = np.nonzero(segment >= 0.6 * peak)[0]
    return lo_i + int(strong[0] if outward < 0 else strong[-1])


def _camera_rectangle(face: Tuple[float, float, float, float],
                      persistence: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """The camera's rectangle (x, y, w, h) in analysis pixels, from the persistent sides around the face."""
    height, width = persistence.shape
    fx, fy, fw, fh = face
    cx, cy = fx + fw / 2.0, fy + fh / 2.0
    rows = persistence[max(0, int(cy - 2.4 * fh)):min(height, int(cy + 2.4 * fh) + 1), :]
    cols = persistence[:, max(0, int(cx - 2.4 * fw)):min(width, int(cx + 2.4 * fw) + 1)]
    # The 60th percentile along each line: a side is an edge along most of its length, and the
    # camera need not fill the whole search window (it can be shorter, or cut by the frame).
    column_profile = _smooth(np.quantile(rows, 0.6, axis=0))
    row_profile = _smooth(np.quantile(cols, 0.6, axis=1))
    sides = {
        "left": _outermost(column_profile, cx - 3.6 * fw, cx - 0.7 * fw, -1),
        "right": _outermost(column_profile, cx + 0.7 * fw, cx + 3.6 * fw, +1),
        "top": _outermost(row_profile, cy - 2.6 * fh, cy - 0.5 * fh, -1),
        "bottom": _outermost(row_profile, cy + 0.8 * fh, cy + 3.6 * fh, +1),
    }
    if sum(value is None for value in sides.values()) > 1:
        return None  # two sides without an edge: no overlay can be bounded from this
    # A side without an edge is placed from the face instead; a side past the frame is the frame's edge.
    x0 = sides["left"] if sides["left"] is not None else max(0.0, cx - 1.7 * fw)
    x1 = sides["right"] if sides["right"] is not None else min(width - 1.0, cx + 1.7 * fw)
    y0 = sides["top"] if sides["top"] is not None else max(0.0, cy - 1.1 * fh)
    y1 = sides["bottom"] if sides["bottom"] is not None else min(height - 1.0, cy + 2.4 * fh)
    w, h = x1 - x0, y1 - y0
    if not (w >= 1.5 * fw and h >= 1.5 * fh and 0.8 <= w / h <= 2.6):
        return None
    if not (x0 <= cx <= x1 and y0 <= cy <= y1):
        return None
    if w * h > 0.6 * width * height:
        return None
    return x0, y0, w, h


def detect_webcam_box(frames_bgr: List[np.ndarray], src_w: int, src_h: int,
                      face_finder: FaceFinder) -> Optional[Box]:
    """The webcam's box in source pixels, from frames of one source (at the analysis width).
    None when no face recurs in the strips, or no rectangle bounds it."""
    return _webcam_from_faces(frames_bgr, [face_finder(frame) for frame in frames_bgr], src_w, src_h)


def _webcam_from_faces(frames_bgr: List[np.ndarray], per_frame: List[List[Box]],
                       src_w: int, src_h: int) -> Optional[Box]:
    """The webcam's box from the analysis frames and the faces found in each of them (analysis px)."""
    if len(frames_bgr) < 3:
        return None
    height, width = frames_bgr[0].shape[:2]
    face = _anchor_face(per_frame, width)
    if face is None:
        return None
    rect = _camera_rectangle(face, _persistence(frames_bgr))
    if rect is None:
        return None
    sx, sy = src_w / float(width), src_h / float(height)
    x, y, w, h = rect
    return round(x * sx), round(y * sy), round(w * sx), round(h * sy)


# --- reading the video ---------------------------------------------------------------- #

def _resized(frame: np.ndarray, width: int) -> np.ndarray:
    height = max(2, int(round(frame.shape[0] * width / float(frame.shape[1]))))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _sample_with_faces(video_path: str, find: FaceFinder, count: int = SAMPLE_FRAMES,
                       width: int = ANALYSIS_WIDTH) -> Tuple[List[np.ndarray], List[List[Box]]]:
    """Frames spread evenly over the video at the analysis width, and the faces found in each one.
    The faces are found on the full-resolution frame: a small webcam's face is too small at 1280."""
    cap = cv2.VideoCapture(video_path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames: List[np.ndarray] = []
        per_frame: List[List[Box]] = []
        if total <= 0:
            return frames, per_frame
        for index in np.linspace(0, total - 1, count + 2)[1:-1].astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            scale = width / float(frame.shape[1])
            frames.append(_resized(frame, width))
            per_frame.append([(int(round(x * scale)), int(round(y * scale)),
                               int(round(w * scale)), int(round(h * scale)))
                              for x, y, w, h in find(frame)])
        return frames, per_frame
    finally:
        cap.release()


def _source_size(video_path: str) -> Tuple[int, int]:
    cap = cv2.VideoCapture(video_path)
    try:
        return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
    finally:
        cap.release()


def find_webcam(video_path: str, src_w: int, src_h: int) -> Optional[Box]:
    """The webcam's box, found in the video itself with MediaPipe faces and persistent edges."""
    with mediapipe_faces() as find:
        frames, per_frame = _sample_with_faces(video_path, find)
    return _webcam_from_faces(frames, per_frame, src_w, src_h)


def calibrated_box(src_w: int, src_h: int) -> Optional[Box]:
    """The box calibrated for this channel (WEBCAM_BOX, set by a live job), when there is one."""
    import camera_inset  # imported here: keeps this module light to import
    return camera_inset.manual_box_from_env(src_w, src_h)


# --- rendering -------------------------------------------------------------------------- #

def _encode(input_video: str, out_path: str, graph: str) -> bool:
    import ffmpeg_utils
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input_video,
           "-filter_complex", graph, "-map", "[v]", "-map", "0:a?",
           *ffmpeg_utils.video_encode_args(ffmpeg_utils.QUALITY), "-pix_fmt", "yuv420p",
           *ffmpeg_utils.audio_encode_args(), *ffmpeg_utils.METADATA_SCRUB,
           "-movflags", "+faststart", out_path]
    proc = subprocess.run(cmd, capture_output=True, timeout=ENCODE_TIMEOUT)
    if proc.returncode != 0 or not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        err = proc.stderr.decode("utf-8", "replace")[-300:]
        print(f"   ❌ Stream layout encode failed: {err}")
        return False
    return True


def render_clip(input_video: str, final_output_video: str,
                out_w: int = OUT_W, out_h: int = OUT_H) -> bool:
    """Render a live clip in the streamer layout. The webcam is detected in the clip first; the
    channel's calibrated box (WEBCAM_BOX) is the fallback. False, with nothing written, when neither
    finds a webcam: the caller then runs the normal reframe."""
    src_w, src_h = _source_size(input_video)
    if not src_w or not src_h:
        return False
    box = None
    source = "detected"
    try:
        box = find_webcam(input_video, src_w, src_h)
    except Exception as e:  # a broken detector must not cost a calibrated channel its layout
        print(f"   ⚠️ Webcam detection failed ({type(e).__name__}: {e}); trying the calibrated box.")
    if box is None:
        box = calibrated_box(src_w, src_h)
        source = "calibrated"
    if box is None:
        print("   📹 No webcam found: the clip takes the normal reframe.")
        return False
    print(f"   📹 Webcam {source} at {box} (source {src_w}x{src_h})")
    return _encode(input_video, final_output_video,
                   filtergraph(src_w, src_h, box, out_w, out_h))
