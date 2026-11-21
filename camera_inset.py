"""Find the webcam inset in a screen recording, and frame the two apart.

The case: one source, the whole screen (a game, a desktop, an editor) with the
person composited into a corner. It is how OBS records and how every stream VOD
looks, and it is the case screencast_layout gets wrong. There the speaker band
is a large crop taken AROUND the face, which on a full-screen source means the
band is mostly more screen: measured on an Excel walkthrough, the output showed
the same spreadsheet twice, once whole and once enlarged.

What this needs instead is the inset's own rectangle, so the two bands can hold
genuinely different things — screen above, person below.

Finding it exactly is harder than it looks. The inset is not always a rectangle
(circles and clipped polygons are common), it has no reliable border, and on
gameplay the background moves as much as the person does, which rules out
temporal-variance tricks. What IS reliable is that the inset is anchored to a
corner and that a person detector fires inside it. So: locate the person, decide
which corner they are nearest, and grow a box from that corner until it covers
them with margin.
"""
import math
import os

from ffmpeg_utils import blurred_backdrop

CORNER_MARGIN = 0.20   # a subject this far from an edge (as a fraction of the
                       # frame) still counts as anchored to it

# An inset subject is SMALL and OFF-CENTRE; a presenter filling the shot is
# neither. Requiring the detection to actually touch two edges was tried first
# and rejected: YOLO returns an upper-body box that stops at the chest, so a
# webcam pinned to the right edge measured 18% clear of the bottom and got
# thrown out. Measured on four real clips, these two properties separate the
# cases where "touching a corner" did not.
MAX_SUBJECT_HEIGHT = 0.35   # fraction of frame height

# Horizontal offset from centre, as a fraction of frame width. HORIZONTAL
# specifically: a composited camera is pinned to the left or right side, while a
# talking head is centred left-to-right even when their face sits high in the
# shot. Accepting offset on either axis let four talking heads through, all of
# them with a face near the top edge. Measured on those nine candidates the two
# groups do not overlap: real insets sat 0.37-0.43 from centre, the talking
# heads 0.01-0.12.
MIN_OFFSET = 0.18

# How much bigger than the detected head/upper body the inset is assumed to be.
# A webcam frames head and shoulders, and detectors return the head or the upper
# body, so the box has to grow to reach the inset's real edges.
INSET_PADDING = float(os.environ.get("INSET_PADDING", "1.45"))

# Guard rails as a fraction of frame height. Below the floor there is nothing
# worth showing; above the ceiling it stopped being an inset and the layout
# should not be used at all.
MIN_INSET_HEIGHT = 0.10
MAX_INSET_HEIGHT = 0.38

# An inset is pinned to the same pixels for the whole recording; a presenter
# walks about. Spread of the detected centres, as a fraction of frame width,
# above which this is a person in a room rather than a composited camera.
# Measured on the two false positives this rule was written for: a real inset
# moved 3-11px across samples, a presenter 316px.
MAX_CENTRE_SPREAD = 0.05

# Room kept above the detection, as a fraction of its height (see inset_box).
FACE_HEADROOM = 0.35
BODY_HEADROOM = 0.10

# A webcam window frames head and shoulders: the face is about a third of its
# width. Used for overlays that float mid-screen, where inset_box's corner
# anchoring does not apply and pushed the crop up onto the ceiling, and to grow
# a face detection to the inset in a corner.
OVERLAY_FACE_WIDTHS = 3.0


def nearest_corner(box, frame_w, frame_h):
    """Which corner the subject sits in: (horizontal, vertical) as strings.

    Returns e.g. ("left", "bottom"). A subject in the middle of the frame is
    reported by its nearest edges anyway; callers use `is_cornered` to reject.
    """
    cx = box[0] + box[2] / 2.0
    cy = box[1] + box[3] / 2.0
    return ("left" if cx < frame_w / 2 else "right",
            "top" if cy < frame_h / 2 else "bottom")


def is_cornered(box, frame_w, frame_h, margin=CORNER_MARGIN):
    """True when the subject looks like a webcam inset rather than the shot.

    Small and off-centre, not "touches two edges": see MAX_SUBJECT_HEIGHT.

    This is a sanity filter, not the decision. Whether the video is a screen
    recording with a camera in it at all is answered upstream by layout_picker;
    a game character standing in a corner would pass this test, and is kept out
    by the fact that nobody asked for this layout on that video.
    """
    x, y, w, h = box
    if h > frame_h * MAX_SUBJECT_HEIGHT:
        return False

    cx = (x + w / 2.0) / frame_w
    off_centre = abs(cx - 0.5) >= MIN_OFFSET

    near_edge = (x <= frame_w * margin or (x + w) >= frame_w * (1 - margin)
                 or y <= frame_h * margin or (y + h) >= frame_h * (1 - margin))
    return off_centre and near_edge


INSET_ASPECT = 16 / 9.0


def _place(x, y, w, h, new_w, new_h, frame_w, frame_h, horizontal, vertical):
    """Top-left corner of a new_w x new_h box grown around the detection."""
    # Anchor: keep the edge the subject is already near, grow the other way.
    if horizontal == "left":
        new_x = max(0, min(x - (new_w - w) / 2.0, frame_w - new_w))
        if x <= frame_w * CORNER_MARGIN:
            new_x = 0
    else:
        new_x = max(0, min(x + w + (new_w - w) / 2.0 - new_w, frame_w - new_w))
        if (x + w) >= frame_w * (1 - CORNER_MARGIN):
            new_x = frame_w - new_w

    # Vertical growth is biased upwards: detectors return the head or the chest,
    # and a portrait needs headroom, not more torso. Splitting the growth evenly
    # cropped the top of the head off on real clips.
    grow = new_h - h
    if vertical == "top":
        new_y = max(0, min(y - grow * 0.6, frame_h - new_h))
        if y <= frame_h * CORNER_MARGIN:
            new_y = 0
    else:
        # Pin to the bottom edge: the inset is there, and the detection sits
        # somewhere inside it rather than at its top.
        new_y = frame_h - new_h
        if (y + h) < frame_h * (1 - CORNER_MARGIN):
            new_y = max(0, min(y - grow * 0.6, frame_h - new_h))
    return new_x, new_y


def inset_box(box, frame_w, frame_h, padding=None, face=False):
    """Estimated inset rectangle (x, y, w, h) around a detected subject.

    Grown from the corner the subject is anchored to, so the box hugs the same
    edges the inset does instead of floating around the face.

    The height comes from assuming a 16:9 inset rather than from scaling the
    detection, because the detection is often a torso: YOLO returned a 246x86
    box for a webcam whose real rectangle was about 325x180, and padding that
    shape upwards still cut the head off. Deriving the height from the width
    and pinning the result to the corner reaches the top of the inset instead.

    ``face`` says the detection is a face box, not YOLO's upper body. A face is
    a third of a webcam's width (OVERLAY_FACE_WIDTHS), not 1/1.45 of it: grown
    by the body padding, a 128px face on a 720p Excel walkthrough gave a
    181px box pinned to the bottom edge that started 41px below the top of
    the face, and the enlarged band showed a forehead-less face (17-oct-2026).
    """
    if padding is None:
        padding = OVERLAY_FACE_WIDTHS if face else INSET_PADDING
    x, y, w, h = box
    horizontal, vertical = nearest_corner(box, frame_w, frame_h)

    new_w = min(frame_w, w * padding)
    # A face box is square, so tripling its height too would make a portrait
    # crop; the 16:9 assumption alone sets the height.
    new_h = new_w / INSET_ASPECT if face else max(h * padding, new_w / INSET_ASPECT)
    new_h = min(frame_h, new_h)
    # Never start below the head. A face box stops at the brows, so the hair
    # needs about a third of its height on top; YOLO's box already starts at
    # the top of the head.
    top_limit = max(0.0, y - h * (FACE_HEADROOM if face else BODY_HEADROOM))

    new_x, new_y = _place(x, y, w, h, new_w, new_h, frame_w, frame_h,
                          horizontal, vertical)
    if new_y > top_limit:
        if new_y + new_h >= frame_h - 1:
            # Pinned to the bottom edge: grow upwards (and keep 16:9) rather
            # than slide off the edge the inset sits on.
            new_h = frame_h - top_limit
            new_w = min(frame_w, max(new_w, new_h * INSET_ASPECT))
            new_x, _ = _place(x, y, w, h, new_w, new_h, frame_w, frame_h,
                              horizontal, vertical)
        new_y = top_limit

    return (int(round(new_x)), int(round(new_y)),
            int(round(new_w)), int(round(new_h)))


def usable(box, frame_h, min_ratio=MIN_INSET_HEIGHT, max_ratio=MAX_INSET_HEIGHT):
    """Whether an inset of this size is worth building a layout around."""
    return min_ratio * frame_h <= box[3] <= max_ratio * frame_h


def person_boxes(frame):
    """Every person in the frame as an upper-body box (x, y, w, h), in frame pixels.

    main.detect_person_yolo returns only the largest person, so a webcam smaller
    than an on-screen character was never considered. This uses the same model
    and the same upper-body approximation (the top 40% of the person box).
    """
    import main as m

    small, scale = m._detection_frame(frame)
    with m.DETECT_LOCK:
        results = m.model(small, verbose=False, classes=[0])  # class 0 is person
    boxes = []
    for result in results or []:
        for det in result.boxes:
            x1, y1, x2, y2 = [int(i * scale) for i in det.xyxy[0]]
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                boxes.append((x1, y1, w, int(h * 0.4)))
    return boxes


def manual_box_from_env(orig_w, orig_h):
    """The webcam box calibrated for this channel, in source pixels, or None.

    WEBCAM_BOX is "x,y,w,h" as fractions of the frame, set per live job from
    webcam_boxes.json. A calibrated box is known, so detection is skipped for it.
    """
    raw = os.environ.get("WEBCAM_BOX", "").strip()
    if not raw:
        return None
    try:
        fx, fy, fw, fh = (float(v) for v in raw.split(","))
    except ValueError:
        return None
    if not (0 <= fx < 1 and 0 <= fy < 1 and 0 < fw <= 1 and 0 < fh <= 1):
        return None
    x, y = int(round(fx * orig_w)), int(round(fy * orig_h))
    w = min(int(round(fw * orig_w)), orig_w - x)
    h = min(int(round(fh * orig_h)), orig_h - y)
    if w < 2 or h < 2:
        return None
    return (x, y, w, h)


def detect(video_path, samples=10):
    """Median inset rectangle across sampled frames, or None.

    None means "no usable inset here", which callers must treat as "use another
    layout" rather than as an error.
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0 or not frame_w:
        cap.release()
        return None
    # A channel with a calibrated box (WEBCAM_BOX, set per live job) skips detection.
    manual = manual_box_from_env(frame_w, frame_h)
    if manual:
        cap.release()
        return manual

    # The models load only when something has to be detected.
    import main as m
    import screencast_layout

    boxes = []
    try:
        for i in range(samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total / samples))
            ok, frame = cap.read()
            if not ok:
                continue
            # Faces first, body as fallback: a webcam inset is small, and on a
            # 1080p source the face inside it is often too few pixels for
            # BlazeFace even at full resolution, while YOLO still finds the
            # person.
            # Every face and every person is a candidate, not only the biggest one:
            # a webcam is small, and an on-screen character (a cutscene face or a
            # standing NPC) is larger and used to win the single slot. The
            # clustering below keeps the place where candidates keep coming back.
            for face in screencast_layout.detect_faces_full_res(frame):
                if is_cornered(face['box'], frame_w, frame_h):
                    boxes.append(inset_box(face['box'], frame_w, frame_h, face=True))
            for body in person_boxes(frame):
                if is_cornered(body, frame_w, frame_h):
                    boxes.append(inset_box(body, frame_w, frame_h, face=False))
    finally:
        cap.release()

    if len(boxes) < max(3, samples // 3):
        return None

    return stable_box(boxes, frame_w, frame_h)


def stable_box(boxes, frame_w, frame_h, min_agree=3):
    """Median of the inset boxes that agree with each other, or None."""
    import numpy as np

    arr = np.array(boxes, dtype=float)

    # Stability gate. Without it, a presenter standing in front of a screen and
    # an ordinary talking head both come back as "insets" and the layout puts a
    # zoom of someone's face under a copy of the same shot.
    #
    # Measured against the plain min-to-max range first, which was far too
    # brittle: one sample landing on a face elsewhere in the frame killed three
    # genuine insets. So cluster around the median instead and require most
    # detections to agree, which tolerates the odd stray without tolerating a
    # subject that actually moves.
    centres_x = arr[:, 0] + arr[:, 2] / 2.0
    centres_y = arr[:, 1] + arr[:, 3] / 2.0
    tolerance = frame_w * MAX_CENTRE_SPREAD
    # Several subjects can be candidates in the same run (a webcam, a cutscene
    # face, a person behind it), so the agreement test runs on the best-supported
    # place rather than on the median of everything. A webcam is pinned, so its
    # candidates pile up in one spot; a wandering presenter's do not.
    near = ((np.abs(centres_x[:, None] - centres_x[None, :]) <= tolerance)
            & (np.abs(centres_y[:, None] - centres_y[None, :]) <= tolerance))
    support = near.sum(axis=1)
    best = int(np.argmax(support))
    if support[best] < max(min_agree, 0.3 * len(boxes)):
        return None

    arr = arr[near[best]]
    median = tuple(int(v) for v in np.median(arr, axis=0))
    if not usable(median, frame_h):
        return None
    return median


def overlay_box(box, frame_w, frame_h, widen=OVERLAY_FACE_WIDTHS):
    """A 16:9 box centred on a detected face (or upper body), clamped."""
    x, y, w, h = box
    new_w = min(frame_w, w * widen)
    new_h = min(frame_h, new_w / INSET_ASPECT)
    cx, cy = x + w / 2.0, y + h / 2.0
    new_x = max(0, min(cx - new_w / 2.0, frame_w - new_w))
    new_y = max(0, min(cy - new_h * 0.45, frame_h - new_h))
    return (int(round(new_x)), int(round(new_y)),
            int(round(new_w)), int(round(new_h)))


def detect_in_scene(video_path, start_f, end_f, samples=5):
    """A presenter window laid over the screen in this one scene, or None.

    ``detect`` is the clip-wide check and insists on a corner, because nothing
    upstream of it says the frame is a screen. Here the shot check already did
    (this runs only on scenes the AI model called "screen"), so a small face that
    stays put is the presenter's camera wherever it floats: a QuickTime or Loom
    window over a document sits mid-left, not in a corner, and the screen-only
    layout shrank both into a 1080px-wide strip (Ty Myers tutorial,
    6-oct-2026). What still rules a subject out: being big (that is the shot,
    not an overlay) or moving between samples.

    The caller only asks where the shot check said a live presenter camera is
    laid over the screen (presenter_cam). That gate is what keeps game
    characters, photos on slides and cover art out; the YOLO body fallback
    stays because BlazeFace misses the webcam face in about half the samples
    of a dim or small window (ticker_VUqnFqEHy2E), and without it a real
    presenter window is lost.
    """
    import cv2
    import numpy as np
    import main as m
    import screencast_layout

    if end_f - 1 < start_f:
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    boxes = []
    try:
        for f_idx in np.linspace(start_f, end_f - 1, samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(f_idx)))
            ok, frame = cap.read()
            if not ok:
                continue
            faces = screencast_layout.detect_faces_full_res(frame)
            if faces:
                box = max(faces, key=lambda c: c['score'])['box']
                widen = OVERLAY_FACE_WIDTHS
            else:
                box = m.detect_person_yolo(frame)
                widen = INSET_PADDING
            if not box or box[3] > frame_h * MAX_SUBJECT_HEIGHT:
                continue
            boxes.append(overlay_box(box, frame_w, frame_h, widen))
    finally:
        cap.release()
    need = samples // 2 + 1
    if len(boxes) < need:
        return None
    return stable_box(boxes, frame_w, frame_h, min_agree=need)


def _centre_inside(det, box, slack):
    cx = det[0] + det[2] / 2.0
    cy = det[1] + det[3] / 2.0
    x, y, w, h = box
    return (x - slack <= cx <= x + w + slack) and (y - slack <= cy <= y + h + slack)


def present_in_scene(video_path, box, start_f, end_f, samples=3):
    """Whether the webcam really sits in ``box`` during this scene.

    ``detect`` finds the box once per clip, from samples spread over every
    scene, and a clip cut from a tutorial mixes shots: the bubble is in the
    corner during the screen recording and gone in the editor shot next to it.
    Before this check every screen scene of a clip with an inset anywhere got
    the INSET layout, and on a scene without the camera the bottom band was an
    enlarged crop of a toolbar (measured on a Canva + QuickTime tutorial,
    6-oct-2026). A face or a person whose centre falls in the box in most
    samples is the answer.
    """
    import cv2
    import numpy as np
    import main as m
    import screencast_layout

    if end_f - 1 < start_f:
        return False
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    slack = max(box[2], box[3]) * 0.1
    hits = 0
    try:
        for f_idx in np.linspace(start_f, end_f - 1, samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(f_idx)))
            ok, frame = cap.read()
            if not ok:
                continue
            # Every face and every person counts: the largest person in the frame is
            # often an on-screen character, and the webcam's own person is smaller.
            dets = [c['box'] for c in screencast_layout.detect_faces_full_res(frame)]
            dets += person_boxes(frame)
            if any(_centre_inside(d, box, slack) for d in dets):
                hits += 1
    finally:
        cap.release()
    return hits * 2 > samples


# Stream layout: the webcam band is the top 30% of the output frame, and the
# gameplay fills the rest (70%). 25% clipped faces on several streams.
# See inset_filtergraph.
WEBCAM_BAND_RATIO = 0.30


def _fill_crop(src_w, src_h, out_w, out_h):
    """The largest centred crop of the source with the output's aspect ratio.

    Returns (w, h, x, y), all even, so the gameplay fills a band without being
    stretched.
    """
    target = out_w / float(out_h)
    if src_w / float(src_h) > target:
        h = src_h - src_h % 2
        w = min(src_w, int(round(src_h * target)))
    else:
        w = src_w - src_w % 2
        h = min(src_h, int(round(src_w / target)))
    w -= w % 2
    h -= h % 2
    x = max(0, (src_w - w) // 2)
    y = max(0, (src_h - h) // 2)
    return w, h, x - x % 2, y - y % 2


def _cover_crop(x, y, w, h, aspect):
    """The largest even crop of the box (x, y, w, h) with the band's aspect, centred on it.

    The crop is trimmed out of the box and never grows past it, so the band is filled
    by the camera alone: the gameplay around the box never shows inside the band. The
    box's narrow axis is kept whole. Its long axis is trimmed mostly from the bottom,
    where the stream's overlay and the desk are, so the head stays inside the band.
    """
    x -= x % 2
    y -= y % 2
    w -= w % 2
    h -= h % 2
    if w / float(h) < aspect:
        crop_w = w
        crop_h = min(h, int(w / aspect) // 2 * 2)
    else:
        crop_h = h
        crop_w = min(w, int(h * aspect) // 2 * 2)
    # Centred across the width; down the height a quarter of the trim comes off the top.
    return x + ((w - crop_w) // 4) * 2, y + ((h - crop_h) // 8) * 2, crop_w, crop_h


def inset_filtergraph(orig_w, orig_h, out_w, out_h, box, camera_ratio=None, webcam_on_top=None):
    """Composites screen and webcam inset vertically on blurred backdrop.

    The streamer webcam sits on top and gameplay/screen below by default (`webcam_on_top=True`).
    When `webcam_on_top=False`, screen on top and webcam below.

    Layout: the webcam band is the top 30% of the frame and the gameplay fills
    the bottom 70%. The webcam band is filled by the camera alone: its box is
    trimmed to the band's shape, centred, so no gameplay shows inside the band.
    The gameplay is a centred crop of the source, so its left and right edges
    (and any HUD in those corners) are lost.
    """
    if webcam_on_top is None:
        webcam_on_top = os.environ.get('WEBCAM_ON_TOP', '1').strip().lower() in ('1', 'true', 'yes')

    box_w = max(2, box[2])
    box_h = max(2, box[3])

    if camera_ratio is None:
        camera_ratio = WEBCAM_BAND_RATIO
    cam_h = int(round(out_h * camera_ratio))
    cam_h -= cam_h % 2

    # The gameplay fills everything the webcam band does not take.
    screen_h = out_h - cam_h
    screen_crop_w, screen_crop_h, screen_crop_x, screen_crop_y = _fill_crop(orig_w, orig_h, out_w, screen_h)

    # The band is filled by the camera alone: the box is trimmed to the band's shape,
    # centred, and scaled up to the band. Nothing is grown past the box, so no gameplay
    # shows inside the band. The box is clamped to the frame first.
    x, y, w, h = box
    w = max(2, min(int(round(w)), orig_w))
    h = max(2, min(int(round(h)), orig_h))
    x = max(0, min(int(round(x)), orig_w - w))
    y = max(0, min(int(round(y)), orig_h - h))
    crop_x, crop_y, crop_w, crop_h = _cover_crop(x, y, w, h, out_w / float(cam_h))
    fit_w, fit_h = out_w, cam_h
    cam_x, cam_y = 0, 0

    if webcam_on_top:
        overlay_chain = (
            f"[bg][cam]overlay=x={cam_x}:y={cam_y}[withcam];"
            f"[withcam][screen]overlay=x=0:y={cam_h},"
            f"setsar=1[v]"
        )
    else:
        overlay_chain = (
            f"[bg][screen]overlay=x=0:y=0[withscreen];"
            f"[withscreen][cam]overlay=x={cam_x}:y={screen_h + cam_y},"
            f"setsar=1[v]"
        )

    return (
        f"[0:v]split=3[bga][sa][ca];"
        # Blurred backdrop so the leftover strip is not a black bar. Scaled by
        # HEIGHT: scaling a 16:9 source to 1080 wide gives 608 tall, and there
        # is no 1920-tall crop to take out of that.
        f"[bga]{blurred_backdrop(out_w, out_h, 14)}[bg];"
        f"[sa]crop=w={screen_crop_w}:h={screen_crop_h}:x={screen_crop_x}:y={screen_crop_y},"
        f"scale={out_w}:{screen_h}[screen];"
        f"[ca]crop=w={crop_w}:h={crop_h}:x={crop_x}:y={crop_y},scale={fit_w}:{fit_h}[cam];"
        f"{overlay_chain}"
    )
