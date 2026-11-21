"""The streamer layout for live clips: the two bands' geometry, the webcam finder on synthetic
frames, and renders through ffmpeg."""
import shutil
import subprocess

import cv2
import numpy as np
import pytest

import stream_layout as sl


# --- geometry -------------------------------------------------------------------------- #

def test_the_whole_webcam_fits_the_top_band_with_its_aspect():
    # A 16:9 camera fills the 1080x576 band by its height; the sides are the blurred copy.
    assert sl.fit_size(1920, 1080, 1080, 576) == (1024, 576)


def test_a_tall_webcam_fits_by_height_and_is_never_cut():
    assert sl.fit_size(480, 640, 1080, 576) == (432, 576)


def test_the_gameplay_is_the_centre_of_the_source():
    x, y, w, h = sl.gameplay_crop(1920, 1080, 1080, 1344)
    assert (y, h) == (0, 1080)
    assert abs((x + w / 2.0) - 960) <= 2
    assert abs(w / h - 1080 / 1344) < 0.01


def test_the_frame_is_30_percent_webcam_and_70_percent_gameplay_and_static():
    graph = sl.filtergraph(1920, 1080, (1440, 720, 360, 270))
    assert "scale=768:576" in graph         # the webcam, fitted into the top band
    assert "crop=1080:576" in graph         # the blurred band behind it
    assert "scale=1080:1344" in graph       # the gameplay, 70% of 1920
    assert "vstack=inputs=2" in graph
    # Nothing per frame: no time or frame-number expressions in the graph.
    assert "t/" not in graph and "n/" not in graph and "if(" not in graph


def test_boxes_are_clamped_to_the_frame_with_even_sides():
    assert sl.clamp_box((1900, 1070, 300, 99), 1920, 1080) == (1620, 980, 300, 98)


# --- finding the webcam ---------------------------------------------------------------- #

def _gameplay(rng, width, height):
    """A changing game picture: new noise every frame, so its edges never stay in place."""
    return cv2.GaussianBlur(rng.integers(0, 255, (height, width, 3), dtype=np.uint8), (9, 9), 0)


def _fake_live(rng, count=16, width=640, height=360, cam=(400, 200, 200, 150)):
    """Frames of a fake live: a bordered webcam with a face-like blob that moves a few pixels,
    over changing gameplay. Also returns the face box a face detector would report per frame."""
    x, y, w, h = cam
    frames, faces = [], []
    for _ in range(count):
        frame = _gameplay(rng, width, height)
        dx, dy = (int(v) for v in rng.integers(-2, 3, 2))
        cam_img = np.full((h, w, 3), (60, 120, 180), np.uint8)
        cv2.ellipse(cam_img, (w // 2 + dx, h // 3 + dy), (30, 38), 0, 0, 360, (210, 170, 140), -1)
        frame[y:y + h, x:x + w] = cam_img
        cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), (255, 255, 255), 3)
        frames.append(frame)
        faces.append([(x + w // 2 - 30 + dx, y + h // 3 - 35 + dy, 60, 70)])
    return frames, faces


def test_the_webcam_is_found_from_its_face_and_its_border():
    frames, faces = _fake_live(np.random.default_rng(3))
    per_frame = iter(faces)
    x, y, w, h = sl.detect_webcam_box(frames, 640, 360, lambda frame: next(per_frame))
    assert abs(x - 400) <= 6 and abs(y - 200) <= 6
    assert abs(w - 200) <= 8 and abs(h - 150) <= 8


def test_a_face_that_does_not_stay_put_is_no_webcam():
    rng = np.random.default_rng(5)
    frames, _ = _fake_live(rng)
    spots = iter([[(int(rng.integers(0, 560)), int(rng.integers(0, 300)), 60, 70)] for _ in frames])
    assert sl.detect_webcam_box(frames, 640, 360, lambda frame: next(spots)) is None


# --- renders ------------------------------------------------------------------------------ #

def _lavfi_clip(path, source, seconds=2):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"{source}:duration={seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
                   check=True)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the ffmpeg CLI")
def test_a_live_clip_renders_at_delivery_size_with_the_calibrated_box(tmp_path, monkeypatch):
    src = tmp_path / "live.mp4"
    _lavfi_clip(src, "testsrc2=size=1920x1080:rate=25")
    monkeypatch.setenv("WEBCAM_BOX", "0.75,0.66,0.19,0.25")
    out = tmp_path / "out.mp4"
    assert sl.render_clip(str(src), str(out)) is True
    cap = cv2.VideoCapture(str(out))
    try:
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 1080
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 1920
    finally:
        cap.release()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the ffmpeg CLI")
def test_no_webcam_means_no_stream_layout_so_the_normal_reframe_runs(tmp_path, monkeypatch):
    pytest.importorskip("mediapipe")
    monkeypatch.delenv("WEBCAM_BOX", raising=False)
    src = tmp_path / "game.mp4"
    _lavfi_clip(src, "testsrc2=size=1920x1080:rate=25")
    out = tmp_path / "out.mp4"
    assert sl.render_clip(str(src), str(out)) is False
    assert not out.exists()


# --- the hook in main ---------------------------------------------------------------------- #

def test_a_live_clip_takes_the_stream_layout_before_the_normal_reframe(tmp_path, monkeypatch):
    main = pytest.importorskip("main")  # needs cv2/mediapipe, absent in minimal CI
    import reframe_v2
    monkeypatch.setenv("IS_STREAM_SOURCE", "1")
    written = []

    def fake_render_clip(input_video, final_output_video, **kwargs):
        written.append(final_output_video)
        return True

    def no_reframe(*args, **kwargs):
        raise AssertionError("the normal reframe must not run when the webcam layout renders")

    monkeypatch.setattr(sl, "render_clip", fake_render_clip)
    monkeypatch.setattr(reframe_v2, "render", no_reframe)
    out = str(tmp_path / "out.mp4")
    assert main.process_video_to_vertical("in.mp4", out, aspect_ratio=main.ASPECT_RATIO) is True
    assert written == [out]


def test_a_live_without_a_webcam_runs_the_normal_reframe(tmp_path, monkeypatch):
    main = pytest.importorskip("main")  # needs cv2/mediapipe, absent in minimal CI
    import reframe_v2
    monkeypatch.setenv("IS_STREAM_SOURCE", "1")
    monkeypatch.setattr(sl, "render_clip", lambda *args, **kwargs: False)
    reframed = []

    def fake_reframe(*args, **kwargs):
        reframed.append(args)
        return True

    monkeypatch.setattr(reframe_v2, "render", fake_reframe)
    out = str(tmp_path / "out.mp4")
    assert main.process_video_to_vertical("in.mp4", out, aspect_ratio=main.ASPECT_RATIO) is True
    assert reframed, "the normal reframe must run when no webcam is found"


def test_other_videos_never_take_the_stream_layout(tmp_path, monkeypatch):
    main = pytest.importorskip("main")  # needs cv2/mediapipe, absent in minimal CI
    import reframe_v2
    monkeypatch.delenv("IS_STREAM_SOURCE", raising=False)
    monkeypatch.setattr(sl, "render_clip", lambda *args, **kwargs: pytest.fail("not for a non-live video"))
    monkeypatch.setattr(reframe_v2, "render", lambda *args, **kwargs: True)
    assert main.process_video_to_vertical("in.mp4", str(tmp_path / "out.mp4"),
                                          aspect_ratio=main.ASPECT_RATIO) is True


# --- detection first, the calibrated box as the fallback ------------------------------------ #

def _captured_graphs(monkeypatch):
    graphs = []
    monkeypatch.setattr(sl, "_source_size", lambda path: (1920, 1080))
    monkeypatch.setattr(sl, "_encode", lambda source, out, graph: graphs.append(graph) or True)
    return graphs


def test_the_detected_webcam_comes_before_the_calibrated_box(monkeypatch):
    graphs = _captured_graphs(monkeypatch)
    monkeypatch.setattr(sl, "find_webcam", lambda video, w, h: (1440, 720, 360, 270))
    monkeypatch.setenv("WEBCAM_BOX", "0.1,0.1,0.2,0.2")
    assert sl.render_clip("live.mp4", "out.mp4") is True
    assert "crop=360:270:1440:720" in graphs[0]
    assert "crop=384:216:192:108" not in graphs[0]


def test_the_calibrated_box_is_the_fallback_when_nothing_is_detected(monkeypatch):
    graphs = _captured_graphs(monkeypatch)
    monkeypatch.setattr(sl, "find_webcam", lambda video, w, h: None)
    monkeypatch.setenv("WEBCAM_BOX", "0.75,0.66,0.19,0.25")
    assert sl.render_clip("live.mp4", "out.mp4") is True
    assert "crop=364:270:1440:712" in graphs[0]


def test_a_broken_detector_falls_back_to_the_calibrated_box(monkeypatch):
    graphs = _captured_graphs(monkeypatch)

    def broken(video, w, h):
        raise RuntimeError("detector unavailable")

    monkeypatch.setattr(sl, "find_webcam", broken)
    monkeypatch.setenv("WEBCAM_BOX", "0.75,0.66,0.19,0.25")
    assert sl.render_clip("live.mp4", "out.mp4") is True
    assert "crop=364:270:1440:712" in graphs[0]


def test_nothing_detected_and_no_calibration_means_no_stream_layout(monkeypatch):
    graphs = _captured_graphs(monkeypatch)
    monkeypatch.setattr(sl, "find_webcam", lambda video, w, h: None)
    monkeypatch.delenv("WEBCAM_BOX", raising=False)
    assert sl.render_clip("live.mp4", "out.mp4") is False
    assert graphs == []
