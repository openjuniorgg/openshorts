"""Tests for Twitch and YouTube live stream reframing and layout preservation."""
import os
import sys
import types
import pytest

import reframe_v2
import twitch_signals


class FT:
    """Stand-in for scenedetect's FrameTimecode."""
    def __init__(self, frames):
        self.frames = frames

    def get_frames(self):
        return self.frames


def scenes_of(*bounds):
    return [(FT(s), FT(e)) for s, e in bounds]


class _StopRender(Exception):
    pass


@pytest.fixture
def fake_main(monkeypatch):
    m = types.ModuleType("main")
    # 3 scenes: 0..90, 90..180, 180..270
    m.detect_scenes = lambda path: (scenes_of((0, 90), (90, 180), (180, 270)), 30.0)
    m.get_video_resolution = lambda path: (1920, 1080)
    m.analyze_scenes_strategy = lambda path, scenes: ['TRACK', 'TRACK', 'GENERAL']
    m.watermark_logo_path = lambda: None

    class FakeCameraman:
        def __init__(self, *args, **kwargs):
            self.crop_width = 608
            self.crop_height = 1080

    m.SmoothedCameraman = FakeCameraman
    m.SpeakerTracker = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "main", m)
    return m


def test_stream_with_webcam_inset_upgrades_scenes_with_webcam(monkeypatch, fake_main):
    """When stream is detected and webcam inset exists, scenes with webcam become INSET,
    scenes without webcam keep their original strategy."""
    monkeypatch.setenv("IS_STREAM_SOURCE", "1")

    # Inset detected at corner
    detected_box = (1500, 800, 350, 200)
    monkeypatch.setattr(reframe_v2.camera_inset, "detect", lambda video: detected_box)

    # Present in scene 0 and scene 2, but NOT in scene 1
    def fake_present(video, box, start_f, end_f):
        if start_f == 0:
            return True
        elif start_f == 90:
            return False
        elif start_f == 180:
            return True
        return False

    monkeypatch.setattr(reframe_v2.camera_inset, "present_in_scene", fake_present)

    # Capture specs passed to encode_scenes
    captured = {}

    def fake_encode_scenes(input_video, specs, orig_w, orig_h, out_w, out_h, fps,
                          workdir, logo, final_output_video):
        captured["specs"] = specs
        raise _StopRender()

    monkeypatch.setattr(reframe_v2, "encode_scenes", fake_encode_scenes)
    # Stop before full analysis trajectory
    monkeypatch.setattr(reframe_v2, "_analyze_trajectory", lambda *a, **k: [100] * 270)
    monkeypatch.setattr(reframe_v2.split_layout, "detect_split_scenes", lambda *a, **k: {})

    inset_graphs = []
    real_inset_filtergraph = reframe_v2.camera_inset.inset_filtergraph

    def spy_inset_filtergraph(*args, **kwargs):
        inset_graphs.append((args, kwargs))
        return real_inset_filtergraph(*args, **kwargs)

    monkeypatch.setattr(reframe_v2.camera_inset, "inset_filtergraph", spy_inset_filtergraph)

    with pytest.raises(_StopRender):
        reframe_v2.render("stream_vod.mp4", "out.mp4", 9 / 16)

    specs = captured["specs"]
    assert len(specs) == 3

    # Scene 0 had webcam -> INSET
    # Scene 1 had no webcam -> retained original strategy ('SPLIT' or 'TRACK'/'GENERAL')
    # Scene 2 had webcam -> INSET
    assert len(inset_graphs) == 2
    for args, kwargs in inset_graphs:
        assert kwargs.get("webcam_on_top") is True
        # Box matches detected_box
        assert args[4] == detected_box

    # In the filtergraph for INSET, verify webcam [cam] is on top and screen on bottom
    g0 = specs[0][2]
    assert "[bg][cam]overlay=x=" in g0  # the fitted camera, centred in its band
    assert "[withcam][screen]overlay=x=0:y=576" in g0  # gameplay starts at 30%


def test_stream_without_webcam_inset_preserves_all_layouts(monkeypatch, fake_main):
    """When stream is detected but no webcam inset exists, original layouts are preserved."""
    monkeypatch.setenv("IS_STREAM_SOURCE", "1")
    monkeypatch.setattr(reframe_v2.camera_inset, "detect", lambda video: None)

    captured = {}

    def fake_encode_scenes(input_video, specs, orig_w, orig_h, out_w, out_h, fps,
                          workdir, logo, final_output_video):
        captured["specs"] = specs
        raise _StopRender()

    monkeypatch.setattr(reframe_v2, "encode_scenes", fake_encode_scenes)
    monkeypatch.setattr(reframe_v2, "_analyze_trajectory", lambda *a, **k: [100] * 270)
    monkeypatch.setattr(reframe_v2.split_layout, "detect_split_scenes", lambda *a, **k: {})

    with pytest.raises(_StopRender):
        reframe_v2.render("stream_vod.mp4", "out.mp4", 9 / 16)

    # None of the scenes should be INSET
    specs = captured["specs"]
    assert len(specs) == 3
    for ss, dur, graph, boxes in specs:
        assert "[withcam]" not in graph


def test_non_stream_without_screencast_skips_inset_detect(monkeypatch, fake_main):
    """When video is not a stream and screencast_layout is disabled, inset detection is skipped."""
    monkeypatch.delenv("IS_STREAM_SOURCE", raising=False)
    monkeypatch.delenv("AUTO_LAYOUT", raising=False)
    monkeypatch.setattr(reframe_v2.screencast_layout, "ENABLED", False)

    called = []
    monkeypatch.setattr(reframe_v2.camera_inset, "detect", lambda video: called.append(video))

    captured = {}

    def fake_encode_scenes(input_video, specs, orig_w, orig_h, out_w, out_h, fps,
                          workdir, logo, final_output_video):
        captured["specs"] = specs
        raise _StopRender()

    monkeypatch.setattr(reframe_v2, "encode_scenes", fake_encode_scenes)
    monkeypatch.setattr(reframe_v2, "_analyze_trajectory", lambda *a, **k: [100] * 270)
    monkeypatch.setattr(reframe_v2.split_layout, "detect_split_scenes", lambda *a, **k: {})

    with pytest.raises(_StopRender):
        reframe_v2.render("regular_video.mp4", "out.mp4", 9 / 16)

    assert called == []  # Not called at all


def test_main_download_detects_stream_sources(monkeypatch):
    """main.py sets IS_STREAM_SOURCE=1 when stream source is detected."""
    if "fcntl" not in sys.modules:
        fcntl = types.ModuleType("fcntl")
        fcntl.flock = lambda *a, **k: None
        fcntl.LOCK_EX = 2
        fcntl.LOCK_NB = 4
        fcntl.LOCK_UN = 8
        monkeypatch.setitem(sys.modules, "fcntl", fcntl)
    if "ultralytics" not in sys.modules:
        ultra = types.ModuleType("ultralytics")
        ultra.YOLO = lambda *a, **k: None
        monkeypatch.setitem(sys.modules, "ultralytics", ultra)
    if "mediapipe" not in sys.modules:
        mp = types.ModuleType("mediapipe")
        mp.solutions = types.SimpleNamespace(face_detection=types.SimpleNamespace(FaceDetection=lambda *a, **k: None))
        monkeypatch.setitem(sys.modules, "mediapipe", mp)

    import main

    monkeypatch.delenv("IS_STREAM_SOURCE", raising=False)

    # Calling download_youtube_video with twitch url sets IS_STREAM_SOURCE
    twitch_url = "https://www.twitch.tv/videos/987654321"
    monkeypatch.setattr(main, "_download_youtube_video", lambda *a, **k: ("vid.mp4", "title"))

    main.download_youtube_video(twitch_url)
    assert os.environ.get("IS_STREAM_SOURCE") == "1"

    monkeypatch.delenv("IS_STREAM_SOURCE", raising=False)
    yt_live_url = "https://www.youtube.com/live/someLiveId"
    main.download_youtube_video(yt_live_url)
    assert os.environ.get("IS_STREAM_SOURCE") == "1"
