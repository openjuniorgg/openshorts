"""Calibrated webcam boxes per channel, and the box a job reads from WEBCAM_BOX."""
import json
import os

import pytest

import camera_inset
import webcam_boxes


def test_a_calibrated_channel_gets_its_box(tmp_path):
    path = tmp_path / "boxes.json"
    path.write_text(json.dumps({"kick:coringa": [0.0, 0.11, 0.24, 0.28]}), encoding="utf-8")
    assert webcam_boxes.lookup("kick", "Coringa", path=str(path)) == "0.0000,0.1100,0.2400,0.2800"


def test_an_uncalibrated_channel_gets_nothing(tmp_path):
    path = tmp_path / "boxes.json"
    path.write_text(json.dumps({"kick:coringa": [0.0, 0.11, 0.24, 0.28]}), encoding="utf-8")
    assert webcam_boxes.lookup("twitch", "coringa", path=str(path)) is None


def test_a_missing_or_broken_file_is_no_box(tmp_path):
    assert webcam_boxes.lookup("kick", "coringa", path=str(tmp_path / "nope.json")) is None
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert webcam_boxes.lookup("kick", "coringa", path=str(bad)) is None


def test_the_shipped_file_covers_the_calibrated_streamers():
    for key in ("kick:coringa", "kick:bistecone", "twitch:alanzoka", "twitch:kaicenat"):
        assert key in webcam_boxes.load()


def test_a_job_reads_the_box_in_its_own_pixels(monkeypatch):
    monkeypatch.setenv("WEBCAM_BOX", "0.0,0.11,0.24,0.28")
    assert camera_inset.manual_box_from_env(1920, 1080) == (0, 119, 461, 302)
    monkeypatch.setenv("WEBCAM_BOX", "0.0,0.11,0.24,0.28")
    assert camera_inset.manual_box_from_env(1280, 720) == (0, 79, 307, 202)


def test_a_bad_box_is_ignored(monkeypatch):
    monkeypatch.setenv("WEBCAM_BOX", "garbage")
    assert camera_inset.manual_box_from_env(1920, 1080) is None
    monkeypatch.setenv("WEBCAM_BOX", "1.5,0,0.2,0.2")
    assert camera_inset.manual_box_from_env(1920, 1080) is None
    monkeypatch.delenv("WEBCAM_BOX", raising=False)
    assert camera_inset.manual_box_from_env(1920, 1080) is None


def test_detect_returns_the_calibrated_box_without_sampling(monkeypatch):
    cv2 = pytest.importorskip("cv2")

    class FakeCapture:
        def __init__(self, path):
            pass

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                cv2.CAP_PROP_FRAME_COUNT: 900,
                cv2.CAP_PROP_FRAME_WIDTH: 1920,
                cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            }[prop]

        def set(self, *args):
            raise AssertionError("a calibrated channel must not sample frames")

        def read(self):
            raise AssertionError("a calibrated channel must not sample frames")

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setenv("WEBCAM_BOX", "0.0,0.11,0.24,0.28")
    assert camera_inset.detect("clip.mp4") == (0, 119, 461, 302)
