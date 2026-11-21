"""Frame sampling for the model: real ffmpeg on a generated clip, with the time of every frame."""
import os
import shutil
import subprocess

import pytest

import vision_frames

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the ffmpeg CLI")


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc=duration=8:size=320x240:rate=10",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return str(path)


def test_frames_come_back_with_their_times(clip, tmp_path):
    out = tmp_path / "frames"
    out.mkdir()
    interval, frames = vision_frames.sample(clip, 0.0, 8.0, str(out), count=4, width=160)

    assert interval == pytest.approx(2.0)
    assert [t for t, _ in frames] == pytest.approx([0.0, 2.0, 4.0, 6.0], abs=0.05)
    for _, jpeg in frames:
        assert jpeg[:3] == b"\xff\xd8\xff"  # a real JPEG


def test_the_interval_never_drops_below_the_minimum(clip, tmp_path):
    out = tmp_path / "frames"
    out.mkdir()
    interval, frames = vision_frames.sample(clip, 0.0, 8.0, str(out), count=40, width=160)
    assert interval == vision_frames.MIN_INTERVAL_SECONDS
    assert len(frames) <= 40


def test_a_stretch_starts_its_times_at_zero(clip, tmp_path):
    out = tmp_path / "frames"
    out.mkdir()
    _interval, frames = vision_frames.sample(clip, 3.0, 8.0, str(out), count=2, width=160)
    assert frames[0][0] == 0.0


def test_labels_name_each_frame_in_order():
    frames = [(0.0, b"a"), (2.0, b"b"), (4.5, b"c")]
    assert vision_frames.labels(frames) == "1=0.0s, 2=2.0s, 3=4.5s"


def test_no_decodable_frames_is_an_error(tmp_path):
    out = tmp_path / "frames"
    out.mkdir()
    with pytest.raises(RuntimeError, match="no frames decoded"):
        vision_frames.sample(str(tmp_path / "missing.mp4"), 0.0, 8.0, str(out), count=4, width=160)
    assert os.listdir(out) == []
