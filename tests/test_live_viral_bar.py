"""A live clip is kept only when Gemini's predicted_score reaches the bar."""
import os
import sys
import tempfile
import types
from unittest.mock import MagicMock

if "boto3" not in sys.modules:
    boto3_mock = types.ModuleType("boto3")
    boto3_mock.client = MagicMock()
    sys.modules["boto3"] = boto3_mock
if "botocore" not in sys.modules:
    botocore_mock = types.ModuleType("botocore")
    botocore_mock.__path__ = []
    botocore_mock.exceptions = types.ModuleType("botocore.exceptions")

    class ClientError(Exception):
        pass

    botocore_mock.exceptions.ClientError = ClientError
    sys.modules["botocore"] = botocore_mock
    sys.modules["botocore.exceptions"] = botocore_mock.exceptions
    botocore_config = types.ModuleType("botocore.config")

    class Config:
        def __init__(self, *args, **kwargs):
            pass

    botocore_config.Config = Config
    botocore_mock.config = botocore_config
    sys.modules["botocore.config"] = botocore_config
if "fcntl" not in sys.modules:
    fcntl = types.ModuleType("fcntl")
    fcntl.flock = lambda *a, **k: None
    sys.modules["fcntl"] = fcntl

import app  # noqa: E402


def test_the_bar_is_80_by_default():
    assert app.LIVE_MIN_PREDICTED_SCORE == 80


def test_scores_at_or_above_the_bar_are_kept():
    assert app.live_clip_is_viral({"predicted_score": 80}) is True
    assert app.live_clip_is_viral({"predicted_score": 92}) is True


def test_scores_below_the_bar_are_dropped():
    assert app.live_clip_is_viral({"predicted_score": 79}) is False
    assert app.live_clip_is_viral({"predicted_score": 75}) is False


def test_missing_or_non_numeric_scores_are_dropped():
    assert app.live_clip_is_viral({}) is False
    assert app.live_clip_is_viral({"predicted_score": None}) is False
    assert app.live_clip_is_viral({"predicted_score": "95"}) is False
    assert app.live_clip_is_viral({"predicted_score": True}) is False
    assert app.live_clip_is_viral(None) is False


def test_dropped_clip_file_is_removed_from_the_job_folder():
    folder = tempfile.mkdtemp(prefix="live_drop_")
    path = os.path.join(folder, "clip_3.mp4")
    with open(path, "wb") as f:
        f.write(b"x")
    app._remove_clip_file(folder, {"video_url": "/videos/abc/clip_3.mp4"})
    assert not os.path.exists(path)
