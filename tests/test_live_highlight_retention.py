"""Raw live clips are swept once they are older than JOB_RETENTION_SECONDS."""
import os
import sys
import tempfile
import time
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


def test_old_raw_clips_are_swept_fresh_ones_kept(monkeypatch):
    folder = tempfile.mkdtemp(prefix="live_highlights_")
    monkeypatch.setenv("LIVE_HIGHLIGHT_DIR", folder)
    monkeypatch.setattr(app, "JOB_RETENTION_SECONDS", 24 * 3600)

    now = time.time()
    old = os.path.join(folder, "live_old.mp4")
    fresh = os.path.join(folder, "live_fresh.mp4")
    for path in (old, fresh):
        with open(path, "wb") as f:
            f.write(b"x")
    os.utime(old, (now - 25 * 3600, now - 25 * 3600))
    os.utime(fresh, (now - 3600, now - 3600))

    removed = app._sweep_live_highlights(now)

    assert removed == 1
    assert not os.path.exists(old)
    assert os.path.exists(fresh)


def test_missing_folder_is_not_an_error(monkeypatch):
    monkeypatch.setenv("LIVE_HIGHLIGHT_DIR", os.path.join(tempfile.gettempdir(), "does_not_exist_live"))
    assert app._sweep_live_highlights(time.time()) == 0
