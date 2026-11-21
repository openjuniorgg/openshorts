"""Live clips: the resume keeps the live flag, a clip Gemini did not rate is never
delivered, and the dashboard's Gemini key reaches the automatic triggers."""
import asyncio
import json
import os
import sys
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

import pytest  # noqa: E402

import app  # noqa: E402
from live_monitor import LiveMonitor  # noqa: E402


JOB_ID = "22222222-3333-4444-5555-666666666666"


def test_resume_manifest_keeps_the_live_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "OUTPUT_DIR", str(tmp_path))
    job_dir = tmp_path / JOB_ID
    job_dir.mkdir()
    app._write_resume_manifest(JOB_ID, ["python", "main.py"], 1, None, None, False,
                               live_highlight=True, submitted_at=1.0)
    data = json.loads((job_dir / app._RESUME_FILE).read_text(encoding="utf-8"))
    assert data["live_highlight"] is True


def test_resume_manifest_defaults_to_not_live(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "OUTPUT_DIR", str(tmp_path))
    job_dir = tmp_path / JOB_ID
    job_dir.mkdir()
    app._write_resume_manifest(JOB_ID, ["python", "main.py"], 2, None, None, False, submitted_at=1.0)
    data = json.loads((job_dir / app._RESUME_FILE).read_text(encoding="utf-8"))
    assert data["live_highlight"] is False


def _finish(tmp_path, monkeypatch, rendered, live=True):
    out = tmp_path / JOB_ID
    out.mkdir()
    (out / "clip_metadata.json").write_text(
        json.dumps({"shorts": [{"start": 0, "end": 30}]}), encoding="utf-8")
    app.jobs[JOB_ID] = {"status": "processing", "logs": [], "output_dir": str(out),
                        "is_live_highlight": live, "result": None}
    monkeypatch.setattr(app, "_clips_actually_rendered", lambda *a, **k: (rendered, 0))
    monkeypatch.setattr(app, "upload_job_artifacts", MagicMock())

    async def run():
        app._finish_job(JOB_ID, 0, str(out))

    asyncio.run(run())
    return app.jobs[JOB_ID]


def test_a_clip_gemini_never_rated_is_never_delivered(tmp_path, monkeypatch):
    job = _finish(tmp_path, monkeypatch, [{"video_url": f"/videos/{JOB_ID}/clip_1.mp4"}])
    assert job["status"] == "failed"
    assert job["result"] is None
    assert any("did not rate" in line for line in job["logs"])


def test_a_rated_clip_below_the_bar_is_dropped(tmp_path, monkeypatch):
    job = _finish(tmp_path, monkeypatch,
                  [{"video_url": f"/videos/{JOB_ID}/clip_1.mp4", "predicted_score": 40}])
    assert job["status"] == "failed"
    assert any("viral score bar" in line for line in job["logs"])


def test_a_rated_clip_at_the_bar_is_delivered(tmp_path, monkeypatch):
    clip = {"video_url": f"/videos/{JOB_ID}/clip_1.mp4", "predicted_score": 80}
    job = _finish(tmp_path, monkeypatch, [clip])
    assert job["status"] == "completed"
    assert job["result"]["clips"] == [clip]


def test_a_non_live_job_keeps_its_clips_as_before(tmp_path, monkeypatch):
    clip = {"video_url": f"/videos/{JOB_ID}/clip_1.mp4"}
    job = _finish(tmp_path, monkeypatch, [clip], live=False)
    assert job["status"] == "completed"


def test_dashboard_key_reaches_automatic_clips_but_not_the_clip_list(tmp_path):
    received = {}

    def on_clip(path, meta):
        received.update(meta)

    monitor = LiveMonitor(channels=["gaules"], on_clip_ready=on_clip,
                          output_dir=str(tmp_path), mock_extraction=True)
    monitor.gemini_api_key = "KEY-FROM-DASHBOARD"
    state = monitor._channels["twitch_gaules"]
    monitor._handle_viral_spike(state, "stream_audio_shout", now=1000.0)
    assert received.get("gemini_api_key") == "KEY-FROM-DASHBOARD"
    assert monitor.captured_clips and all("gemini_api_key" not in c for c in monitor.captured_clips)
