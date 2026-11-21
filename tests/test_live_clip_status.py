"""The live clip list must show a job's real state, its last log line, and the
download links of a finished render (result.clips[].video_url)."""
import asyncio
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


def _enqueue_one():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41")
        clip = f.name
    try:
        meta = {"channel": "gaules", "timestamp": time.time(), "duration": 45, "reason": "test"}
        return asyncio.run(app._enqueue_live_clip_async(clip, meta))
    finally:
        try:
            os.remove(clip)
        except OSError:
            pass


def _record_for(job_id):
    status = asyncio.run(app.live_monitor_status(request=None))
    return next(c for c in status["captured_clips"] if c.get("job_id") == job_id)


def setup_function():
    app._live_captured_clips.clear()
    app.live_monitor_instance = None


def test_completed_render_exposes_its_clip_links():
    job_id = _enqueue_one()
    job = app.jobs[job_id]
    job["status"] = "completed"
    job["result"] = {"clips": [{"video_url": f"/videos/{job_id}/clip_1.mp4"}], "cost_analysis": None}

    rec = _record_for(job_id)
    assert rec["status"] == "completed"
    assert rec["result"]["clips"][0]["video_url"].endswith("clip_1.mp4")


def test_running_job_reports_its_last_log_line():
    job_id = _enqueue_one()
    app.jobs[job_id]["status"] = "processing"
    app.jobs[job_id]["logs"].append("Transcribing audio...")

    rec = _record_for(job_id)
    assert rec["status"] == "processing"
    assert rec["last_log"] == "Transcribing audio..."


def test_failed_job_reports_the_reason():
    job_id = _enqueue_one()
    app.jobs[job_id]["status"] = "failed"
    app.jobs[job_id]["logs"].append("Process failed with exit code 1")

    rec = _record_for(job_id)
    assert rec["status"] == "failed"
    assert "exit code 1" in rec["error"]
