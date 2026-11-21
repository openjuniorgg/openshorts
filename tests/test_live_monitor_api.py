"""Tests for LiveMonitor API endpoints in app.py:
- POST /api/live-monitor/start
- POST /api/live-monitor/stop
- GET /api/live-monitor/status
- Automatic clip enqueueing with IS_STREAM_SOURCE=1 and WEBCAM_ON_TOP=1
"""
import asyncio
import os
import sys
import tempfile
import time
import types
from unittest.mock import MagicMock, patch

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

import pytest
from fastapi.testclient import TestClient

import app


@pytest.fixture(autouse=True)
def _no_live_network(monkeypatch):
    # These tests check the endpoints, not the live transcript or the GTA VI lookup.
    monkeypatch.setenv("LIVE_TRANSCRIPT", "0")
    monkeypatch.setenv("LIVE_GAME_FILTER", "0")


@pytest.fixture
def client():
    # Reset live monitor state before and after each test
    if app.live_monitor_instance is not None:
        try:
            app.live_monitor_instance.stop()
        except Exception:
            pass
    app.live_monitor_instance = None
    app._live_captured_clips.clear()
    
    with TestClient(app.app) as c:
        yield c

    if app.live_monitor_instance is not None:
        try:
            app.live_monitor_instance.stop()
        except Exception:
            pass
    app.live_monitor_instance = None
    app._live_captured_clips.clear()


def test_live_monitor_status_initial_stopped(client):
    res = client.get("/api/live-monitor/status")
    assert res.status_code == 200
    data = res.json()
    assert data["running"] is False
    assert data["channels"] == []
    assert data["captured_clips"] == []


def test_live_monitor_start_empty_channels(client):
    res = client.post("/api/live-monitor/start", json={"channels": []})
    assert res.status_code == 400
    assert "cannot be empty" in res.json()["detail"].lower()


def test_live_monitor_start_and_status(client):
    with patch("live_monitor.LiveMonitor._run_monitoring_loop"):
        res = client.post("/api/live-monitor/start", json={
            "channels": ["twitch.tv/gaules", "alanzoka"],
            "highlight_seconds": 45
        })
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "started"
        assert data["running"] is True
        assert len(data["channels"]) == 2
        assert data["highlight_seconds"] == 45
        assert app.live_monitor_instance is not None

        # Check status endpoint reflects running
        status_res = client.get("/api/live-monitor/status")
        assert status_res.status_code == 200
        status_data = status_res.json()
        assert status_data["running"] is True
        assert len(status_data["channels"]) == 2
        # channels is clean list of string URLs
        assert isinstance(status_data["channels"][0], str)
        # streams is rich list of stream objects
        assert len(status_data["streams"]) == 2
        first_stream = status_data["streams"][0]
        assert "embed_url" in first_stream
        assert "platform" in first_stream
        assert "stream_audio_status" in first_stream


def test_live_monitor_start_multi_platform_channels(client):
    with patch("live_monitor.LiveMonitor._run_monitoring_loop"):
        channels = [
            "https://kick.com/bistecone",
            "https://kick.com/coringa",
            "https://www.twitch.tv/alanzoka",
            "https://www.youtube.com/@LofiGirl/live"
        ]
        res = client.post("/api/live-monitor/start", json={"channels": channels})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "started"
        assert data["running"] is True
        assert len(data["channels"]) >= 3
        assert isinstance(data["channels"][0], str)
        assert len(data["streams"]) >= 3
        embed_urls = [s["embed_url"] for s in data["streams"]]
        assert any("kick.com" in u for u in embed_urls)
        assert any("twitch.tv" in u for u in embed_urls)
        assert any("youtube.com" in u for u in embed_urls)


def test_live_monitor_start_deduplication(client):
    with patch("live_monitor.LiveMonitor._run_monitoring_loop"):
        channels = [
            "https://kick.com/bistecone",
            "kick.com/bistecone",
            "kick:bistecone",
            "https://www.twitch.tv/alanzoka",
            "alanzoka",
            "twitch:alanzoka"
        ]
        res = client.post("/api/live-monitor/start", json={"channels": channels})
        assert res.status_code == 200
        data = res.json()
        assert data["running"] is True
        assert len(data["channels"]) == 2
        assert len(data["streams"]) == 2
        assert set(s["platform"] for s in data["streams"]) == {"kick", "twitch"}


def test_live_monitor_stop(client):
    with patch("live_monitor.LiveMonitor._run_monitoring_loop"):
        client.post("/api/live-monitor/start", json={"channels": ["tarik"]})
        assert app.live_monitor_instance is not None
        assert app.live_monitor_instance.is_running is True

        res = client.post("/api/live-monitor/stop")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "stopped"
        assert data["running"] is False
        assert app.live_monitor_instance is None


def test_live_monitor_stop_when_already_stopped(client):
    res = client.post("/api/live-monitor/stop")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "stopped"
    assert data["running"] is False


def test_live_monitor_clip_not_running(client):
    res = client.post("/api/live-monitor/clip", json={"channel": "kick.com/bistecone"})
    assert res.status_code == 400
    assert "not running" in res.json()["detail"].lower()


def test_live_monitor_clip_empty_channel(client):
    with patch("live_monitor.LiveMonitor._run_monitoring_loop"):
        client.post("/api/live-monitor/start", json={"channels": ["alanzoka"]})
        res = client.post("/api/live-monitor/clip", json={"channel": ""})
        assert res.status_code == 400
        assert "channel must be specified" in res.json()["detail"].lower()


def test_live_monitor_clip_success(client):
    with patch("live_monitor.LiveMonitor._run_monitoring_loop"):
        client.post("/api/live-monitor/start", json={"channels": ["kick.com/bistecone"]})
        assert app.live_monitor_instance is not None

        with patch.object(app.live_monitor_instance, "trigger_clip", return_value=None) as mock_trigger:
            res = client.post("/api/live-monitor/clip", json={
                "channel": "kick.com/bistecone",
                "reason": "manual_instant_highlight"
            })
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "clipping"
            assert data["channel"] == "kick.com/bistecone"
            mock_trigger.assert_called_once_with("kick.com/bistecone", reason="manual_instant_highlight")


@pytest.mark.asyncio
async def test_enqueue_live_clip_async():
    # Create a temporary dummy highlight video clip
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41")
        temp_clip = f.name

    try:
        meta = {
            "channel": "gaules",
            "platform": "twitch",
            "url": "https://www.twitch.tv/gaules",
            "timestamp": time.time(),
            "duration": 45,
            "reason": "hype_emote_spike",
        }

        job_id = await app._enqueue_live_clip_async(temp_clip, meta)
        assert job_id is not None
        assert job_id in app.jobs

        job = app.jobs[job_id]
        assert job["status"] == "queued"
        assert job["channel"] == "gaules"
        assert job["is_live_highlight"] is True
        assert job.get("source_type") == "live_stream_audio_highlight"
        assert job["attestation"].get("source_type") == "live_stream_audio_highlight"
        
        # Verify vertical format and streamer reframe layout environment variables
        assert job["env"].get("IS_STREAM_SOURCE") == "1"
        assert job["env"].get("WEBCAM_ON_TOP") == "1"
        assert "--format" in job["cmd"]
        format_idx = job["cmd"].index("--format")
        assert job["cmd"][format_idx + 1] == "vertical"

        # Verify added to _live_captured_clips
        assert len(app._live_captured_clips) > 0
        latest_clip = app._live_captured_clips[-1]
        assert latest_clip["job_id"] == job_id
        assert latest_clip["channel"] == "gaules"
    finally:
        if os.path.exists(temp_clip):
            try:
                os.remove(temp_clip)
            except OSError:
                pass


def test_start_keeps_the_dashboard_key_in_memory(client):
    with patch("live_monitor.LiveMonitor._run_monitoring_loop"):
        res = client.post("/api/live-monitor/start", json={"channels": ["alanzoka"]},
                          headers={"X-Gemini-Key": "KEY-FROM-DASHBOARD"})
        assert res.status_code == 200
        assert app.live_monitor_instance.gemini_api_key == "KEY-FROM-DASHBOARD"


def test_start_defaults_to_a_120_second_highlight(client):
    with patch("live_monitor.LiveMonitor._run_monitoring_loop"):
        res = client.post("/api/live-monitor/start", json={"channels": ["alanzoka"]})
        assert res.status_code == 200
        assert res.json()["highlight_seconds"] == 120
