"""Regression tests for the live monitor fixes: channels stuck in cooldown, silent
failures, clip window before the trigger, first-reading spikes, duplicate loops,
offline HLS retries and API-key leakage into the clip list."""
import asyncio
import os
import shutil
import tempfile
from unittest.mock import patch

import pytest

import live_monitor
from live_monitor import ChannelState, LiveMonitor


@pytest.fixture
def out_dir():
    d = tempfile.mkdtemp(prefix="test_live_fixes_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _baseline_then_shout(monitor, cid, t0=500.0):
    for i in range(4):
        monitor.record_stream_audio_sample(cid, mean_db=-28.0, max_db=-18.0, timestamp=t0 + i * 4.0)
    monitor.record_stream_audio_sample(cid, mean_db=-11.0, max_db=-2.0, timestamp=t0 + 16.0)


def test_channel_returns_to_active_after_a_clip(out_dir):
    monitor = LiveMonitor(channels=["https://kick.com/bistecone"], output_dir=out_dir, mock_extraction=True)
    monitor._running = True
    _baseline_then_shout(monitor, "kick_bistecone")

    state = monitor._channels["kick_bistecone"]
    assert len(monitor.captured_clips) == 1
    # Before the fix this stayed "cooldown" forever and the channel was never sampled again.
    assert state.status == "active"


def test_failed_extraction_is_reported_not_silent(out_dir):
    monitor = LiveMonitor(channels=["https://kick.com/bistecone"], output_dir=out_dir, mock_extraction=False)
    with patch.object(live_monitor, "resolve_streamlink_hls", return_value=None):
        monitor.trigger_clip("bistecone", reason="manual_test")

    failed = [c for c in monitor.captured_clips if c.get("status") == "failed"]
    assert len(failed) == 1
    assert "HLS" in failed[0]["error"]
    state = monitor._channels["kick_bistecone"]
    assert state.clip_failures == 1
    assert state.last_error and "HLS" in state.last_error


def test_clip_window_starts_before_the_trigger(out_dir):
    monitor = LiveMonitor(channels=["https://kick.com/bistecone"], output_dir=out_dir,
                          mock_extraction=False, highlight_seconds=45, pre_roll_seconds=15)
    seen = {}

    def fake_extract(hls_url, output_path, duration, live_start_index):
        seen.update(duration=duration, live_start_index=live_start_index)
        return True

    with patch.object(live_monitor, "resolve_streamlink_hls", return_value="https://cdn/x.m3u8"), \
         patch.object(live_monitor, "extract_highlight_clip", side_effect=fake_extract):
        monitor.trigger_clip("bistecone", reason="manual_test")

    assert seen["duration"] == 45
    assert seen["live_start_index"] <= -7  # about 15 s of 2 s segments before the live edge


def test_first_audio_reading_never_fires():
    state = ChannelState("c", "kick", "https://kick.com/c")
    state.record_audio_sample(mean_db=-10.0, max_db=-1.0, timestamp=100.0)
    spike, _ = state.check_audio_spike(now=100.0)
    assert spike is False


def test_api_key_goes_to_callback_but_not_to_clip_list(out_dir):
    received = {}

    def on_clip(path, meta):
        received.update(meta)

    monitor = LiveMonitor(channels=["gaules"], on_clip_ready=on_clip, output_dir=out_dir, mock_extraction=True)
    monitor.trigger_clip("gaules", reason="manual_test", ai_api_key="SECRET-KEY")

    assert received.get("ai_api_key") == "SECRET-KEY"
    assert all("ai_api_key" not in c for c in monitor.captured_clips)
    assert "SECRET-KEY" not in repr(monitor.get_status())


def test_restart_does_not_leave_two_loops(out_dir):
    async def scenario():
        monitor = LiveMonitor(channels=["gaules"], output_dir=out_dir, mock_extraction=True)
        await monitor.start()
        first_loop = monitor._loop_task
        monitor.stop()
        await monitor.start()
        await asyncio.sleep(0.05)
        assert first_loop.done()
        assert monitor._loop_task is not first_loop
        monitor.stop()
        await asyncio.sleep(0.05)

    asyncio.run(scenario())


def test_offline_channel_is_not_resolved_every_tick(out_dir):
    monitor = LiveMonitor(channels=["gaules"], output_dir=out_dir, mock_extraction=False)
    monitor._running = True
    state = monitor._channels["twitch_gaules"]

    calls = []

    def offline(url, prefer=None):
        calls.append(url)
        return None

    async def scenario():
        with patch.object(live_monitor, "fetch_stream_metadata", return_value={"title": "GTA 6", "category": "Grand Theft Auto VI"}), patch.object(live_monitor, "resolve_streamlink_hls", side_effect=offline):
            await monitor._sample_channel_audio_task("twitch_gaules", state)
            await monitor._sample_channel_audio_task("twitch_gaules", state)
            await monitor._sample_channel_audio_task("twitch_gaules", state)

    asyncio.run(scenario())
    assert len(calls) == 1  # the next lookup waits OFFLINE_RETRY_SECONDS
    assert state.last_error and "not live" in state.last_error
