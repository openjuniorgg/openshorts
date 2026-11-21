"""Unit tests for live_monitor.py: live stream monitoring, velocity spike detection, and highlight clipping."""
import asyncio
import os
import shutil
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

import live_monitor
from live_monitor import (
    ChannelState,
    LiveMonitor,
    extract_highlight_clip,
    get_embed_url,
    is_hype_message,
    parse_channel,
    resolve_streamlink_hls,
    sample_stream_audio_loudness,
)


@pytest.fixture
def temp_output_dir():
    d = tempfile.mkdtemp(prefix="test_live_monitor_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 1. Channel Parsing Tests
# --------------------------------------------------------------------------- #

def test_parse_twitch_channel_names():
    res1 = parse_channel("gaules")
    assert res1 is not None
    assert res1["platform"] == "twitch"
    assert res1["channel"] == "gaules"
    assert res1["url"] == "https://www.twitch.tv/gaules"
    assert res1["id"] == "twitch_gaules"

    res2 = parse_channel("alanzoka")
    assert res2 is not None
    assert res2["platform"] == "twitch"
    assert res2["channel"] == "alanzoka"

    res3 = parse_channel("shroud")
    assert res3 is not None
    assert res3["channel"] == "shroud"


def test_parse_twitch_urls():
    # HTTPS full URL
    res = parse_channel("https://www.twitch.tv/tarik")
    assert res is not None
    assert res["platform"] == "twitch"
    assert res["channel"] == "tarik"
    assert res["url"] == "https://www.twitch.tv/tarik"

    # URL without scheme
    res2 = parse_channel("twitch.tv/ninja")
    assert res2 is not None
    assert res2["platform"] == "twitch"
    assert res2["channel"] == "ninja"

    # Mobile URL
    res3 = parse_channel("https://m.twitch.tv/fps_shaka")
    assert res3 is not None
    assert res3["channel"] == "fps_shaka"

    # URL with query params
    res4 = parse_channel("https://www.twitch.tv/pokimane?referrer=raid")
    assert res4 is not None
    assert res4["channel"] == "pokimane"

    # URL with chat / popout
    res5 = parse_channel("https://www.twitch.tv/popout/xqc/chat")
    assert res5 is not None
    assert res5["channel"] == "xqc"


def test_parse_prefixed_channels():
    res_t = parse_channel("twitch:gaules")
    assert res_t is not None
    assert res_t["platform"] == "twitch"
    assert res_t["channel"] == "gaules"

    res_yt = parse_channel("youtube:@StreamerLive")
    assert res_yt is not None
    assert res_yt["platform"] == "youtube"
    assert "StreamerLive" in res_yt["channel"]


def test_parse_kick_urls():
    res1 = parse_channel("https://kick.com/bistecone")
    assert res1 is not None
    assert res1["platform"] == "kick"
    assert res1["channel"] == "bistecone"
    assert res1["url"] == "https://kick.com/bistecone"
    assert res1["id"] == "kick_bistecone"

    res2 = parse_channel("https://kick.com/coringa")
    assert res2 is not None
    assert res2["platform"] == "kick"
    assert res2["channel"] == "coringa"

    res3 = parse_channel("kick:bistecone")
    assert res3 is not None
    assert res3["platform"] == "kick"
    assert res3["channel"] == "bistecone"

    res4 = parse_channel("kick.com/coringa")
    assert res4 is not None
    assert res4["platform"] == "kick"
    assert res4["channel"] == "coringa"


def test_parse_youtube_urls():
    res = parse_channel("https://www.youtube.com/@LofiGirl/live")
    assert res is not None
    assert res["platform"] == "youtube"
    assert "LofiGirl" in res["channel"]
    assert res["url"] == "https://www.youtube.com/@LofiGirl/live"

    res2 = parse_channel("https://www.youtube.com/watch?v=M576WGiDBdQ")
    assert res2 is not None
    assert res2["platform"] == "youtube"
    assert res2["channel"] == "M576WGiDBdQ"
    assert res2["id"] == "youtube_M576WGiDBdQ"

    res3 = parse_channel("https://youtu.be/M576WGiDBdQ")
    assert res3 is not None
    assert res3["platform"] == "youtube"
    assert res3["channel"] == "M576WGiDBdQ"

    res4 = parse_channel("https://www.youtube.com/live/M576WGiDBdQ")
    assert res4 is not None
    assert res4["platform"] == "youtube"
    assert res4["channel"] == "M576WGiDBdQ"

    assert res2 is not None
    assert res2["platform"] == "youtube"


def test_parse_invalid_channels():
    assert parse_channel(None) is None
    assert parse_channel("") is None
    assert parse_channel("   ") is None
    assert parse_channel("https://www.twitch.tv/videos/123456") is None
    assert parse_channel("https://www.twitch.tv/directory") is None
    assert parse_channel("https://unsupported-site.com/stream") is None


# --------------------------------------------------------------------------- #
# 2. Hype Emote and Message Detection Tests
# --------------------------------------------------------------------------- #

def test_is_hype_message_emotes():
    assert is_hype_message("OMEGALUL THAT WAS INSANE") is True
    assert is_hype_message("KEKW KEKW KEKW") is True
    assert is_hype_message("POG what a shot!") is True
    assert is_hype_message("BIG W IN THE CHAT") is True
    assert is_hype_message("pogchamp dude") is True
    assert is_hype_message("5head move") is True


def test_is_hype_message_clip_commands():
    assert is_hype_message("!clip that now") is True
    assert is_hype_message("CLIP IT CLIP IT") is True
    assert is_hype_message("alguem clipe isso pfv") is True


def test_is_hype_message_repeated_laughter():
    assert is_hype_message("kkkkkkkkkkkkkkkkkkkk") is True
    assert is_hype_message("hahahahahahaha") is True
    assert is_hype_message("omgggggggg") is True


def test_is_hype_message_normal_messages():
    assert is_hype_message("hello everyone") is False
    assert is_hype_message("what game is this?") is False
    assert is_hype_message("good morning streamer") is False
    assert is_hype_message("") is False


# --------------------------------------------------------------------------- #
# 3. Rolling Velocity and Metrics Calculation Tests
# --------------------------------------------------------------------------- #

def test_channel_state_metrics_empty():
    state = ChannelState("test_chan", "twitch", "https://twitch.tv/test_chan")
    vel, hype, base = state.update_metrics(now=100.0)
    assert vel == 0.0
    assert hype == 0.0
    assert base == 0.0


def test_channel_state_velocity_rolling_10s():
    state = ChannelState("test_chan", "twitch", "https://twitch.tv/test_chan")
    base_t = 1000.0

    # Add 20 messages between t=991.0 and t=1000.0 (all inside the last 10s)
    for i in range(20):
        t = base_t - 9.0 + (i * 0.4)
        state.record_message("LUL that was cool", timestamp=t)

    vel, hype, base = state.update_metrics(now=base_t)
    # 20 messages / 10s = 2.0 messages/sec
    assert vel == pytest.approx(2.0, rel=1e-2)
    assert hype == 1.0  # all messages had "LUL"


def test_channel_state_pruning_old_messages():
    state = ChannelState("test_chan", "twitch", "https://twitch.tv/test_chan")
    now = 500.0

    # Message from 90s ago (should be pruned)
    state.record_message("old message", timestamp=now - 90.0)
    # Message from 20s ago (outside 10s window, inside 60s baseline)
    state.record_message("baseline message", timestamp=now - 20.0)
    # Message from 5s ago (inside 10s window)
    state.record_message("recent message", timestamp=now - 5.0)

    vel, hype, base = state.update_metrics(now=now)
    # Old message was pruned: remaining messages are 2 (at -20s and -5s)
    assert len(state.messages) == 2
    # Only the -5s message is within the 10s window -> 1 / 10 = 0.1 msg/s
    assert vel == pytest.approx(0.1, rel=1e-2)


# --------------------------------------------------------------------------- #
# 4. Spike Detection Tests
# --------------------------------------------------------------------------- #

def test_spike_detection_hype_density():
    state = ChannelState("test_chan", "twitch", "https://twitch.tv/test_chan")
    now = 100.0

    # 15 messages in the last 5 seconds, all with hype emotes
    for i in range(15):
        state.record_message("KEKW POG LUL", timestamp=now - 4.0 + (i * 0.2))

    spike, reason = state.check_spike(
        now=now,
        cooldown_seconds=60.0,
        min_velocity=1.0,
        min_hype_ratio=0.30
    )
    assert spike is True
    assert reason is not None
    assert "hype_emote_spike" in reason


def test_spike_detection_chat_clip_commands():
    state = ChannelState("test_chan", "twitch", "https://twitch.tv/test_chan")
    now = 100.0

    # 4 users saying !clip
    for i in range(4):
        state.record_message(f"viewer_{i}: !clip that!", timestamp=now - 3.0 + i * 0.5)

    spike, reason = state.check_spike(now=now, cooldown_seconds=60.0)
    assert spike is True
    assert "chat_clip_command" in reason


def test_spike_detection_normal_chat_no_spike():
    state = ChannelState("test_chan", "twitch", "https://twitch.tv/test_chan")
    now = 100.0

    # Low frequency, normal conversational messages
    state.record_message("hey streamer", timestamp=now - 8.0)
    state.record_message("what is this game", timestamp=now - 4.0)

    spike, reason = state.check_spike(now=now, cooldown_seconds=60.0, min_velocity=2.0)
    assert spike is False
    assert reason is None


# --------------------------------------------------------------------------- #
# 5. Cooldown Logic Tests
# --------------------------------------------------------------------------- #

def test_cooldown_prevents_duplicate_spikes():
    state = ChannelState("test_chan", "twitch", "https://twitch.tv/test_chan")
    t0 = 200.0

    # First spike at t0
    for i in range(10):
        state.record_message("POG CHAMP LUL", timestamp=t0 - 2.0 + (i * 0.1))

    spike1, reason1 = state.check_spike(now=t0, cooldown_seconds=60.0, min_velocity=1.0)
    assert spike1 is True
    # Mark cooldown start
    state.last_clip_time = t0

    # Second spike attempt at t0 + 20s (during 60s cooldown)
    t1 = t0 + 20.0
    for i in range(15):
        state.record_message("KEKW KEKW KEKW", timestamp=t1 - 2.0 + (i * 0.1))

    spike2, reason2 = state.check_spike(now=t1, cooldown_seconds=60.0, min_velocity=1.0)
    assert spike2 is False  # Blocked by cooldown!

    # Third spike attempt at t0 + 65s (cooldown expired)
    t2 = t0 + 65.0
    for i in range(10):
        state.record_message("LULW OMG INSANE", timestamp=t2 - 2.0 + (i * 0.1))

    spike3, reason3 = state.check_spike(now=t2, cooldown_seconds=60.0, min_velocity=1.0)
    assert spike3 is True  # Allowed!


def test_cooldown_per_channel_isolation():
    chan_a = ChannelState("channel_a", "twitch", "https://twitch.tv/channel_a")
    chan_b = ChannelState("channel_b", "twitch", "https://twitch.tv/channel_b")

    t = 300.0
    for i in range(10):
        chan_a.record_message("POG", timestamp=t - 2.0 + (i * 0.1))
        chan_b.record_message("POG", timestamp=t - 2.0 + (i * 0.1))

    spike_a, _ = chan_a.check_spike(now=t, cooldown_seconds=60.0, min_velocity=1.0)
    assert spike_a is True
    chan_a.last_clip_time = t  # A enters cooldown

    # Channel B should still be allowed to spike independently
    spike_b, _ = chan_b.check_spike(now=t, cooldown_seconds=60.0, min_velocity=1.0)
    assert spike_b is True


# --------------------------------------------------------------------------- #
# 6. LiveMonitor Management & Lifecycle Tests
# --------------------------------------------------------------------------- #

def test_monitor_channel_management(temp_output_dir):
    monitor = LiveMonitor(
        channels=["gaules", "https://twitch.tv/alanzoka", "https://youtube.com/@LofiGirl/live"],
        output_dir=temp_output_dir,
        mock_extraction=True
    )

    status = monitor.get_status()
    assert status["running"] is False
    assert len(status["channels"]) == 3

    # Add new channel
    monitor.add_channel("twitch:tarik")
    status = monitor.get_status()
    assert len(status["channels"]) == 4

    # Remove channel
    removed = monitor.remove_channel("gaules")
    assert removed is True
    status = monitor.get_status()
    assert len(status["channels"]) == 3
    assert not any(c["name"] == "gaules" for c in status["channels"])


def test_monitor_multiplatform_kick_twitch_youtube(temp_output_dir):
    user_channels = [
        "https://kick.com/bistecone",
        "https://kick.com/coringa",
        "https://www.youtube.com/watch?v=M576WGiDBdQ",
        "https://www.twitch.tv/alanzoka",
    ]
    monitor = LiveMonitor(
        channels=user_channels,
        output_dir=temp_output_dir,
        mock_extraction=True
    )
    status = monitor.get_status()
    assert len(status["channels"]) == 4
    platforms = {c["platform"] for c in status["channels"]}
    assert platforms == {"kick", "youtube", "twitch"}
    names = {c["name"] for c in status["channels"]}
    assert "bistecone" in names
    assert "coringa" in names
    assert "alanzoka" in names
    assert "M576WGiDBdQ" in names


def test_monitor_start_and_stop_synchronously(temp_output_dir):
    monitor = LiveMonitor(
        channels=["gaules"],
        output_dir=temp_output_dir,
        mock_extraction=True
    )

    assert monitor.is_running is False
    monitor.start()
    assert monitor.is_running is True
    assert monitor.get_status()["running"] is True

    monitor.stop()
    assert monitor.is_running is False
    assert monitor.get_status()["running"] is False


@pytest.mark.asyncio
async def test_monitor_start_and_stop_async(temp_output_dir):
    monitor = LiveMonitor(
        channels=["alanzoka"],
        output_dir=temp_output_dir,
        mock_extraction=True
    )

    await monitor.start()
    assert monitor.is_running is True

    await monitor.stop()
    assert monitor.is_running is False


# --------------------------------------------------------------------------- #
# 7. Simulated Clip Extraction & Callback Tests
# --------------------------------------------------------------------------- #

def test_manual_trigger_clip_mock(temp_output_dir):
    dispatched_clips = []

    def on_clip(path, meta):
        dispatched_clips.append((path, meta))

    monitor = LiveMonitor(
        channels=["gaules"],
        on_clip_ready=on_clip,
        highlight_seconds=30,
        output_dir=temp_output_dir,
        mock_extraction=True
    )

    monitor.trigger_clip("gaules", reason="manual_test")

    status = monitor.get_status()
    assert len(status["captured_clips"]) == 1
    clip_info = status["captured_clips"][0]
    assert os.path.exists(clip_info["clip_path"])
    assert clip_info["channel"] == "gaules"
    assert clip_info["reason"] == "manual_test"
    assert clip_info["duration"] == 30

    assert len(dispatched_clips) == 1
    assert dispatched_clips[0][0] == clip_info["clip_path"]


@pytest.mark.asyncio
async def test_trigger_clip_async_return_value(temp_output_dir):
    dispatched = []

    def on_clip(path, meta):
        dispatched.append(path)

    monitor = LiveMonitor(
        channels=["https://kick.com/bistecone"],
        on_clip_ready=on_clip,
        output_dir=temp_output_dir,
        mock_extraction=True
    )
    clip_path = await monitor.trigger_clip("bistecone", reason="manual_instant_highlight")
    assert clip_path is not None
    assert os.path.exists(clip_path)
    assert "live_bistecone_" in clip_path
    assert len(dispatched) == 1
    assert dispatched[0] == clip_path


@pytest.mark.asyncio
async def test_async_callback_on_spike(temp_output_dir):
    dispatched_events = []

    async def async_on_clip(path, meta):
        dispatched_events.append((path, meta))

    monitor = LiveMonitor(
        channels=["tarik"],
        on_clip_ready=async_on_clip,
        highlight_seconds=45,
        output_dir=temp_output_dir,
        mock_extraction=True
    )

    now = 1000.0
    # Inject 25 hype messages to exceed velocity threshold (25 / 10s = 2.5 msg/s > 2.0)
    for i in range(25):
        monitor.record_chat_message("tarik", "POGGERS WHAT A FLICK KEKW", timestamp=now - 2.0 + (i * 0.05))

    # Allow event loop tasks to run
    await asyncio.sleep(0.05)

    status = monitor.get_status()
    assert len(status["captured_clips"]) == 1
    assert len(dispatched_events) == 1
    assert os.path.exists(dispatched_events[0][0])
    assert dispatched_events[0][1]["channel"] == "tarik"


# --------------------------------------------------------------------------- #
# 8. Streamlink and FFmpeg Extraction Helpers Tests
# --------------------------------------------------------------------------- #

def test_resolve_streamlink_hls_mock():
    with patch("streamlink.Streamlink") as MockStreamlink:
        mock_session = MagicMock()
        MockStreamlink.return_value = mock_session

        mock_stream = MagicMock()
        mock_stream.to_manifest_url.return_value = "https://cdn.twitch.tv/hls/test.m3u8"
        mock_session.streams.return_value = {"best": mock_stream}

        hls_url = resolve_streamlink_hls("https://twitch.tv/shroud")
        assert hls_url == "https://cdn.twitch.tv/hls/test.m3u8"


def test_extract_highlight_clip_mock_ffmpeg(temp_output_dir):
    output_file = os.path.join(temp_output_dir, "test_clip.mp4")

    def fake_subprocess_run(cmd, **kwargs):
        assert "ffmpeg" in cmd[0]
        assert "-live_start_index" in cmd
        assert "-3" in cmd
        assert "-c" in cmd
        assert "copy" in cmd
        assert "-t" in cmd
        assert "45" in cmd
        # Simulate ffmpeg writing output file
        with open(output_file, "wb") as f:
            f.write(b"fake_mp4_content")
        return MagicMock(returncode=0, stderr=b"")

    with patch("subprocess.run", side_effect=fake_subprocess_run):
        ok = extract_highlight_clip(
            hls_url="https://cdn.twitch.tv/hls/test.m3u8",
            output_path=output_file,
            duration=45,
            live_start_index=-3
        )
        assert ok is True
        assert os.path.exists(output_file)


# --------------------------------------------------------------------------- #
# 9. Live Player Embed URL Tests
# --------------------------------------------------------------------------- #

def test_get_embed_url_twitch():
    url = get_embed_url({"platform": "twitch", "channel": "alanzoka"}, parent="openshorts.local")
    assert "player.twitch.tv" in url
    assert "channel=alanzoka" in url
    assert "parent=openshorts.local" in url
    assert "muted=true" in url


def test_get_embed_url_kick():
    url = get_embed_url({"platform": "kick", "channel": "bistecone"})
    assert "player.kick.com/bistecone" in url
    assert "muted=true" in url


def test_get_embed_url_youtube():
    url = get_embed_url({"platform": "youtube", "channel": "M576WGiDBdQ"})
    assert "youtube.com/embed/M576WGiDBdQ" in url
    assert "autoplay=1" in url
    assert "mute=1" in url


def test_parse_channel_includes_embed_url():
    parsed = parse_channel("https://kick.com/coringa")
    assert parsed is not None
    assert "embed_url" in parsed
    assert "player.kick.com/coringa" in parsed["embed_url"]


# --------------------------------------------------------------------------- #
# 10. In-Stream Audio Excitement & Shout Detection Tests
# --------------------------------------------------------------------------- #

def test_audio_sample_normal_conversational_no_spike():
    state = ChannelState("test_chan", "twitch", "https://twitch.tv/test_chan")
    t0 = 100.0

    # Normal talking / stream gameplay baseline (-28 dB mean, -18 dB peak)
    for i in range(5):
        state.record_audio_sample(mean_db=-28.0, max_db=-18.0, timestamp=t0 + i * 4.0)

    spike, reason = state.check_audio_spike(now=t0 + 20.0)
    assert spike is False
    assert reason is None
    assert state.audio_status == "monitoring"


def test_audio_sample_streamer_shout_triggers_spike():
    state = ChannelState("test_chan", "twitch", "https://twitch.tv/test_chan")
    t0 = 200.0

    # 4 normal baseline samples around -26 dB
    for i in range(4):
        state.record_audio_sample(mean_db=-26.0, max_db=-16.0, timestamp=t0 + i * 4.0)

    # Streamer shouts / reacts excitedly: peak jumps to -2.5 dB (+23.5 dB jump)
    t_shout = t0 + 16.0
    state.record_audio_sample(mean_db=-12.0, max_db=-2.5, timestamp=t_shout)

    spike, reason = state.check_audio_spike(now=t_shout)
    assert spike is True
    assert reason is not None
    assert "stream_audio_shout" in reason
    assert state.audio_status == "peak"


def test_audio_sample_sustained_burst_triggers_spike():
    state = ChannelState("test_chan", "kick", "https://kick.com/test_chan")
    t0 = 300.0

    # Baseline around -30 dB
    for i in range(4):
        state.record_audio_sample(mean_db=-30.0, max_db=-20.0, timestamp=t0 + i * 4.0)

    # Sustained shouting / scream burst (-13 dB mean, +17 dB jump)
    t_burst = t0 + 16.0
    state.record_audio_sample(mean_db=-13.0, max_db=-7.0, timestamp=t_burst)

    spike, reason = state.check_audio_spike(now=t_burst)
    assert spike is True
    assert "stream" in reason


def test_audio_spike_respects_cooldown():
    state = ChannelState("test_chan", "twitch", "https://twitch.tv/test_chan")
    t0 = 400.0

    for i in range(4):
        state.record_audio_sample(mean_db=-25.0, max_db=-15.0, timestamp=t0 + i * 4.0)

    # Shout 1 at t0 + 16s
    t1 = t0 + 16.0
    state.record_audio_sample(mean_db=-10.0, max_db=-1.0, timestamp=t1)
    spike1, _ = state.check_audio_spike(now=t1, cooldown_seconds=60.0)
    assert spike1 is True
    state.last_clip_time = t1  # Start cooldown

    # Shout 2 at t1 + 20s (inside 60s cooldown)
    t2 = t1 + 20.0
    state.record_audio_sample(mean_db=-9.0, max_db=-0.5, timestamp=t2)
    spike2, _ = state.check_audio_spike(now=t2, cooldown_seconds=60.0)
    assert spike2 is False
    assert state.audio_status == "cooldown"

    # Shout 3 at t1 + 65s (cooldown expired)
    t3 = t1 + 65.0
    state.record_audio_sample(mean_db=-9.0, max_db=-1.0, timestamp=t3)
    spike3, _ = state.check_audio_spike(now=t3, cooldown_seconds=60.0)
    assert spike3 is True


def test_sample_stream_audio_loudness_parses_ffmpeg_output():
    fake_stderr = (
        b"[Parsed_volumedetect_0 @ 0000...] n_samples: 88200\n"
        b"[Parsed_volumedetect_0 @ 0000...] mean_volume: -24.3 dB\n"
        b"[Parsed_volumedetect_0 @ 0000...] max_volume: -3.8 dB\n"
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr=fake_stderr)
        mean_v, max_v = sample_stream_audio_loudness("https://cdn.example.com/live.m3u8", duration=3)
        assert mean_v == -24.3
        assert max_v == -3.8


def test_record_stream_audio_sample_dispatches_clip(temp_output_dir):
    dispatched = []

    def on_clip(path, meta):
        dispatched.append((path, meta))

    monitor = LiveMonitor(
        channels=["https://kick.com/bistecone"],
        on_clip_ready=on_clip,
        output_dir=temp_output_dir,
        mock_extraction=True
    )

    t0 = 500.0
    # Baseline
    for i in range(4):
        monitor.record_stream_audio_sample("kick_bistecone", mean_db=-28.0, max_db=-18.0, timestamp=t0 + i * 4.0)

    # In-stream audio shout spike
    t_shout = t0 + 16.0
    monitor.record_stream_audio_sample("kick_bistecone", mean_db=-11.0, max_db=-2.0, timestamp=t_shout)

    status = monitor.get_status()
    assert len(status["captured_clips"]) == 1
    clip_info = status["captured_clips"][0]
    assert clip_info["channel"] == "bistecone"
    assert "stream_audio" in clip_info["reason"]
    assert clip_info["source_type"] == "live_stream_audio_highlight"
    assert len(dispatched) == 1


def test_status_includes_embed_url_and_audio_metrics(temp_output_dir):
    monitor = LiveMonitor(
        channels=["https://kick.com/coringa", "https://twitch.tv/alanzoka"],
        output_dir=temp_output_dir,
        mock_extraction=True
    )
    status = monitor.get_status()
    assert len(status["channels"]) == 2

    kick_chan = next(c for c in status["channels"] if c["platform"] == "kick")
    assert "player.kick.com/coringa" in kick_chan["embed_url"]
    assert "audio_level" in kick_chan
    assert "audio_peak_db" in kick_chan
    assert "audio_status" in kick_chan

