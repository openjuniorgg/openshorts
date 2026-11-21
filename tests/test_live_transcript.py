"""Live transcript chunks: keyword candidates, the GTA VI gate, no local score.

No GPU, ffmpeg or network is used: the ASR and the stream metadata are replaced.
"""
import asyncio
import shutil
import tempfile
from unittest.mock import patch

import pytest

import live_monitor
from live_monitor import LiveMonitor
from live_transcript import chunk_is_candidate, detect_speech_hits


@pytest.fixture
def out_dir():
    d = tempfile.mkdtemp(prefix="test_live_transcript_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def seg(text, start=0.0):
    return {"text": text, "start": start, "end": start + 1.0}


def _monitor(out_dir):
    return LiveMonitor(channels=["https://kick.com/bistecone"], output_dir=out_dir, mock_extraction=True)


# --- candidates ----------------------------------------------------------- #

def test_silent_or_empty_chunk_is_not_a_candidate():
    assert chunk_is_candidate([]) is False
    assert chunk_is_candidate([seg("   ")]) is False


def test_one_weak_word_is_not_enough():
    assert chunk_is_candidate([seg("that was insane")]) is False


def test_two_weak_hits_make_a_candidate():
    assert chunk_is_candidate([seg("insane, the cops are here")]) is True


def test_a_clip_request_alone_is_a_candidate():
    assert chunk_is_candidate([seg("clipa isso")]) is True


def test_a_scream_alone_is_a_candidate():
    assert chunk_is_candidate([seg("AAAAHHHH")]) is True


def test_portuguese_laughter_and_reaction_together():
    assert chunk_is_candidate([seg("kkkkkk nossa, que isso")]) is True


def test_repeated_word_counts_once():
    hits = detect_speech_hits([seg("kkkk kkkk kkkk")])
    assert hits == {"laughter": ["kkkk"]}
    assert chunk_is_candidate([seg("kkkk kkkk kkkk")]) is False


def test_hits_are_grouped_by_category():
    hits = detect_speech_hits([seg("headshot, GG, the police are chasing us")])
    assert "game_event" in hits and "reaction" in hits


# --- monitor: candidates start clips, Gemini decides --------------------- #

def test_candidate_chunk_starts_a_clip(out_dir):
    monitor = _monitor(out_dir)
    state = monitor._channels["kick_bistecone"]
    monitor._on_transcript_chunk("kick_bistecone", state, [seg("insane, the cops are here")], 1000.0)
    assert len(monitor.captured_clips) == 1
    assert "live_speech" in monitor.captured_clips[0]["reason"]


def test_non_candidate_chunk_starts_nothing(out_dir):
    monitor = _monitor(out_dir)
    state = monitor._channels["kick_bistecone"]
    monitor._on_transcript_chunk("kick_bistecone", state, [seg("so the plan is to go left")], 1000.0)
    assert monitor.captured_clips == []


def test_candidate_respects_the_cooldown(out_dir):
    monitor = _monitor(out_dir)
    state = monitor._channels["kick_bistecone"]
    state.last_clip_time = 1000.0
    monitor._on_transcript_chunk("kick_bistecone", state, [seg("clipa isso")], 1000.0)
    assert monitor.captured_clips == []


def test_transcript_keeps_timestamped_lines(out_dir):
    monitor = _monitor(out_dir)
    state = monitor._channels["kick_bistecone"]
    monitor._on_transcript_chunk("kick_bistecone", state, [seg("hello chat", start=2.0)], 500.0)
    assert list(state.transcript) == [(502.0, "hello chat")]


def test_no_hourly_or_count_limit_on_automatic_clips(out_dir):
    monitor = _monitor(out_dir)
    state = monitor._channels["kick_bistecone"]
    for i in range(5):
        monitor._on_transcript_chunk("kick_bistecone", state, [seg("clipa isso")], 1000.0 + i * 100.0)
    assert len(monitor.captured_clips) == 5


# --- GTA VI gate ---------------------------------------------------------- #

def test_a_gta5_stream_never_clips(out_dir):
    monitor = _monitor(out_dir)
    state = monitor._channels["kick_bistecone"]
    state.game_ok = False
    monitor._on_transcript_chunk("kick_bistecone", state, [seg("clipa isso")], 1000.0)
    assert monitor.captured_clips == []


def test_gta6_check_reads_title_and_category(out_dir):
    monitor = _monitor(out_dir)
    state = monitor._channels["kick_bistecone"]
    with patch.object(live_monitor, "fetch_stream_metadata",
                      return_value={"title": "GTA 6 day one", "category": "Grand Theft Auto VI"}):
        assert asyncio.run(monitor._check_game(state)) is True
    assert state.game_ok is True
    assert state.game_category == "Grand Theft Auto VI"


def test_non_gta6_stream_is_not_monitored(out_dir):
    monitor = _monitor(out_dir)
    state = monitor._channels["kick_bistecone"]
    with patch.object(live_monitor, "fetch_stream_metadata",
                      return_value={"title": "GTA V RP", "category": "Grand Theft Auto V"}):
        assert asyncio.run(monitor._check_game(state)) is False
    assert "not GTA VI" in state.last_error


def test_unreadable_stream_is_not_monitored(out_dir):
    monitor = _monitor(out_dir)
    state = monitor._channels["kick_bistecone"]
    with patch.object(live_monitor, "fetch_stream_metadata", return_value=None):
        assert asyncio.run(monitor._check_game(state)) is False


def test_game_gate_can_be_turned_off(out_dir, monkeypatch):
    monkeypatch.setenv("LIVE_GAME_FILTER", "0")
    monitor = _monitor(out_dir)
    state = monitor._channels["kick_bistecone"]
    with patch.object(live_monitor, "fetch_stream_metadata", side_effect=AssertionError("no lookup")):
        assert asyncio.run(monitor._check_game(state)) is True


def test_offline_channel_has_no_metadata(monkeypatch):
    class FakePlugin:
        def __init__(self, session, url):
            pass

        def streams(self):
            return {}  # what streamlink returns for an offline channel

        def get_title(self):
            return None

        def get_category(self):
            return None

    class FakeSession:
        def resolve_url(self, url):
            return ("fake", FakePlugin, url)

    class FakeStreamlink:
        def Streamlink(self):
            return FakeSession()

    monkeypatch.setattr(live_monitor, "streamlink", FakeStreamlink())
    assert live_monitor.fetch_stream_metadata("https://www.twitch.tv/offline") is None


def test_live_channel_has_its_title_and_category(monkeypatch):
    class FakePlugin:
        def __init__(self, session, url):
            pass

        def streams(self):
            return {"best": object()}

        def get_title(self):
            return "GTA 6 day one"

        def get_category(self):
            return "Grand Theft Auto VI"

    class FakeSession:
        def resolve_url(self, url):
            return ("fake", FakePlugin, url)

    class FakeStreamlink:
        def Streamlink(self):
            return FakeSession()

    monkeypatch.setattr(live_monitor, "streamlink", FakeStreamlink())
    meta = live_monitor.fetch_stream_metadata("https://www.twitch.tv/live")
    assert meta == {"title": "GTA 6 day one", "category": "Grand Theft Auto VI"}
