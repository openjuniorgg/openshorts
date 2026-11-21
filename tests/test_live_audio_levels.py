"""Audio levels of a live come from its audio chunks (no second stream per live)."""
import asyncio
import math
import os
import shutil
import tempfile
import wave
from array import array

import pytest

import live_monitor
from live_monitor import LiveMonitor, chunk_window_levels

RATE = 16000


def write_wav(path, amplitude, seconds=10.0, freq=440.0):
    n = int(seconds * RATE)
    frames = array("h", (int(amplitude * 32767 * math.sin(2 * math.pi * freq * i / RATE)) for i in range(n)))
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(frames.tobytes())


@pytest.fixture
def work_dir():
    d = tempfile.mkdtemp(prefix="test_live_audio_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_a_half_scale_sine_has_the_known_levels(work_dir):
    path = os.path.join(work_dir, "chunk.wav")
    write_wav(path, amplitude=0.5)
    levels = chunk_window_levels(path)
    assert len(levels) == 3
    for _start, mean_db, max_db in levels:
        assert max_db == pytest.approx(20 * math.log10(0.5), abs=0.2)          # peak: -6.02 dBFS
        assert mean_db == pytest.approx(20 * math.log10(0.5 / math.sqrt(2)), abs=0.3)  # RMS: -9.03 dBFS


def test_window_offsets_cover_the_chunk(work_dir):
    path = os.path.join(work_dir, "chunk.wav")
    write_wav(path, amplitude=0.1)
    starts = [start for start, _m, _x in chunk_window_levels(path)]
    assert starts == pytest.approx([0.0, 10.0 / 3, 20.0 / 3], abs=0.01)


def test_silence_is_floored(work_dir):
    path = os.path.join(work_dir, "silent.wav")
    write_wav(path, amplitude=0.0)
    for _start, mean_db, max_db in chunk_window_levels(path):
        assert mean_db == -90.0
        assert max_db == -90.0


def test_non_16_bit_audio_gives_no_levels(work_dir):
    path = os.path.join(work_dir, "eight_bit.wav")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(RATE)
        wf.writeframes(bytes([128]) * RATE)
    assert chunk_window_levels(path) == []


def test_the_probe_loop_is_off_by_default(work_dir):
    monitor = LiveMonitor(channels=["https://kick.com/bistecone"], output_dir=work_dir, mock_extraction=True)
    assert monitor.audio_probe_loop is False


def test_a_loud_chunk_after_quiet_ones_starts_a_clip(work_dir):
    monitor = LiveMonitor(channels=["https://kick.com/bistecone"], output_dir=work_dir, mock_extraction=True)
    state = monitor._channels["kick_bistecone"]

    async def run():
        for i, amplitude in enumerate([0.02, 0.02, 0.9]):
            path = os.path.join(work_dir, f"chunk_{i}.wav")
            write_wav(path, amplitude=amplitude)
            await monitor._consume_chunk("kick_bistecone", state, path, 1000.0 + i * 10.0)
            assert not os.path.exists(path)  # consumed chunks are removed
        await asyncio.sleep(0.05)  # let the clip task finish

    asyncio.run(run())
    assert len(monitor.captured_clips) == 1
    assert "stream_audio_shout" in monitor.captured_clips[0]["reason"]


def test_quiet_chunks_alone_start_no_clip(work_dir):
    monitor = LiveMonitor(channels=["https://kick.com/bistecone"], output_dir=work_dir, mock_extraction=True)
    state = monitor._channels["kick_bistecone"]

    async def run():
        for i in range(3):
            path = os.path.join(work_dir, f"quiet_{i}.wav")
            write_wav(path, amplitude=0.02)
            await monitor._consume_chunk("kick_bistecone", state, path, 1000.0 + i * 10.0)
        await asyncio.sleep(0.05)

    asyncio.run(run())
    assert monitor.captured_clips == []
