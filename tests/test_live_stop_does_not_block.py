"""Stopping the live monitor answers at once, even while an ASR chunk is still decoding.

stop() used to call transcribe_backends.release_models() on the event loop. That call
waits for the ASR gate, so one chunk still being transcribed froze every request, the
stop included, until the decode ended. The release now runs on its own thread.
"""
import asyncio
import shutil
import sys
import tempfile
import threading
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
from live_monitor import LiveMonitor  # noqa: E402


def test_stop_answers_while_an_asr_chunk_still_holds_the_gate(monkeypatch):
    out_dir = tempfile.mkdtemp(prefix="live_stop_")
    gate = threading.Event()
    released = threading.Event()

    def release_models():
        gate.wait(5)  # a chunk still decoding holds the ASR gate
        released.set()

    fake_backends = types.ModuleType("transcribe_backends")
    fake_backends.release_models = release_models
    monkeypatch.setitem(sys.modules, "transcribe_backends", fake_backends)

    monitor = LiveMonitor(channels=["https://kick.com/bistecone"], output_dir=out_dir, mock_extraction=True)
    monitor._running = True
    monitor._asr_used = True
    monkeypatch.setattr(app, "live_monitor_instance", monitor)
    threading.Timer(1.0, gate.set).start()

    ticks_while_gate_closed = []

    async def ticker():
        for _ in range(10):
            if not gate.is_set():
                ticks_while_gate_closed.append(time.monotonic())
            await asyncio.sleep(0.02)

    async def scenario():
        started = time.monotonic()
        stop = asyncio.ensure_future(app.live_monitor_stop())
        ticking = asyncio.ensure_future(ticker())
        result = await stop
        elapsed = time.monotonic() - started
        await ticking
        return result, elapsed

    try:
        result, elapsed = asyncio.run(scenario())
        assert result == {"status": "stopped", "running": False}
        assert elapsed < 0.5, f"stop took {elapsed:.2f}s: it waited for the ASR release"
        assert len(ticks_while_gate_closed) >= 5, "the event loop served nothing while stop ran"
        assert released.wait(5), "the ASR release never ran"
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
