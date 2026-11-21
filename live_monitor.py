"""Ultra-optimized live stream monitoring engine for Twitch, Kick, and YouTube Live.

Features:
- In-Stream Audio Peak & Loudness Excitement Detection (streamer shouts, screaming, reaction peaks via FFmpeg -af volumedetect -vn)
- Embedded Player Metadata Generator (direct Twitch, Kick, and YouTube player URLs for live visual frontend grids)
- On-Demand Stream-Copy Highlight Clipping (Streamlink HLS resolver + FFmpeg -c copy, 30-60s)
- Zero continuous video decoding, zero continuous disk writes, <1% CPU overhead.
"""
from __future__ import annotations

import asyncio
import math
import wave
from array import array
import collections
import inspect
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import parse_qs, urlparse

try:
    import streamlink
except ImportError:
    streamlink = None

try:
    import websockets
except ImportError:
    websockets = None

from live_keywords import is_candidate, is_gta6_stream
from live_transcript import detect_speech_hits

logger = logging.getLogger("live_monitor")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(asctime)s] [live_monitor] [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# Standard viral / hype emote triggers
DEFAULT_HYPE_EMOTES: Set[str] = {
    "LUL", "LULW", "OMEGALUL", "KEKW", "KEK", "POG", "POGGERS", "POGCHAMP",
    "W", "CLIP", "CLIPIT", "WTF", "GG", "LETSGO", "MONKAS", "5HEAD", "AYAYA",
    "EZ", "HYPE", "OMG", "SHEESH", "GIGA", "GIGACHAD", "HOLY", "KKKK", "KKKKK",
    "MDS", "CLIPTHIS", "CLIPTHAT", "LETSGOOO", "CLAP", "CLIPPED", "POGGIES"
}

# Regex to parse Twitch IRC messages
_IRC_MSG_RE = re.compile(r"^:?(?P<user>[^!]+)![^\s]+ PRIVMSG #(?P<channel>[^\s]+) :(?P<content>.*)$")

# Twitch IRC endpoints
TWITCH_WS_URL = "wss://irc-ws.chat.twitch.tv:443"
TWITCH_IRC_HOST = "irc.chat.twitch.tv"
TWITCH_IRC_PORT = 6667
TWITCH_ANON_NICK = "justinfan12345"

# Seconds a channel waits before asking for its HLS again after a miss (offline channels).
OFFLINE_RETRY_SECONDS = 60.0
# Rough HLS segment length, used to turn "seconds before the trigger" into an
# ffmpeg -live_start_index (Twitch parts are ~2 s; other platforms vary).
HLS_SEGMENT_SECONDS_ESTIMATE = 2.0
# The audio detector needs this many earlier samples before it may fire.
MIN_AUDIO_BASELINE_SAMPLES = 3
# Live transcript: seconds of audio per ASR call, and the local text score that makes a candidate.
TRANSCRIPT_CHUNK_SECONDS = 10
# Game gate (GTA VI only): how often a stream's title and category are read again.
GAME_RECHECK_OK = 120.0
GAME_RECHECK_NOT = 60.0


class _AwaitableWrapper:
    """Helper enabling methods to be both called synchronously or awaited."""
    def __init__(self, coro_fn=None, result=None):
        self._coro_fn = coro_fn
        self._result = result

    def __await__(self):
        if self._coro_fn is not None:
            return self._coro_fn().__await__()
        async def _noop():
            return self._result
        return _noop().__await__()


def get_embed_url(channel_info: Union[Dict[str, str], str], parent: str = "localhost") -> str:
    """Returns the direct player embed URL for Twitch, Kick, or YouTube.

    Used by the frontend to render real-time interactive multi-stream tiles.
    """
    if isinstance(channel_info, str):
        parsed = parse_channel(channel_info)
    else:
        parsed = channel_info

    if not parsed:
        return ""

    platform = str(parsed.get("platform", "")).lower()
    name = str(parsed.get("channel") or parsed.get("name") or "").strip()
    url = str(parsed.get("url", "")).strip()

    if platform == "twitch":
        clean_name = name.lower().lstrip("@")
        return f"https://player.twitch.tv/?channel={clean_name}&parent={parent}&muted=true"
    elif platform == "kick":
        clean_name = name.lower().lstrip("@")
        return f"https://player.kick.com/{clean_name}?muted=true"
    elif platform == "youtube":
        vid_match = re.search(r"(?:v=|\/live\/|\/watch\?v=|\.be\/)([a-zA-Z0-9_-]{11})", url or name)
        if vid_match:
            return f"https://www.youtube.com/embed/{vid_match.group(1)}?autoplay=1&mute=1"
        clean_name = name.lstrip("@")
        return f"https://www.youtube.com/embed/{clean_name}?autoplay=1&mute=1"

    return parsed.get("url", "")


def parse_channel(channel_or_url: Optional[str]) -> Optional[Dict[str, str]]:
    """Parses a live channel input (name or URL) into a standardized metadata dict.

    Supports:
      - Twitch: 'gaules', 'alanzoka', 'https://twitch.tv/alanzoka', 'twitch:gaules'
      - Kick: 'https://kick.com/bistecone', 'kick.com/coringa', 'kick:bistecone'
      - YouTube: 'https://youtube.com/@channel/live', 'https://www.youtube.com/watch?v=xyz', 'https://youtu.be/xyz'
      - Prefixes: 'twitch:channel', 'kick:channel', 'youtube:channel'

    Returns:
      {'platform': 'twitch'|'kick'|'youtube', 'channel': str, 'url': str, 'id': str, 'embed_url': str} or None
    """
    if not channel_or_url:
        return None

    raw = str(channel_or_url).strip()
    if not raw:
        return None

    raw_lower = raw.lower()

    # Prefix checks
    if raw_lower.startswith("kick:"):
        name = raw.split(":", 1)[1].strip().lstrip("#/@").lower()
        if not name:
            return None
        item = {
            "platform": "kick",
            "channel": name,
            "url": f"https://kick.com/{name}",
            "id": f"kick_{name}"
        }
        item["embed_url"] = get_embed_url(item)
        return item

    if raw_lower.startswith("twitch:"):
        name = raw.split(":", 1)[1].strip().lstrip("#/@").lower()
        if not name:
            return None
        item = {
            "platform": "twitch",
            "channel": name,
            "url": f"https://www.twitch.tv/{name}",
            "id": f"twitch_{name}"
        }
        item["embed_url"] = get_embed_url(item)
        return item

    if raw_lower.startswith("youtube:"):
        val = raw.split(":", 1)[1].strip()
        if not val:
            return None
        yt_url = val if val.startswith("http") else f"https://www.youtube.com/{val}"
        chan_name = val.lstrip("@").replace("/", "_")
        item = {
            "platform": "youtube",
            "channel": chan_name,
            "url": yt_url,
            "id": f"youtube_{chan_name}"
        }
        item["embed_url"] = get_embed_url(item)
        return item

    # URL prefixing if protocol omitted
    if "://" not in raw:
        if any(d in raw_lower for d in ("twitch.tv", "kick.com", "youtube.com", "youtu.be")):
            raw = "https://" + raw
            raw_lower = raw.lower()

    try:
        parsed = urlparse(raw)
    except Exception:
        parsed = None

    if parsed and parsed.netloc:
        host = parsed.netloc.lower()
        path = parsed.path.strip("/")

        # Kick URL
        if "kick.com" in host:
            parts = [p for p in path.split("/") if p and p.lower() not in ("popout", "chatroom", "chat")]
            if parts:
                name = parts[0].lower().lstrip("@")
                if name in ("video", "videos", "categories", "privacy-policy", "terms-of-service"):
                    return None
                item = {
                    "platform": "kick",
                    "channel": name,
                    "url": f"https://kick.com/{name}",
                    "id": f"kick_{name}"
                }
                item["embed_url"] = get_embed_url(item)
                return item
            return None

        # Twitch URL
        if any(h in host for h in ("twitch.tv", "m.twitch.tv")):
            parts = [p for p in path.split("/") if p and p.lower() not in ("popout", "chat")]
            if parts:
                name = parts[0].lower().lstrip("@")
                if name in ("videos", "video", "directory", "downloads", "p"):
                    return None
                item = {
                    "platform": "twitch",
                    "channel": name,
                    "url": f"https://www.twitch.tv/{name}",
                    "id": f"twitch_{name}"
                }
                item["embed_url"] = get_embed_url(item)
                return item
            return None

        # YouTube URL
        if any(h in host for h in ("youtube.com", "youtu.be", "m.youtube.com")):
            # Check ?v= query parameter
            qs = parse_qs(parsed.query)
            if "v" in qs and qs["v"]:
                video_id = qs["v"][0].strip()
                item = {
                    "platform": "youtube",
                    "channel": video_id,
                    "url": raw,
                    "id": f"youtube_{video_id}"
                }
                item["embed_url"] = get_embed_url(item)
                return item

            # Check youtu.be/<id>
            if "youtu.be" in host and path:
                video_id = path.split("/")[0].strip()
                item = {
                    "platform": "youtube",
                    "channel": video_id,
                    "url": raw,
                    "id": f"youtube_{video_id}"
                }
                item["embed_url"] = get_embed_url(item)
                return item

            # Check /live/<id>
            parts = [p for p in path.split("/") if p]
            if parts:
                if parts[0].lower() == "live" and len(parts) > 1:
                    live_id = parts[1].strip()
                    item = {
                        "platform": "youtube",
                        "channel": live_id,
                        "url": raw,
                        "id": f"youtube_{live_id}"
                    }
                    item["embed_url"] = get_embed_url(item)
                    return item
                chan_name = path.replace("/live", "").lstrip("@").replace("/", "_") or "live"
                item = {
                    "platform": "youtube",
                    "channel": chan_name,
                    "url": raw,
                    "id": f"youtube_{chan_name}"
                }
                item["embed_url"] = get_embed_url(item)
                return item

    # Plain channel name assumed to be Twitch channel
    clean_name = raw.strip().lstrip("#/@").lower()
    if re.match(r"^[a-zA-Z0-9_]{3,30}$", clean_name):
        item = {
            "platform": "twitch",
            "channel": clean_name,
            "url": f"https://www.twitch.tv/{clean_name}",
            "id": f"twitch_{clean_name}"
        }
        item["embed_url"] = get_embed_url(item)
        return item

    return None


def is_hype_message(text: str, custom_emotes: Optional[Set[str]] = None) -> bool:
    """Detects whether a chat message is hype / viral."""
    if not text:
        return False

    emotes = custom_emotes or DEFAULT_HYPE_EMOTES
    tokens = re.findall(r"[A-Za-z0-9_!]+", text)
    if any(tok.upper() in emotes for tok in tokens):
        return True

    text_lower = text.lower()
    if any(k in text_lower for k in ("!clip", "clip it", "clip this", "clipe", "clipa", "clip that")):
        return True

    # Repeated laughter or hype chars e.g. "kkkkkk", "wwwww", "hahahaha", "omgggg"
    if re.search(r"k{4,}|w{4,}|(?:ha){3,}|(?:ja){3,}|omg{3,}|lmao{2,}", text_lower):
        return True

    return False


def resolve_streamlink_hls(stream_url: str, prefer: Tuple[str, ...] = ("best", "worst")) -> Optional[str]:
    """Uses Streamlink to resolve a live stream URL to its direct HLS manifest (.m3u8).

    Supports Twitch, Kick, and YouTube Live natively.
    """
    if streamlink is None:
        logger.warning("Streamlink is not installed; cannot resolve HLS URL.")
        return None

    try:
        session = streamlink.Streamlink()
        session.set_option("twitch-disable-ads", True)
        session.set_option("twitch-low-latency", True)
        session.set_option("retry-streams", 2)
        session.set_option("retry-max", 3)

        streams = session.streams(stream_url)
        if not streams:
            return None

        stream = next((streams[q] for q in prefer if q in streams), None) or next(iter(streams.values()), None)
        if not stream:
            return None

        if hasattr(stream, "to_manifest_url"):
            return stream.to_manifest_url()
        if hasattr(stream, "url"):
            return stream.url
        if hasattr(stream, "multivariant") and stream.multivariant:
            return getattr(stream.multivariant, "uri", None)
    except Exception as e:
        logger.warning(f"Streamlink resolve error for {stream_url}: {e}")
    return None


def sample_stream_audio_loudness(
    hls_url: str,
    duration: int = 3,
    timeout: int = 10
) -> Tuple[Optional[float], Optional[float]]:
    """Samples stream audio loudness metrics via FFmpeg volumedetect (-vn, zero video decode).

    Ultra-lightweight (<0.5% CPU) audio-only probe returning (mean_volume_db, max_volume_db).
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "info",
        "-live_start_index", "-1",
        "-i", hls_url,
        "-t", str(duration),
        "-vn",
        "-af", "volumedetect",
        "-f", "null",
        "-"
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        stderr_txt = proc.stderr.decode("utf-8", errors="replace")

        mean_vol = None
        max_vol = None
        for line in stderr_txt.splitlines():
            if "mean_volume:" in line:
                m = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", line)
                if m:
                    mean_vol = float(m.group(1))
            elif "max_volume:" in line:
                m = re.search(r"max_volume:\s*([-\d.]+)\s*dB", line)
                if m:
                    max_vol = float(m.group(1))

        return mean_vol, max_vol
    except Exception as e:
        logger.debug(f"Audio sampling failed for {hls_url}: {e}")
        return None, None


def extract_highlight_clip(
    hls_url: str,
    output_path: str,
    duration: int = 45,
    live_start_index: int = -3
) -> bool:
    """Extracts ONLY a 30-60s highlight clip from a live HLS playlist via FFmpeg stream-copy.

    No full stream is recorded, zero continuous disk writes, zero GPU NVENC load.
    Command: ffmpeg -y -hide_banner -loglevel warning -live_start_index -3 -i <hls_url> -t <duration> -c copy output.mp4
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "warning",
        "-live_start_index", str(live_start_index),
        "-i", hls_url,
        "-t", str(duration),
        "-c", "copy",
        "-movflags", "+faststart",
        output_path
    ]

    try:
        logger.info(f"Extracting live highlight ({duration}s) to {output_path}...")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=duration + 30)
        if proc.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"Successfully extracted {output_path} ({os.path.getsize(output_path)} bytes)")
            return True
        else:
            err = proc.stderr.decode("utf-8", errors="replace")
            logger.warning(f"FFmpeg live extraction failed (code {proc.returncode}): {err}")
    except Exception as e:
        logger.warning(f"FFmpeg highlight extraction error: {e}")

    return False


AUDIO_WINDOWS_PER_CHUNK = 3  # ~3.3 s windows per 10 s chunk: the granularity of the old probe


def chunk_window_levels(wav_path: str, windows: int = AUDIO_WINDOWS_PER_CHUNK) -> List[Tuple[float, float, float]]:
    """Loudness of each equal window of a 16-bit PCM WAV chunk.

    Returns (window_start_seconds, mean_db, max_db) per window. mean_db is the RMS
    level and max_db the peak, in dBFS, the same meaning as ffmpeg's volumedetect
    (mean_volume, max_volume). Silence is floored at -90 dB.
    """
    with wave.open(wav_path, "rb") as wf:
        if wf.getsampwidth() != 2:
            return []
        channels = wf.getnchannels()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    samples = array("h")
    samples.frombytes(frames)
    if channels > 1:
        samples = samples[::channels]
    total = len(samples)
    if total == 0 or rate <= 0:
        return []
    seconds_per_window = total / float(rate) / windows
    size = max(1, total // windows)
    out: List[Tuple[float, float, float]] = []
    for i in range(windows):
        end = total if i == windows - 1 else (i + 1) * size
        seg = samples[i * size:end]
        n = len(seg)
        if n == 0:
            continue
        peak = max(abs(min(seg)), abs(max(seg)))
        power = sum(x * x for x in seg) / float(n)
        mean_db = 10.0 * math.log10(power / (32768.0 ** 2)) if power > 0 else -90.0
        max_db = 20.0 * math.log10(peak / 32768.0) if peak > 0 else -90.0
        out.append((i * seconds_per_window, max(mean_db, -90.0), max(max_db, -90.0)))
    return out


def fetch_stream_metadata(stream_url: str) -> Optional[Dict[str, str]]:
    """Title and category of a live URL, as the streamlink plugin for it reports them.

    The plugin fills in its title and category while it resolves the stream, so
    one resolve gives both. None when the platform cannot be read (offline,
    unsupported, or a network error).
    """
    if streamlink is None:
        return None
    try:
        session = streamlink.Streamlink()
        _name, plugin_cls, resolved = session.resolve_url(stream_url)
        plugin = plugin_cls(session, resolved)
        streams = plugin.streams()
        if not streams:
            return None  # offline: streamlink finds nothing to play
        return {"title": plugin.get_title() or "", "category": plugin.get_category() or ""}
    except Exception as e:
        logger.info(f"Stream metadata unavailable for {stream_url}: {e}")
        return None


class ChannelState:
    """Per-channel rolling state for in-stream audio loudness and viral spike detection."""

    def __init__(self, channel: str, platform: str, url: str, embed_url: Optional[str] = None):
        self.channel = channel
        self.platform = platform
        self.url = url
        self.embed_url = embed_url or get_embed_url({"channel": channel, "platform": platform, "url": url})
        self.status = "idle"  # "active", "connecting", "idle", "cooldown", "error"

        # Chat metrics
        self.messages: Deque[Tuple[float, bool, str]] = collections.deque()  # (ts, is_hype, text)
        self.total_messages: int = 0
        self.current_velocity: float = 0.0
        self.current_hype_ratio: float = 0.0
        self.baseline_velocity: float = 0.0

        # In-Stream Audio Loudness metrics
        self.audio_samples: Deque[Tuple[float, float, float]] = collections.deque()  # (ts, mean_db, max_db)
        self.current_mean_volume: float = -35.0
        self.current_max_volume: float = -35.0
        self.baseline_mean_volume: float = -30.0
        self.audio_status: str = "monitoring"  # "monitoring", "analyzing", "peak", "cooldown"

        # Clipping state
        self.last_clip_time: float = 0.0
        self.clip_count: int = 0
        self.last_spike_time: Optional[float] = None
        self.last_spike_reason: Optional[str] = None

        # Live transcript: recent (absolute time, text) lines and the last chunk's text score
        self.transcript: Deque[Tuple[float, str]] = collections.deque(maxlen=200)
        self.last_speech_hits: Dict[str, List[str]] = {}

        # Game gate: None until checked, then True only for a GTA VI stream
        self.game_ok: Optional[bool] = None
        self.game_title: str = ""
        self.game_category: str = ""
        self.game_checked_at: float = 0.0

        # Cached HLS URL
        self.cached_hls_url: Optional[str] = None
        self.hls_url_timestamp: float = 0.0
        self.hls_retry_at: float = 0.0  # no new HLS lookup before this time (offline backoff)

        # Last problem seen on this channel, shown in the status so failures are not silent
        self.last_error: Optional[str] = None
        self.last_error_at: Optional[float] = None
        self.clip_failures: int = 0

    def set_error(self, message: str, now: Optional[float] = None):
        self.last_error = message
        self.last_error_at = now if now is not None else time.time()

    @property
    def audio_level_normalized(self) -> float:
        """Normalized 0.0 - 1.0 audio energy level for frontend visualizer pulse."""
        # Maps -60 dB .. 0 dB to 0.0 .. 1.0
        return max(0.0, min(1.0, (self.current_mean_volume + 60.0) / 60.0))

    # --- Audio In-Stream Loudness Tracking --- #

    def record_audio_sample(self, mean_db: float, max_db: float, timestamp: Optional[float] = None):
        """Records an audio loudness sample from the in-stream audio probe."""
        ts = timestamp if timestamp is not None else time.time()
        self.audio_samples.append((ts, mean_db, max_db))
        self.current_mean_volume = mean_db
        self.current_max_volume = max_db
        self._prune_audio(ts)
        self.update_audio_metrics(ts)

    def _prune_audio(self, now: float, max_age: float = 90.0):
        while self.audio_samples and (now - self.audio_samples[0][0]) > max_age:
            self.audio_samples.popleft()

    def update_audio_metrics(self, now: Optional[float] = None) -> Tuple[float, float, float]:
        """Calculates (current_mean_db, current_max_db, baseline_mean_db)."""
        current_time = now if now is not None else time.time()
        self._prune_audio(current_time)

        if not self.audio_samples:
            return self.current_mean_volume, self.current_max_volume, self.baseline_mean_volume

        # Baseline is the moving average of mean_volume across samples in the window (excluding the latest 4s)
        baseline_samples = [s[1] for s in self.audio_samples if (current_time - s[0]) > 4.0]
        if baseline_samples:
            self.baseline_mean_volume = sum(baseline_samples) / len(baseline_samples)
        else:
            self.baseline_mean_volume = sum(s[1] for s in self.audio_samples) / len(self.audio_samples)

        return self.current_mean_volume, self.current_max_volume, self.baseline_mean_volume

    def check_audio_spike(
        self,
        now: Optional[float] = None,
        cooldown_seconds: float = 60.0,
        peak_threshold_db: float = -8.0,
        min_jump_db: float = 10.0
    ) -> Tuple[bool, Optional[str]]:
        """Detects in-stream excitement: streamer shout, scream, loud reaction, energy burst."""
        current_time = now if now is not None else time.time()
        if (current_time - self.last_clip_time) < cooldown_seconds:
            self.audio_status = "cooldown"
            return False, None

        self.update_audio_metrics(current_time)

        if not self.audio_samples:
            self.audio_status = "monitoring"
            return False, None

        # Decide against the samples BEFORE the latest one. Comparing a reading with a
        # baseline that already contains it made the first reading fire by itself.
        # Too few earlier samples means no verdict yet.
        prior = list(self.audio_samples)[:-1]
        if len(prior) < MIN_AUDIO_BASELINE_SAMPLES:
            self.audio_status = "monitoring"
            return False, None
        self.baseline_mean_volume = sum(s[1] for s in prior) / len(prior)

        jump = self.current_max_volume - self.baseline_mean_volume

        # Condition 1: Streamer shout peak (max_volume >= -8 dB and jumped >= 10 dB above baseline)
        if self.current_max_volume >= peak_threshold_db and jump >= min_jump_db:
            reason = f"stream_audio_shout (peak={self.current_max_volume:.1f}dB, baseline={self.baseline_mean_volume:.1f}dB, jump=+{jump:.1f}dB)"
            self.last_spike_time = current_time
            self.last_spike_reason = reason
            self.audio_status = "peak"
            return True, reason

        # Condition 2: Sustained loudness burst (mean volume jumped >= 8 dB above baseline and >= -14 dB)
        mean_jump = self.current_mean_volume - self.baseline_mean_volume
        if self.current_mean_volume >= -14.0 and mean_jump >= 8.0:
            reason = f"stream_energy_burst (mean={self.current_mean_volume:.1f}dB, baseline={self.baseline_mean_volume:.1f}dB, jump=+{mean_jump:.1f}dB)"
            self.last_spike_time = current_time
            self.last_spike_reason = reason
            self.audio_status = "peak"
            return True, reason

        self.audio_status = "monitoring"
        return False, None

    # --- Chat Metrics Tracking --- #

    def record_message(self, text: str, timestamp: Optional[float] = None) -> bool:
        """Records a new chat message in the rolling window."""
        ts = timestamp if timestamp is not None else time.time()
        is_hype = is_hype_message(text)
        self.messages.append((ts, is_hype, text))
        self.total_messages += 1
        self._prune(ts)
        return is_hype

    def _prune(self, now: float, max_age: float = 60.0):
        while self.messages and (now - self.messages[0][0]) > max_age:
            self.messages.popleft()

    def update_metrics(self, now: Optional[float] = None) -> Tuple[float, float, float]:
        """Calculates (velocity_10s, hype_ratio_10s, baseline_60s)."""
        current_time = now if now is not None else time.time()
        self._prune(current_time)

        recent_10s = [m for m in self.messages if (current_time - m[0]) <= 10.0]
        self.current_velocity = len(recent_10s) / 10.0

        if recent_10s:
            hype_count = sum(1 for m in recent_10s if m[1])
            self.current_hype_ratio = hype_count / len(recent_10s)
        else:
            self.current_hype_ratio = 0.0

        if self.messages:
            oldest_ts = self.messages[0][0]
            span = max(1.0, current_time - oldest_ts)
            self.baseline_velocity = len(self.messages) / span
        else:
            self.baseline_velocity = 0.0

        return self.current_velocity, self.current_hype_ratio, self.baseline_velocity

    def check_spike(
        self,
        now: Optional[float] = None,
        cooldown_seconds: float = 60.0,
        min_velocity: float = 2.0,
        min_hype_ratio: float = 0.30,
        surge_multiplier: float = 2.5
    ) -> Tuple[bool, Optional[str]]:
        """Determines if a viral spike occurred in the channel."""
        current_time = now if now is not None else time.time()
        velocity, hype_ratio, baseline = self.update_metrics(current_time)

        # Respect cooldown
        if (current_time - self.last_clip_time) < cooldown_seconds:
            return False, None

        recent_10s = [m for m in self.messages if (current_time - m[0]) <= 10.0]
        if len(recent_10s) < 3:
            return False, None

        # Check explicit clip commands
        clip_calls = sum(
            1 for m in recent_10s
            if any(k in m[2].lower() for k in ("!clip", "clip it", "clipe", "clipa", "clip this"))
        )
        if clip_calls >= 3:
            reason = f"chat_clip_command (calls={clip_calls}, velocity={velocity:.1f}/s)"
            self.last_spike_time = current_time
            self.last_spike_reason = reason
            return True, reason

        # Check hype emote spike
        if len(recent_10s) >= 4 and hype_ratio >= min_hype_ratio and velocity >= min_velocity:
            reason = f"hype_emote_spike (hype={hype_ratio*100:.0f}%, velocity={velocity:.1f}/s)"
            self.last_spike_time = current_time
            self.last_spike_reason = reason
            return True, reason

        # Check velocity surge
        if len(recent_10s) >= 8 and baseline > 0 and (velocity / max(baseline, 0.4)) >= surge_multiplier:
            reason = f"velocity_surge (velocity={velocity:.1f}/s, baseline={baseline:.1f}/s)"
            self.last_spike_time = current_time
            self.last_spike_reason = reason
            return True, reason

        return False, None


def _release_asr_models():
    """Drops the live transcript's ASR models. Runs on its own thread (see LiveMonitor.stop)."""
    try:
        from transcribe_backends import release_models
        release_models()
    except Exception as e:
        logger.debug(f"release_models after live transcript failed: {e}")


class LiveMonitor:
    """Unified ultra-optimized live stream monitoring engine.

    Monitors in-stream audio excitement and stream events across multiple channels without continuous stream recording.
    Detects viral hype spikes and extracts 30-60s highlight clips on-demand.
    """

    def __init__(
        self,
        channels: Optional[List[str]] = None,
        on_clip_ready: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        highlight_seconds: int = 120,
        cooldown_seconds: int = 60,
        detection_mode: str = "hybrid",  # 'hybrid' (stream audio + chat), 'stream_audio', or 'chat'
        audio_peak_threshold_db: float = -8.0,
        audio_jump_threshold_db: float = 10.0,
        spike_velocity_threshold: float = 2.0,
        spike_hype_ratio_threshold: float = 0.30,
        output_dir: Optional[str] = None,
        mock_extraction: bool = False,
        pre_roll_seconds: int = 15,
        transcriber: Optional[Callable[[str], Dict[str, Any]]] = None,
    ):
        self.on_clip_ready = on_clip_ready
        self.highlight_seconds = max(15, min(120, highlight_seconds))
        self.cooldown_seconds = max(10, cooldown_seconds)
        self.detection_mode = detection_mode
        self.audio_peak_threshold_db = audio_peak_threshold_db
        self.audio_jump_threshold_db = audio_jump_threshold_db
        self.spike_velocity_threshold = spike_velocity_threshold
        self.spike_hype_ratio_threshold = spike_hype_ratio_threshold
        self.mock_extraction = mock_extraction
        # The spike is detected seconds after it happened: start the clip this long before the trigger.
        self.pre_roll_seconds = max(0, min(60, int(pre_roll_seconds)))

        # Live transcript (audio chunks -> ASR -> local score). Never in mock mode, so tests
        # do not start ffmpeg or load a model.
        self.transcriber = transcriber
        self.transcribe_enabled = os.environ.get("LIVE_TRANSCRIPT", "1") != "0" and not mock_extraction
        # Only GTA VI streams are monitored (LIVE_GAME_FILTER=0 turns the gate off, for tests).
        # Audio levels come from the audio chunks of each live (one ffmpeg per live).
        # The old per-live probe spawned a second ffmpeg every few seconds and timed out under load.
        self.audio_enabled = not mock_extraction
        self.audio_probe_loop = False
        self.game_filter = os.environ.get("LIVE_GAME_FILTER", "1") != "0"
        # The Gemini key the dashboard sent with the start request (memory only).
        self.gemini_api_key: Optional[str] = None
        self._transcript_tasks: Dict[str, asyncio.Task] = {}
        self._chunk_procs: Dict[str, subprocess.Popen] = {}
        self._asr_used = False

        self.output_dir = output_dir or os.environ.get("LIVE_HIGHLIGHT_DIR") or os.path.join(os.getcwd(), "uploads", "live_highlights")
        os.makedirs(self.output_dir, exist_ok=True)

        self._channels: Dict[str, ChannelState] = {}
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None
        self._audio_loop_task: Optional[asyncio.Task] = None
        self._ws_client = None
        self._tcp_writer = None
        self._stop_event = asyncio.Event()
        self.captured_clips: List[Dict[str, Any]] = []

        if channels:
            for ch in channels:
                self.add_channel(ch)

    @property
    def is_running(self) -> bool:
        return self._running

    def add_channel(self, channel_or_url: str) -> Optional[Dict[str, str]]:
        """Adds a channel or live URL to monitoring. Dynamically joins if running."""
        parsed = parse_channel(channel_or_url)
        if not parsed:
            logger.warning(f"Invalid channel format: {channel_or_url}")
            return None

        channel_id = parsed["id"]
        if channel_id not in self._channels:
            state = ChannelState(
                channel=parsed["channel"],
                platform=parsed["platform"],
                url=parsed["url"],
                embed_url=parsed.get("embed_url")
            )
            state.status = "active" if self._running else "idle"
            self._channels[channel_id] = state
            logger.info(f"Added live channel: {parsed['channel']} ({parsed['platform']})")

            # If IRC is currently running, dynamically join the channel
            if self._running and parsed["platform"] == "twitch":
                self._send_irc_join(parsed["channel"])

        return parsed

    def remove_channel(self, channel_or_url: str) -> bool:
        """Removes a channel from monitoring."""
        parsed = parse_channel(channel_or_url)
        channel_id = parsed["id"] if parsed else channel_or_url

        if channel_id in self._channels:
            state = self._channels.pop(channel_id)
            if self._running and state.platform == "twitch":
                self._send_irc_part(state.channel)
            logger.info(f"Removed live channel: {state.channel}")
            return True
        return False

    def get_status(self) -> Dict[str, Any]:
        """Returns the monitor's current status, channels, and captured clips."""
        now = time.time()
        channels_info = []
        for cid, state in self._channels.items():
            vel, hype, base = state.update_metrics(now)
            channels_info.append({
                "id": cid,
                "name": state.channel,
                "channel": state.channel,
                "platform": state.platform,
                "url": state.url,
                "embed_url": state.embed_url,
                "status": state.status,
                "audio_status": state.audio_status,
                "audio_level": round(state.audio_level_normalized, 2),
                "audio_peak_db": round(state.current_max_volume, 1),
                "audio_baseline_db": round(state.baseline_mean_volume, 1),
                "messages_count": state.total_messages,
                "velocity": round(vel, 2),
                "hype_ratio": round(hype, 2),
                "baseline_velocity": round(base, 2),
                "last_spike": state.last_spike_time,
                "last_spike_reason": state.last_spike_reason,
                "clips_captured": state.clip_count,
                "clip_failures": state.clip_failures,
                "last_error": state.last_error,
                "last_error_at": state.last_error_at,
                "game_ok": state.game_ok,
                "game_title": state.game_title,
                "game_category": state.game_category,
            })

        return {
            "running": self._running,
            "channels": channels_info,
            "captured_clips": list(self.captured_clips),
            "transcribing": bool(self._transcript_tasks) and self._running,
            "detection_mode": self.detection_mode,
            "highlight_seconds": self.highlight_seconds,
            "cooldown_seconds": self.cooldown_seconds,
        }

    def _resolve_channel_state(self, channel_or_url: str) -> Optional[ChannelState]:
        """Resolves ChannelState by channel_id, channel name, or raw URL."""
        if channel_or_url in self._channels:
            return self._channels[channel_or_url]

        target_lower = str(channel_or_url).strip().lower()
        for s in self._channels.values():
            if s.channel.lower() == target_lower or s.url.lower() == target_lower:
                return s

        parsed = parse_channel(channel_or_url)
        if parsed and parsed["id"] in self._channels:
            return self._channels[parsed["id"]]

        if parsed:
            added = self.add_channel(channel_or_url)
            if added:
                return self._channels.get(added["id"])

        return None

    def record_stream_audio_sample(
        self,
        channel_or_url: str,
        mean_db: float,
        max_db: float,
        timestamp: Optional[float] = None
    ):
        """Allows feeding audio loudness metrics directly (for in-stream audio analysis & tests)."""
        state = self._resolve_channel_state(channel_or_url)
        if state:
            state.record_audio_sample(mean_db, max_db, timestamp)
            spike, reason = state.check_audio_spike(
                now=timestamp,
                cooldown_seconds=self.cooldown_seconds,
                peak_threshold_db=self.audio_peak_threshold_db,
                min_jump_db=self.audio_jump_threshold_db
            )
            if spike:
                self._handle_viral_spike(state, reason or "stream_audio_shout", now=timestamp)

    def record_chat_message(self, channel_or_url: str, text: str, timestamp: Optional[float] = None):
        """Allows injecting messages (e.g. from tests or alternative chat feeds)."""
        state = self._resolve_channel_state(channel_or_url)
        if state:
            state.record_message(text, timestamp)
            if self.detection_mode in ("chat", "hybrid"):
                spike, reason = state.check_spike(
                    now=timestamp,
                    cooldown_seconds=self.cooldown_seconds,
                    min_velocity=self.spike_velocity_threshold,
                    min_hype_ratio=self.spike_hype_ratio_threshold
                )
                if spike:
                    self._handle_viral_spike(state, reason or "viral_spike", now=timestamp)

    def _record_failure(self, state: ChannelState, trigger_time: float, reason: str, error: str):
        """Keeps a failed attempt visible in the clip list (no job, no file)."""
        state.clip_failures += 1
        self.captured_clips.append({
            "channel": state.channel,
            "platform": state.platform,
            "url": state.url,
            "timestamp": trigger_time,
            "duration": self.highlight_seconds,
            "reason": reason,
            "status": "failed",
            "error": error,
            "source_type": "live_stream_audio_highlight",
        })
        logger.warning(f"Live clip failed for {state.channel}: {error}")

    def _handle_viral_spike(self, state: ChannelState, reason: str, now: Optional[float] = None, **kwargs):
        """Triggers highlight extraction and invokes on_clip_ready.

        Only GTA VI streams are clipped. Nothing is limited by count: Gemini's
        predicted_score decides which moments become clips.
        """
        trigger_time = now or time.time()
        if state.game_ok is False:
            logger.info(f"Not a GTA VI stream, no clip for {state.channel}: {reason}")
            return None
        state.last_clip_time = trigger_time
        # The Gemini key only goes to the callback; it must never be stored in captured_clips
        # (the status endpoint returns that list to the browser).
        api_key = kwargs.pop("gemini_api_key", None) or self.gemini_api_key
        # "cooldown" only while this extraction is in flight; reset in `finally` below.
        state.status = "cooldown"

        async def _async_spike():
            try:
                clip_path = await self._extract_highlight(state, trigger_time)
                if not clip_path:
                    self._record_failure(state, trigger_time, reason,
                                         state.last_error or "clip extraction failed")
                    return None

                state.clip_count += 1
                meta = {
                    "clip_path": clip_path,
                    "channel": state.channel,
                    "platform": state.platform,
                    "url": state.url,
                    "embed_url": state.embed_url,
                    "timestamp": trigger_time,
                    "duration": self.highlight_seconds,
                    "reason": reason,
                    "velocity": state.current_velocity,
                    "hype_ratio": state.current_hype_ratio,
                    "audio_peak_db": state.current_max_volume,
                    "source_type": "live_stream_audio_highlight"
                }
                if kwargs:
                    meta.update(kwargs)
                self.captured_clips.append(meta)
                logger.info(f"Captured live clip #{state.clip_count} for {state.channel}: {clip_path}")

                if self.on_clip_ready:
                    callback_meta = dict(meta)
                    if api_key:
                        callback_meta["gemini_api_key"] = api_key
                    try:
                        if inspect.iscoroutinefunction(self.on_clip_ready):
                            await self.on_clip_ready(clip_path, callback_meta)
                        else:
                            res = self.on_clip_ready(clip_path, callback_meta)
                            if inspect.isawaitable(res):
                                await res
                    except Exception as e:
                        logger.error(f"Error executing on_clip_ready callback: {e}")
                return clip_path
            except Exception as e:
                state.set_error(f"clip error: {e}")
                logger.error(f"Failed extracting highlight for {state.channel}: {e}")
                self._record_failure(state, trigger_time, reason, str(e))
                return None
            finally:
                # Before this fix the channel stayed "cooldown" forever and was never sampled again.
                state.status = "active" if self._running else "idle"

        # Dispatch async task if event loop is running, else run in loop
        try:
            loop = asyncio.get_running_loop()
            return loop.create_task(_async_spike())
        except RuntimeError:
            return asyncio.run(_async_spike())

    async def _extract_highlight(self, state: ChannelState, trigger_time: float) -> Optional[str]:
        """Resolves HLS playlist and extracts 30-60s highlight without full recording."""
        """Resolves the HLS playlist and records a highlight window that STARTS before the trigger.

        The spike was detected seconds after it happened, so the window opens
        ``pre_roll_seconds`` earlier and runs for ``highlight_seconds``. On failure
        ``state.last_error`` says why.
        """
        filename = f"live_{state.channel}_{int(trigger_time)}.mp4"
        output_path = os.path.join(self.output_dir, filename)

        if self.mock_extraction or os.environ.get("MOCK_STREAM_EXTRACTION") == "1":
            # For testing: create a simulated MP4 placeholder file
            with open(output_path, "wb") as f:
                f.write(b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41")
            state.last_error = None
            return output_path

        hls_url = await asyncio.to_thread(resolve_streamlink_hls, state.url)
        if not hls_url:
            state.set_error("could not resolve a live HLS URL (offline, unsupported or Streamlink error)")
            logger.warning(f"Could not resolve live HLS URL for {state.url}")
            return None

        start_index = -max(1, round(self.pre_roll_seconds / HLS_SEGMENT_SECONDS_ESTIMATE))
        success = await asyncio.to_thread(
            extract_highlight_clip,
            hls_url=hls_url,
            output_path=output_path,
            duration=self.highlight_seconds,
            live_start_index=start_index
        )
        if not success:
            state.set_error("FFmpeg could not capture the live stream (see the server log)")
            return None
        state.last_error = None
        return output_path

    def trigger_clip(self, channel_or_url: str, reason: str = "manual", **kwargs):
        """Manually forces an immediate clip extraction for a channel, bypassing cooldown."""
        state = self._resolve_channel_state(channel_or_url)
        if not state:
            logger.warning(f"trigger_clip: could not resolve channel {channel_or_url}")
            return _AwaitableWrapper(result=None)

        state.last_spike_reason = reason
        task = self._handle_viral_spike(state, reason, **kwargs)

        async def _coro():
            if asyncio.iscoroutine(task) or isinstance(task, asyncio.Task):
                return await task
            return task

        return _AwaitableWrapper(_coro, result=task)

    def _send_irc_join(self, channel: str):
        join_cmd = f"JOIN #{channel.lower()}\r\n"
        if self._ws_client:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._ws_client.send(join_cmd))
            except Exception:
                pass
        elif self._tcp_writer:
            try:
                self._tcp_writer.write(join_cmd.encode("utf-8"))
            except Exception:
                pass

    def _send_irc_part(self, channel: str):
        part_cmd = f"PART #{channel.lower()}\r\n"
        if self._ws_client:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._ws_client.send(part_cmd))
            except Exception:
                pass
        elif self._tcp_writer:
            try:
                self._tcp_writer.write(part_cmd.encode("utf-8"))
            except Exception:
                pass

    def _cancel_tasks(self) -> List[asyncio.Task]:
        """Cancels the background loops, so a stop/start can never leave two copies running."""
        tasks = [t for t in (self._loop_task, self._audio_loop_task, *self._transcript_tasks.values())
                 if t is not None and not t.done()]
        for t in tasks:
            t.cancel()
        for proc in list(self._chunk_procs.values()):
            if proc.poll() is None:
                proc.kill()
        self._chunk_procs.clear()
        self._transcript_tasks.clear()
        self._loop_task = None
        self._audio_loop_task = None
        return tasks

    def start(self):
        """Starts monitoring. Can be called synchronously or awaited with `await monitor.start()`."""
        if self._running:
            return _AwaitableWrapper(result=self)

        self._running = True
        self._stop_event.clear()
        self._cancel_tasks()  # replaced below by fresh loops
        for s in self._channels.values():
            s.status = "active"

        async def _start_async():
            if self._loop_task is None or self._loop_task.done():
                self._loop_task = asyncio.create_task(self._run_monitoring_loop())
            if self._audio_loop_task is None or self._audio_loop_task.done():
                self._audio_loop_task = asyncio.create_task(self._run_audio_sampling_loop())
            return self

        try:
            loop = asyncio.get_running_loop()
            self._loop_task = loop.create_task(self._run_monitoring_loop())
            self._audio_loop_task = loop.create_task(self._run_audio_sampling_loop())
            self._start_transcripts()
        except RuntimeError:
            pass

        return _AwaitableWrapper(_start_async, result=self)

    def stop(self):
        """Gracefully stops monitoring. Can be called synchronously or awaited."""
        if not self._running:
            return _AwaitableWrapper(result=self)

        self._running = False
        self._stop_event.set()
        cancelled = self._cancel_tasks()
        if self._asr_used:
            # Give the GPU back: the live transcript held the ASR model while running.
            # Off the event loop: release_models() waits for any chunk still decoding,
            # and a blocked loop would stall every request, this stop included.
            threading.Thread(target=_release_asr_models, name="live-asr-release", daemon=True).start()
            self._asr_used = False
        for s in self._channels.values():
            s.status = "idle"

        async def _stop_async():
            for t in cancelled:
                try:
                    await t
                except asyncio.CancelledError:
                    pass
            self._close_connections()
            return self

        self._close_connections()
        return _AwaitableWrapper(_stop_async, result=self)

    def _close_connections(self):
        if self._tcp_writer:
            try:
                self._tcp_writer.close()
            except Exception:
                pass
            self._tcp_writer = None
        self._ws_client = None

    # --- Live transcript: audio chunks -> ASR -> local score ----------------- #

    def _default_transcriber(self, path: str) -> Dict[str, Any]:
        """The clip pipeline's own ASR: GPU gated, language auto-detected, same output shape."""
        from transcribe_backends import transcribe_media
        return transcribe_media(path)

    def _start_transcripts(self):
        """Starts one audio session per monitored channel (needs a running loop).

        A session is one long-running ffmpeg writing audio-only chunks. The chunks give
        the live's loudness (always) and its speech (when ASR is on).
        """
        if not self.audio_enabled:
            return
        if self.transcriber is None and self.transcribe_enabled:
            import importlib.util
            if importlib.util.find_spec("faster_whisper") is None:
                logger.warning("faster-whisper is not installed: speech is not transcribed; audio levels still run.")
                self.transcribe_enabled = False
            else:
                self.transcriber = self._default_transcriber
        self._asr_used = self.transcribe_enabled and self.transcriber is not None
        loop = asyncio.get_running_loop()
        for cid, state in self._channels.items():
            task = self._transcript_tasks.get(cid)
            if task is None or task.done():
                self._transcript_tasks[cid] = loop.create_task(self._run_transcript(cid, state))

    async def _check_game(self, state: ChannelState) -> bool:
        """True while the stream is GTA VI. Its title and category are read again
        every couple of minutes (every minute while it is not GTA VI)."""
        if not self.game_filter:
            state.game_ok = True
            return True
        now = time.time()
        interval = GAME_RECHECK_OK if state.game_ok else GAME_RECHECK_NOT
        if state.game_checked_at and now - state.game_checked_at < interval:
            return bool(state.game_ok)
        meta = await asyncio.to_thread(fetch_stream_metadata, state.url)
        state.game_checked_at = now
        if meta is None:
            state.game_ok = False
            state.set_error("offline, or its title and category could not be read; not monitored", now)
            return False
        state.game_title = meta["title"]
        state.game_category = meta["category"]
        state.game_ok = is_gta6_stream(meta["title"], meta["category"])
        if state.game_ok:
            if state.last_error and state.last_error.startswith("not GTA VI"):
                state.last_error = None
        else:
            state.set_error("not GTA VI (title or category); not monitored", now)
        return state.game_ok

    async def _run_transcript(self, cid: str, state: ChannelState):
        """Keeps the live's audio-only chunks flowing while it is GTA VI.

        Offline or non-GTA VI streams are polled, not recorded. Each finished chunk
        is transcribed and checked for keywords.
        """
        seg_dir = os.path.join(self.output_dir, "chunks", cid)
        try:
            while self._running:
                if not await self._check_game(state):
                    await asyncio.sleep(30.0)
                    continue
                hls = await asyncio.to_thread(resolve_streamlink_hls, state.url, ("audio_only", "worst", "best"))
                if not hls:
                    state.set_error("live transcript: no audio stream yet (retrying)")
                    await asyncio.sleep(30.0)
                    continue
                await self._transcript_session(cid, state, hls, seg_dir)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            state.set_error(f"live transcript stopped: {e}")
            logger.warning(f"Live transcript stopped for {state.channel}: {e}")
        finally:
            shutil.rmtree(seg_dir, ignore_errors=True)

    async def _transcript_session(self, cid: str, state: ChannelState, hls: str, seg_dir: str):
        """One ffmpeg run: audio-only chunks written to seg_dir until the stream ends,
        the game changes, or the monitor stops."""
        shutil.rmtree(seg_dir, ignore_errors=True)
        os.makedirs(seg_dir, exist_ok=True)
        started = time.time()
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", hls, "-vn", "-ac", "1", "-ar", "16000",
            "-f", "segment", "-segment_time", str(TRANSCRIPT_CHUNK_SECONDS), "-reset_timestamps", "1",
            os.path.join(seg_dir, "chunk_%05d.wav"),
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._chunk_procs[cid] = proc
        try:
            while self._running and proc.poll() is None:
                await asyncio.sleep(1.0)
                if not await self._check_game(state):
                    break
                names = sorted(n for n in os.listdir(seg_dir) if n.startswith("chunk_") and n.endswith(".wav"))
                for name in names[:-1]:  # the newest file may still be being written
                    index = int(name[6:11])
                    await self._consume_chunk(
                        cid, state, os.path.join(seg_dir, name), started + index * TRANSCRIPT_CHUNK_SECONDS)
        finally:
            if proc.poll() is None:
                proc.kill()
            self._chunk_procs.pop(cid, None)

    async def _consume_chunk(self, cid: str, state: ChannelState, path: str, t0: float):
        """One finished audio chunk: its loudness per window and, when ASR is on, its speech.

        The loudness comes from the chunk itself, so no second stream is opened per live.
        """
        try:
            levels = await asyncio.to_thread(chunk_window_levels, path)
            window = TRANSCRIPT_CHUNK_SECONDS / AUDIO_WINDOWS_PER_CHUNK
            for start, mean_db, max_db in levels:
                self.record_stream_audio_sample(cid, mean_db, max_db, timestamp=t0 + start + window)
            if self.transcriber is not None and self.transcribe_enabled:
                transcript = await asyncio.to_thread(self.transcriber, path)
                self._on_transcript_chunk(cid, state, (transcript or {}).get("segments") or [], t0)
        except Exception as e:
            state.set_error(f"live audio chunk failed: {e}")
            logger.warning(f"Live audio chunk failed for {state.channel}: {e}")
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def _on_transcript_chunk(self, cid: str, state: ChannelState, segments: List[Dict[str, Any]], t0: float):
        """Keeps the text, checks it for keywords, and sends candidates to the pipeline.

        Nothing here rates the moment. A candidate starts a clip job; Gemini's
        predicted_score decides whether the clip is kept.
        """
        for seg in segments:
            text = str(seg.get("text", "")).strip()
            if text:
                state.transcript.append((t0 + float(seg.get("start", 0.0)), text))

        hits = detect_speech_hits(segments)
        state.last_speech_hits = hits
        if not is_candidate(hits):
            return
        now = t0 + TRANSCRIPT_CHUNK_SECONDS
        if now - state.last_clip_time < self.cooldown_seconds:
            return
        described = "; ".join(f"{cat}: {', '.join(words[:3])}" for cat, words in hits.items())
        self._handle_viral_spike(state, f"live_speech ({described})", now=now)

    async def _sample_channel_audio_task(self, cid: str, state: ChannelState):
        """Samples audio for a single channel asynchronously."""
        try:
            if not self._running or state.status == "cooldown":
                return

            if not await self._check_game(state):
                return

            now = time.time()
            if not state.cached_hls_url or (now - state.hls_url_timestamp) > 90.0:
                if now < state.hls_retry_at:
                    return  # offline backoff: do not ask Streamlink every tick
                hls = await asyncio.to_thread(resolve_streamlink_hls, state.url, ("audio_only", "worst", "best"))
                state.cached_hls_url = hls
                state.hls_url_timestamp = now
                if not hls:
                    state.hls_retry_at = now + OFFLINE_RETRY_SECONDS
                    state.set_error(f"not live, or no HLS available (retrying in {int(OFFLINE_RETRY_SECONDS)} s)", now)
                    return
                state.hls_retry_at = 0.0

            if state.cached_hls_url and self._running:
                mean_vol, max_vol = await asyncio.to_thread(
                    sample_stream_audio_loudness,
                    state.cached_hls_url,
                    3,
                    8
                )
                if mean_vol is not None and max_vol is not None:
                    state.last_error = None
                    self.record_stream_audio_sample(cid, mean_vol, max_vol, now=time.time())
                else:
                    state.set_error("audio probe failed (will retry)")
        except Exception as e:
            logger.debug(f"Audio sample failed for {state.channel}: {e}")

    async def _run_audio_sampling_loop(self):
        """Background loop sampling in-stream audio loudness concurrently across all active channels."""
        logger.info("In-stream audio sampling loop started.")
        while self._running:
            try:
                if self.audio_probe_loop and self.detection_mode in ("stream_audio", "hybrid") and not self.mock_extraction:
                    # Concurrently sample all active channels via asyncio.gather so 4 streams sample in 3s (not 12s)
                    active_items = list(self._channels.items())
                    if active_items:
                        tasks = [
                            self._sample_channel_audio_task(cid, state)
                            for cid, state in active_items
                            if state.status != "cooldown"
                        ]
                        if tasks:
                            await asyncio.gather(*tasks, return_exceptions=True)

                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=6.0)
                    break
                except asyncio.TimeoutError:
                    pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Audio sampling loop tick error: {e}")
                await asyncio.sleep(2.0)

        logger.info("In-stream audio sampling loop stopped.")

    async def _run_monitoring_loop(self):
        """Main background loop handling Twitch IRC and multi-platform channels."""
        logger.info("LiveMonitor background loop started.")
        while self._running:
            try:
                # Twitch Channels
                twitch_channels = [
                    s.channel for s in self._channels.values() if s.platform == "twitch"
                ]

                if twitch_channels and self.detection_mode in ("chat", "hybrid"):
                    # Prefer WebSocket (port 443), fall back to raw IRC socket (port 6667)
                    connected = False
                    if websockets is not None:
                        try:
                            await self._listen_twitch_websocket(twitch_channels)
                            connected = True
                        except Exception as e:
                            logger.warning(f"Twitch WebSocket error, trying raw IRC: {e}")

                    if not connected and self._running:
                        try:
                            await self._listen_twitch_irc_socket(twitch_channels)
                        except Exception as e:
                            logger.warning(f"Twitch raw IRC socket error: {e}")

                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=3.0)
                    break
                except asyncio.TimeoutError:
                    pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"LiveMonitor loop error: {e}")
                await asyncio.sleep(2.0)

        logger.info("LiveMonitor background loop stopped.")

    async def _listen_twitch_websocket(self, channels: List[str]):
        """Connects to Twitch IRC via WebSocket on port 443."""
        async with websockets.connect(TWITCH_WS_URL, close_timeout=2) as ws:
            self._ws_client = ws
            await ws.send(f"PASS oauth:{TWITCH_ANON_NICK}\r\n")
            await ws.send(f"NICK {TWITCH_ANON_NICK}\r\n")

            for ch in channels:
                await ws.send(f"JOIN #{ch.lower()}\r\n")
            logger.info(f"Connected to Twitch IRC WebSocket for channels: {channels}")

            async for message in ws:
                if not self._running:
                    break
                lines = message.split("\r\n")
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("PING"):
                        pong_resp = line.replace("PING", "PONG", 1) + "\r\n"
                        await ws.send(pong_resp)
                        continue
                    self._process_raw_irc_line(line)

    async def _listen_twitch_irc_socket(self, channels: List[str]):
        """Fallback raw TCP socket listener for Twitch IRC."""
        reader, writer = await asyncio.open_connection(TWITCH_IRC_HOST, TWITCH_IRC_PORT)
        self._tcp_writer = writer

        writer.write(f"PASS oauth:{TWITCH_ANON_NICK}\r\n".encode("utf-8"))
        writer.write(f"NICK {TWITCH_ANON_NICK}\r\n".encode("utf-8"))
        for ch in channels:
            writer.write(f"JOIN #{ch.lower()}\r\n".encode("utf-8"))
        await writer.drain()

        logger.info(f"Connected to Twitch raw IRC TCP for channels: {channels}")

        while self._running:
            line_bytes = await reader.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line.startswith("PING"):
                pong_resp = line.replace("PING", "PONG", 1) + "\r\n"
                writer.write(pong_resp.encode("utf-8"))
                await writer.drain()
                continue
            self._process_raw_irc_line(line)

    def _process_raw_irc_line(self, line: str):
        """Parses PRIVMSG line and updates corresponding channel's state."""
        clean = line
        if clean.startswith("@"):
            parts = clean.split(" ", 1)
            if len(parts) > 1:
                clean = parts[1]

        m = _IRC_MSG_RE.match(clean)
        if not m:
            return

        channel_name = m.group("channel").lower()
        content = m.group("content")
        cid = f"twitch_{channel_name}"

        state = self._channels.get(cid)
        if state:
            now = time.time()
            state.record_message(content, now)
            if self.detection_mode in ("chat", "hybrid"):
                spike, reason = state.check_spike(
                    now=now,
                    cooldown_seconds=self.cooldown_seconds,
                    min_velocity=self.spike_velocity_threshold,
                    min_hype_ratio=self.spike_hype_ratio_threshold
                )
                if spike:
                    self._handle_viral_spike(state, reason or "viral_spike", now=now)
