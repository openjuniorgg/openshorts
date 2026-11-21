import os
import json
import re
import subprocess
import tempfile
import time
from typing import List
from pydantic import BaseModel

import ai_provider
import vision_frames
from edit_builder import build_filter_string
from ffmpeg_utils import video_encode_args, QUALITY, METADATA_SCRUB


def _int_env(name, default):
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


# The edit passes look at this many frames of the clip, spread evenly.
EDITOR_FRAMES = _int_env("EDITOR_FRAMES", 24)
EDITOR_FRAME_WIDTH = _int_env("EDITOR_FRAME_WIDTH", 512)


class EditDecision(BaseModel):
    type: str
    start: float
    end: float
    strength: float = 0.0
    reason: str = ""


class EditPlan(BaseModel):
    edits: List[EditDecision]


def _clip_frames(video_path, duration):
    """``(interval, [(t, jpeg), ...])`` for the whole clip, for the edit passes."""
    with tempfile.TemporaryDirectory(prefix="editor_frames_") as tmp:
        return vision_frames.sample(video_path, 0.0, float(duration), tmp,
                                    count=EDITOR_FRAMES, width=EDITOR_FRAME_WIDTH)


class VideoEditor:
    def __init__(self, ai=None):
        # The model the request chose in Settings, else the server's configured one.
        if ai is None and ai_provider.configured():
            ai = ai_provider.resolve()
        self.config = ai if (ai is not None and ai.ready) else None
        self.model_name = self.config.model if self.config else ""

    def get_ffmpeg_filter(self, video_path, duration, fps=30, width=None, height=None, transcript=None, has_captions=False):
        """Asks the model for an edit decision list, then builds the FFmpeg filter
        deterministically (edit_builder) so syntax and zoom limits are always safe."""
        if width is None or height is None:
            # Keep prompt usable even if caller didn't pass dimensions.
            width, height = 1080, 1920

        transcript_text = json.dumps(transcript) if transcript else "Not available."

        caption_rule = ""
        if has_captions:
            caption_rule = (
                "\n        CRITICAL: This video already has burned-in captions and/or a hook text. "
                "Any zoom would crop or shift them off screen. Do NOT use zoom_in, punch_in or zoom_pulse — "
                "only color_pop, bw_moment, flash and vignette are allowed.\n"
            )

        prompt = f"""
        You are a viral short-form video editor (TikTok / Reels / Shorts). You receive still frames of the clip, in order, with their times, and its transcript.
        Decide WHERE a small set of tasteful, high-impact effects belongs. You do NOT write FFmpeg —
        you return an edit decision list, and a deterministic renderer applies it safely.

        Video duration: {duration:.1f} seconds. Resolution: {width}x{height}.

        Available effect types:
        - "zoom_in": slow push-in across the segment. Builds tension or focus on a key statement. strength 0.06-0.15, segment 1.5-6s.
        - "punch_in": instant tighter framing for the whole segment. Emphasis, punchlines, "listen to this" moments. strength 0.06-0.15, segment 0.8-5s.
        - "zoom_pulse": quick in-and-out pulse peaking mid-segment. Beat drops, single impactful words. strength 0.04-0.10, segment 0.4-1.2s.
        - "color_pop": richer saturation/contrast. Energy, excitement, product/visual highlights. strength 0.2-1.0.
        - "bw_moment": black & white. Drama, serious quotes, flashbacks.
        - "flash": brief bright flash at segment start. Hard-cut energy, reveals. Use at most 2.
        - "vignette": darkened edges. Focus, intimacy, storytelling moments.

        Rules:
        1. Match effects to the CONTENT: place them on the exact words or moments they emphasize, using the transcript timestamps.
        2. Less is more: 2-6 edits per 30 seconds of video. A random effect is worse than no effect.
        3. Never overlap two zoom-type edits (zoom_in/punch_in/zoom_pulse).
        4. Calm, serious delivery => few or no motion effects. High-energy content => more punch.
        {caption_rule}
        TRANSCRIPT (with timestamps, the context of what is being said):
        {transcript_text}

        Return JSON only:
        {{"edits": [{{"type": "punch_in", "start": 3.2, "end": 5.4, "strength": 0.1, "reason": "punchline"}}]}}
        If no effects genuinely improve the video, return {{"edits": []}}.
        """

        if not self.config:
            print("⚠️ No AI provider is configured — keeping the clip untouched.")
            return None
        interval, frames = _clip_frames(video_path, duration)
        prompt += f"\nFRAMES (image order; seconds from the start of this clip): {vision_frames.labels(frames)}\n"
        print("🤖 Asking the model for an edit decision list...")
        answer, _cost = ai_provider.generate_json(
            prompt, EditPlan, images=[jpeg for _, jpeg in frames], config=self.config, temperature=0.2)
        raw_edits = answer.get("edits") or []

        filter_string, applied = build_filter_string(
            raw_edits, duration=duration, fps=fps, width=width, height=height,
            has_captions=has_captions,
        )
        if not filter_string:
            print("ℹ️ The model suggested no (valid) edits — keeping the clip untouched.")
            return {"filter_string": None, "edits": []}
        print(f"🎯 Applying {len(applied)} edits: " + ", ".join(f"{e['type']}@{e['start']:.1f}s" for e in applied))
        return {"filter_string": filter_string, "edits": applied}

    def get_effects_config(self, video_path, duration, fps=30, width=None, height=None, transcript=None):
        """Asks the model for a structured EffectsConfig JSON for Remotion rendering."""
        if width is None or height is None:
            width, height = 1080, 1920

        transcript_text = json.dumps(transcript) if transcript else "Not available."

        prompt = f"""
        You are an expert video editor analyzing a video and its transcript to generate dynamic visual effects for a Remotion-based renderer.

        Video Duration: {duration} seconds.
        Video FPS: {fps}
        Video Resolution: {width}x{height}

        TRANSCRIPT (Context of what is being said):
        {transcript_text}

        Your task is to produce a structured JSON describing time-based effect segments that cover the FULL video duration.

        Each segment has these fields:
        - "startSec" (number): Start time in seconds.
        - "endSec" (number): End time in seconds.
        - "zoom" (number): Zoom level. 1.0 = no zoom, max 1.5. Use subtle values like 1.05-1.2 for most cases.
        - "zoomCenterX" (number): Horizontal focus point for zoom, 0.0 (left) to 1.0 (right). 0.5 = center.
        - "zoomCenterY" (number): Vertical focus point for zoom, 0.0 (top) to 1.0 (bottom). 0.5 = center.
        - "brightness" (number): Brightness multiplier. 1.0 = normal. Range 0.8-1.2.
        - "contrast" (number): Contrast multiplier. 1.0 = normal. Range 0.8-1.3.
        - "saturate" (number): Saturation multiplier. 1.0 = normal. Range 0.8-1.3.

        Instructions:
        1. ANALYZE the video content and transcript to understand mood, pacing, and key moments.
        2. Apply CONTEXTUAL effects aligned with speech and action:
           - Use slow, subtle zooms toward the speaker's face during speaking moments.
           - Emphasize key moments, punchlines, or dramatic beats with slightly stronger zoom or contrast.
           - Keep transitions smooth — avoid jarring jumps between segments.
           - If nothing significant is happening, keep values at defaults (zoom 1.0, all multipliers 1.0).
        3. Segments MUST cover the entire video duration from 0 to {duration} seconds with no gaps.
        4. Prefer fewer, longer segments with gradual changes over many rapid short segments.
        5. Output ONLY valid JSON, no explanations.

        Output format:
        {{
            "segments": [
                {{
                    "startSec": 0,
                    "endSec": 3.5,
                    "zoom": 1.0,
                    "zoomCenterX": 0.5,
                    "zoomCenterY": 0.5,
                    "brightness": 1.0,
                    "contrast": 1.0,
                    "saturate": 1.0
                }}
            ]
        }}
        """

        if not self.config:
            print("⚠️ No AI provider is configured — no effects config.")
            return None
        interval, frames = _clip_frames(video_path, duration)
        prompt += f"\nFRAMES (image order; seconds from the start of this clip): {vision_frames.labels(frames)}\n"
        print("🤖 Asking the model for a Remotion effects config...")
        text, _cost = ai_provider.generate_text(
            prompt, images=[jpeg for _, jpeg in frames], config=self.config, temperature=0.4)
        print(f"🔍 DEBUG: Model raw response:\n{text}")
        try:
            return ai_provider.parse_json_response_text(text)
        except ValueError:
            print(f"❌ Failed to parse effects config JSON: {text}")
            return None

    @staticmethod
    def _split_filter_chain(filter_string: str) -> list[str]:
        """Split a -vf filter chain on commas, respecting single-quoted substrings."""
        parts: list[str] = []
        start = 0
        in_quote = False
        for i, ch in enumerate(filter_string):
            if ch == "'":
                in_quote = not in_quote
            elif ch == "," and not in_quote:
                parts.append(filter_string[start:i])
                start = i + 1
        parts.append(filter_string[start:])
        return parts

    @classmethod
    def _enforce_zoompan_output_size(cls, filter_string: str, width: int, height: int) -> str:
        """Force any zoompan filter to output the same geometry as the input clip."""
        parts = cls._split_filter_chain(filter_string)
        out_parts: list[str] = []
        for part in parts:
            if "zoompan=" in part:
                # Force s=WxH inside zoompan options (digitsxdigits only).
                if re.search(r":s=\d+x\d+", part):
                    part = re.sub(r":s=\d+x\d+", f":s={width}x{height}", part)
                else:
                    part = f"{part}:s={width}x{height}"
            out_parts.append(part)
        return ",".join(out_parts)

    @staticmethod
    def _sanitize_filter_string(filter_string: str) -> str:
        """
        Best-effort sanitizer for model-generated FFmpeg expressions.
        Converts comparison operators (t<3, on>=75, etc.) into FFmpeg expr functions (lt(), gte(), ...),
        which are far more reliably parsed across FFmpeg builds.
        """
        s = filter_string

        # Order matters: handle >= / <= before > / <
        patterns: list[tuple[re.Pattern[str], str]] = [
            (re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_]\w*)\s*>=\s*(-?\d+(?:\.\d+)?)"), r"gte(\1,\2)"),
            (re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_]\w*)\s*<=\s*(-?\d+(?:\.\d+)?)"), r"lte(\1,\2)"),
            (re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_]\w*)\s*>\s*(-?\d+(?:\.\d+)?)"), r"gt(\1,\2)"),
            (re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_]\w*)\s*<\s*(-?\d+(?:\.\d+)?)"), r"lt(\1,\2)"),
        ]
        for pat, repl in patterns:
            s = pat.sub(repl, s)

        return s

    # The filters the edit builder emits (edit_builder.build_filter_string),
    # plus the ones this module adds itself. A filter string is run only if
    # every filter in it is one of these: none of them can open a file, and
    # anything else (a filter that reads a file, an instance name, a label, a
    # second chain) means the string was not built here. The self-repair path
    # is the one that returns free text from the model.
    _ALLOWED_FILTERS = frozenset({"eq", "zoompan", "vignette", "hue", "setsar",
                                  "format", "scale", "fps", "null"})

    @classmethod
    def _touches_files(cls, filter_string) -> bool:
        """True unless the string is a plain chain of the allowed filters."""
        s = str(filter_string or "")
        if not s.strip() or any(c in s for c in "@\\[];\n\r\x00"):
            return True
        for part in cls._split_filter_chain(s):
            name = part.strip().split("=", 1)[0].strip()
            if name not in cls._ALLOWED_FILTERS:
                return True
        return False

    @staticmethod
    def _ffmpeg_env():
        """The environment for an ffmpeg that runs a generated filter: the
        process's own, without anything named like a credential."""
        env = {k: v for k, v in os.environ.items()
               if not re.search(r"KEY|SECRET|TOKEN|PASSWORD|DATABASE|COOKIE|PROXY|DSN", k, re.I)}
        env["LANG"] = "C.UTF-8"
        env["LC_ALL"] = "C.UTF-8"
        return env

    @staticmethod
    def _test_filter(input_path, filter_string, env):
        """Dry-run the filter on the first 2 seconds (no output written).

        Catches broken model-generated filter syntax in seconds instead of
        failing after a full-length encode. Returns (ok, stderr_tail)."""
        cmd = [
            'ffmpeg', '-v', 'error', '-t', '2',
            '-i', input_path,
            '-vf', filter_string,
            '-f', 'null', '-',
        ]
        try:
            result = subprocess.run(
                [a.encode('utf-8') if isinstance(a, str) else a for a in cmd],
                env=env, capture_output=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            return False, "filter dry-run timed out after 120s"
        if result.returncode == 0:
            return True, ""
        stderr_text = (result.stderr or b"").decode(errors="replace")
        return False, stderr_text[-1500:]

    def _repair_filter(self, filter_string, error_text, width, height):
        """One self-repair round-trip: show the model the FFmpeg error and ask for
        a corrected filter string. Returns the new string or None."""
        if not self.config:
            return None
        prompt = f"""
        The following FFmpeg -vf filter string you generated fails to run.

        FILTER:
        {filter_string}

        FFMPEG ERROR:
        {error_text}

        Fix the filter. Keep the same creative intent, obey the same rules as before:
        - exact output resolution {width}x{height} (zoompan must set s={width}x{height}),
        - no bare comparison operators (use between/lt/lte/gt/gte),
        - expression values in single quotes.
        Output JSON only: {{"filter_string": "..."}}
        """
        try:
            text, _cost = ai_provider.generate_text(prompt, config=self.config, temperature=0.2)
            repaired = ai_provider.parse_json_response_text(text).get("filter_string")
            return repaired if isinstance(repaired, str) and repaired.strip() else None
        except Exception as e:
            print(f"⚠️ Filter self-repair failed: {e}")
            return None

    def apply_edits(self, input_path, output_path, filter_data):
        """Executes FFmpeg with the generated filter."""
        
        if (filter_data and filter_data.get("filter_string")
                and self._touches_files(filter_data["filter_string"])):
            print("⚠️ The filter names a file or a command filter. Not run; copying original.")
            filter_data = None
        if not filter_data or not filter_data.get("filter_string"):
            print("⚠️ No filter string found. Copying original.")
            try:
                subprocess.run(['ffmpeg', '-y', '-i', input_path, '-c', 'copy',
                                *METADATA_SCRUB, output_path], timeout=1800)
            except subprocess.TimeoutExpired:
                raise RuntimeError("FFmpeg copy timed out after 1800s.")
            return

        filter_string = filter_data["filter_string"]
        
        # Get input dimensions so we can enforce geometry (avoid broken aspect ratios).
        try:
            probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', input_path]
            res_out = subprocess.check_output(probe_cmd, env={**os.environ, "LANG": "C.UTF-8"}, timeout=60).decode().strip()
            w, h = map(int, res_out.split('x'))
        except Exception as e:
            print(f"⚠️ Could not probe resolution: {e}")
            w, h = None, None

        # Sanitize common expression pitfalls (e.g., t<3 / on>=75) before executing FFmpeg.
        sanitized = self._sanitize_filter_string(filter_string)
        if sanitized != filter_string:
            print("🧼 Sanitized AI Filter (converted comparisons to lt/lte/gt/gte functions)")
            print(f"🧼 Before: {filter_string}")
            print(f"🧼 After:  {sanitized}")
            filter_string = sanitized

        # Enforce zoompan output size to preserve aspect ratio / resolution.
        if w and h:
            enforced = self._enforce_zoompan_output_size(filter_string, w, h)
            if enforced != filter_string:
                print(f"📐 Enforced zoompan output size to {w}x{h}")
                filter_string = enforced

            # Ensure square pixels (avoid weird display stretching in some players).
            if "setsar=" not in filter_string:
                filter_string = f"{filter_string},setsar=1"

        # Use explicit environment with UTF-8 to avoid ascii errors in subprocess
        # On some minimal docker images, we need to ensure we use a UTF-8 locale
        # (C.UTF-8 is the safe one there). No credentials: ffmpeg needs none.
        env = self._ffmpeg_env()

        # Dry-run the filter on 2 seconds before committing to a full encode;
        # on failure, give the model one self-repair attempt with the real error.
        ok, error_text = self._test_filter(input_path, filter_string, env)
        if not ok:
            print(f"⚠️ AI filter failed dry-run: {error_text}")
            repaired = self._repair_filter(filter_string, error_text, w or 1080, h or 1920)
            if repaired and self._touches_files(repaired):
                print("⚠️ The repaired filter names a file or a command filter. Dropped.")
                repaired = None
            if repaired:
                repaired = self._sanitize_filter_string(repaired)
                if w and h:
                    repaired = self._enforce_zoompan_output_size(repaired, w, h)
                    if "setsar=" not in repaired:
                        repaired = f"{repaired},setsar=1"
                ok, error_text = self._test_filter(input_path, repaired, env)
                if ok:
                    print("🔧 Self-repaired AI filter passed dry-run.")
                    filter_string = repaired
            if not ok:
                raise RuntimeError(f"AI filter failed validation even after self-repair: {error_text}")

        print(f"🎬 Executing AI Filter: {filter_string}")

        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-vf', filter_string,
            *video_encode_args(QUALITY),
            '-c:a', 'copy',
            *METADATA_SCRUB,
            '-movflags', '+faststart',
            output_path
        ]
        
        try:
            # We must encode arguments if filesystem is ascii but we have unicode chars
            # But subprocess in Python 3 handles unicode args by encoding them with os.fsencode().
            # If sys.getfilesystemencoding() is ascii, this fails.
            # We can't change fs encoding at runtime easily.
            # Workaround: pass bytes directly? subprocess allows bytes in args.
            
            # Convert command elements to bytes assuming utf-8 if they are strings
            cmd_bytes = []
            for arg in cmd:
                if isinstance(arg, str):
                    cmd_bytes.append(arg.encode('utf-8'))
                else:
                    cmd_bytes.append(arg)
            
            subprocess.run(cmd_bytes, check=True, env=env, timeout=1800)
        except subprocess.TimeoutExpired:
            print("❌ FFmpeg filter timed out after 1800s.")
            raise RuntimeError("FFmpeg filter timed out after 1800s.")
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg failed: {e}")
            raise e

if __name__ == "__main__":
    pass