"""Live transcript chunks: keyword hits for the candidate gate.

The live monitor transcribes short audio chunks with the clip pipeline's own
ASR (transcribe_backends). This module reads those chunks and reports which
keyword categories were said. It never rates a moment: Gemini does that, later,
in the clip job.
"""
from __future__ import annotations

from typing import Any, Dict, List

from live_keywords import find_hits, is_candidate


def chunk_text(segments: List[Dict[str, Any]]) -> str:
    """The text of one transcript chunk, segments joined in order."""
    return " ".join(str(s.get("text", "")).strip() for s in segments or [] if str(s.get("text", "")).strip())


def detect_speech_hits(segments: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Keyword hits in one chunk, by category. Empty when nothing matched."""
    return find_hits(chunk_text(segments))


def chunk_is_candidate(segments: List[Dict[str, Any]]) -> bool:
    return is_candidate(detect_speech_hits(segments))
