"""Lifetime AI cost: the total of every answer the ledger has recorded.

ai_provider.record_usage appends one JSON line per answer to the ledger file. This module
reads it back for the dashboard's header total. It reads only the bytes added since its
last read, so a long ledger costs little per request. A ledger that shrinks (someone
cleared the runtime data) starts the count again.
"""
import json
import os
import threading
from typing import Any, Dict, Optional

import ai_provider

_lock = threading.Lock()
_state: Dict[str, Any] = {"path": None, "offset": 0, "totals": None}


def _empty() -> Dict[str, Any]:
    return {"total_usd": 0.0, "calls": 0, "cloud_calls": 0, "local_calls": 0,
            "input_tokens": 0, "output_tokens": 0, "since": None}


def _fold(totals: Dict[str, Any], row: Dict[str, Any]) -> None:
    totals["total_usd"] += float(row.get("cost_usd") or 0.0)
    totals["calls"] += 1
    totals["local_calls" if row.get("local") else "cloud_calls"] += 1
    totals["input_tokens"] += int(row.get("input_tokens") or 0)
    totals["output_tokens"] += int(row.get("output_tokens") or 0)
    ts = row.get("ts")
    if ts is not None and (totals["since"] is None or ts < totals["since"]):
        totals["since"] = ts


def summary(path: Optional[str] = None) -> Dict[str, Any]:
    """Totals over the whole ledger. ``total_usd`` sums the cost OpenRouter reported for
    each answer; local (Ollama) answers count in calls and tokens, at no cost."""
    path = path or ai_provider.usage_log_path()
    with _lock:
        if _state["path"] != path or _state["totals"] is None:
            _state.update(path=path, offset=0, totals=_empty())
        try:
            size = os.path.getsize(path)
        except OSError:
            _state.update(offset=0, totals=_empty())
            return dict(_state["totals"], total_usd=0.0)
        if size < _state["offset"]:
            _state.update(offset=0, totals=_empty())
        if size > _state["offset"]:
            with open(path, "rb") as f:
                f.seek(_state["offset"])
                chunk = f.read(size - _state["offset"])
            end = chunk.rfind(b"\n") + 1  # only whole lines; a line still being written waits
            for raw in chunk[:end].splitlines():
                try:
                    row = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    continue
                if isinstance(row, dict):
                    _fold(_state["totals"], row)
            _state["offset"] += end
        totals = dict(_state["totals"])
    totals["total_usd"] = round(totals["total_usd"], 6)
    return totals
