"""
Streaming subprocess helpers for tool execution.

Provides optional line-by-line callbacks for live UI progress and logs while
keeping compatibility with plain blocking execution.
"""

from __future__ import annotations

import re
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping


ProgressCallback = Callable[[Dict[str, Any]], None]
CancelCheck = Callable[[], bool]

_PERCENT_RE = re.compile(r"(?P<pct>\d{1,3}(?:\.\d+)?)\s*%")
_FRACTION_RE = re.compile(r"\b(?P<num>\d{1,5})\s*/\s*(?P<den>\d{1,5})\b")


def _safe_emit(callback: ProgressCallback | None, payload: Dict[str, Any]) -> None:
    if callback is None:
        return
    try:
        callback(payload)
    except Exception:
        # UI callback failures should not crash pipeline tools.
        return


def _extract_progress(line: str) -> float | None:
    text = str(line or "")
    m = _PERCENT_RE.search(text)
    if m:
        try:
            pct = float(m.group("pct"))
            if pct < 0:
                return 0.0
            if pct > 100:
                return 1.0
            return pct / 100.0
        except Exception:
            pass

    m = _FRACTION_RE.search(text)
    if m:
        try:
            num = float(m.group("num"))
            den = float(m.group("den"))
            if den <= 0:
                return None
            ratio = num / den
            if ratio < 0:
                return 0.0
            if ratio > 1:
                return 1.0
            return ratio
        except Exception:
            return None

    return None


def run_subprocess_streaming(
    *,
    cmd: Iterable[str],
    cwd: str | Path | None = None,
    tool: str,
    stage: str,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
    tail_lines: int = 200,
    env: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    """
    Run a subprocess and optionally stream logs/progress through callback events.
    """
    safe_cmd = [str(part) for part in cmd]
    _safe_emit(
        progress_callback,
        {
            "kind": "tool_start",
            "tool": tool,
            "stage": stage,
            "message": f"Starting: {' '.join(safe_cmd)}",
            "level": "info",
            "progress": 0.0,
        },
    )

    if cancel_check is not None and cancel_check():
        _safe_emit(
            progress_callback,
            {
                "kind": "tool_interrupt",
                "tool": tool,
                "stage": stage,
                "message": "Interrupted before subprocess start.",
                "level": "warning",
            },
        )
        return {"returncode": 130, "interrupted": True, "log_line_count": 0, "stdout_tail": []}

    process = subprocess.Popen(
        safe_cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=dict(env) if env is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    tail = deque(maxlen=max(10, int(tail_lines)))
    line_count = 0
    interrupted = False
    last_progress = 0.0

    assert process.stdout is not None
    while True:
        if cancel_check is not None and cancel_check():
            interrupted = True
            try:
                process.terminate()
            except Exception:
                pass
            break

        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                break
            time.sleep(0.02)
            continue

        text = line.rstrip("\r\n")
        line_count += 1
        tail.append(text)
        _safe_emit(
            progress_callback,
            {
                "kind": "tool_log",
                "tool": tool,
                "stage": stage,
                "message": text,
                "level": "info",
            },
        )

        parsed_progress = _extract_progress(text)
        if parsed_progress is not None and parsed_progress >= last_progress:
            last_progress = parsed_progress
            _safe_emit(
                progress_callback,
                {
                    "kind": "tool_progress",
                    "tool": tool,
                    "stage": stage,
                    "message": text,
                    "level": "info",
                    "progress": float(parsed_progress),
                },
            )

    if interrupted:
        try:
            process.wait(timeout=3)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
            process.wait(timeout=3)

    returncode = int(process.wait())
    if interrupted:
        _safe_emit(
            progress_callback,
            {
                "kind": "tool_interrupt",
                "tool": tool,
                "stage": stage,
                "message": "Tool execution interrupted.",
                "level": "warning",
            },
        )
    elif returncode == 0:
        _safe_emit(
            progress_callback,
            {
                "kind": "tool_end",
                "tool": tool,
                "stage": stage,
                "message": "Tool completed successfully.",
                "level": "success",
                "progress": 1.0,
            },
        )
    else:
        _safe_emit(
            progress_callback,
            {
                "kind": "tool_error",
                "tool": tool,
                "stage": stage,
                "message": f"Tool failed with exit code {returncode}.",
                "level": "error",
            },
        )

    return {
        "returncode": returncode,
        "interrupted": interrupted,
        "log_line_count": line_count,
        "stdout_tail": list(tail),
    }
