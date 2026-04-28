#!/usr/bin/env python3
"""Launch FastAPI + Next.js dev server together, then open the browser."""
import subprocess
import sys
import signal
import threading
import time
import webbrowser
import shutil
import os
from pathlib import Path

ROOT = Path(__file__).parent
FRONTEND = ROOT / "frontend"

API_PORT  = 8000
NEXT_PORT = 3000
UI_URL    = f"http://localhost:{NEXT_PORT}"

def resolve_api_python() -> str:
    """Resolve the Python interpreter for FastAPI.

    Priority: ENSAGENT_API_PYTHON env var > ensagent conda env > current interpreter.
    """
    override = os.environ.get("ENSAGENT_API_PYTHON", "").strip()
    if override and Path(override).exists():
        return override
    # Try ensagent conda env first — it has fastapi, litellm, and all pipeline deps
    conda_root = os.environ.get("CONDA_ROOT") or os.environ.get("MAMBA_ROOT_PREFIX") or ""
    for base in [conda_root, Path(sys.executable).parents[2] if sys.executable else ""]:
        if not base:
            continue
        candidate = Path(base) / "envs" / "ensagent" / ("python.exe" if sys.platform == "win32" else "bin/python")
        if candidate.exists():
            return str(candidate)
    if sys.executable:
        return sys.executable
    found = shutil.which("python")
    if found:
        return found
    return "python"


def resolve_powershell_executable() -> str:
    return shutil.which("pwsh") or shutil.which("powershell") or "powershell"


API_PYTHON = resolve_api_python()
POWERSHELL = resolve_powershell_executable()


def _wait_and_open(url: str, timeout: int = 30) -> None:
    """Poll until the Next.js dev server is ready, then open the browser."""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            webbrowser.open(url)
            print(f"Browser opened → {url}")
            return
        except Exception:
            time.sleep(1)
    print(f"[warn] Timed out waiting for {url}; open it manually.")


def _kill_port(port: int) -> None:
    """Kill any process listening on *port* (Windows-safe)."""
    try:
        result = subprocess.run(
            [POWERSHELL, "-NoProfile", "-Command",
             f"Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue"
             f" | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique"
             f" | ForEach-Object {{ Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }}"],
            capture_output=True, timeout=10,
        )
    except Exception:
        pass  # best-effort


def main():
    procs = []

    # Clean up any stale server processes on both ports
    print("Clearing ports…")
    _kill_port(API_PORT)
    _kill_port(NEXT_PORT)
    time.sleep(1)

    # FastAPI — use ensagent env which has litellm and all pipeline deps
    print(f"Using Python: {API_PYTHON}")
    api_proc = subprocess.Popen(
        [API_PYTHON, "-m", "uvicorn", "api.main:app",
         "--host", "0.0.0.0", "--port", str(API_PORT), "--reload"],
        cwd=ROOT,
    )
    procs.append(api_proc)
    print(f"FastAPI  → http://localhost:{API_PORT}")

    # Give API a moment to start, then health-check
    time.sleep(2)
    if api_proc.poll() is not None:
        print(f"\n[ERROR] FastAPI process exited immediately (code {api_proc.returncode}).")
        print(f"  Python used: {API_PYTHON}")
        print(f"  Make sure the 'ensagent' conda env is set up and has fastapi/uvicorn installed.")
        print(f"  Or set ENSAGENT_API_PYTHON to the correct Python path.")
        for p in procs:
            p.terminate()
        sys.exit(1)

    import urllib.request
    try:
        urllib.request.urlopen(f"http://localhost:{API_PORT}/health", timeout=5)
        print("FastAPI  ✓ health check passed")
    except Exception:
        print(f"[WARN] FastAPI health check failed — API may still be starting up.")

    # Next.js
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    next_proc = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=FRONTEND,
    )
    procs.append(next_proc)
    print(f"Next.js  → {UI_URL}")

    # Open browser once Next.js is ready (non-blocking)
    threading.Thread(target=_wait_and_open, args=(UI_URL,), daemon=True).start()

    def _shutdown(sig, frame):
        print("\nShutting down…")
        for p in procs:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Keep running until a process exits
    for p in procs:
        p.wait()


if __name__ == "__main__":
    main()
