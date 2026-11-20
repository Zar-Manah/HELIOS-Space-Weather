#!/usr/bin/env python3
"""Optional localhost bridge for the HELIOS v3.4 cockpit.

The cockpit (`Zar-Manah_HELIOS.html`) is self-contained and runs a physics-only
standalone mode with live NOAA SWPC telemetry.  This bridge is optional: it
exposes the real v3.4 five-stage pipeline (`predict.py` + the 14 frozen
`.joblib` artefacts) on localhost so the cockpit can switch to "Full ML"
without any web framework, build step or external dependency.

    python3 server.py --port 8787

Endpoints
    GET /api/health
    GET /api/forecast?speed=800&width=60&lat=0&lon=0[&...]
    GET /

The server binds 127.0.0.1 by default.  It refuses non-loopback addresses:
the bridge is a local convenience, not a public service.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BASE = Path(__file__).resolve().parent
PREDICT = BASE / "predict.py"
COCKPIT = BASE / "Zar-Manah_HELIOS.html"
MODELS = BASE / "models" / "v3"
VERSION = "3.4.0"
MAX_CONCURRENT = threading.Semaphore(2)
CACHE_TTL_S = 120.0
CACHE_MAX = 32
_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_LOCK = threading.Lock()
LAUNCH_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$")

# Query key -> predict.py flag.  Only these keys are accepted.
FLAGS = {
    "speed": "--cme_speed",
    "width": "--cme_width",
    "lat": "--cme_lat",
    "lon": "--cme_lon",
    "accel": "--cme_accel",
    "v20r": "--cme_v20r",
    "dv2": "--cme_dv2",
    "mass": "--cme_mass",
    "energy": "--cme_energy",
    "sharp_r": "--cme_sharp_r",
    "ar": "--cme_ar",
    "cpa": "--cme_cpa",
}
BOUNDS = {
    "speed": (1.0, 5000.0),
    "width": (0.0, 360.0),
    "lat": (-90.0, 90.0),
    "lon": (-180.0, 180.0),
    "accel": (-10000.0, 10000.0),
    "v20r": (1.0, 5000.0),
    "dv2": (-10000.0, 10000.0),
    "mass": (0.0, 1e30),
    "energy": (0.0, 1e40),
    "sharp_r": (0.0, 1e30),
    "ar": (0.0, 20000.0),
    "cpa": (0.0, 360.0),
}


def _finite(value: float, key: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    lo, hi = BOUNDS[key]
    if value < lo or value > hi:
        raise ValueError(f"{key} out of range [{lo}, {hi}]")
    return value


def build_args(query: dict[str, list[str]]) -> list[str]:
    """Translate the whitelisted query string into predict.py arguments."""
    def one(key: str) -> str | None:
        values = query.get(key)
        if not values:
            return None
        value = values[-1].strip()
        return value or None

    speed = one("speed")
    if speed is None:
        raise ValueError("speed is required (launch speed in km/s)")
    args: list[str] = ["--live", "--cme", "--json"]
    for key, flag in FLAGS.items():
        raw = one(key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"{key} is not a number") from exc
        args.extend([flag, str(_finite(value, key))])
    halo = (one("halo") or "").lower()
    if halo in ("1", "true", "yes", "on"):
        args.append("--halo")
    launch = one("launch_time")
    if launch is not None:
        if not LAUNCH_RE.match(launch):
            raise ValueError("launch_time must look like 2026-11-19T00:00")
        args.extend(["--launch_time", launch])
    return args


def run_predict(args: list[str]) -> dict:
    """Run predict.py with one thread and return its JSON report."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    with MAX_CONCURRENT:
        try:
            proc = subprocess.run(
                [sys.executable, str(PREDICT), *args],
                cwd=str(BASE),
                capture_output=True,
                text=True,
                timeout=240,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("predict.py exceeded the 240 s timeout") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-1600:]
        raise RuntimeError(f"predict.py exited {proc.returncode}: {tail}")
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        tail = (proc.stdout or "").strip()[-800:]
        raise RuntimeError(f"predict.py returned invalid JSON: {exc}; tail={tail!r}") from exc
    report["bridge"] = {
        "mode": "full-ml",
        "served_by": "server.py",
        "version": VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    return report


def cached_forecast(key: str, args: list[str]) -> tuple[dict, bool]:
    """Return (report, cached).  A 20 s predict.py run is reused for 2 min."""
    now = time.monotonic()
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit and (now - hit[0]) <= CACHE_TTL_S:
            return hit[1], True
    report = run_predict(args)
    with _CACHE_LOCK:
        if len(_CACHE) >= CACHE_MAX:
            oldest = min(_CACHE, key=lambda k: _CACHE[k][0])
            _CACHE.pop(oldest, None)
        _CACHE[key] = (time.monotonic(), report)
    return report, False


class Handler(BaseHTTPRequestHandler):
    server_version = "HELIOSBridge/3.4"

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Cache-Control", "no-store")

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=float).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _text(self, code: int, text: str) -> None:
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _html(self) -> None:
        if not COCKPIT.exists():
            self._text(404, "Zar-Manah_HELIOS.html is not next to server.py\n")
            return
        body = COCKPIT.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802 (http.server API)
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            models = sorted(MODELS.glob("*.joblib"))
            self._json(200, {
                "ok": True,
                "version": VERSION,
                "models": len(models),
                "python": sys.version.split()[0],
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "full-ml",
            })
            return
        if parsed.path == "/api/forecast":
            query = parse_qs(parsed.query, keep_blank_values=False)
            try:
                args = build_args(query)
                key = "&".join(args)
                report, was_cached = cached_forecast(key, args)
            except ValueError as exc:
                self._json(400, {"error": str(exc), "version": VERSION})
                return
            except Exception as exc:  # noqa: BLE001 (surface the real failure)
                self._json(502, {"error": str(exc), "version": VERSION})
                return
            report.setdefault("bridge", {})["cached"] = was_cached
            self._json(200, report)
            return
        if parsed.path in ("/", "/Zar-Manah_HELIOS.html"):
            self._html()
            return
        if parsed.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        self._json(404, {"error": "not found", "version": VERSION})

    def log_message(self, fmt: str, *args) -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
        print(f"[{stamp}] {self.address_string()} {fmt % args}", file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="HELIOS v3.4 localhost bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        parser.error("the bridge only binds loopback addresses (127.0.0.1)")
    if not PREDICT.exists():
        parser.error(f"predict.py not found next to server.py: {PREDICT}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"HELIOS v3.4 bridge on http://{args.host}:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
