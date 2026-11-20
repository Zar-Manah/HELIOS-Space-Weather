#!/bin/bash
# Zar-Manah HELIOS v3.4 — double-click launcher (macOS).
#
# Starts the optional localhost bridge (server.py -> predict.py + the 14 frozen
# .joblib artefacts) and opens the cockpit in the default browser.  If the
# bridge cannot start, the standalone HTML is opened directly: it still runs
# the exact v3.4 physics with live NOAA SWPC telemetry, just without the ML.
#
# Stop the bridge later with:  ./Zar-Manah_HELIOS.command --stop

set -u
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR" || exit 1

PORT=8787
LOG="${TMPDIR:-/tmp}/helios_bridge.log"
URL="http://127.0.0.1:${PORT}/"

if [ "${1:-}" = "--stop" ]; then
  pkill -f "server.py --port ${PORT}" 2>/dev/null && echo "HELIOS bridge stopped." || echo "HELIOS bridge was not running."
  exit 0
fi

healthy() {
  python3 - "$URL" <<'PY' >/dev/null 2>&1
import json, sys, urllib.request
with urllib.request.urlopen(sys.argv[1] + "api/health", timeout=1.5) as r:
    sys.exit(0 if json.load(r).get("ok") else 1)
PY
}

if healthy; then
  echo "HELIOS bridge already running on port ${PORT}."
elif command -v python3 >/dev/null 2>&1; then
  echo "Starting HELIOS bridge on ${URL} ..."
  nohup python3 server.py --port "$PORT" >>"$LOG" 2>&1 &
  disown 2>/dev/null || true
  for _ in $(seq 1 40); do
    healthy && break
    sleep 0.25
  done
  healthy || echo "Bridge did not answer; falling back to the standalone HTML. Log: $LOG"
else
  echo "python3 not found; opening the standalone HTML (physics + live L1 only)."
fi

if healthy; then
  open "$URL"
else
  open "$DIR/Zar-Manah_HELIOS.html"
fi
