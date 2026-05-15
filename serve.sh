#!/usr/bin/env bash
# Serve the SMIRK web demo with stdlib http.server.
#
# Before first run, ensure web/assets/face_landmarker.task points at the existing
# MediaPipe model:
#   mkdir -p web/assets
#   ln -sf "$(pwd)/assets/face_landmarker.task" web/assets/face_landmarker.task
#
# Then open http://localhost:8080/ in a browser (Chrome/Firefox/Edge).
#
# IMPORTANT: open it via localhost or 127.0.0.1. The webcam (getUserMedia) API is
# secure-context-only, so http://<LAN-IP>:8080 will fail with "Cannot read
# properties of undefined (reading 'getUserMedia')". For remote access you'd need
# HTTPS (e.g. via a tunnel like cloudflared/ngrok).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec python3 -m http.server --directory "$HERE/web" 8080
