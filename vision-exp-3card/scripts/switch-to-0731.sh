#!/bin/bash
# Switch GPUs 0-2 back to the 0731 production setup.
# Stops the Vision-Exp container if running, then runs the UNMODIFIED 0731 production
# launcher (the same one your desktop shortcut uses).
set -euo pipefail
if docker ps --format '{{.Names}}' | grep -qx dsv4-vision-3card; then
  echo "stopping Vision-Exp (dsv4-vision-3card)..."
  docker stop -t 60 dsv4-vision-3card >/dev/null
fi
exec "$HOME/CMP-170HX-PROJECT/vllm-dsv4/phase5-launch-dspark-production.sh" "$@"
