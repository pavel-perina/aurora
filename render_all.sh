#!/bin/sh
# Render all aurora presets as 4K 16-bit PNG (plus JPEG previews).
# Usage: ./render_all.sh [size] [extra aurora.py args...]
#   e.g. ./render_all.sh 7680x4320 --density 1.2
set -e
cd "$(dirname "$0")"
SIZE="${1:-3840x2160}"
[ $# -gt 0 ] && shift

for scene in vista bronze ember orchid glacier; do
    uv run python aurora.py --scene "$scene" --size "$SIZE" --ss 2 --preview "$@"
done
