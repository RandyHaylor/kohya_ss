#!/usr/bin/env bash
# Launch the Kohya GUI with the CUDA caching allocator configured to use
# expandable segments. This reduces VRAM fragmentation, which caused an
# out-of-memory crash mid-training (step 201) despite ~1.5 GB reserved-but-
# unallocated. The training subprocess inherits this env var because the GUI
# launches it with os.environ.copy() (kohya_gui/common_gui.py).
#
# Usage: ./gui-uv-with-expandable-cuda-segments.sh [any gui-uv.sh args]
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec ./gui-uv.sh "$@"
