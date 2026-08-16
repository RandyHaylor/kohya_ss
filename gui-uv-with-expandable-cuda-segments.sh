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

# Keep the sd-scripts submodule on our Anima branch so runs use our code (CFG blend, ER-SDE, etc.).
# Run this launcher from the superproject's anima-cfg-ersde-tooling branch: that branch pins this
# submodule commit, so gui-uv.sh's `git submodule update` checks out our code rather than upstream.
git -C "$SCRIPT_DIR/sd-scripts" checkout anima-cfg-ersde-tooling

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec ./gui-uv.sh "$@"
