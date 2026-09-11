#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <cpu|mps|cuda> <run-id> [experiment arguments...]" >&2
  exit 2
fi

DEVICE=$1
RUN_ID=$2
shift 2

case "$DEVICE" in
  cpu|mps|cuda) ;;
  *)
    echo "Unsupported device: $DEVICE" >&2
    exit 2
    ;;
esac

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Run id must contain only letters, digits, dot, underscore, or hyphen." >&2
  exit 2
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ARTIFACT_BASE=${ARTIFACT_ROOT:-"$REPO_ROOT/artifacts"}
RUN_DIR="$ARTIFACT_BASE/phase0/tiny-decoder-fixed-v1/$RUN_ID"

PYTHON_BIN=""
for candidate in "${CONDA_PREFIX:-}/bin/python" "$REPO_ROOT/.venv/bin/python" "$(command -v python 2>/dev/null || true)" "$(command -v python3 2>/dev/null || true)"; do
  if [[ -n "$candidate" && -x "$candidate" ]] && "$candidate" -c 'import torch' >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    break
  fi
done
if [[ -z "$PYTHON_BIN" ]]; then
  echo "A Python interpreter with PyTorch is required. Install the matching PyTorch build in .venv, then rerun." >&2
  exit 127
fi

if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing run directory: $RUN_DIR" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
"$PYTHON_BIN" "$REPO_ROOT/scripts/capture_environment.py" > "$RUN_DIR/environment.json"
git -C "$REPO_ROOT" diff --binary > "$RUN_DIR/working-tree.patch"

{
  printf '%q %q --device %q --output-dir %q' \
    "$PYTHON_BIN" "$REPO_ROOT/experiments/phase0/tiny_decoder.py" "$DEVICE" "$RUN_DIR"
  printf ' %q' "$@"
  printf '\n'
} > "$RUN_DIR/command.sh"

"$PYTHON_BIN" "$REPO_ROOT/experiments/phase0/tiny_decoder.py" \
  --device "$DEVICE" \
  --output-dir "$RUN_DIR" \
  "$@" 2>&1 | tee "$RUN_DIR/stdout.log"

echo "Artifacts: $RUN_DIR"
