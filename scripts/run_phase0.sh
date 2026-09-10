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

if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing run directory: $RUN_DIR" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
python "$REPO_ROOT/scripts/capture_environment.py" > "$RUN_DIR/environment.json"
git -C "$REPO_ROOT" diff --binary > "$RUN_DIR/working-tree.patch"

{
  printf 'python %q --device %q --output-dir %q' \
    "$REPO_ROOT/experiments/phase0/tiny_decoder.py" "$DEVICE" "$RUN_DIR"
  printf ' %q' "$@"
  printf '\n'
} > "$RUN_DIR/command.sh"

python "$REPO_ROOT/experiments/phase0/tiny_decoder.py" \
  --device "$DEVICE" \
  --output-dir "$RUN_DIR" \
  "$@" 2>&1 | tee "$RUN_DIR/stdout.log"

echo "Artifacts: $RUN_DIR"

