#!/usr/bin/env bash
# Serve Laguna S 2.1 Chadrock ROCmFP4 StrixKVSpine V4 with its validated profile.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-laguna-strix-vulkan}"
BIN="${BIN:-$BUILD_DIR/bin/llama-server}"
MODEL="${MODEL:-${1:-}}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
ALIAS="${ALIAS:-laguna-s21-rocmfp4-strixkvspine-v4}"
DEVICE="${DEVICE:-Vulkan0}"
CTX_SIZE="${CTX_SIZE:-262144}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
THREADS="${THREADS:-16}"
THREADS_BATCH="${THREADS_BATCH:-16}"

if [[ -z "$MODEL" ]]; then
    echo "usage: MODEL=/path/to/laguna-v4.gguf $0" >&2
    echo "   or: $0 /path/to/laguna-v4.gguf" >&2
    exit 2
fi
if [[ ! -r "$MODEL" ]]; then
    echo "model is not readable: $MODEL" >&2
    exit 2
fi
if [[ ! -x "$BIN" ]]; then
    echo "llama-server not found: $BIN" >&2
    echo "build it with scripts/build-laguna-strix-vulkan.sh" >&2
    exit 2
fi

if [[ "${VERIFY_SHA256:-0}" == "1" ]]; then
    EXPECTED_SHA256="ea1d854a72c47ec8e72c16ea91b8ff3cd5e1620b834df175f683c86f27dc26d6"
    ACTUAL_SHA256="$(sha256sum "$MODEL" | awk '{print $1}')"
    if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
        echo "SHA-256 mismatch for $MODEL" >&2
        echo "expected: $EXPECTED_SHA256" >&2
        echo "actual:   $ACTUAL_SHA256" >&2
        exit 3
    fi
fi

server_args=(
    --model "$MODEL" \
    --alias "$ALIAS" \
    --host "$HOST" \
    --port "$PORT" \
    --jinja \
    --ctx-size "$CTX_SIZE" \
    --parallel 1 \
    --n-gpu-layers 999 \
    --device "$DEVICE" \
    --split-mode row \
    --flash-attn on \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --batch-size "$BATCH_SIZE" \
    --ubatch-size "$UBATCH_SIZE" \
    --threads "$THREADS" \
    --threads-batch "$THREADS_BATCH" \
    --temp 1.0 \
    --top-p 1.0 \
    --top-k 20 \
    --min-p 0.0 \
    --repeat-penalty 1.0 \
    --seed 42 \
    --reasoning off \
    --reasoning-format none \
    --reasoning-budget 0 \
    --no-mmproj \
    --spec-type none \
    --metrics
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "$BIN" "${server_args[@]}"
    printf '\n'
    exit 0
fi

exec "$BIN" "${server_args[@]}"
