#!/usr/bin/env bash
# Serve Laguna S 2.1 Chadrock ROCmFP4 StrixKVSpine V4 with its validated profile.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_RELEASE="V2"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-laguna-strix-vulkan}"
BIN="${BIN:-${6:-$BUILD_DIR/bin/llama-server}}"
MODEL="${MODEL:-${1:-}}"
HOST="${HOST:-${3:-127.0.0.1}}"
PORT="${PORT:-${4:-8080}}"
ALIAS="${ALIAS:-${2:-laguna-s21-rocmfp4-strixkvspine-v4}}"
DEVICE="${DEVICE:-Vulkan0}"
STABILITY_MODE="${STABILITY_MODE:-safe}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
THREADS="${THREADS:-16}"
THREADS_BATCH="${THREADS_BATCH:-16}"
FLASH_ATTN="${FLASH_ATTN:-on}"
SPLIT_MODE="${SPLIT_MODE:-row}"
CACHE_TYPE_K="${CACHE_TYPE_K:-f16}"
CACHE_TYPE_V="${CACHE_TYPE_V:-f16}"

case "$STABILITY_MODE" in
    safe)
        CTX_SIZE="${CTX_SIZE:-${5:-131072}}"
        UBATCH_SIZE="${UBATCH_SIZE:-512}"
        # Vulkan dispatch split values.
        # Standard V2 (10/4) was validated on Mesa 26.1.2 only and was observed
        # to be flaky on Mesa 25.3.x with kernels <6.x: single FA dispatches
        # can still exceed the 2s amdgpu lockup_timeout and trigger
        # VK_ERROR_DEVICE_LOST. The (4/1) split is the conservative profile
        # verified 4/4 across N=4 fresh-server 103k prefill runs on the
        # Ryzen AI Max+ 395 / Radeon 8060S with Mesa 25.3.6 / kernel 6.19.x.
        VK_MAX_NODES_PER_SUBMIT="${VK_MAX_NODES_PER_SUBMIT:-4}"
        VK_FA_MAX_WORKGROUPS_X_PER_DISPATCH="${VK_FA_MAX_WORKGROUPS_X_PER_DISPATCH:-1}"
        ;;
    performance)
        CTX_SIZE="${CTX_SIZE:-${5:-262144}}"
        UBATCH_SIZE="${UBATCH_SIZE:-512}"
        VK_MAX_NODES_PER_SUBMIT="${VK_MAX_NODES_PER_SUBMIT:-32}"
        VK_FA_MAX_WORKGROUPS_X_PER_DISPATCH="${VK_FA_MAX_WORKGROUPS_X_PER_DISPATCH:-0}"
        ;;
    custom)
        : "${CTX_SIZE:?CTX_SIZE is required with STABILITY_MODE=custom}"
        : "${UBATCH_SIZE:?UBATCH_SIZE is required with STABILITY_MODE=custom}"
        : "${VK_MAX_NODES_PER_SUBMIT:?VK_MAX_NODES_PER_SUBMIT is required with STABILITY_MODE=custom}"
        : "${VK_FA_MAX_WORKGROUPS_X_PER_DISPATCH:?VK_FA_MAX_WORKGROUPS_X_PER_DISPATCH is required with STABILITY_MODE=custom}"
        ;;
    *)
        echo "invalid STABILITY_MODE: $STABILITY_MODE (expected safe, performance, or custom)" >&2
        exit 2
        ;;
esac

for numeric_setting in CTX_SIZE BATCH_SIZE UBATCH_SIZE THREADS THREADS_BATCH VK_MAX_NODES_PER_SUBMIT; do
    numeric_value="${!numeric_setting}"
    if [[ ! "$numeric_value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$numeric_setting must be a positive integer, got: $numeric_value" >&2
        exit 2
    fi
done

if [[ ! "$VK_FA_MAX_WORKGROUPS_X_PER_DISPATCH" =~ ^[0-9]+$ ]]; then
    echo "VK_FA_MAX_WORKGROUPS_X_PER_DISPATCH must be a non-negative integer, got: $VK_FA_MAX_WORKGROUPS_X_PER_DISPATCH" >&2
    exit 2
fi

case "$FLASH_ATTN" in
    on|off) ;;
    *)
        echo "FLASH_ATTN must be on or off, got: $FLASH_ATTN" >&2
        exit 2
        ;;
esac

case "$SPLIT_MODE" in
    none|layer|row) ;;
    *)
        echo "SPLIT_MODE must be none, layer, or row, got: $SPLIT_MODE" >&2
        exit 2
        ;;
esac

case "$CACHE_TYPE_K" in
    f16|q8_0) ;;
    *)
        echo "CACHE_TYPE_K must be f16 or q8_0, got: $CACHE_TYPE_K" >&2
        exit 2
        ;;
esac

case "$CACHE_TYPE_V" in
    f16|q8_0) ;;
    *)
        echo "CACHE_TYPE_V must be f16 or q8_0, got: $CACHE_TYPE_V" >&2
        exit 2
        ;;
esac

export GGML_VK_MAX_NODES_PER_SUBMIT="$VK_MAX_NODES_PER_SUBMIT"
export GGML_VK_FA_MAX_WORKGROUPS_X_PER_DISPATCH="$VK_FA_MAX_WORKGROUPS_X_PER_DISPATCH"

if [[ -z "$MODEL" ]]; then
    echo "usage: MODEL=/path/to/laguna-v4.gguf $0" >&2
    echo "   or: $0 /path/to/laguna-v4.gguf" >&2
    echo "   or as BACKEND_LAUNCH_SCRIPT: $0 MODEL ALIAS HOST PORT CTX BIN" >&2
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
    --split-mode "$SPLIT_MODE" \
    --flash-attn "$FLASH_ATTN" \
    --cache-type-k "$CACHE_TYPE_K" \
    --cache-type-v "$CACHE_TYPE_V" \
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

echo "Laguna Vulkan stability settings:" >&2
echo "  runtime_release=$RUNTIME_RELEASE" >&2
echo "  mode=$STABILITY_MODE context=$CTX_SIZE batch=$BATCH_SIZE ubatch=$UBATCH_SIZE" >&2
echo "  cache_type_k=$CACHE_TYPE_K cache_type_v=$CACHE_TYPE_V" >&2
echo "  flash_attn=$FLASH_ATTN split_mode=$SPLIT_MODE max_nodes_per_submit=$VK_MAX_NODES_PER_SUBMIT" >&2
echo "  fa_max_workgroups_x_per_dispatch=$VK_FA_MAX_WORKGROUPS_X_PER_DISPATCH" >&2
echo "  binary=$BIN" >&2
echo "  binary_sha256=$(sha256sum "$BIN" | awk '{print $1}')" >&2

if [[ "$STABILITY_MODE" == "performance" ]]; then
    echo "  WARNING: the 256K performance lane is experimental and has not passed the V2 full-depth stability gate" >&2
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "$BIN" "${server_args[@]}"
    printf '\n'
    exit 0
fi

exec "$BIN" "${server_args[@]}"
