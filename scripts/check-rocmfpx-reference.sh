#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-rocmfpx-reference}"
CC_BIN="${CC:-cc}"

COMPILE_FLAGS=(-ffunction-sections -fdata-sections)
LINK_FLAGS=(-pthread)

case "$(uname -s)" in
    Linux*)
        LINK_FLAGS+=(-Wl,--gc-sections -ldl)
        ;;
    Darwin*)
        LINK_FLAGS+=(-Wl,-dead_strip)
        ;;
esac

mkdir -p "$BUILD_DIR"

"$CC_BIN" \
    -std=c11 \
    -Wall \
    -Wextra \
    -pedantic \
    -Wno-unused-function \
    "${COMPILE_FLAGS[@]}" \
    -D_GNU_SOURCE \
    '-DGGML_VERSION="reference"' \
    '-DGGML_COMMIT="reference"' \
    -I"$ROOT/ggml/include" \
    -I"$ROOT/ggml/src" \
    -I"$ROOT/ggml/rocmfpx" \
    "$ROOT/ggml/rocmfpx/rocmfpx.c" \
    "$ROOT/ggml/rocmfpx/test_rocmfpx.c" \
    "$ROOT/ggml/src/ggml.c" \
    "${LINK_FLAGS[@]}" \
    -lm \
    -o "$BUILD_DIR/test-rocmfpx"

"$BUILD_DIR/test-rocmfpx"
