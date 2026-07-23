#!/usr/bin/env bash
# Build the production Laguna S 2.1 ROCmFP4 runtime for Strix Halo/Vulkan.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-laguna-strix-vulkan}"
JOBS="${JOBS:-$(nproc)}"

if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake is required" >&2
    exit 1
fi
if ! command -v ninja >/dev/null 2>&1; then
    echo "ninja is required" >&2
    exit 1
fi
if ! command -v glslc >/dev/null 2>&1; then
    echo "glslc is required (Ubuntu package: glslc)" >&2
    exit 1
fi

cmake -S "$ROOT" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_BACKEND_DL=OFF \
    -DGGML_VULKAN=ON \
    -DGGML_HIP=OFF \
    -DGGML_CUDA=OFF \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_WEBUI=OFF \
    -DLLAMA_USE_PREBUILT_WEBUI=OFF \
    -DLLAMA_BUILD_TESTS=ON \
    -DGGML_BUILD_TESTS=OFF

cmake --build "$BUILD_DIR" -j "$JOBS" --target \
    llama-server \
    llama-cli \
    llama-bench \
    llama-quantize \
    test-chat-auto-parser \
    test-llama-archs

echo
echo "Laguna ROCmFP4 Vulkan build ready:"
echo "  $BUILD_DIR/bin/llama-server"
echo "  $BUILD_DIR/bin/llama-cli"
echo "  $BUILD_DIR/bin/llama-bench"
echo "  $BUILD_DIR/bin/llama-quantize"
