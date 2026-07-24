#!/usr/bin/env bash
# Build the production Laguna S 2.1 ROCmFP4 runtime for Strix Halo/Vulkan.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-laguna-strix-vulkan}"
JOBS="${JOBS:-$(nproc)}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
VULKAN_DEBUG="${VULKAN_DEBUG:-OFF}"
VULKAN_MEMORY_DEBUG="${VULKAN_MEMORY_DEBUG:-OFF}"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "this production build currently targets Linux x86-64" >&2
    exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake is required" >&2
    echo "run: scripts/install-laguna-vulkan-deps.sh --install" >&2
    exit 1
fi
if ! command -v ninja >/dev/null 2>&1; then
    echo "ninja is required" >&2
    echo "run: scripts/install-laguna-vulkan-deps.sh --install" >&2
    exit 1
fi
if ! command -v glslc >/dev/null 2>&1; then
    echo "glslc is required" >&2
    echo "run: scripts/install-laguna-vulkan-deps.sh --install" >&2
    exit 1
fi

if [[ -z "${CMAKE_PREFIX_PATH:-}" && -d "${HOME:-}/.nix-profile" ]]; then
    export CMAKE_PREFIX_PATH="${HOME}/.nix-profile"
fi

cmake -S "$ROOT" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_BACKEND_DL=OFF \
    -DGGML_VULKAN=ON \
    -DGGML_VULKAN_DEBUG="$VULKAN_DEBUG" \
    -DGGML_VULKAN_MEMORY_DEBUG="$VULKAN_MEMORY_DEBUG" \
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
echo "  build type: $BUILD_TYPE"
echo "  Vulkan debug: $VULKAN_DEBUG"
echo "  Vulkan memory debug: $VULKAN_MEMORY_DEBUG"
echo "  $BUILD_DIR/bin/llama-server"
echo "  $BUILD_DIR/bin/llama-cli"
echo "  $BUILD_DIR/bin/llama-bench"
echo "  $BUILD_DIR/bin/llama-quantize"
