#!/usr/bin/env bash
# Build the validated HIP-only DualView runtime for Strix Halo (gfx1151).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BUILD_DIR="${BUILD_DIR:-$ROOT/build-strix-dualview}"
HIP_ARCH="${CMAKE_HIP_ARCHITECTURES:-gfx1151}"
JOBS="${JOBS:-$(nproc)}"

if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake was not found. Install the build dependencies in docs/DUALVIEW.md." >&2
    exit 2
fi

if [[ -n "${CMAKE_HIP_COMPILER:-}" ]]; then
    if [[ ! -x "$CMAKE_HIP_COMPILER" ]]; then
        echo "CMAKE_HIP_COMPILER is not executable: $CMAKE_HIP_COMPILER" >&2
        exit 2
    fi
elif command -v amdclang++ >/dev/null 2>&1; then
    export CMAKE_HIP_COMPILER="$(command -v amdclang++)"
elif [[ -x /opt/rocm/bin/amdclang++ ]]; then
    export CMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++
else
    echo "No ROCm HIP compiler found. Install ROCm or set CMAKE_HIP_COMPILER." >&2
    exit 2
fi

ROCM_ROOT="${ROCM_ROOT:-$(cd "$(dirname "$CMAKE_HIP_COMPILER")/.." && pwd)}"
DUALVIEW_HIP_FLAGS="${CMAKE_HIP_FLAGS:-}"
DUALVIEW_HIP_FLAGS+=" -DGGML_ROCMFPX_RDNA35_MMID_MAX_BATCH=5"
DUALVIEW_HIP_FLAGS+=" -DGGML_ROCMFPX_MOE_MMVQ_ROWS_PER_BLOCK=4"

echo "DualView target: $HIP_ARCH"
echo "HIP compiler: $CMAKE_HIP_COMPILER"
echo "ROCm root: $ROCM_ROOT"
echo "Build directory: $BUILD_DIR"

GENERATOR_ARGS=()
if command -v ninja >/dev/null 2>&1; then
    GENERATOR_ARGS=(-G Ninja)
fi

cmake -S "$ROOT" -B "$BUILD_DIR" "${GENERATOR_ARGS[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_COMPILER="$CMAKE_HIP_COMPILER" \
    -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM_ROOT" \
    -DCMAKE_PREFIX_PATH="$ROCM_ROOT" \
    -DCMAKE_HIP_ARCHITECTURES="$HIP_ARCH" \
    -DCMAKE_HIP_FLAGS="$DUALVIEW_HIP_FLAGS" \
    -DAMDGPU_TARGETS="$HIP_ARCH" \
    -DGPU_TARGETS="$HIP_ARCH" \
    -DGGML_HIP=ON \
    -DGGML_HIP_FORCE_MMQ=ON \
    -DGGML_HIP_ROCWMMA_FATTN=OFF \
    -DGGML_VULKAN=OFF \
    -DGGML_CUDA=OFF \
    -DGGML_NATIVE=OFF \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_WEBUI=OFF \
    -DLLAMA_USE_PREBUILT_WEBUI=OFF \
    -DLLAMA_BUILD_TESTS=ON \
    -DGGML_BUILD_TESTS=OFF

TARGETS=(
    llama-cli
    llama-server
    llama-quantize
    llama-bench
    llama-perplexity
    test-q7-q8-view
    test-quantize-fns
    test-backend-ops
)

if [[ $# -gt 0 ]]; then
    TARGETS=("$@")
fi

cmake --build "$BUILD_DIR" -j "$JOBS" --target "${TARGETS[@]}"

echo "DualView binaries are in: $BUILD_DIR/bin"
