#!/usr/bin/env bash
set -euo pipefail

readonly repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ck_root="${KAIRIC_CK_ROOT:-$repo_root/third_party/composable_kernel}"
readonly ck_commit=fdf4bb7fcc984811cef48ce817d89aac064b984a
readonly ck_patch="$repo_root/patches/composable-kernel-gfx1151-iu4.patch"
readonly build_dir="${BUILD_DIR:-$repo_root/build-kairic}"
readonly rocm="${ROCM_PATH:-/opt/rocm}"
readonly jobs="${JOBS:-$(nproc)}"

for tool in cmake git ninja; do
  command -v "$tool" >/dev/null || {
    echo "missing required tool: $tool" >&2
    exit 1
  }
done

[[ -f "$ck_patch" ]] || {
  echo "missing Composable Kernel patch: $ck_patch" >&2
  exit 1
}

if [[ ! -d "$ck_root/.git" ]]; then
  mkdir -p "$(dirname -- "$ck_root")"
  git clone https://github.com/ROCm/composable_kernel.git "$ck_root"
fi

git -C "$ck_root" checkout "$ck_commit"
if git -C "$ck_root" diff --quiet; then
  git -C "$ck_root" apply --check "$ck_patch"
  git -C "$ck_root" apply "$ck_patch"
elif git -C "$ck_root" apply --reverse --check "$ck_patch"; then
  echo "Composable Kernel release patch is already applied"
else
  echo "Composable Kernel checkout does not match the pinned patch" >&2
  exit 1
fi

if [[ -x "$rocm/bin/amdclang++" ]]; then
  hip_compiler="$rocm/bin/amdclang++"
elif [[ -x "$rocm/llvm/bin/clang++" ]]; then
  hip_compiler="$rocm/llvm/bin/clang++"
else
  echo "cannot find a HIP compiler below ROCM_PATH=$rocm" >&2
  exit 1
fi

if [[ -n "${CC:-}" ]]; then
  c_compiler="$(command -v "$CC")"
elif [[ -x /usr/bin/gcc ]]; then
  c_compiler=/usr/bin/gcc
else
  c_compiler="$(command -v gcc)"
fi
if [[ -n "${CXX:-}" ]]; then
  cxx_compiler="$(command -v "$CXX")"
elif [[ -x /usr/bin/g++ ]]; then
  cxx_compiler=/usr/bin/g++
else
  cxx_compiler="$(command -v g++)"
fi

export ROCM_PATH="$rocm"
export PATH="$rocm/bin:$rocm/lib/llvm/bin:$rocm/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$rocm/lib:$rocm/lib/rocm_sysdeps/lib:$rocm/lib/llvm/lib:$rocm/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
readonly hip_flags="-DGGML_ROCMFPX_RDNA35_MMID_MAX_BATCH=5 -DGGML_ROCMFPX_MOE_MMVQ_ROWS_PER_BLOCK=4${KAIRIC_HIP_TOOLCHAIN_FLAGS:+ $KAIRIC_HIP_TOOLCHAIN_FLAGS}"

cmake -S "$repo_root" -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$c_compiler" \
  -DCMAKE_CXX_COMPILER="$cxx_compiler" \
  -DCMAKE_HIP_COMPILER="$hip_compiler" \
  -DCMAKE_HIP_FLAGS="$hip_flags" \
  -DCMAKE_PREFIX_PATH="$rocm" \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_CPU=ON -DGGML_OPENMP=ON -DGGML_HIP=ON -DGGML_CUDA=OFF \
  -DGGML_VULKAN=OFF -DGGML_HIP_FORCE_MMQ=ON -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON \
  -DGGML_HIP_ROCWMMA_FATTN=OFF -DGGML_NATIVE=ON \
  -DAMDGPU_TARGETS=gfx1151 -DGPU_BUILD_TARGETS=gfx1151 \
  -DPROMPTFORGE_CK_ROOT="$ck_root" \
  -DLLAMA_BUILD_WEBUI=OFF \
  -DGGML_BUILD_TESTS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=ON

cmake --build "$build_dir" --target llama-server -j"$jobs"

"$build_dir/bin/llama-server" --version
"$build_dir/bin/llama-server" --help | grep -A1 -- '--kairic-edge'
ldd "$build_dir/bin/llama-server" 2>&1 | tee "$build_dir/llama-server.ldd.txt"
if grep -Eiq 'not found|version .* required by .* not found' "$build_dir/llama-server.ldd.txt"; then
  echo "runtime dependency verification failed" >&2
  exit 1
fi

echo "Kairic Edge runtime built at $build_dir/bin/llama-server"
