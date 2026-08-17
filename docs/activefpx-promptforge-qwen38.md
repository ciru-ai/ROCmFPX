# ActiveFPX + PromptForge runtime for Qwen3.8-27B

This tagged runtime is the matching inference implementation for
`Qwen3.8-27B-CIRU-ActiveFPX-PromptForge.gguf` and its PromptForge companion
views. It is derived from ROCmFPX and keeps llama.cpp's native target and
draft-MTP execution paths while adding prompt-only AMD gfx1151 routes.

## What the runtime adds

- PromptForge FFN routing for the validated 2,048-row prompt block, the
  2,044-row context-checkpoint block, and the 1,476-row tail.
- A fused gate/up path, fused SwiGLU-to-down packing, and accelerated down
  projection.
- A merged QKV/Z projection path for the model's recurrent Gated DeltaNet
  layers.
- Request-level route telemetry and fail-closed shape checks.
- Native compact execution for generated tokens, MTP verification, unsupported
  row shapes, and every path outside the qualified prompt routes.

PromptForge loads its prepacked companion views once at startup. The GGUF
remains the source of model behavior; the sidecars are serving-time compute
views for the exact published artifact.

## Pinned dependencies

- Runtime base: `5d04ce30c831879424a43f01aa2cda440000e271`
- Composable Kernel: `fdf4bb7fcc984811cef48ce817d89aac064b984a`
- Validated GPU: AMD Radeon 8060S / gfx1151
- Validated HIP toolchain: TheRock ROCm 7.15 development build

## Build

### Easiest non-NixOS path: Ubuntu 24.04

For a Ryzen AI Max / Radeon 8060S system, use Ubuntu 24.04 and follow AMD's
[current Ryzen ROCm installation guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/install-ryzen.html).
Install the host build tools:

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build git python3
```

Confirm that ROCm is installed in the standard location and recognizes the
gfx1151 GPU:

```bash
export ROCM_PATH=/opt/rocm
test -x "$ROCM_PATH/llvm/bin/clang++"
"$ROCM_PATH/bin/rocminfo" | grep -m1 gfx1151
```

Then clone both pinned trees:

```bash
git clone https://github.com/ciru-ai/ROCmFPX.git
cd ROCmFPX
git checkout qwen3.8-activefpx-promptforge-v1

git clone https://github.com/ROCm/composable_kernel.git ../composable_kernel
git -C ../composable_kernel checkout fdf4bb7fcc984811cef48ce817d89aac064b984a
```

Configure and build:

```bash
export ROCM_PATH=/opt/rocm

cmake -S . -B build-promptforge -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$ROCM_PATH/llvm/bin/clang" \
  -DCMAKE_CXX_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
  -DCMAKE_HIP_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
  -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
  -DGGML_HIP=ON \
  -DGGML_CUDA=OFF \
  -DGGML_VULKAN=OFF \
  -DGGML_HIP_FORCE_MMQ=ON \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_MMQ_MFMA=ON \
  -DGGML_HIP_NO_VMM=ON \
  -DGGML_HIP_ROCWMMA_FATTN=OFF \
  -DGGML_NATIVE=ON \
  -DAMDGPU_TARGETS=gfx1151 \
  -DGPU_BUILD_TARGETS=gfx1151 \
  -DPROMPTFORGE_CK_ROOT="$PWD/../composable_kernel" \
  -DGGML_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_SERVER=ON

cmake --build build-promptforge --target llama-server -j"$(nproc)"
```

The release binary was validated on NixOS with the pinned TheRock toolchain.
The Ubuntu path above is the intended portable source build for gfx1151. Other
Linux distributions can use the same CMake command when their HIP toolchain
supports gfx1151; change `ROCM_PATH` to the toolchain prefix. Those combinations
are not currently validated by Ciru.

## Required environment

```bash
export PROMPTFORGE_SIDECAR=/models/Qwen3.8-27B-CIRU-ActiveFPX-PromptForge-FFN.pfs
export PROMPTFORGE_GDN_SIDECAR=/models/Qwen3.8-27B-CIRU-ActiveFPX-PromptForge-GDN.pfs
export PROMPTFORGE_MODE=m2048_fused_tail1476
export GGML_CUDA_GRAPH_OPT=0
```

Keep llama.cpp context checkpoints enabled so growing conversations can reuse
their processed prefix. The validated setting is:

```bash
--ctx-checkpoints 32 --cache-ram 8192 --cache-prompt
```

Do not set `--ctx-checkpoints 0`: that disables the checkpoint state required
to restore the target, draft-MTP, and speculative-serving prefix. The runtime's
dedicated 2,044-row PromptForge route preserves accelerated prefill when the
checkpoint scheduler reserves the final four tokens.

Use the launch settings published on the model card. PromptForge is specialized
for this model and these companion views; startup fails closed when required
files or qualified kernel routes are unavailable.
