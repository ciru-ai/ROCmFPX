# DualView runtime guide

DualView lets one GGUF model present a compact Q7 representation to the decode
path and an exact signed-Q8 representation to the prefill path. It was created
for the asymmetry inside autoregressive inference:

- Prompt processing is wide matrix work and can exploit AMD INT8 dot/WMMA
  throughput.
- Token generation repeatedly streams weights and benefits from the smaller Q7
  representation.

The stored `Q7_0_ROCMFPX` block is authoritative. DualView sign-extends its
seven-bit signed values into the standard Q8 execution layout and reuses the
same FP16 scales. No second quantization or new rounding occurs at this
conversion boundary.

## Current support boundary

| Surface | Status |
|---|---|
| Linux, ROCm/HIP, Strix Halo `gfx1151` | Validated |
| Radeon 8060S / Ryzen AI Max 300 UMA | Primary tested target |
| Other AMD HIP targets | Compile/fallback only; the Q8 compute shadow is currently hard-gated to `gfx1151` |
| Windows HIP SDK `gfx1151` | Experimental and not yet release-validated |
| CUDA, Metal, Vulkan | The GGUF type has reference/fallback coverage, but the DualView Q8 compute shadow is currently a HIP path |
| Stock upstream llama.cpp | Unsupported: this model requires the CIRU ROCmFPX fork |

The retained Ornith model can occupy about 68–73 GB of GTT in the measured
profiles because the runtime materializes a Q8 compute shadow. Its 33.54 GB
integrated GGUF file is not its live memory footprint.

## Ubuntu 24.04 installation

Install the current AMD ROCm Core SDK for your system. Confirm that the runtime
sees `gfx1151`:

```bash
/opt/rocm/bin/hipconfig --full
/opt/rocm/bin/rocminfo | grep -m1 gfx
```

Install the ordinary build dependencies:

```bash
sudo apt update
sudo apt install -y build-essential cmake git ninja-build pkg-config \
  libcurl4-openssl-dev
```

Clone and build the tested release branch:

```bash
git clone --branch dualview --single-branch \
  https://github.com/ciru-ai/ROCmFPX.git
cd ROCmFPX
JOBS="$(nproc)" ./scripts/build-strix-dualview.sh
```

If ROCm is installed outside `/opt/rocm`, provide its compiler:

```bash
CMAKE_HIP_COMPILER=/path/to/amdclang++ \
  JOBS="$(nproc)" ./scripts/build-strix-dualview.sh
```

The CIRU validation build used a TheRock `gfx1151` toolchain. Standard ROCm is
the mainstream installation path; compiler/runtime version changes should be
treated as a new validation point.

## Verify the build

The format and conversion tests do not load a model:

```bash
./build-strix-dualview/bin/test-q7-q8-view
./build-strix-dualview/bin/test-quantize-fns
```

Confirm the GPU backend before loading the 35B artifact:

```bash
./build-strix-dualview/bin/llama-bench --list-devices
```

The output must identify the ROCm device. `No devices found` is a failed
environment, not a CPU fallback result.

## Run the model without MTP

Target-only is the strongest choice for prompt-heavy requests whose answers are
too short to repay speculative-prefill overhead:

```bash
export GGML_ROCM_GFX1151_Q7_Q8_VIEW=no-output
# Recommended for large UMA loads:
export GGML_HIP_ENABLE_UNIFIED_MEMORY=1
# Only for an older ROCm stack that does not identify the APU as gfx1151:
# export HSA_OVERRIDE_GFX_VERSION=11.5.1

./build-strix-dualview/bin/llama-server \
  -m /path/to/Ornith1.0-35b-CIRU-DUALVIEW-FPX7+Q8-MTP.gguf \
  --alias Ornith1.0-35b-CIRU-DUALVIEW-FPX7+Q8-MTP \
  --host 127.0.0.1 --port 8080 --jinja \
  --reasoning on --reasoning-format deepseek --reasoning-budget -1 \
  -dev ROCm0 -sm none -ngl 999 -fa on \
  -n 16384 -c 131072 -b 2048 -ub 512 -t 16 -tb 16 \
  -ctk f16 -ctv f16 --parallel 1 --metrics --mmap --no-repack \
  --no-cache-prompt --no-context-shift -fit off \
  --spec-type none
```

`no-output` keeps the target output tensor on its compact Q7 path while
creating Q8 views for the eligible prompt-processing tensors. This was the
retained policy.

## Run with the integrated official Q8 MTP head

```bash
export GGML_ROCM_GFX1151_Q7_Q8_VIEW=no-output
# Recommended for large UMA loads:
export GGML_HIP_ENABLE_UNIFIED_MEMORY=1
# Only for an older ROCm stack that does not identify the APU as gfx1151:
# export HSA_OVERRIDE_GFX_VERSION=11.5.1

./build-strix-dualview/bin/llama-server \
  -m /path/to/Ornith1.0-35b-CIRU-DUALVIEW-FPX7+Q8-MTP.gguf \
  --alias Ornith1.0-35b-CIRU-DUALVIEW-FPX7+Q8-MTP \
  --host 127.0.0.1 --port 8080 --jinja \
  --reasoning on --reasoning-format deepseek --reasoning-budget -1 \
  -dev ROCm0 -sm none -ngl 999 -fa on \
  -n 16384 -c 131072 -b 2048 -ub 512 -t 16 -tb 16 \
  -ctk f16 -ctv f16 --parallel 1 --metrics --mmap --no-repack \
  --no-cache-prompt --no-context-shift -fit off \
  --spec-type draft-mtp \
  --spec-draft-p-min 0.50 \
  --spec-draft-n-max 7 \
  -ctkd f16 -ctvd f16
```

Use depth 6 as the tested general short-context preset. Depth 7 won the tested
16K, 32K, and 64K effective-generation curves. MTP is target-verified: accepted
tokens are checked by the byte-identical target weights. It is not guaranteed
to reproduce a bit-identical greedy transcript because multi-token
verification changes reduction and batching paths.

The MTP state reset in this branch is validated for uncached requests beginning
at position zero. Cached-prefix restoration at a nonzero position has not been
validated, which is why the public command disables prompt caching and context
shift.

## Quantize a Q7 source model

The new GGUF preset is:

```bash
./build-strix-dualview/bin/llama-quantize \
  source-bf16.gguf output-q7.gguf Q7_0_ROCMFPX
```

The public Ornith release is not a uniform Q7 conversion. It is a
topology-aware mixed model with selected canonical-Q8 precision islands. See
[`DUALVIEW-ORNITH-35B-RESEARCH.md`](DUALVIEW-ORNITH-35B-RESEARCH.md) before
attempting to reproduce that recipe.

## Windows status

AMD's current HIP SDK lists `gfx1151` APUs as supported on Windows. This
DualView release has not yet completed a clean Windows compile and model-load
qualification, and CMake's HIP-language behavior differs from Linux. The code
is published for Windows contributors, but the supported copy-paste path is
Ubuntu/Linux until a Windows build is verified end to end.

## Why the environment switch is explicit

The Q8 compute shadow materially increases residency. Automatically enabling it
for every Q7 model would surprise users who selected Q7 primarily to fit a
memory limit. `GGML_ROCM_GFX1151_Q7_Q8_VIEW=no-output` therefore remains an
explicit serving policy rather than a global default.
