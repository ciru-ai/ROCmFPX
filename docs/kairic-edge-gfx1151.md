# Kairic Edge on AMD Strix Halo (`gfx1151`)

This guide builds and runs the immutable source release for Qwen3.8-27B IU4
Kairic Edge. It is the known-best public Prompt Forge / Dual View runner
corresponding to the qualified Kairic Edge release.

Version 1.2 changes exact 65-row n-gram verification to the compact
authoritative path by default. The model and PromptForge sidecars are unchanged.
The previous native IU4 M65 verifier is retained only for controlled diagnostics
because it changed a reproduced greedy target token.

## Scope

Kairic Edge routes supported prompt and multi-token verification shapes through
AMD RDNA 3.5 native IU4 matrix compute. It is a guarded hybrid path: operations
outside the qualified shape/quality envelope retain their established fallback.
The current native sidecars do not accelerate M1 target decode.

Validated device:

```text
AMD Ryzen AI Max+ 395
Radeon 8060S
RDNA 3.5
gfx1151
40 compute units in AMD's published inventory
```

The release was certified with:

```text
TheRock             7.15.0a20260718
AMD clang           23.0.0
HIP                 7.15.0
GCC / G++           13.3.0
CMake               4.4.2
Ninja               1.13.0
Composable Kernel   fdf4bb7fcc984811cef48ce817d89aac064b984a + release patch
GPU target          gfx1151
```

The portable `/opt/rocm` procedure below may work with another compatible ROCm
release, but only the toolchain above is release-certified. A locally rebuilt
binary is not expected to match the frozen validation binary byte for byte.
On NixOS, export a matched GCC 13 `CC` and `CXX` explicitly; do not pair the
certified HIP compiler with an unrelated newer host C++ standard library.
If the HIP compiler cannot discover Nix's GCC/glibc headers, pass the matching
`--gcc-toolchain` and `-idirafter` paths through
`KAIRIC_HIP_TOOLCHAIN_FLAGS`; the release script appends that value only to HIP
compilation.

## Install build dependencies

Ubuntu 24.04:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git ninja-build pkg-config libssl-dev
```

Install a ROCm distribution that supports Radeon 8060S / `gfx1151`, then make
its root visible as `ROCM_PATH`. A complete TheRock tree or a compatible system
ROCm tree can be used.

## Clone the immutable release

```bash
git clone https://github.com/ciru-ai/ROCmFPX.git
cd ROCmFPX
git checkout kairic-edge-qwen38-27b-v1.2
git status --short
```

The last command should print nothing.

## Prepare Composable Kernel

```bash
git clone https://github.com/ROCm/composable_kernel.git third_party/composable_kernel
git -C third_party/composable_kernel checkout fdf4bb7fcc984811cef48ce817d89aac064b984a
git -C third_party/composable_kernel apply \
  ../../patches/composable-kernel-gfx1151-iu4.patch
```

The patch SHA-256 is:

```text
ea57bca3793d35d884fc156b6c0bc33d8e672e3bb49f66ff67338503029d6af5
```

Verify before building:

```bash
sha256sum patches/composable-kernel-gfx1151-iu4.patch
git -C third_party/composable_kernel diff --check
```

## Build

The release script owns the certified CMake options:

```bash
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
JOBS="$(nproc)" scripts/build-kairic-edge-gfx1151.sh
```

Or configure directly:

```bash
if [[ -x "$ROCM_PATH/bin/amdclang++" ]]; then
  HIP_COMPILER="$ROCM_PATH/bin/amdclang++"
else
  HIP_COMPILER="$ROCM_PATH/llvm/bin/clang++"
fi
C_COMPILER="${CC:-/usr/bin/gcc}"
CXX_COMPILER="${CXX:-/usr/bin/g++}"

cmake -S . -B build-kairic -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$C_COMPILER" \
  -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
  -DCMAKE_HIP_COMPILER="$HIP_COMPILER" \
  -DCMAKE_HIP_FLAGS='-DGGML_ROCMFPX_RDNA35_MMID_MAX_BATCH=5 -DGGML_ROCMFPX_MOE_MMVQ_ROWS_PER_BLOCK=4' \
  -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_CPU=ON -DGGML_OPENMP=ON -DGGML_HIP=ON -DGGML_CUDA=OFF \
  -DGGML_VULKAN=OFF -DGGML_HIP_FORCE_MMQ=ON -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON \
  -DGGML_HIP_ROCWMMA_FATTN=OFF -DGGML_NATIVE=ON \
  -DAMDGPU_TARGETS=gfx1151 -DGPU_BUILD_TARGETS=gfx1151 \
  -DPROMPTFORGE_CK_ROOT="$PWD/third_party/composable_kernel" \
  -DLLAMA_BUILD_WEBUI=OFF \
  -DGGML_BUILD_TESTS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=ON

cmake --build build-kairic --target llama-server -j"$(nproc)"
```

## Build verification

```bash
./build-kairic/bin/llama-server --version
./build-kairic/bin/llama-server --help | grep -A1 -- '--kairic-edge'
ldd ./build-kairic/bin/llama-server 2>&1 | tee /tmp/kairic-edge-ldd.txt
! grep -Eiq 'not found|version .* required by .* not found' /tmp/kairic-edge-ldd.txt
```

The v1.2 release-candidate qualification used these binary identities:

```text
llama-server SHA-256    9a307481c268b631e8bcb0a1e68aa9ae3adb77de00db87646c41827528ca6661
libllama-common SHA-256 17776460e3a91ee2ee2e7b52cc230e93a86d97a86c11d85e9365eec8d54ec39b
libggml-hip SHA-256     25bd836bdbb5d0275f4ed9c61a801c15458d28cbbf72f11ca4c1f5aab1c1f9af
model SHA-256         360caf7381907c3eca7ac0afd1228efc016af747f3f38637fb1c7f94daabac2a
```

These hashes identify the certified build; a portable rebuild records the
public release commit and local compiler identity instead.

## Download model artifacts

Download all four files from
[jcbtc/Qwen3.8-27B-IU4-Kairic-Edge](https://huggingface.co/jcbtc/Qwen3.8-27B-IU4-Kairic-Edge):

```text
Qwen3.8-27B-IU4-Kairic-Edge.gguf
Qwen3.8-27B-Kairic-IU4-FFN.pfs
Qwen3.8-27B-Kairic-IU4-GDN.pfs
Qwen3.8-27B-Kairic-IU4-GDN-Output.pfs
```

Verify their SHA-256 values against the model card before launch.

## Run the recommended live profile

```bash
export LLAMA_SERVER="$PWD/build-kairic/bin/llama-server"
export MODEL_PATH=/path/to/Qwen3.8-27B-IU4-Kairic-Edge.gguf
export KAIRIC_FFN_SIDECAR=/path/to/Qwen3.8-27B-Kairic-IU4-FFN.pfs
export KAIRIC_GDN_SIDECAR=/path/to/Qwen3.8-27B-Kairic-IU4-GDN.pfs
export KAIRIC_GDN_OUTPUT_SIDECAR=/path/to/Qwen3.8-27B-Kairic-IU4-GDN-Output.pfs
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

scripts/run-kairic-edge-gfx1151.sh
```

The runner binds `127.0.0.1:8080` by default and enables:

- full `ROCm0` offload and flash attention;
- 262,144 context with one slot;
- batch 2,048 and ubatch 512;
- 16 target threads and 32 batch threads;
- F16 target and draft KV;
- Kairic Edge;
- explicit native MTP depth 4;
- n-gram match/min/max 24/64/64 with strict compact M65 verification;
- 8,192 MiB prompt cache, idle-slot persistence, and 32 context checkpoints;
- metrics;
- deterministic temperature 0 / top-p 1 / top-k 0 / min-p 0; and
- reasoning disabled.

The server log must contain:

```text
Kairic Edge: enabled (ngram=24/64/64, M65 verifier=strict-compact)
```

Do not set `KAIRIC_UNSAFE_NATIVE_M65_VERIFY=1` in a live profile. That diagnostic
override restores the superseded native M65 verifier, which is faster but changed
a reproduced greedy target token.

Fast greedy mode is the default. It accepts exactly one unmodified greedy
completion and intentionally rejects request features that require full target
logits, including sampling, penalties, probabilities, grammar-constrained tool
calls, logit bias, LoRA, and a reasoning budget.

Override `HOST`, `PORT`, `CONTEXT`, or `CACHE_RAM` only if you understand the
memory/security tradeoff. Binding beyond localhost exposes an unauthenticated
OpenAI-compatible endpoint unless you add a trusted proxy.

## Smoke test and prompt-cache verification

Health and identity:

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/v1/models
curl -fsS http://127.0.0.1:8080/slots
curl -fsS http://127.0.0.1:8080/metrics | grep -E 'prompt|cache|tokens'
```

Make the same request twice with `cache_prompt` enabled:

```bash
request='{
  "model":"main",
  "prompt":"Repeatable cache smoke: explain why packed four-bit compute matters.",
  "n_predict":64,
  "temperature":0,
  "cache_prompt":true
}'

curl -fsS http://127.0.0.1:8080/completion \
  -H 'Content-Type: application/json' -d "$request" > /tmp/kairic-cold.json
curl -fsS http://127.0.0.1:8080/completion \
  -H 'Content-Type: application/json' -d "$request" > /tmp/kairic-warm.json
```

Confirm the warm response and server metrics report restored/cached prompt
tokens, then inspect the server log for route fallback, swap activity, HIP
errors, and device loss. Context checkpoints are required for this model's
recurrent state; do not remove `-ctxcp 32` while using the prompt cache.

## Request-level chat sampling

Sampling and tool calling require compatibility mode. Set it before starting
the runner:

```bash
export KAIRIC_EDGE_COMPATIBILITY_MODE=1
scripts/run-kairic-edge-gfx1151.sh
```

Then ordinary non-thinking chat can use the upstream Qwen recommendation at
request level:

```json
{
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "min_p": 0.0,
  "presence_penalty": 1.5,
  "cache_prompt": true
}
```

Compatibility mode disables only the target greedy argmax fast path; Kairic
Edge, native MTP4, prompt caching, context checkpoints, and the other qualified
launch settings remain enabled. In the release gate, sampled chat and a forced
tool call both returned HTTP 200, and HumanEval 0–9 passed 10/10 Base and 10/10
Plus with byte-identical output to fast mode. It measured 41.87 versus 46.37
generated tokens/s on that short coding subset, a 9.70% reduction, so it is
opt-in rather than the default. This subset is a compatibility smoke test, not
a leaderboard score or a universal throughput estimate. The sanitized result
is preserved in
[`docs/evidence/kairic-edge-compatibility-v1.1.json`](evidence/kairic-edge-compatibility-v1.1.json).

## v1.2 M65 verification gate

The frozen structured-generation repro produced the same no-spec target response
hash in all six candidate runs. Five warm strict-M65 rows took 22.31–22.38 seconds
(22.34 seconds mean), accepted 9,255/9,280 drafted tokens (99.73%), and retained a
mean accepted length of 64.83 tokens. The strict route is approximately 5–8%
slower than the unsafe native M65 verifier on this workload and approximately
7.19x faster than speculation off.

The target response itself contains two known arithmetic mistakes. The release
gate is target equivalence: speculative verification must preserve the target
model's tokens rather than silently changing them.

## Evidence boundary

The release is validated on one Strix Halo system and one serving slot. The
instruction, operator, prompt, cache, and full-suite measurements use different
boundaries and must remain labeled. Do not turn an instruction-rate peak into a
sustained model-throughput claim, infer energy efficiency without sustained
power integration, or describe this hybrid route as whole-model native INT4.
