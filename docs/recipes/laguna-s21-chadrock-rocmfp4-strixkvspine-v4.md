# Laguna S 2.1 Chadrock ROCmFP4 StrixKVSpine V4 — Runtime V2

Runtime V2 and serving recipe for the Laguna S 2.1 118B-A8B ROCmFP4 artifact:

```text
laguna-s-2.1-ROCmFP4-StrixKVSpine-v4.gguf
SHA-256 ea1d854a72c47ec8e72c16ea91b8ff3cd5e1620b834df175f683c86f27dc26d6
```

This model uses the `laguna` GGUF architecture and ROCmFP4 tensor types. Use
this ROCmFPX branch rather than stock llama.cpp.

The V2 update changes the Vulkan runtime and serving defaults. The GGUF weights
are unchanged from the first release, so existing downloads do not need to be
replaced.

## V2 patch notes

The first runtime release could lose the Vulkan device during a very deep
Flash Attention prefill on RADV/Strix Halo. Reducing the graph-node submission
ceiling alone did not fix the problem: both 100-node and 10-node 128K controls
timed out after approximately 77–78 minutes.

Runtime V2:

- splits a large Flash Attention X grid into shorter Vulkan dispatch commands
  while preserving global workgroup IDs and output offsets;
- adds `GGML_VK_FA_MAX_WORKGROUPS_X_PER_DISPATCH` and
  `GGML_VK_MAX_NODES_PER_SUBMIT` controls;
- ports the FLOP-aware graph submission heuristic;
- latches the first DeviceLost result, stops further queue work, and makes
  lost-device cleanup best-effort;
- records graph node/operator context around the first failure;
- adds driver-recovery preflight, portable crash diagnostics, and bounded
  restart/backoff supervision;
- makes 128K the safe default and labels 256K as experimental.

| Serving behavior | First release | Runtime V2 |
| --- | --- | --- |
| Default context | 262,144 | **131,072 validated safe lane** |
| Ubatch | 512 | 512 |
| Graph nodes per submit | 100 | 10 |
| FA workgroups per dispatch | Unbounded | 4 |
| DeviceLost handling | Secondary exceptions possible | Sticky fatal latch and bounded teardown |
| Crash evidence | Manual | Automatic support bundle |
| 256K status | Advertised as tested | Experimental pending a full-depth gate |

Validation on Ryzen AI Max+ 395 / Radeon 8060S with Mesa RADV 26.1.2:

| Gate | Prompt processing | Generation | Result |
| --- | ---: | ---: | --- |
| 8K, three matched passes | 352.38 tok/s | 35.64 tok/s | Pass |
| 64K, one full prefill | 267.27 tok/s | 35.63 tok/s | Pass |
| 128K, one full prefill | 195.70 tok/s | 35.62 tok/s | Pass |

### Mesa 25.3.x / kernel 6.19.x revision note

The Mesa 26.1.2 row above assumes a kernel+Vulkan stack that has shipped several
upstream ring-stall fixes for gfx1151. Earlier stacks (Mesa 25.3.x with
`linux-firmware` < 20251201, kernel 6.19.x without the 6.19.12 device-lost
backport, etc.) still trip the 2-second `amdgpu` `lockup_timeout` watchdog on
the 100 k+ V2 lane because the FA split is purely in-command-buffer and does not
emit progress fences between the dispatched workgroups. The conservative safe
defaults for these earlier stacks are:

| Setting | Mesa 26.1.2 | Mesa 25.3.x |
| --- | --- | --- |
| `GGML_VK_MAX_NODES_PER_SUBMIT` | 10 | 4 |
| `GGML_VK_FA_MAX_WORKGROUPS_X_PER_DISPATCH` | 4 | 1 |

Verification on Ryzen AI Max+ 395 / Radeon 8060S with Mesa RADV 25.3.6 /
kernel 6.19.12-200.fc43, N=4 fresh-server 103k prefill runs:

| Run | http | time | ptok/s | kernel ring timeout |
| --- | ---: | ---: | ---: | --- |
| 1 | 200 | 565s | 185.87 | none |
| 2 | 200 | 569s | 186.15 | none |
| 3 | 200 | 572s | 185.37 | none |
| 4 | 200 | 568s | 185.40 | none |

By contrast, the unhardened Mesa-26.1.2 defaults (10/4) on the same stack fail
deterministically: 1/2 observed `Compute error` after 463 s, with kernel
`comp_1.X.Y timeout` followed by `device wedged, but recovered through reset`.
The 4/1 split lowers the per-submission work below the 2-second watchdog
ceiling without altering generation correctness — match-on-string sampling
matches the Mesa-26.1.2 128K reference output.

The 8K V2 row improved prompt processing by 10.57% over the matched unsplit
10-node control (318.70 tok/s), with effectively unchanged generation speed.
Deterministic split and unsplit test generations were byte-identical after
removing their timing lines.

### KV cache quantization and prefix caching

The runtime exposes `CACHE_TYPE_K` and `CACHE_TYPE_V` to choose the KV cache
quantization. The default is `q8_0` on both K and V. Choices validated on the
Ryzen AI Max+ 395 / Radeon 8060S with Mesa 25.3.6 / kernel 6.19.x:

| Cache type | Per-token decode at 92k cached (cold) | Notes |
| --- | ---: | --- |
| `f16` | not measured on this Mesa | 2× VRAM, no benefit on this stack |
| `q8_0` | 22.8 tok/s | safe default |
| `q4_0` | 23.7 tok/s | ~4% faster decode, small quality hit |

The launch script accepts `CACHE_TYPE_K=q4_0 CACHE_TYPE_V=q4_0` to opt in.
Set either independently — mixed types are also valid.

The default llama-server prompt cache is enabled (`--cache-prompt`,
`--cache-idle-slots`). When a request's `messages` array shares a prefix with
the cached slot, the matched tokens are reused from the slot's KV cache and
only the delta (the new user turn + chat template) is pre-filled. Verified on
the runtime build at 92k cached:

| Phase | Cold request | Prefix-matched follow-up |
| --- | --- | --- |
| Prefill | 92,453 tokens in 425 s (217 tok/s) | 1 token in 55 ms (cache_n = 92,452) |
| Decode | 11 tokens in 467 ms (23.5 tok/s) | 11 tokens in 445 ms (24 tok/s) |
| Total | 426 s | <1 s |

The between-step latency drops ~430× on the workflow where the same long
document is re-sent with a small follow-up. Do **not** add `--no-cache-idle-slots`
or `--cache-prompt off` to the runtime — that disables the fast path.

#### Caveat: SWA invalidation on long generations

The Laguna model is ISWA — 12 full-attention layers interleaved with 36
sliding-window-attention (SWA) layers, `n_swa = 512`. The prompt cache only
stores KV for the full-attention layers. The SWA layers' keys are anchored to
the slot's live `pos_min`, so once a slot has generated past 512 tokens the
prefix cache for that position is invalidated. After that, only the first
~`n_swa` tokens of the matched prefix are re-usable; the rest is forced
re-prefill. The server logs this as:

```
slot update_slots: id 0 | task N | n_past = 2920,
    slot.prompt.tokens.size() = 61432, seq_id = 0, pos_min = 59896, n_swa = 512
slot update_slots: id 0 | task N | forcing full prompt re-processing due to
    lack of cache data (likely due to SWA or hybrid/recurrent memory,
    see https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
```

In practice, the 430× speedup is only realized on a fresh slot — the first
request after server boot, or after the slot has been released between turns
without any generation. Long generations that push past the SWA window
discard the cache. Multi-turn "send the same document again" workflows
should expect to re-prefill the full document on the second request onwards.

Workarounds, none of which are currently promoted in the V2 safe lane:

1. `--swa-full` disables SWA, making the entire model full-attention. The
   cache would then be reusable, but: (a) doubles VRAM use, (b) breaks the
   Mesa 25.3.x stability gate that the V2 lane was hardened against, (c)
   pushes a 131k context window out of the 128 GiB unified memory budget.
2. `--parallel N` splits the cache across slots, but does not help the
   single-client "same prefix again" workflow.
3. Persisting the slot between requests is not exposed via the HTTP API;
   the slot is released on every response.

## Supported release target

- AMD Ryzen AI Max+ 395 / Radeon 8060S Strix Halo
- Vulkan backend (`Vulkan0`)
- Linux x86-64
- 128 GB unified memory for the validated 131,072-token safe profile
- one server slot

Smaller contexts can be selected through `CTX_SIZE` when less memory is
available. The model supports a 262,144-token context, but the V2 256K runtime
lane is not promoted until it passes the same full-depth stability gate.

## Linux support

The production runtime is built from portable Linux source rather than a
distro-specific binary.

| Distribution | Installation path | Validation status |
| --- | --- | --- |
| Ubuntu 24.04 LTS / Debian 12+ | Native `apt` packages | Primary documented path |
| Fedora 42+ | Native `dnf` packages | Supported build path |
| Arch / Manjaro | Native `pacman` packages | Supported build path |
| NixOS | Native Nix packages | Production build validated |

The helper detects these Linux families, prints the exact package-manager
commands by default, and installs them only when explicitly requested:

```bash
scripts/install-laguna-vulkan-deps.sh
scripts/install-laguna-vulkan-deps.sh --install
```

Manual commands:

### Ubuntu, Debian, Linux Mint, and Pop!_OS

```bash
sudo apt-get update
sudo apt-get install -y \
  git cmake ninja-build build-essential glslc \
  libvulkan-dev vulkan-tools spirv-headers mesa-vulkan-drivers
```

### Fedora, Rocky Linux, and AlmaLinux

```bash
sudo dnf install -y \
  git cmake ninja-build gcc gcc-c++ glslc \
  vulkan-loader-devel vulkan-headers spirv-headers \
  vulkan-tools mesa-vulkan-drivers
```

### Arch, Manjaro, and EndeavourOS

```bash
sudo pacman -S --needed \
  git cmake ninja base-devel shaderc \
  vulkan-icd-loader vulkan-headers spirv-headers \
  vulkan-tools vulkan-radeon
```

### NixOS

```bash
nix --extra-experimental-features 'nix-command flakes' profile add \
  nixpkgs#git nixpkgs#cmake nixpkgs#ninja nixpkgs#gcc \
  nixpkgs#shaderc nixpkgs#vulkan-headers nixpkgs#vulkan-loader \
  nixpkgs#spirv-headers
```

Confirm that Linux can see the AMD Vulkan device:

```bash
vulkaninfo --summary
```

## Build

Clone the V2 release branch and build a static Vulkan runtime:

```bash
git clone --branch agent/laguna-radv-device-lost-20260724 --depth 1 \
  https://github.com/ciru-ai/ROCmFPX.git
cd ROCmFPX
JOBS=8 scripts/build-laguna-strix-vulkan.sh
```

The build produces:

```text
build-laguna-strix-vulkan/bin/llama-server
build-laguna-strix-vulkan/bin/llama-cli
build-laguna-strix-vulkan/bin/llama-bench
build-laguna-strix-vulkan/bin/llama-quantize
```

The exact release commit is pinned on the Hugging Face model card. After
cloning, confirm it before building:

```bash
git rev-parse HEAD
```

Run the two release checks from the repository root:

```bash
build-laguna-strix-vulkan/bin/test-chat-auto-parser
build-laguna-strix-vulkan/bin/test-llama-archs
```

## Run the validated 128K V2 profile

The supervised launcher is recommended on RADV. It performs the Vulkan/driver
preflight, invokes the safe runner, captures DeviceLost evidence, and preserves
bounded restart state:

```bash
scripts/run-laguna-vulkan-supervised.sh \
  /path/to/laguna-s-2.1-ROCmFP4-StrixKVSpine-v4.gguf
```

Diagnostics default to
`${XDG_STATE_HOME:-$HOME/.local/state}/rocmfpx/laguna-vulkan-failures`.
Override that location with `DIAGNOSTIC_ROOT`.

The direct runner uses the same V2 safe defaults without supervision:

```bash
scripts/run-laguna-s21-rocmfp4-v4.sh \
  /path/to/laguna-s-2.1-ROCmFP4-StrixKVSpine-v4.gguf
```

The server listens on `127.0.0.1:8080`. Override portable settings through
environment variables:

```bash
HOST=0.0.0.0 PORT=8080 DEVICE=Vulkan0 \
  scripts/run-laguna-s21-rocmfp4-v4.sh /path/to/model.gguf
```

Verify the 60.945 GiB model once before serving:

```bash
VERIFY_SHA256=1 scripts/run-laguna-s21-rocmfp4-v4.sh /path/to/model.gguf
```

The release profile uses temperature 1.0, top-p 1.0, top-k 20, min-p 0,
F16/F16 KV, Flash Attention, and thinking off. This sampler stopped naturally
on all seven prompts that looped under strict greedy decoding.

The safe defaults are:

```text
STABILITY_MODE=safe
CTX_SIZE=131072
UBATCH_SIZE=512
VK_MAX_NODES_PER_SUBMIT=10
VK_FA_MAX_WORKGROUPS_X_PER_DISPATCH=4
```

The launcher exports the two `GGML_VK_*` variables consumed by the runtime.

### Experimental 256K lane

The model's 256K context remains available for explicit testing, but it is not
the V2 safe default:

```bash
STABILITY_MODE=performance \
  scripts/run-laguna-s21-rocmfp4-v4.sh /path/to/model.gguf
```

This mode prints an experimental-lane warning. Do not advertise it as stable
until repeated 256K prefill, multi-turn, and cache-replay gates complete
without a compute-ring timeout or DeviceLost.

Check readiness:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
```

## Recipe

StrixKVSpine V4 protects attention K/V and gates, dense block 0, shared
experts, a nine-layer expert-down spine, and output-sensitive tensors.
Attention Q/O and non-spine packed experts use the fast ROCmFP4 path. The
output tensor is Q6_K.

## Credits

Poolside created and released Laguna S 2.1. Charlie
(`charlie12345` / `caf`) created and maintains the ROCmFP4 codebook and
experimental ROCmFPX work that made this release possible. Ciru developed the
Laguna-specific StrixKVSpine recipe, calibration, production profile, and
validation.
