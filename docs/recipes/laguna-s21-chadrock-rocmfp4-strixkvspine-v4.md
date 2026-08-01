# Laguna S 2.1 Chadrock ROCmFP4 StrixKVSpine V4 — Runtime V3

Runtime V3 and serving recipe for the Laguna S 2.1 118B-A8B ROCmFP4 artifact:

```text
laguna-s-2.1-ROCmFP4-StrixKVSpine-v4.gguf
SHA-256 ea1d854a72c47ec8e72c16ea91b8ff3cd5e1620b834df175f683c86f27dc26d6
```

This model uses the `laguna` GGUF architecture and ROCmFP4 tensor types. Use
this ROCmFPX branch rather than stock llama.cpp.

Runtime V3 adds tool-call grammar/parser and portable-template fixes on top of
the V2 Vulkan runtime. The GGUF weights are unchanged, so existing downloads do
not need to be replaced.

## V3 patch notes

Runtime V3:

- ports llama.cpp PR #24835 so generated JSON values no longer carry trailing
  grammar whitespace that the final PEG tool parser rejects;
- aligns PEG JSON-array comma whitespace handling with the generated grammar;
- adds a Laguna/Pi regression for an `edit` call containing an
  `array<object>` argument and source-code strings;
- removes the render-only `{% generation %}` extension from the Laguna S 2.1
  template, preserving the rendered prompt while allowing standard Jinja and
  MiniJinja engines such as Hipfire to parse it;
- retains the V2 RADV DeviceLost safeguards and validated 128K defaults.

## V2 Vulkan patch notes

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

The 8K V2 row improved prompt processing by 10.57% over the matched unsplit
10-node control (318.70 tok/s), with effectively unchanged generation speed.
Deterministic split and unsplit test generations were byte-identical after
removing their timing lines.

## Supported release target

- AMD Ryzen AI Max+ 395 / Radeon 8060S Strix Halo
- Vulkan backend (`Vulkan0`)
- Linux x86-64
- 128 GB unified memory for the validated 131,072-token safe profile
- one server slot

Smaller contexts can be selected through `CTX_SIZE` when less memory is
available. The model supports a 262,144-token context, but the 256K runtime lane
is not promoted until it passes the same full-depth stability gate.

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

Clone the V3 release branch and build a static Vulkan runtime:

```bash
git clone --branch agent/laguna-s21-runtime-v3 --depth 1 \
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

Run the focused release checks from the repository root:

```bash
build-laguna-strix-vulkan/bin/test-json-schema-to-grammar
build-laguna-strix-vulkan/bin/test-chat-auto-parser
build-laguna-strix-vulkan/bin/test-chat --template poolside-Laguna-S-2.1.jinja
build-laguna-strix-vulkan/bin/test-llama-archs
```

## Run the validated 128K V3 profile

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

The direct runner uses the same safe defaults without supervision:

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
CTX_CHECKPOINTS=0
VK_MAX_NODES_PER_SUBMIT=10
VK_FA_MAX_WORKGROUPS_X_PER_DISPATCH=4
```

The launcher exports the two `GGML_VK_*` variables consumed by the runtime.
Context checkpoints are disabled for Laguna because the on-device checkpoint
invalidation path can crash during a divergent long-context request. This does
not disable the normal KV cache. `CTX_CHECKPOINTS` remains available for
explicit diagnostic testing, but values above zero are not part of the
validated profile until repeated 128K multi-turn and cache-replay gates pass.

### Experimental 256K lane

The model's 256K context remains available for explicit testing, but it is not
the safe default:

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

## Hipfire template compatibility

Hipfire releases that do not implement the Hugging Face `{% generation %}`
extension otherwise reject the embedded template and silently raw-encode the
user text, producing incoherent output. Runtime V3's Laguna S template avoids
that extension.

For an existing `.mq4` file with the older embedded template, point Hipfire at
the V3 template and force a fresh one-shot daemon:

```bash
HIPFIRE_LOCAL=1 \
HIPFIRE_CHAT_TEMPLATE_FILE="$PWD/models/templates/poolside-Laguna-S-2.1.jinja" \
  hipfire run ~/.hipfire/models/laguna-s21.mq4 --max-tokens 64 \
  "The capital of France is"
```

The override fixes prompt rendering without re-quantizing the weights. If a
background Hipfire daemon is already running, stop it with `hipfire stop` before
testing without `HIPFIRE_LOCAL=1`.

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
