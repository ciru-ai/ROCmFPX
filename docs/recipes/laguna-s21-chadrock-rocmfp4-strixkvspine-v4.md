# Laguna S 2.1 Chadrock ROCmFP4 StrixKVSpine V4

Production runtime and serving recipe for the Laguna S 2.1 118B-A8B ROCmFP4
artifact:

```text
laguna-s-2.1-ROCmFP4-StrixKVSpine-v4.gguf
SHA-256 ea1d854a72c47ec8e72c16ea91b8ff3cd5e1620b834df175f683c86f27dc26d6
```

This model uses the `laguna` GGUF architecture and ROCmFP4 tensor types. Use
this ROCmFPX branch rather than stock llama.cpp.

## Supported release target

- AMD Ryzen AI Max+ 395 / Radeon 8060S Strix Halo
- Vulkan backend (`Vulkan0`)
- Linux x86-64
- 128 GB unified memory for the demonstrated 262,144-token profile
- one server slot

Smaller contexts can be selected through `CTX_SIZE` when less memory is
available.

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

Clone the release branch and build a static Vulkan runtime:

```bash
git clone --branch agent/laguna-s21-production-runtime --depth 1 \
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

## Run the validated 262K profile

```bash
scripts/run-laguna-s21-rocmfp4-v4.sh \
  /path/to/laguna-s-2.1-ROCmFP4-StrixKVSpine-v4.gguf
```

The server listens on `127.0.0.1:8080`. Override any portable setting through
environment variables:

```bash
HOST=0.0.0.0 PORT=8080 CTX_SIZE=131072 DEVICE=Vulkan0 \
  scripts/run-laguna-s21-rocmfp4-v4.sh /path/to/model.gguf
```

Verify the 60.945 GiB model once before serving:

```bash
VERIFY_SHA256=1 scripts/run-laguna-s21-rocmfp4-v4.sh /path/to/model.gguf
```

The release profile uses temperature 1.0, top-p 1.0, top-k 20, min-p 0,
F16/F16 KV, Flash Attention, and thinking off. This sampler stopped naturally
on all seven prompts that looped under strict greedy decoding.

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
