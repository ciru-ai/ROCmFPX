# Kairic Edge v1.3 runtime patch — September 24, 2026

This runtime update preserves the Kairic Edge GGUF, all three PromptForge
sidecars, and v1.2's strict compact M65 verification. It combines selected
Ciru, [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX), and
[llama.cpp](https://github.com/ggml-org/llama.cpp) fixes with measured prefill
improvements for Radeon 8060S / `gfx1151`.

Source: [`kairic-edge-qwen38-27b-v1.3`](https://github.com/ciru-ai/ROCmFPX/tree/kairic-edge-qwen38-27b-v1.3).
Downloads and checksums: [GitHub release](https://github.com/ciru-ai/ROCmFPX/releases/tag/kairic-edge-qwen38-27b-v1.3) and
[Hugging Face runtime/v1.3](https://huggingface.co/jcbtc/Qwen3.8-27B-IU4-Kairic-Edge/tree/main/runtime/v1.3).
The original model card and previous release notes are retained as historical
evidence; use this version's source and settings when trying the new runtime.

## Changes

- Synchronize reduction scratch-buffer reuse in normalization and softmax;
  retain Ciru's existing negative-infinity handling.
- Correct the F16/BF16 activation pointer passed through the BLAS callback,
  clear stale graph output state, skip inactive draft result pointers, and
  prevent inherited embedding/pooling settings in draft contexts.
- Default the launcher to `KAIRIC_GDN_MCOL=8` and
  `PROMPTFORGE_SMALLM_CK_VARIANT=1`, with explicit `1` / `0` fallbacks.
  The kernel retains its conservative default and guarded shape fallback.
- Isolate launcher settings from inherited experiments, force strict M65,
  and make compatibility mode disable both greedy argmax shortcuts and
  backend draft sampling.

The [source-selection audit](https://github.com/ciru-ai/ROCmFPX/blob/kairic-edge-qwen38-27b-v1.3/docs/kairic-update-sources.md)
links the imported fixes and explains deferred changes. The original HIP
fast-math policy is retained: removing it changed deterministic token IDs and
regressed generation throughput in the discarded candidate.

## Measured result

Same-binary, uncached A-B-B-A screens used a 3,917-token prompt, zero cached
tokens, and 128 generated tokens. Each setting was measured twice.

| Isolated change | Before PP | After PP | Mean change |
| --- | ---: | ---: | ---: |
| MCOL1 → MCOL8, CK0 | 478.50 tok/s | 509.44 tok/s | +6.47% |
| CK0 → CK1, MCOL8 | 508.40 tok/s | 522.87 tok/s | +2.85% |

Both screens preserved the output text and all 128 generated token IDs.
MCOL's mirrored gains were +8.39% and +4.60%; CK's were +2.72% and +2.98%.
These are bounded prefill measurements, not general speedups or a compounded
headline gain. Generation throughput stayed approximately unchanged.

Independent baseline/candidate loads in A-C-C-A order, with MCOL1/CK0 in both
arms, matched every generated token on the 498- and 3,917-token prompts.
Measured cached generation was 47.17 → 47.11 tok/s and 56.46 → 56.30 tok/s,
respectively (within 0.3% of v1.2).

Final backend checks passed **360/360 cases**: GDN MCOL8 52, invalid-MCOL
fallback 52, softmax/group norm 220, and bounded F16/BF16 matrix multiplication
36. Compatibility smoke passed forced tool arguments, sampled output, direct
`/completion` JSON-schema structure, and two cached continuations.
The existing nested-schema `/v1/chat/completions` sampler-initialization issue
was reproduced on v1.2 and the discarded candidate; this release does not
claim that endpoint issue is fixed. Schema structure is not a task-quality score.

The new qualification used **32,768 context**, one slot, F16 target/draft KV,
batch 2048, ubatch 512, threads 16/32, MTP4, thinking off, and strict M65 on
Radeon 8060S / `gfx1151`, NixOS, TheRock 7.15.0a20260718, HIP Clang 23,
and GCC 13.4. It did not rerun the full quality suite or the 256K context
ladder. Earlier card results remain attributed to their original versions.

Machine-readable measurements, test counts, and binary identities are in
[`validation.json`](https://huggingface.co/jcbtc/Qwen3.8-27B-IU4-Kairic-Edge/resolve/main/runtime/v1.3/validation.json).

## Build and run

The recommended distribution is the pinned source. The
[build guide](https://github.com/ciru-ai/ROCmFPX/blob/kairic-edge-qwen38-27b-v1.3/docs/kairic-edge-gfx1151.md)
includes an Ubuntu 24.04 dependency recipe and `/opt/rocm` build path.
The actual qualified environment is listed above; the Ubuntu recipe was not
separately validated in this update.

```bash
git clone --depth 1 --branch kairic-edge-qwen38-27b-v1.3 https://github.com/ciru-ai/ROCmFPX.git
cd ROCmFPX
export ROCM_PATH=/opt/rocm
# On NixOS, set matching GCC 13 CC/CXX and toolchain flags as in the build guide.
scripts/build-kairic-edge-gfx1151.sh

export LLAMA_SERVER="$PWD/build-kairic/bin/llama-server"
export MODEL_PATH=/path/to/Qwen3.8-27B-IU4-Kairic-Edge.gguf
export KAIRIC_FFN_SIDECAR=/path/to/Qwen3.8-27B-Kairic-IU4-FFN.pfs
export KAIRIC_GDN_SIDECAR=/path/to/Qwen3.8-27B-Kairic-IU4-GDN.pfs
export KAIRIC_GDN_OUTPUT_SIDECAR=/path/to/Qwen3.8-27B-Kairic-IU4-GDN-Output.pfs
CONTEXT=32768 scripts/run-kairic-edge-gfx1151.sh
```

The runner retains the previous 262,144 context default for compatibility;
the explicit `CONTEXT=32768` above matches this update's validation.
For sampling, tools, or grammar requests, add
`KAIRIC_EDGE_COMPATIBILITY_MODE=1`. To compare the conservative settings, add
`KAIRIC_GDN_MCOL=1 PROMPTFORGE_SMALLM_CK_VARIANT=0`.

The `kairic-edge-v1.3-gfx1151-nixos-qualification.tar.gz` asset preserves the
exact tested executables and complete matching project-library set. It is a
**NixOS qualification snapshot**, not a portable Linux binary distribution:
its ELF loader and dependencies require the recorded Nix store objects and
a compatible TheRock tree. See its `DEPENDENCIES.txt`; it does not bundle
ROCm or the Nix runtime. Rebuild from source on another installation.
The binaries report version `0 (unknown)` because the qualified build used
a source archive without Git metadata; `provenance.json` and SHA-256 hashes
bind that build to the published source. Keep all project libraries together.

## Unchanged model artifacts

| Artifact | SHA-256 |
| --- | --- |
| `Qwen3.8-27B-IU4-Kairic-Edge.gguf` | `360caf7381907c3eca7ac0afd1228efc016af747f3f38637fb1c7f94daabac2a` |
| `Qwen3.8-27B-Kairic-IU4-FFN.pfs` | `adcbb90a7b429a30a2a39043366d68320d72e8b4816a0f498e882b2f80a2ba2b` |
| `Qwen3.8-27B-Kairic-IU4-GDN.pfs` | `82f931316f1c895da104915dec4697163808d06f0e6b2dc027cee7aa3afc0f0e` |
| `Qwen3.8-27B-Kairic-IU4-GDN-Output.pfs` | `3b07e7b176559e4402924ba0c368532fa6f02118a33c71e70974c809bf6208a3` |

Model licensing and credits remain as documented in the original card.
Runtime source and third-party components retain their upstream licenses.
