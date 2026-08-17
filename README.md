# ROCmFPX by Ciru

**Ciru's NixOS-based AMD inference lab for low-bit formats, mixed-precision
execution, specialized GPU paths, and reproducible model releases.**

This repository is the maintained home of Ciru's ROCmFPX work. It carries
format and kernel development, model-specific serving runtimes, quantization
research, validation infrastructure, and the release recipes behind Ciru's
published GGUFs.

ROCmFPX is experimental. Formats, kernels, and runtime specialization can
change between release lines. Use the tag and model-specific documentation for
reproducible results.

## What Ciru Built

ROCmFP2 is one part of the work, not the whole story.

| Area | Ciru's work |
|---|---|
| **ROCmFP2** | Designed and implemented the 2.50-bpw S40 format, frozen codebook, CPU reference path, HIP/ROCm MMQ and MMVQ paths, quantization support, dispatch, and correctness tests |
| **ROCmFP3** | Added Python GGUF dequantization and specialized the Vulkan dequant, matvec, and packed execution paths; continued the scale-search and backend performance work |
| **ROCmFP6** | Built Strix quality recipes, enabled and tuned HIP/ROCm MMQ and MMVQ execution, fixed CPU and cross-backend endpoint semantics, and added safety and regression coverage |
| **ROCmFP7 / DualView** | Designed and implemented the signed Q7 format and the Q7-decode/Q8-prefill execution architecture, including compute views, caches, GPU kernels, tests, model research, and the Ornith 35B release |
| **ActiveFPX PromptForge** | Built the Qwen3.8-27B prompt-specialized runtime with fused FFN routes, merged recurrent QKV/Z projection, prepacked compute views, route telemetry, shape guards, checkpoint preservation, and the validated MTP argmax path |
| **MTP and serving** | Fixed partial-draft state tracking and M-RoPE hybrid batches; built dynamic drafting, request-level controls, stateful SSD prompt caching, and repeated-request correctness work |
| **Quantization research** | Built ranked tensor-policy tooling, architecture/topology recipe contracts, reconstruction-error scale-search improvements, precision-island studies, and artifact-level release maps |
| **Runtime reliability** | Fixed ROCm fast-math edge cases, row-group typing, GPU dispatch and stride bugs, portable cache synchronization, and preservation regressions during upstream rebases |
| **Maintained fork** | Forward-ported the ROCmFPX stack onto newer official `llama.cpp` lines while preserving Ciru features, tests, CI, and model compatibility |

The work spans GGML format definitions, reference math, quantization, CPU
execution, HIP/ROCm and Vulkan kernels, model loading, speculative decoding,
server state, CI, model research, and release engineering.

## Current Ciru Release Lines

| Line | Purpose | Reference |
|---|---|---|
| **ROCmFPX main** | Maintained integration line for the format family, kernels, serving work, and upstream preservation | [`main`](https://github.com/ciru-ai/ROCmFPX/tree/main) |
| **ActiveFPX PromptForge v2.2** | Current Qwen3.8-27B specialized prefill runtime with checkpoint and ROCm MTP fixes | [`qwen3.8-activefpx-promptforge-v2.2`](https://github.com/ciru-ai/ROCmFPX/tree/qwen3.8-activefpx-promptforge-v2.2) |
| **DualView Ornith 35B v1** | Reproducible Q7 decode + exact Q8 prefill runtime and integrated MTP release | [`dualview-ornith-35b-v1`](https://github.com/ciru-ai/ROCmFPX/releases/tag/dualview-ornith-35b-v1) |

## ActiveFPX PromptForge

ActiveFPX PromptForge is Ciru's current model-specialized serving work for
`Qwen3.8-27B-CIRU-ActiveFPX-PromptForge.gguf`.

It adds qualified `gfx1151` prompt routes for the published artifact:

- fused gate/up projection;
- fused SwiGLU-to-down packing and accelerated down projection;
- merged QKV/Z projection for recurrent Gated DeltaNet layers;
- dedicated routes for the 2,048-row prompt block, the 2,044-row checkpoint
  block, and the 1,476-row tail;
- prepacked companion compute views loaded once at startup;
- request-level route telemetry and fail-closed shape checks;
- native compact execution outside the qualified prompt shapes;
- checkpoint-safe target/draft state and a validated greedy MTP argmax path.

The published GGUF remains the source of model behavior. PromptForge sidecars
are serving-time compute views for that exact artifact.

The current release was validated in Ciru's NixOS environment on Radeon 8060S
(`gfx1151`) with a pinned TheRock ROCm 7.15 development toolchain and pinned
Composable Kernel revision. Use the tagged document for the exact dependency
and runtime contract:

- [ActiveFPX PromptForge runtime guide](https://github.com/ciru-ai/ROCmFPX/blob/qwen3.8-activefpx-promptforge-v2.2/docs/activefpx-promptforge-qwen38.md)

## DualView

DualView is Ciru's mixed-physical-view execution architecture. One GGUF keeps
signed Q7 codes as its source of truth and exposes those same integers to the
GPU in two forms:

- **Decode:** packed `Q7_0_ROCMFPX` to reduce bytes read per generated token.
- **Prefill:** an exact signed-Q8 compute view for native INT8 dot/WMMA paths.

The Q7-to-Q8 view change sign-extends the stored integer and reuses the same
scale. It introduces no second quantization or additional rounding. It is
lossless relative to the stored Q7 code; Q7 itself remains a quantization of
the source model.

The first public target is
**Ornith1.0-35B-CIRU-DUALVIEW-FPX7+Q8-MTP**. Ciru's retained quality-max target
uses 362 Q7 tensors, 70 selected canonical-Q8 precision islands, and 301 F32
tensors, with the official Q8 MTP head integrated into the release artifact.

Matched full-model results on Radeon 8060S / `gfx1151`:

| Artifact | PP4096 | TG256 |
|---|---:|---:|
| **Ciru quality-max DualView target** | **1,236.156 tok/s** | **48.049 tok/s** |
| Original Q7S8 DualView | 1,202.550 tok/s | 48.719 tok/s |
| Official pure Q8_0 control | 1,184.626 tok/s | 43.470 tok/s |

The retained target reduced KLD by 15.26% versus the original Q7S8 model in
the recorded 24,576-token quality study. The research record also reports
residency costs, MTP curves through 64K context, precision-island ablations,
and the cases where MTP prefill overhead loses on total wall time.

- [DualView architecture and runtime guide](docs/DUALVIEW.md)
- [Complete Ornith 35B research record](docs/DUALVIEW-ORNITH-35B-RESEARCH.md)
- [Visual DualView explainer](https://llm.ciru.ai/dualview)

## ROCmFPX Format Work

| Format | Native block BPW | Ciru work in this tree |
|---|---:|---|
| `Q2_0_ROCMFPX` | 2.50 | Core format, reference math, quantizer, HIP/ROCm kernels, dispatch, and tests |
| `Q3_0_ROCMFPX` | 3.50 | Python dequantization, specialized Vulkan execution, packed-path and scale-search performance |
| `Q4_0_ROCMFP4` / `FAST` | 4.50 / 4.25 | Integration, conversion, regression coverage, serving, and reproducibility work built on the original ROCmFP4 format |
| `Q6_0_ROCMFPX` | 6.50 | Strix recipes, GPU execution tuning, endpoint corrections, and cross-backend validation |
| `Q7_0_ROCMFPX` | 7.50 | Ciru format and the authoritative stored representation for DualView |
| `Q8_0_ROCMFPX` | 8.25 | Reference and cross-backend semantics used in quality and execution studies |

That is the complete tensor-format family currently implemented in the Ciru
tree: FP2, FP3, FP4, FP6, FP7, and FP8. There is no registered ROCmFP5 tensor
type in the current source.

All six base formats are available directly through `llama-quantize`:

```bash
llama-quantize source-BF16.gguf output-FP2.gguf Q2_0_ROCMFPX
llama-quantize source-BF16.gguf output-FP3.gguf Q3_0_ROCMFPX
llama-quantize source-BF16.gguf output-FP4.gguf Q4_0_ROCMFP4
llama-quantize source-BF16.gguf output-FP6.gguf Q6_0_ROCMFPX
llama-quantize source-BF16.gguf output-FP7.gguf Q7_0_ROCMFPX
llama-quantize source-BF16.gguf output-FP8.gguf Q8_0_ROCMFPX
```

These are GGUF model-weight formats. Runtime cache types are separate and are
not presented as ROCmFPX model formats.

## Ciru's Build and Validation Environment

Ciru develops and validates this work on NixOS. The current hosts are:

- `ciru` — NixOS 26.05;
- `dunamis` — NixOS 26.11, the primary model and benchmark host.

Release dependencies are pinned per runtime. TheRock ROCm and other toolchain
revisions belong in the matching tagged guide; this README does not substitute
generic distribution instructions for the environment in which Ciru produced
the results.

Clone Ciru's repository and select the line you intend to reproduce:

```bash
git clone https://github.com/ciru-ai/ROCmFPX.git
cd ROCmFPX

# Maintained integration line
git checkout main

# Or pin a published runtime
git checkout qwen3.8-activefpx-promptforge-v2.2
git checkout dualview-ornith-35b-v1
```

Follow the documentation for the selected tag. Do not mix a model artifact,
sidecars, and runtime from different release lines.

## Ciru Models and Recipe Work

The artifact-level recipe catalog lives in
[`docs/recipes/README.md`](docs/recipes/README.md). It keeps model identity,
architecture, topology, tensor policy, and measured BPW separate.

Representative Ciru/JCBTC releases documented by this work:

| Release | Research or recipe line |
|---|---|
| **Qwen3.8-27B CIRU ActiveFPX PromptForge** | ActiveFPX compact decode + specialized prompt compute views |
| **Ornith1.0-35B CIRU DualView FPX7+Q8 MTP** | DualView quality-max mixed Q7/Q8 target |
| [`Step-3.7-Flash ROCmFPX Q3 QualityPlus`](https://huggingface.co/jcbtc/Step-3.7-Flash-ROCmFPX-Q3-QualityPlus) | Step MoE Q3 QualityPlus |
| [`Chadrock 35B Ace Saber ROCmFP4/MoEQuality`](https://huggingface.co/jcbtc/chadrock-35b-ace-saber-rocmfp4-mtp) | Qwen MoE Strix Lean and MoEQuality |
| [`Qwable 27B ROCmFPX UltraQuality`](https://huggingface.co/jcbtc/Qwable-27B-Chadrock-ROCmFPX-ULTRAQUALITY-7.61BPW) | Qwen dense UltraQuality |
| [`Qwen3.6 35B Crown Halo Dynamic`](https://huggingface.co/jcbtc/qwen3.6-35b-a3b-crown-halo-mtp-dynamic) | Qwen MoE dynamic runtime work |
| [`Chadrock v2 27B ROCmFP6`](https://huggingface.co/jcbtc/Chadrockv2-Qwen3.6-27B-ROCmFP6-STRIX-QUALITY) | ROCmFP6 Strix Quality |
| [`Qwable 5 27B ROCmFP6`](https://huggingface.co/jcbtc/Qwable-5-27B-Chadrock-v2-ROCmFP6-QUALITY) | ROCmFP6 Quality |

The ranked-policy and recipe-map tooling makes the selected tensor policy
reviewable and reproducible instead of hiding it behind a filename.

## Validation and Evidence

Core checks and evidence paths include:

```text
scripts/check-rocmfp2-reference.sh
scripts/check-rocmfpx-reference.sh
scripts/sweep-rocmfpx-backend-ops.sh
scripts/check-rocmfpx-ranked-policy.sh
scripts/check-rocmfpx-preservation.py
scripts/check-release-recipe-map.py
tests/test-q7-q8-view.cpp
tests/test-raw-fp32-logits.cpp
```

Key research and maintenance records:

- [`docs/ROCmFPX-PRESERVATION-AUDIT.md`](docs/ROCmFPX-PRESERVATION-AUDIT.md)
- [`docs/ROCmFPX-EXPERIMENT.md`](docs/ROCmFPX-EXPERIMENT.md)
- [`docs/ROCmFP6-COMPETITIVE-EXPERIMENT.md`](docs/ROCmFP6-COMPETITIVE-EXPERIMENT.md)
- [`docs/ROCmFP6-DECODE-FOLLOWUP.md`](docs/ROCmFP6-DECODE-FOLLOWUP.md)
- [`docs/ROCmFPX-SERVING.md`](docs/ROCmFPX-SERVING.md)
- [`UPSTREAM.md`](UPSTREAM.md)

## Repository Map

- `ggml/rocmfpx/` — ROCmFPX formats, reference math, and Q7/Q8 view contract
- `ggml/rocmfp4/` — original ROCmFP4 family integrated into this tree
- `ggml/src/ggml-cuda/` — HIP/ROCm kernels, DualView, and PromptForge paths
- `ggml/src/ggml-vulkan/` — ROCmFPX Vulkan kernels and shaders
- `common/` and `tools/server/` — speculative execution and serving state
- `docs/recipes/` — Ciru release recipes and artifact mappings
- `docs/DUALVIEW*.md` — DualView design, results, and reproduction record

## Credit and Lineage

- [`charlie12345`](https://github.com/charlie12345) created the original
  ROCmFP4 format. Ciru retains that origin credit explicitly.
- Ciru created and maintains the work presented on this page, including the
  ROCmFPX extensions, kernels, runtime systems, DualView, ActiveFPX PromptForge,
  research records, and Ciru/JCBTC release recipes described above.
- This repository is based on `llama.cpp`; upstream authors and contributors
  retain their authorship and MIT license credit in `AUTHORS`, `LICENSE`, Git
  history, and `THIRD_PARTY_NOTICES.md`.

## License

MIT. See [`LICENSE`](LICENSE).
