# Ornith 1.0 35B CIRU DualView research record

This document records the retained model recipe and the public-grade results
behind `Ornith1.0-35b-CIRU-DUALVIEW-FPX7+Q8-MTP`.

## Artifact identity

Integrated target plus MTP:

- Size: 33,536,832,416 bytes (31.23 GiB)
- SHA-256: `7834fb92d451235c123973ba40eccea156db28af8d42ae13178c7efcc4d0177a`
- 753 tensors
- 35.51B integrated parameters
- 733 byte-identical target tensors
- 20 appended byte-identical official Q8 MTP tensors

Quality-max target inside the integrated artifact:

- Size: 32,638,875,232 bytes
- SHA-256: `7f2770fd4f9166c968b50620803740a18e1d91d3dff89c9b0abf50d8e7c9c620`
- 362 `Q7_0_ROCMFPX` tensors
- 70 canonical `Q8_0` precision-island tensors
- 301 F32 tensors
- Approximately 7.533 effective bits per target parameter

The 70 promoted target tensors are the 30 Mamba `ssm_out.weight` projections
and Q/K/V/O for all ten full-attention blocks. They cover 524,288,000 weights.
Every selected payload is byte-identical to the pinned official Q8 donor; every
other target payload is byte-identical to the Q7S8 base.

The integrated Q8 MTP head adds 20 `blk.40.*` tensors from the compatible
official Qwen3.6 35B-A3B MTP donor. The target is still the Ornith model and
remains byte-identical.

## Numerical search

All arms were scored against the saved official-Q8 probability stream over 48
WikiText-2 chunks: 24,576 input tokens, `n_ctx=512`, `n_seq=4`, batch 2048,
F16 KV, full ROCm offload, seed 42.

| Arm | Mean KLD | Change vs Q7S8 | PPL | RMS delta-p | Same top |
|---|---:|---:|---:|---:|---:|
| Official pure Q8 reference | 0 | — | 8.432309 | 0 | 100% |
| Q7S8 baseline | 0.011560 | — | 8.475987 | 2.869% | 95.041% |
| Core Q8 | 0.008789 | -23.97% | 8.465803 | 2.523% | 96.111% |
| Gate experts Q8 | 0.011967 | +3.52% | 8.470177 | 2.952% | 95.139% |
| Up experts Q8 | 0.012153 | +5.13% | 8.490332 | 2.919% | 95.196% |
| Down experts Q8 | 0.011462 | -0.85% | 8.465847 | 2.980% | 95.106% |
| Core + down experts Q8 | 0.008571 | -25.86% | 8.444275 | 2.524% | 96.193% |
| Shared experts Q8 | 0.011683 | +1.06% | 8.488710 | 2.947% | 95.147% |
| Full Mamba Q8 | 0.009443 | -18.31% | 8.472397 | 2.585% | 95.711% |
| Full attention Q8 | 0.011026 | -4.62% | 8.478844 | 2.754% | 95.286% |
| Mamba state Q8 | 0.011655 | +0.82% | 8.483718 | 2.846% | 94.975% |
| Mamba input Q8 | 0.011142 | -3.62% | 8.475264 | 2.819% | 95.229% |
| Mamba output Q8 | 0.010426 | -9.81% | 8.463171 | 2.722% | 95.425% |
| **Mamba output + full attention Q8** | **0.009796** | **-15.26%** | **8.461721** | **2.595%** | **95.621%** |

The retained target reduces KLD 15.26% and RMS probability error 9.55% versus
the original Q7S8 model. It is also 32.57% lower in KLD than the tested Q6_K
control. The scalar Q6 perplexity happens to be slightly closer to Q8, while
its distribution fidelity is materially worse; neither metric should be
collapsed into a blanket quality claim.

The search proves that nominal precision is not monotonic with graph-level
quality. Promoting routed gate/up/shared tensors and small Mamba state
projections made KLD worse. Precision islands must be selected as coupled graph
motifs.

## Target-only speed

Matched protocol: Radeon 8060S / `gfx1151`, ROCm0, full offload, flash
attention, F16/F16 KV, batch 2048, microbatch 512, 16 threads, PP4096 and
TG256. The retained target and Q7S8 rows use three independent clean loads and
nine native samples.

| Artifact | PP4096 tok/s | TG256 tok/s | Notes |
|---|---:|---:|---|
| **Quality-max DualView target** | **1,236.156** | **48.049** | retained |
| Original Q7S8 DualView | 1,202.550 | 48.719 | nine samples |
| Official pure Q8_0 | 1,184.626 | 43.470 | matched control |
| AMD-safe UD-Q8_K_XL | 1,071.744 | 42.679 | three samples |
| Actual Q6_K | 1,110.903 | 52.458 | three samples |

Retained target ranges:

- PP4096: 1,219.43–1,249.91 tok/s
- TG256: 47.783–48.233 tok/s

Compared with official pure Q8_0, the retained target is 4.35% faster in
PP4096 and 10.53% faster in native TG256 while its target file is about 11.5%
smaller. Compared with AMD-safe Q8_K_XL, it is 15.34% faster in prefill and
12.58% faster in decode.

This is a speed/quality/residency trade, not free compression. The retained
target measured 68,202,450,944 bytes of GTT because its Q8 compute views are
materialized.

## Integrated MTP curves

Effective generation throughput is measured from served, target-verified MTP
requests—not standalone `llama-bench` TG.

| Prompt | Profile | Prefill tok/s | Effective generation tok/s |
|---:|---|---:|---:|
| ~1K | target only | 1,112.85 | 46.71 |
| ~1K | depth 6 | 964.85 | **84.40** |
| ~1K | depth 7 | 962.68 | 81.91 |
| 16K | target only | 1,036.55 | 44.08 |
| 16K | depth 7 | 856.14 | **78.86** |
| 32K | target only | 867.80 | 41.36 |
| 32K | depth 7 | 739.37 | **73.70** |
| 64K | target only | 652.01 | 37.05 |
| 64K | depth 7 | 563.32 | **57.82** |

Depth 6 improves the short-context mean by 80.7%. Depth 7 improves effective
generation by 78.2% at 32K and 56.0% at 64K. Workload acceptance matters: the
64K depth-7 split was 39.18 tok/s on prose and 76.46 tok/s on code.

The fastest decode profile is not automatically the fastest request. For the
measured 64K prompt plus 256-token continuation:

| Profile | Mean wall time |
|---|---:|
| Target only | **107.50 s** |
| Q8 MTP, depth 7 | 121.35 s |

MTP was 12.89% slower end to end because its prefill overhead exceeded the
time saved on a short answer. Measured-rate extrapolation puts break-even near
1,072 output tokens for the code prompt and 11,259 for prose, but those
thresholds have not been confirmed with a long-generation wall-time sweep.

The official Q8 MTP head was retained after Q7 and Q5 head ablations. Q7
materialized its own Q8 compute view and used more GTT without winning decode.
Q5 briefly led a short probe, then lost the 64K+1024 confirmation:

| Head, 64K code + 1024 output | Prefill tok/s | Effective generation tok/s |
|---|---:|---:|
| **Official Q8 head** | **576.32** | **77.83** |
| Q5_K head | 556.07 | 77.20 |

## Agent behavior

These are small confirmation sets, not statistical rankings.

| Suite | Original Q7S8 | Quality-max target | Integrated MTP depth 7 |
|---|---:|---:|---:|
| HermesAgent-20 | 79.17 mean | **84.00 mean** (85/88/79) | 83.00 mean (87/78/84) |
| Tool-Eval hard | 76.67 mean | 75.67 mean (70/70/87) | 87 in one pass |

The MTP Tool-Eval result equals the best target-only trial, but one pass is not
enough to claim a stable uplift. Exception scans were clean in the retained
Hermes runs.

## Runtime correctness

Repeated MTP requests originally exposed a cross-request state leak:
target/draft KV was cleared on slot reuse, but external `pending_h` state was
not. The release resets that carry state whenever a fresh sequence begins at
position zero. Repeated corrected workloads then produced one content hash per
arm.

This validates uncached position-zero requests. Cached-prefix restoration at a
nonzero position still needs explicit MTP state serialization or
reconstruction and is not claimed.

## Reproducibility identity

- ROCmFPX base revision: `5d04ce30c831879424a43f01aa2cda440000e271`
- Tested runtime snapshot: `9d91bc168`
- GPU target: `gfx1151`
- Validation toolchain: TheRock `7.15.0a20260718`
- DualView policy: `GGML_ROCM_GFX1151_Q7_Q8_VIEW=no-output`
- Target/draft KV: F16
- Full ROCm offload, flash attention, one slot
- Prompt cache and context shift disabled for the validated MTP path

The public model card links back to this document so benchmark headlines remain
auditable rather than detached from their protocol and caveats.
