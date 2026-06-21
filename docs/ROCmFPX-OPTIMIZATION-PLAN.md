# ROCmFPX Optimization Plan

Last updated: 2026-06-20

## Goal

Optimize ROCmFPX for Strix Halo. Do not broaden scope to Metal or unrelated
hardware. Treat changes as useful only when they produce clear speed, quality,
reproducibility, or workflow evidence for this target.

## Current Proven Work

- FP3 served MTP can be much faster than plain decode when acceptance is high.
  The strongest local FP3 rows use `Vulkan0`, `f16/f16` target and draft KV,
  `b2048/u512`, `draft-mtp`, `n_max=4`, `p_min=0.75`, and disabled draft
  backend sampling.
- Request-scoped speculative controls are implemented in the server path:
  `speculative.n_max`, `speculative.n_min`, and `speculative.p_min`.
  A request may lower the active draft policy without restarting the server,
  while the CLI `--spec-draft-n-max` remains the allocation cap.
- The FP3 context ladder showed the short-context profile remains strong
  through roughly 32k prompt tokens, then prompt processing and draft
  acceptance taper at 64k and 128k.
- FP3 scale/imatrix quality work has small-model evidence only. Do not claim
  35B quality transfer without a matched 35B quality run.

## Workstreams

### 1. Dynamic Context Drafting Profiles

Purpose: choose draft policy from actual prompt length instead of using one
global MTP setting.

Candidate tiers:

| Prompt range | Starting policy | Reason |
|---:|---|---|
| up to ~32k | `n_max=4`, `p_min=0.75` | Proven strong decode and high acceptance on FP3 35B. |
| 64k+ | `n_max=2`, `p_min=0.0` | 75k sweep improved decode from 53.67 to 65.93 tok/s. |
| 128k+ | `n_max=2`, `p_min=0.0` | Best measured 128k wall time at gen512, gen1024, and gen2048. |

Immediate test matrix:

- Done: `n_max=4`, `p_min=0.75` - current short/medium winner and baseline.
- Done: `n_max=4`, `p_min=0.0` - too much wasted draft work at 75k.
- Done: `n_max=3`, `p_min=0.0` - improves over n4 but loses to n2.
- Done: `n_max=2`, `p_min=0.0` - best 75k decode row, best 128k wall row,
  including the longer gen2048 check.
- Done: `n_max=1`, `p_min=0.0` - near-tie at 75k, best 128k decode row.
- Done: `n_max=2`, `p_min=0.25` - tied/slightly slower than `n2,p0`.
- Done: dynamic `n_max=0`, `p_min=0.0` - not useful at 128k/gen1024 because
  PP was unchanged while decode regressed.

Promotion rule:

- Compare served API rows only.
- Keep prompt text, model, context allocation, KV, batch/ubatch, backend, and
  generation length fixed.
- Choose by total wall time first when the target use case is interactive
  long-context response, then decode tok/s, TTFP, and acceptance.
- Re-run the winner at least twice before calling it a profile default.

Current long-context results:

| Prompt tokens | Policy | Decode tok/s | Total s | Accepted |
|---:|---|---:|---:|---:|
| 75023 | `n4,p0.75` | 53.67 | 129.06 | 196 / 220 |
| 75023 | `n2,p0` | 65.93 | 126.62 | 299 / 423 |
| 75023 | `n1,p0` | 65.54 | 126.37 | 230 / 280 |
| 128278 | `n4,p0.75` | 46.08 | 282.73 | 245 / 287 |
| 128278 | `n2,p0` | 51.80 | 281.07 | 295 / 431 |
| 128278 | `n1,p0` | 53.21 | 281.87 | 233 / 278 |
| 128278 / gen1024 | `n2,p0` | 53.24 | 290.30 | 604 / 837 |
| 128278 / gen1024 | `n1,p0` | 53.30 | 291.00 | 468 / 554 |
| 128278 / gen1024 | dynamic `n0,p0` | 37.25 | 298.96 | 0 / 0 |
| 128278 / gen2048 | dynamic `n2,p0` | 57.27 | 307.64 | 1272 / 1549 |
| 128278 / gen2048 | dynamic `n1,p0` | 54.17 | 308.96 | 969 / 1077 |

The gen2048 rows were run on one cap4 MTP server with per-request overrides.
That confirms the dynamic path does not require restarting or serving a second
profile to lower long-context draft depth.

Dynamic drafting should be selected at request admission. Count prompt tokens
before `/completion`, then attach request-level settings such as
`speculative.n_max` and `speculative.p_min`. Mid-job switching is not part of
the current plan because request-level selection is simpler and already matches
the server patch.

The helper `scripts/rocmfpx-draft-profile.py` emits the request JSON for the
current policy:

```bash
scripts/rocmfpx-draft-profile.py --prompt-tokens 128278
scripts/rocmfpx-draft-profile.py \
  --base-url http://127.0.0.1:18231 \
  --api-key local \
  --prompt-file /path/to/prompt.txt
```

The server should still start with a high enough allocation cap, for example
`--spec-draft-n-max 4`; the helper only lowers request settings.

Serving instructions and request examples are in
[`docs/ROCmFPX-SERVING.md`](ROCmFPX-SERVING.md).

### 2. FP3 Speed PRs

Keep only material patches:

- Generic request-level speculative tuning, with tests for request clamping and
  generation settings visibility.
- Opt-in speed profile scripts and docs that reproduce measured rows.
- Kernel/shader changes only when backed by same-harness speed wins and no
  quality regression.

Avoid:

- Formatting-only or style-only PRs.
- Changing default behavior from a one-off benchmark.
- Promoting CPU-only optimizations for GPU inference unless profiling shows CPU
  overhead on the served path.

### 3. FP4 Transfer

What transfers:

- Request-level speculative controls are quant agnostic and should help FP4
  serving experiments.
- The served MTP benchmark shell transfers: one slot, metrics, strict no-cache
  tests, fixed prompts, and draft counters.
- A sweep harness that varies `n_max` and `p_min` is useful for FP4.

What does not transfer automatically:

- The FP3 `p_min=0.75` winner is not the FP4 winner by default.
- FP3 bit packing and shader changes are FP3-specific until separately proven
  on FP4.
- FP4 27B and 35B A3B have different historical MTP preferences; keep their
  profiles separate.
- FP4 should not inherit the FP3 long-context `n2,p0` policy by assumption.
  A quick FP4 35B MTP run on 2026-06-20 favored `n4,p0`: at 3946 prompt tokens
  `n4,p0` reached 141.77 decode tok/s versus 119.29 for `n2,p0`, and the
  intentionally limited 75023-token check had `n4,p0` at 79.23 versus 77.11
  for `n3,p0`.
- ACE/SABER FP4 also benefits from the same request-level controls, but the
  best quick setting differed from the first FP4 run. On a 3946-token text
  prompt, one slot, no prompt cache, f16/f16 KV, and `b2048/u512`,
  `n4,p0.25` reached 143.08 decode tok/s at gen512 and repeated at
  141.77 decode tok/s at gen2048. The no-draft baselines were 72.57 and
  72.04 decode tok/s. The old production-style `b8192/u2048` shape had worse
  prompt processing on this prompt: 708-822 PP tok/s versus roughly
  1064-1088 PP tok/s on `b2048/u512`.

Likely FP4 starting points from existing evidence:

- 27B ROCmFP4: `n_max=4`, low `p_min`, q4 target/draft KV in promoted guards.
- 35B A3B ROCmFP4: more conservative draft settings; test against the local
  profile rather than copying FP3 `p_min=0.75`.
- For patched request-level experiments, use a separate cap4 sweep profile
  instead of overwriting production FP4 profiles. The current Qwen3.6 35B FP4
  cap4 sweep profile is
  `/home/crown/machine-setup/model-profiles/qwen3.6-35b-a3b-mtp-chadrock-rocmfp4-vulkan-cap4-sweep.env`.
  ACE/SABER now has the same treatment at
  `/home/crown/machine-setup/model-profiles/chadrock-qwen36-35b-ace-saber-rocmfp4-vulkan-cap4-text32k.env`.
  That profile is intentionally text-only and leaves the known-good production
  d2 vision profile unchanged.

### 4. Quality Guardrails

- Do not present perplexity/quality wins from a small model as a large-model
  guarantee.
- Keep speed and quality evidence separate.
- For any public FP3/FP4 profile, add behavior/no-thinking checks separately
  from throughput rows.

### 5. Evidence Hygiene

- Headline MTP speed only from served API or CLI guard rows with draft counters.
- Record model path, backend, context allocation, prompt tokens, generated
  tokens, KV, batch/ubatch, draft settings, prompt cache state, TTFP, total
  time, prompt tok/s, decode tok/s, and draft accepted/generated.
- Write a short human summary next to every benchmark SQLite/JSONL run.
- Keep PR descriptions tied to specific rows, not broad claims.

## Current Next Steps

1. Run a quality/behavior guard against the ACE/SABER text cap4 profile before
   treating it as anything beyond a speed profile.
2. Repeat the winning long-context request policy to estimate variance.
3. Run the added server/API speculative-knob test under an HTTPS-capable test
   build before submitting the generic tuning PR.
4. Wire the request-profile helper into the preferred serving client or
   benchmark command path.
5. Create an FP4 sweep script or doc arm that reuses the generic request knobs
   but starts from FP4-specific historical settings.
