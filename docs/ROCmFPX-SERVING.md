# ROCmFPX MTP Serving

This guide covers the request-level MTP controls added for ROCmFPX serving:

```text
speculative.n_max
speculative.n_min
speculative.p_min
```

These fields let a request lower the active draft policy without restarting the
server. The server startup value for `--spec-draft-n-max` is still the
allocation cap, so start the server at the highest draft depth you plan to test.

## Build

For Strix Halo, build the server with the normal ROCmFP4/ROCmFPX script:

```bash
env JOBS=16 scripts/build-strix-rocmfp4-mtp.sh llama-server llama-bench
```

The served MTP helper expects this binary by default:

```text
build-vulkan-server-noui/bin/llama-server
```

Set `BUILD_DIR` or `BIN` if your build directory is different:

```bash
BUILD_DIR="$PWD/build-strix-rocmfp4" \
  scripts/run-rocmfpx-fp3-mtp-server-speed-profile.sh
```

## Start The FP3 Speed Profile

The helper script starts a single-slot OpenAI-compatible server with metrics,
MTP enabled, prompt-cache isolation for benchmarking, and request-level draft
overrides available:

```bash
MODEL=/path/to/model-Q3_0_ROCMFPX.gguf \
PORT=18180 \
CTX_SIZE=135168 \
SPEC_DRAFT_N_MAX=4 \
SPEC_DRAFT_N_MIN=0 \
SPEC_DRAFT_P_MIN=0.75 \
SPEC_DRAFT_P_SPLIT=0.10 \
scripts/run-rocmfpx-fp3-mtp-server-speed-profile.sh
```

Useful defaults in that script:

```text
DEVICE=Vulkan0
BATCH_SIZE=2048
UBATCH_SIZE=512
CACHE_TYPE_K=f16
CACHE_TYPE_V=f16
THREADS=16
THREADS_BATCH=32
STRICT_BENCH=1
```

`STRICT_BENCH=1` disables prompt-cache reuse and sets slot prompt similarity to
zero so repeated benchmark rows are easier to compare. For interactive serving,
set `STRICT_BENCH=0` if you want normal prompt cache behavior.

## Per-Request Overrides

Use the request keys on `/completion`:

```bash
curl -sS http://127.0.0.1:18180/completion \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Write a concise technical note about ROCmFPX MTP serving.",
    "n_predict": 512,
    "temperature": 0,
    "ignore_eos": true,
    "speculative.n_max": 2,
    "speculative.n_min": 0,
    "speculative.p_min": 0.0
  }'
```

The response `generation_settings` should echo the effective values. If
`speculative.n_max` is higher than the server cap, it is clamped to the cap.
`speculative.n_min` is clamped to `0..n_max`, and `speculative.p_min` is
clamped to `0.0..1.0`.

OpenAI chat-compatible requests use the same keys in the top-level payload:

```bash
curl -sS http://127.0.0.1:18180/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "rocmfpx-fp3-mtp-speed",
    "messages": [
      {"role": "user", "content": "Summarize the request-level MTP knobs."}
    ],
    "max_tokens": 512,
    "temperature": 0,
    "speculative.n_max": 2,
    "speculative.n_min": 0,
    "speculative.p_min": 0.0
  }'
```

## Dynamic Prompt-Length Policy

The FP3 long-context sweep currently supports this request-admission policy:

| Prompt tokens | Request policy |
|---:|---|
| `< 49152` | `speculative.n_max=4`, `speculative.n_min=0`, `speculative.p_min=0.75` |
| `>= 49152` | `speculative.n_max=2`, `speculative.n_min=0`, `speculative.p_min=0.0` |

Use the helper to emit the JSON fields:

```bash
scripts/rocmfpx-draft-profile.py --prompt-tokens 128278
```

Or let it count tokens through a running server:

```bash
scripts/rocmfpx-draft-profile.py \
  --base-url http://127.0.0.1:18180 \
  --prompt-file /path/to/prompt.txt
```

Attach the emitted JSON fields to the request before sending `/completion` or
`/v1/chat/completions`. The current implementation chooses policy at request
admission; it does not switch draft settings mid-generation.

## FP4 Transfer

The request-level controls are quant-agnostic and can be used for ROCmFP4 MTP
serving, but FP3 winners should not be copied blindly to FP4 profiles.

Current measured FP4 starting points:

| Model family | Starting point |
|---|---|
| Qwen3.6 35B A3B ROCmFP4 | `n_max=4`, `n_min=0`, `p_min=0.0`, `b2048/u512`, `f16/f16` target and draft KV |
| ACE/SABER 35B ROCmFP4 text profile | `n_max=4`, `n_min=0`, `p_min=0.25`, `b2048/u512`, `f16/f16` target and draft KV |
| Qwable 5 27B Coder ROCmFP4 | `n_max=6`, `n_min=0`, `p_min=0.0`, `p_split=0.20`, target KV `q8`, draft KV `f16`, `b2048/u512` |

For FP4 tests, start the server with a cap at least as high as the deepest
candidate:

```bash
SPEC_DRAFT_N_MAX=6 \
SPEC_DRAFT_N_MIN=0 \
SPEC_DRAFT_P_MIN=0.0 \
SPEC_DRAFT_P_SPLIT=0.20 \
MODEL=/path/to/model-Q4_0_ROCMFP4.gguf \
scripts/run-rocmfpx-fp3-mtp-server-speed-profile.sh
```

Then vary `speculative.n_max` and `speculative.p_min` per request.

## Validation

Before reporting a serving result, record:

- model path and alias
- server binary path and commit
- backend device
- context allocation
- prompt tokens and generated tokens
- target and draft KV cache types
- batch and ubatch
- startup MTP cap and per-request speculative fields
- prompt-cache setting
- decode tok/s, prompt tok/s, TTFP, total time
- draft accepted and draft generated counters

For quick server checks:

```bash
curl -sS http://127.0.0.1:18180/health
curl -sS http://127.0.0.1:18180/props | jq '.default_generation_settings'
curl -sS http://127.0.0.1:18180/metrics | head
```

Use served API rows or a CLI guard with draft counters for headline MTP speed.
Do not use standalone `llama-bench` TG as the headline for MTP serving.
