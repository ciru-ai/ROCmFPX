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

This repository branch is the CHADROCK/llama.cpp runner used for the reproduced
serving rows. Clone the branch, then build `llama-server` from this tree:

```bash
git clone https://github.com/ciru-ai/ROCmFPX.git
cd ROCmFPX
git checkout rocmfpx-mtp-serving-controls
```

For Strix Halo, build the runner with the normal ROCmFP4/ROCmFPX script:

```bash
env JOBS=16 scripts/build-strix-rocmfp4-mtp.sh llama-server llama-bench
```

The runner binary is:

```text
build-strix-rocmfp4/bin/llama-server
```

Set `BUILD_DIR` or `BIN` if your build directory is different. The helper
scripts only wrap this locally built `llama-server`; they do not use a separate
runtime.

```bash
BUILD_DIR="$PWD/build-strix-rocmfp4" \
  scripts/run-rocmfpx-mtp-server.sh
```

## Start A CHADROCK MTP Runner

The helper script starts a single-slot OpenAI-compatible server with metrics,
MTP enabled, prompt-cache isolation for benchmarking, and request-level draft
overrides available:

```bash
MODEL=/path/to/model.gguf \
PORT=18180 \
CTX_SIZE=135168 \
SPEC_DRAFT_N_MAX=4 \
SPEC_DRAFT_N_MIN=0 \
SPEC_DRAFT_P_MIN=0.75 \
SPEC_DRAFT_P_SPLIT=0.10 \
scripts/run-rocmfpx-mtp-server.sh
```

Useful defaults in that script:

```text
DEVICE=Vulkan0
BATCH_SIZE=2048
UBATCH_SIZE=512
CACHE_TYPE_K=f16
CACHE_TYPE_V=f16
CACHE_TYPE_K_DRAFT=f16
CACHE_TYPE_V_DRAFT=f16
THREADS=16
THREADS_BATCH=32
STRICT_BENCH=1
```

`STRICT_BENCH=1` disables prompt-cache reuse and sets slot prompt similarity to
zero so repeated benchmark rows are easier to compare. For interactive serving,
set `STRICT_BENCH=0` if you want normal prompt cache behavior.

## Recreate The 35B ROCmFP4 Rows

The 35B ~140 tok/s rows were produced by the patched CHADROCK runner with one
server slot, no prompt cache, Vulkan0, and served `/completion` requests.

Qwen3.6 35B A3B ROCmFP4 starting point:

```bash
MODEL=/path/to/Qwen3.6-35B-A3B-MTP-BF16-to-ROCmFP4-STRIX_LEAN.gguf \
ALIAS=chadrock-35b-rocmfp4-cap4 \
PORT=18180 \
CTX_SIZE=135168 \
DEVICE=Vulkan0 \
SPEC_DRAFT_DEVICE=Vulkan0 \
BATCH_SIZE=2048 \
UBATCH_SIZE=512 \
CACHE_TYPE_K=f16 \
CACHE_TYPE_V=f16 \
CACHE_TYPE_K_DRAFT=f16 \
CACHE_TYPE_V_DRAFT=f16 \
SPEC_DRAFT_N_MAX=4 \
SPEC_DRAFT_N_MIN=0 \
SPEC_DRAFT_P_MIN=0.0 \
SPEC_DRAFT_P_SPLIT=0.10 \
STRICT_BENCH=1 \
scripts/run-rocmfpx-mtp-server.sh
```

ACE/SABER 35B ROCmFP4 text winner:

```bash
MODEL=/path/to/Qwen3.6-35B-A3B-NSC-ACE-SABER-MTP-F16-to-ROCmFP4-STRIX_LEAN.gguf \
ALIAS=chadrock-35b-ace-saber-rocmfp4-cap4 \
PORT=18180 \
CTX_SIZE=32768 \
DEVICE=Vulkan0 \
SPEC_DRAFT_DEVICE=Vulkan0 \
BATCH_SIZE=2048 \
UBATCH_SIZE=512 \
CACHE_TYPE_K=f16 \
CACHE_TYPE_V=f16 \
CACHE_TYPE_K_DRAFT=f16 \
CACHE_TYPE_V_DRAFT=f16 \
SPEC_DRAFT_N_MAX=4 \
SPEC_DRAFT_N_MIN=0 \
SPEC_DRAFT_P_MIN=0.25 \
SPEC_DRAFT_P_SPLIT=0.10 \
NO_MMPROJ=1 \
STRICT_BENCH=1 \
scripts/run-rocmfpx-mtp-server.sh
```

Use request payloads with `n_predict=512` or `max_tokens=512`,
`temperature=0`, `ignore_eos=true`, and:

```json
{
  "speculative.n_max": 4,
  "speculative.n_min": 0,
  "speculative.p_min": 0.25
}
```

The measured ACE/SABER row was `143.08` decode tok/s at gen512 and repeated at
`141.77` decode tok/s at gen2048 on the 3946-token prompt.

## Recreate The 27B ROCmFP4 Row

The >50 tok/s 27B row used the same patched runner, but a deeper cap and
separate target/draft KV types:

```bash
MODEL=/path/to/Qwable-5-27B-Coder-BF16-to-ROCmFP4-STRIX_LEAN.gguf \
ALIAS=qwable5-27b-coder-rocmfp4-cap6 \
PORT=18180 \
CTX_SIZE=32768 \
DEVICE=Vulkan0 \
SPEC_DRAFT_DEVICE=Vulkan0 \
BATCH_SIZE=2048 \
UBATCH_SIZE=512 \
CACHE_TYPE_K=q8_0 \
CACHE_TYPE_V=q8_0 \
CACHE_TYPE_K_DRAFT=f16 \
CACHE_TYPE_V_DRAFT=f16 \
SPEC_DRAFT_N_MAX=6 \
SPEC_DRAFT_N_MIN=0 \
SPEC_DRAFT_P_MIN=0.0 \
SPEC_DRAFT_P_SPLIT=0.20 \
STRICT_BENCH=1 \
scripts/run-rocmfpx-mtp-server.sh
```

Use this request policy:

```json
{
  "speculative.n_max": 6,
  "speculative.n_min": 0,
  "speculative.p_min": 0.0
}
```

The measured 27B rows on the 3946-token prompt were `52.67` decode tok/s,
`53.39` decode tok/s on an exact repeat, and `52.08` decode tok/s at gen2048.
The no-MTP served control was `12.78` decode tok/s.

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
    "model": "chadrock-35b-ace-saber-rocmfp4-cap4",
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

## Measured Starting Points

Current measured ROCmFP4 starting points:

| Model family | Starting point |
|---|---|
| Qwen3.6 35B A3B ROCmFP4 | `n_max=4`, `n_min=0`, `p_min=0.0`, `b2048/u512`, `f16/f16` target and draft KV |
| ACE/SABER 35B ROCmFP4 text profile | `n_max=4`, `n_min=0`, `p_min=0.25`, `b2048/u512`, `f16/f16` target and draft KV |
| Qwable 5 27B Coder ROCmFP4 | `n_max=6`, `n_min=0`, `p_min=0.0`, `p_split=0.20`, target KV `q8_0`, draft KV `f16`, `b2048/u512` |

For MTP tests, start the server with a cap at least as high as the deepest
candidate:

```bash
SPEC_DRAFT_N_MAX=6 \
SPEC_DRAFT_N_MIN=0 \
SPEC_DRAFT_P_MIN=0.0 \
SPEC_DRAFT_P_SPLIT=0.20 \
MODEL=/path/to/model-Q4_0_ROCMFP4.gguf \
scripts/run-rocmfpx-mtp-server.sh
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
