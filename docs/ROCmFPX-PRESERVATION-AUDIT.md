# ROCmFPX Preservation Audit

This audit records the feature contracts that must survive upstream `llama.cpp`
synchronization. The audited pre-sync baseline is `9d91bc16`; the current-tree
baseline at the time of the audit is `8ffb2cd5`.

## Preserved or restored contracts

| Contract | Pre-sync source | Current status | Automated guard |
| --- | --- | --- | --- |
| Importance-matrix weighted FP3/FP6 scale search | `864f263c` | Preserved | weighted-search symbols |
| FP3 Vulkan and FP6 Vulkan decode | `553504bd`, `c3342eea` | Preserved; cooperative-matrix registrations restored | registration count per custom type |
| TurboQuant KV and Vulkan turbo3/turbo4 | `d859c9e6`, `d0141e86` | Preserved | runtime/shader assets |
| Request-level MTP controls | `c226d1fe` | Restored and forward-ported to the current schema | schema, response, and effective-control symbols |
| Dynamic drafting policy and feedback wrapper | `7be63047`, `fb717944`, `ac7e2599` | Restored | source markers plus wrapper smoke |
| FP3/FP6 decoded-value hot paths | `547321dc` | Preserved | hot-path symbols |
| Independent on-device checkpoint snapshots | `c823b4cb` | Restored; restore mismatch now fails safely | storage API and server fallback markers |
| EAGLE3/Qwen speculative state and Step MTP heads | `caa16f4c`, `7a4f009e`, `11d76c24` | Preserved across current model refactors | model/speculative-state symbols |
| ROCm 7.2 negative-infinity handling | `5d04ce30` | Missing call sites restored | affected CUDA/HIP sources reject literal `-INFINITY` |
| Hybrid M-RoPE MTP batch acceptance | `e0eefaf2` via merge `2864b1a8` | Restored | batch-condition marker |
| DualView Q7/Q8 runtime and tests | `9d91bc16` | Preserved | runtime and test assets |
| Raw FP32 all-logits diagnostic | pre-sync test tooling | Restored and registered as a manual build target | source and CMake registration |
| Ranked quantization-policy fixtures | pre-sync test tooling | Restored | fixture presence and smoke test |
| FP3 MTP speed profile | pre-sync serving tooling | Restored | launcher presence and target wrapper marker |

Run the source-level contract audit with:

```bash
python3 scripts/check-rocmfpx-preservation.py
```

It is also invoked by `scripts/check-rocmfpx-all.sh` and by the ROCmFPX GitHub
Actions reference job. The workflow watches the common, context, server, test,
CUDA, and Vulkan paths that implement these contracts; this prevents a future
sync from bypassing the audit merely because `ggml/rocmfpx/` itself was not
changed.

## Superseded material

The old monolithic DeepSeek V4 converter wrappers were not restored. They call
removed `--deepseek4-max-layers` and `--deepseek4-include-mtp` options and would
publish broken instructions in the current modular converter. The historical
port report is explicitly marked archival. The current generic converter MTP
capability guard supersedes the old model-specific converter-option guard.

The former WebUI subtree was likewise replaced by the current `tools/ui`
implementation. Those deleted paths are upstream structure changes, not lost
ROCmFPX runtime behavior.

## Validation boundary

Source guards, shell/Python smoke checks, the release recipe map, a local
CPU-only MSVC build of `llama-server`, `test-backend-ops`, and
`test-raw-fp32-logits`, and the CPU backend-op suite were used for this audit.
GPU runtime validation remains a separate explicit gate.
