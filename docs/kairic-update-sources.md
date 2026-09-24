# Kairic update source selection — 2026-09-24

Base: Ciru Kairic v1.2, `205a3e5f40e5542e2f2eb68e3d3f81f918b1d895`.
All changes are in a separate checkout and build. Existing release artifacts,
model weights, sidecars, and installed libraries are preserved.

Inspected public tips: Ciru `112629f1ed1a`, Charlie `fb08d7cdb670`,
llama.cpp `70596c4dcb`. The parent fork is `charlie12345/ROCmFPX`.

| Source | Decision and evidence |
| --- | --- |
| [llama.cpp 9bd4c09ea5](https://github.com/ggml-org/llama.cpp/commit/9bd4c09ea571a9020f30eeef169b552625b5b5a4) / Ciru main / Charlie 2e3eca5217 | Port reduction scratch-buffer synchronization and double buffering. Preserve Ciru's negative-infinity helper when adapting softmax. Prevents write/read races without changing reduction arithmetic. |
| [Charlie 3fca7f4bbc](https://github.com/charlie12345/ROCmFPX/commit/3fca7f4bbc3b076a1809bad7b94a4c53e4e54b36) | Import partial-thread-block softmax regression cases. |
| [Charlie b6a2db0eea](https://github.com/charlie12345/ROCmFPX/commit/b6a2db0eea1ae5e52cbaaa18e30ad6734f2c8ea7) | Import cublas/hipblas activation pointer fix: the callback receives F32 data regardless of the original tensor type. |
| [Charlie b728bb21d4](https://github.com/charlie12345/ROCmFPX/commit/b728bb21d449523524d171ac5ea9133533a19197) | Clear stale `t_h_pre_norm` in graph reset while preserving Ciru's extra graph outputs. |
| [Charlie 8e6277f855](https://github.com/charlie12345/ROCmFPX/commit/8e6277f855df2a27ce072525ed19f3adc4138c47) | Tested removal of HIP `-ffast-math` in the first full build. Served TG regressed about 12% on both frozen workloads, so this compiler-policy change is NOT promoted. Preserve the qualified Kairic compile policy and its explicit negative-infinity safeguards. Retain the strict build/patch for a separately justified correctness investigation. |
| [llama.cpp f466cfa38f](https://github.com/ggml-org/llama.cpp/commit/f466cfa38fac99e80a2aa4b58b3203b33872fe9c) | Skip inactive draft slots before dereferencing the optional result pointer. |
| [llama.cpp 2c6b141efb](https://github.com/ggml-org/llama.cpp/commit/2c6b141efb3b0868fd39d3cae73f69606e1d654c) | Adapt embedding/pooling reset to this older server's separate draft and native-MTP initialization paths. |
| Ciru September 10–13 local E1 patch | Final same-binary cold-prefill A-B-B-A at MCOL8 measured +2.85% PP (508.4 to 522.9 tok/s), with identical output and all 128 token IDs. Promote CK1 in the launcher; keep CK0 as explicit fallback and library default. Evidence is workload-specific. |
| Ciru September 10–13 local MCOL patch | Fix invalid-value/unsupported-shape grid truncation; remove unsafe eight-warp launch toggle; keep the 96-token decode boundary. GPU-vs-CPU boundary/snapshot cases passed. The final same-binary cold-prefill A-B-B-A measured +6.47% PP at 3,917 prompt tokens, with identical text and 128 token IDs and unchanged TG. Promote MCOL8 in the launcher; preserve MCOL1 as explicit fallback and library default. |
| Ciru Signal launcher / September Hermes handoff | Port environment isolation, force strict M65, and disable both argmax shortcuts plus draft backend sampling in compatibility mode. |

## Deliberately deferred

- E5 attention and PFK late-Q6 bypass: saved audit found changed recall output;
  the 21K stimulus repeated one word and the compounded speed claim double-counted
  changes. They are not evidence of a generally better runner.
- Charlie ROCmI4 quantization and experimental IU4 dispatch: different formats
  and execution paths, requiring new model artifacts and numerical qualification.
- Charlie RDNA3.5 MMQ geometry: plausible but lacks isolated Kairic evidence;
  do not replace our existing geometry on an untested speed claim.
- New upstream GDN normalization (`5fdfa62829`): changes target-model numerics;
  keep separate from this preservation-focused runner update.
- Broader MTP rollback/vision refactors: our branch has specialized checkpoint
  and M65 handling. These require a distinct integration and full state audit.
- CUDA-only synchronization, Spark prefetch/crossover, other architectures,
  model converters, and web UI changes: outside this gfx1151 Kairic runner.

The source audit alone establishes concrete bug mechanisms, not a measured
speed gain. Final build and serving evidence are in [the v1.3 release notes](kairic-edge-v1.3.md)
and [validation JSON](kairic-edge-v1.3-validation.json).

The final candidate excludes the strict-math compiler-policy change after the
first final gate. This narrowing preserves all targeted source fixes and leaves
the original compiler arithmetic policy intact. Do not attribute the first
candidate's benchmark rows to the narrowed final build.
