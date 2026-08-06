# Upstream synchronization

ROCmFPX `main` is built directly on official
[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) history. Ciru's
ROCmFPX formats, kernels, tests, scripts, and documentation are maintained as a
focused layer on top of that source.

## Current baseline

- Upstream commit: `15586e2d7165570fb3aa7c26e0d442e289ef69de`
- Upstream state: tag `b10297` plus one commit
- Upstream commit date: 2026-08-06
- Includes the upstream decode overhaul through `b10218`

The upstream commit is an ancestor of the published Ciru commit, rather than a
copied source snapshot. Verify that relationship after cloning:

```bash
git merge-base --is-ancestor 15586e2d7165570fb3aa7c26e0d442e289ef69de main
git merge-base --is-ancestor de699957 main # upstream b10218
```

A successful command exits with status zero.

## Attribution

Upstream llama.cpp commits retain their original authorship. ROCmFPX-specific
commits and public project materials are authored and published by Ciru. See
[`docs/UPSTREAM-ATTRIBUTION.md`](docs/UPSTREAM-ATTRIBUTION.md) for additional
credit and licensing information.
