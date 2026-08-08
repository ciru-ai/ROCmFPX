#!/usr/bin/env python3
"""Source-level guard for Ciru ROCmFPX features carried across upstream syncs."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def read(path: str) -> str:
    full = ROOT / path
    if not full.is_file():
        raise AssertionError(f"missing required file: {path}")
    return full.read_text(encoding="utf-8")


def require(path: str, *needles: str) -> None:
    text = read(path)
    for needle in needles:
        if needle not in text:
            raise AssertionError(f"{path}: missing preservation marker: {needle}")


def main() -> int:
    # Quantizer accuracy and hot paths.
    require(
        "ggml/rocmfpx/rocmfpx.c",
        "rocmfpx_fp3_block_weighted_mse_for_scale",
        "rocmfpx_fp6_block_weighted_mse_for_scale",
        "rocmfpx_fp3_decoded_mag",
        "rocmfpx_fp6_decoded_value",
    )
    require(
        "ggml/src/ggml-cuda/dequantize.cuh",
        "rocmfpx_get_fp3_code_cuda",
        "rocmfpx_get_fp6_code_cuda",
    )
    require("ggml/rocmfpx/rocmfpx_hip_codebook.cuh", "rocmfpx_pack4_fp3_codes", "rocmfpx_pack4_fp6_codes")

    # TurboQuant and DualView runtime assets.
    for path in (
        "ggml/src/ggml-vulkan/vulkan-shaders/dequant_turbo3_0.comp",
        "ggml/src/ggml-vulkan/vulkan-shaders/dequant_turbo4_0.comp",
        "ggml/src/ggml-cuda/q7-q8-view.cuh",
        "ggml/src/ggml-cuda/q7-panel16.cuh",
        "tests/test-turboquant.cpp",
        "tests/test-q7-q8-view.cpp",
        "tests/test-raw-fp32-logits.cpp",
    ):
        read(path)

    # All custom quant types must be registered in all four cooperative-matrix
    # MUL_MAT / MUL_MAT_ID paths (KHR f16/f32 and NV f16/f32).
    vk = read("ggml/src/ggml-vulkan/ggml-vulkan.cpp")
    create_mm2 = [line for line in vk.splitlines() if "CREATE_MM2" in line]
    for qtype in (
        "GGML_TYPE_Q4_0_ROCMFP4",
        "GGML_TYPE_Q4_0_ROCMFP4_FAST",
        "GGML_TYPE_Q2_0_ROCMFPX",
        "GGML_TYPE_Q3_0_ROCMFPX",
        "GGML_TYPE_Q6_0_ROCMFPX",
        "GGML_TYPE_Q8_0_ROCMFPX",
    ):
        count = sum(qtype in line for line in create_mm2)
        if count < 4:
            raise AssertionError(f"ggml-vulkan.cpp: {qtype} has only {count} cooperative-matrix registrations")

    # Per-request drafting, adaptive wrapper, and response reporting.
    require(
        "common/speculative.cpp",
        "common_speculative_effective_n_max",
        "common_speculative_effective_n_min",
        "common_speculative_effective_p_min",
    )
    require("tools/server/server-schema.cpp", 'field_num("speculative.n_max"', 'field_num("speculative.p_split"')
    require("tools/server/server-task.cpp", '{"speculative.n_max"', '{"speculative.p_split"')
    require("scripts/rocmfpx-dynamic-draft.py", "def adapt_policy", "speculative.n_max")
    require("scripts/check-rocmfpx-dynamic-draft.sh", "ROCmFPX dynamic drafting smoke passed")
    require(
        "scripts/run-rocmfpx-fp3-mtp-server-speed-profile.sh",
        "rocmfpx-fp3-mtp-speed",
        "--spec-type draft-mtp",
    )

    # Independent device-resident checkpoints and safe restore failure handling.
    require(
        "include/llama.h",
        "llama_state_seq_storage_init",
        "llama_state_seq_get_data_ext_storage",
        "llama_state_seq_set_data_ext_storage",
    )
    require(
        "common/common.cpp",
        "llama_state_seq_storage_clone",
        "target checkpoint restore failed",
        "draft checkpoint restore failed",
    )
    require(
        "tools/server/server-context.cpp",
        "LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE",
        "forcing full prompt re-processing",
    )

    # Model-specific speculative state and hybrid M-RoPE behavior.
    require("src/llama-batch.cpp", "batch.token && !batch.embd")
    require("src/llama-hparams.h", "n_embd_inp_enc_impl")
    require("common/speculative.cpp", "chain_heads", "common_speculative_get_state", "common_speculative_set_state")
    require("src/models/step35.cpp", "nextn_layer_offset")

    # Diagnostics, ranked-policy fixtures, and the guard itself must stay wired
    # into build/validation entry points.
    require("tests/CMakeLists.txt", "test-raw-fp32-logits.cpp")
    require("tests/fixtures/rocmfpx-ranked-policy/attention-rank.sample.csv", "blk.0.attn_qkv.weight")
    require("tests/fixtures/rocmfpx-ranked-policy/base.tensor-type.sample.txt", "attn_qkv")
    require("scripts/check-rocmfpx-all.sh", "check-rocmfpx-preservation.py")
    require(".github/workflows/check-rocmfpx.yml", "ROCmFPX preservation audit", "check-rocmfpx-preservation.py")

    # ROCm 7.2 fast-math must not receive literal -infinity sentinels in the
    # affected kernels. Comments are stripped before checking.
    require("ggml/src/ggml-cuda/common.cuh", "ggml_cuda_negative_infinity")
    for path in (
        "ggml/src/ggml-cuda/cross-entropy-loss.cu",
        "ggml/src/ggml-cuda/dsv4.cu",
        "ggml/src/ggml-cuda/dsv4-hc.cu",
        "ggml/src/ggml-cuda/softmax.cu",
        "ggml/src/ggml-cuda/topk-moe.cu",
    ):
        text = re.sub(r"//.*", "", read(path))
        if "-INFINITY" in text:
            raise AssertionError(f"{path}: literal -INFINITY remains in device code")

    print("ROCmFPX preservation audit passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
