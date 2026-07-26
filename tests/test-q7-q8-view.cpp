#include "ggml.h"

#include "../ggml/rocmfpx/q7-q8-view.h"

#undef NDEBUG
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void pack_group(
        uint8_t destination[QS_ROCMFP7_GROUP],
        const int8_t source[QG_ROCMFP7]) {
    memset(destination, 0, QS_ROCMFP7_GROUP);
    for (int index = 0; index < QG_ROCMFP7; ++index) {
        const uint8_t code = (uint8_t) source[index] & 0x7fu;
        const int bit_pos = 7 * index;
        for (int bit = 0; bit < 7; ++bit) {
            if ((code & (1u << bit)) != 0) {
                const int output_bit = bit_pos + bit;
                destination[output_bit >> 3] |=
                    (uint8_t) (1u << (output_bit & 7));
            }
        }
    }
}

int main() {
    size_t checked_view_size = 0;
    assert(rocmfpx_q7_q8_view_size(
        3,
        2,
        &checked_view_size));
    assert(
        checked_view_size ==
        6 * ROCMFPX_Q7_Q8_VIEW_MACRO_BYTES);
    assert(!rocmfpx_q7_q8_view_size(
        INT64_MAX,
        2,
        &checked_view_size));
    assert(!rocmfpx_q7_q8_view_size(
        INT64_MAX,
        1,
        &checked_view_size));
    assert(!rocmfpx_q7_q8_view_size(
        1,
        INT64_MAX,
        &checked_view_size));
    assert(!rocmfpx_q7_q8_view_size(
        0,
        1,
        &checked_view_size));
    assert(!rocmfpx_q7_q8_view_size(1, 1, nullptr));

    assert(rocmfpx_q7_q8_view_mmq_addressable(248320, 16));
    // Ornith/Qwen3.5 MoE expert storage is a contiguous 3D tensor. The
    // compute view flattens rows-per-expert * expert-count while preserving
    // the same row-major ordering and channel stride.
    assert(rocmfpx_q7_q8_view_mmq_addressable(2048 * 256, 8));
    assert(!rocmfpx_q7_q8_view_mmq_addressable(INT_MAX, 1));
    assert(!rocmfpx_q7_q8_view_mmq_addressable(
        (int64_t) INT_MAX + 1,
        1));
    assert(!rocmfpx_q7_q8_view_mmq_addressable(
        1,
        (int64_t) INT_MAX / QK_ROCMFP7 + 1));

    block_rocmfp7 source = {};
    uint8_t destination[ROCMFPX_Q7_Q8_VIEW_MACRO_BYTES] = {};

    for (int group = 0; group < NG_ROCMFP7; ++group) {
        const uint16_t scale_bits =
            (uint16_t) (0x8001u + 0x111u * group);
        memcpy(&source.d[group], &scale_bits, sizeof(scale_bits));

        int8_t codes[QG_ROCMFP7];
        for (int index = 0; index < QG_ROCMFP7; ++index) {
            // Across the eight groups this covers all signed Q7 codes.
            codes[index] =
                (int8_t) (-64 + ((group * QG_ROCMFP7 + index) & 127));
        }
        pack_group(
            source.qs + group * QS_ROCMFP7_GROUP,
            codes);
    }

    rocmfpx_q7_expand_macro_to_q8_view(&source, destination);

    for (int group = 0; group < NG_ROCMFP7; ++group) {
        const uint8_t * q8_block =
            destination + group * ROCMFPX_Q7_Q8_VIEW_BLOCK_BYTES;
        assert(memcmp(
            q8_block,
            &source.d[group],
            sizeof(source.d[group])) == 0);
        const int8_t * q8_codes =
            (const int8_t *) (q8_block + sizeof(source.d[group]));
        for (int index = 0; index < QG_ROCMFP7; ++index) {
            const int8_t expected =
                (int8_t) (-64 + ((group * QG_ROCMFP7 + index) & 127));
            assert(q8_codes[index] == expected);
        }
    }

    constexpr int64_t nrows = 3;
    constexpr int64_t macros_per_row = 2;
    constexpr int64_t macro_count = nrows * macros_per_row;
    block_rocmfp7 matrix_source[macro_count] = {};
    uint8_t matrix_destination[
        macro_count * ROCMFPX_Q7_Q8_VIEW_MACRO_BYTES];
    memset(matrix_destination, 0xa5, sizeof(matrix_destination));

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t macro_index = 0;
             macro_index < macros_per_row;
             ++macro_index) {
            const int64_t linear_index =
                row * macros_per_row + macro_index;
            block_rocmfp7 & source_macro =
                matrix_source[linear_index];
            for (int group = 0; group < NG_ROCMFP7; ++group) {
                const uint16_t scale_bits =
                    (uint16_t) (
                        0x0101u +
                        0x101u * linear_index +
                        0x11u * group);
                memcpy(
                    &source_macro.d[group],
                    &scale_bits,
                    sizeof(scale_bits));

                int8_t codes[QG_ROCMFP7];
                for (int index = 0;
                     index < QG_ROCMFP7;
                     ++index) {
                    codes[index] =
                        (int8_t) (
                            -64 +
                            ((17 * linear_index +
                              QG_ROCMFP7 * group +
                              index) &
                             127));
                }
                pack_group(
                    source_macro.qs +
                        group * QS_ROCMFP7_GROUP,
                    codes);
            }
        }
    }

    assert(rocmfpx_q7_expand_rows_to_q8_view(
        matrix_source,
        matrix_destination,
        nrows,
        macros_per_row));

    uint8_t overflow_destination = 0;
    assert(!rocmfpx_q7_expand_rows_to_q8_view(
        matrix_source,
        &overflow_destination,
        INT64_MAX,
        2));

    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t macro_index = 0;
             macro_index < macros_per_row;
             ++macro_index) {
            const int64_t linear_index =
                row * macros_per_row + macro_index;
            const block_rocmfp7 & source_macro =
                matrix_source[linear_index];
            const uint8_t * destination_macro =
                matrix_destination +
                linear_index * ROCMFPX_Q7_Q8_VIEW_MACRO_BYTES;
            for (int group = 0; group < NG_ROCMFP7; ++group) {
                const uint8_t * q8_block =
                    destination_macro +
                    group * ROCMFPX_Q7_Q8_VIEW_BLOCK_BYTES;
                assert(memcmp(
                    q8_block,
                    &source_macro.d[group],
                    sizeof(source_macro.d[group])) == 0);
                const int8_t * q8_codes =
                    (const int8_t *) (
                        q8_block + sizeof(source_macro.d[group]));
                for (int index = 0;
                     index < QG_ROCMFP7;
                     ++index) {
                    const int8_t expected =
                        (int8_t) (
                            -64 +
                            ((17 * linear_index +
                              QG_ROCMFP7 * group +
                              index) &
                             127));
                    assert(q8_codes[index] == expected);
                }
            }
        }
    }

    constexpr int64_t expert_count = 2;
    constexpr int64_t rows_per_expert = 3;
    constexpr int64_t expert_macros_per_row = 1;
    constexpr int64_t flat_expert_rows =
        expert_count * rows_per_expert;
    block_rocmfp7 expert_source[
        flat_expert_rows * expert_macros_per_row] = {};
    uint8_t expert_destination[
        flat_expert_rows *
        expert_macros_per_row *
        ROCMFPX_Q7_Q8_VIEW_MACRO_BYTES] = {};

    for (int64_t expert = 0; expert < expert_count; ++expert) {
        for (int64_t row = 0; row < rows_per_expert; ++row) {
            const int64_t flat_row =
                expert * rows_per_expert + row;
            block_rocmfp7 & source_macro =
                expert_source[flat_row];
            for (int group = 0; group < NG_ROCMFP7; ++group) {
                const uint16_t scale_bits =
                    (uint16_t) (
                        0x2200u +
                        0x100u * expert +
                        0x10u * row +
                        group);
                memcpy(
                    &source_macro.d[group],
                    &scale_bits,
                    sizeof(scale_bits));
                int8_t codes[QG_ROCMFP7];
                for (int index = 0; index < QG_ROCMFP7; ++index) {
                    codes[index] =
                        (int8_t) (
                            -64 +
                            ((31 * expert +
                              7 * row +
                              QG_ROCMFP7 * group +
                              index) &
                             127));
                }
                pack_group(
                    source_macro.qs +
                        group * QS_ROCMFP7_GROUP,
                    codes);
            }
        }
    }

    assert(rocmfpx_q7_expand_rows_to_q8_view(
        expert_source,
        expert_destination,
        flat_expert_rows,
        expert_macros_per_row));
    for (int64_t expert = 0; expert < expert_count; ++expert) {
        for (int64_t row = 0; row < rows_per_expert; ++row) {
            const int64_t flat_row =
                expert * rows_per_expert + row;
            const block_rocmfp7 & source_macro =
                expert_source[flat_row];
            const uint8_t * destination_macro =
                expert_destination +
                flat_row * ROCMFPX_Q7_Q8_VIEW_MACRO_BYTES;
            for (int group = 0; group < NG_ROCMFP7; ++group) {
                const uint8_t * q8_block =
                    destination_macro +
                    group * ROCMFPX_Q7_Q8_VIEW_BLOCK_BYTES;
                assert(memcmp(
                    q8_block,
                    &source_macro.d[group],
                    sizeof(source_macro.d[group])) == 0);
                const int8_t * q8_codes =
                    (const int8_t *) (
                        q8_block + sizeof(source_macro.d[group]));
                for (int index = 0; index < QG_ROCMFP7; ++index) {
                    const int8_t expected =
                        (int8_t) (
                            -64 +
                            ((31 * expert +
                              7 * row +
                              QG_ROCMFP7 * group +
                              index) &
                             127));
                    assert(q8_codes[index] == expected);
                }
            }
        }
    }

    printf("Q7 -> standard-Q8 compute-view mapping: ok\n");
    return 0;
}
