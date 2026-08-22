#pragma once

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cfloat>

namespace promptforge_iu4 {

// One long K segment is the performance-first contract. It keeps scale and
// zero-point correction out of the WMMA inner loop and leaves quality recovery
// to offline weight optimization and bounded keeper corrections.
constexpr int kSegments = 1;
constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kThreads = 64;
constexpr int kWordsPerChunk = 16;
constexpr int kPack = 8;
constexpr int kNSubtiles = kTileN / 16;
constexpr int kMTilesPerWave = 2;
constexpr int kHadamardBlock = 1024;
constexpr int kQuantSegment = 256;

using fp16_t = __half;
using bf16_t = __hip_bfloat16;
using vint2 = int __attribute__((ext_vector_type(2)));
using vint4 = int __attribute__((ext_vector_type(4)));
using vint8 = int __attribute__((ext_vector_type(8)));

struct packed_matrix {
    const uint32_t * weights; // [segment][N/64][K/segments/8][64]
    const float * scales;     // [segment][N]
    const int32_t * sums;     // [segment][N]
};

struct packed_activations {
    uint32_t * values; // [segment][M][K/segments/8]
    float * scales;    // [segment][M]
    int32_t * zeros;   // [segment][M]
};

constexpr std::size_t packed_activation_bytes(int rows, int cols) {
    return static_cast<std::size_t>(rows) * cols / 2;
}

constexpr std::size_t activation_scale_bytes(int rows) {
    return static_cast<std::size_t>(kSegments) * rows * sizeof(float);
}

constexpr std::size_t activation_zero_bytes(int rows) {
    return static_cast<std::size_t>(kSegments) * rows * sizeof(int32_t);
}

namespace detail {

__host__ __device__ static inline float hadamard_sign(int index, uint32_t seed) {
    uint32_t value = static_cast<uint32_t>(index) + seed;
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return (value & 1U) ? -1.0f : 1.0f;
}

__host__ __device__ static inline int c_frag_row(int lane, int reg) {
    return 2 * reg + ((lane & 31) >> 4);
}

__host__ __device__ static inline int c_frag_col(int lane) {
    return lane & 15;
}

__device__ __forceinline__ uint32_t insert_u4(uint32_t word, int value, int index) {
    return word | (static_cast<uint32_t>(value) << (4 * index));
}

template <int K>
__global__ void __launch_bounds__(256, 1) pack_input_u4(
        const float * __restrict__ input,
        const fp16_t * __restrict__ channel_scales,
        uint32_t * __restrict__ packed,
        float * __restrict__ scales,
        int32_t * __restrict__ zeros,
        int rows) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    if (threadIdx.x == 0) __builtin_trap();
    return;
#else
    constexpr int segment_k = K / kSegments;
    constexpr int words_per_segment = segment_k / kPack;
    const int row = static_cast<int>(blockIdx.x);
    const int segment = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int k0 = segment * segment_k;
    __shared__ float lo_smem[256];
    __shared__ float hi_smem[256];

    float lo = FLT_MAX;
    float hi = -FLT_MAX;
    for (int k = tid; k < segment_k; k += 256) {
        const int col = k0 + k;
        const float channel = channel_scales ? __half2float(channel_scales[col]) : 1.0f;
        const float value = input[static_cast<std::size_t>(row) * K + col] * channel;
        lo = fminf(lo, value);
        hi = fmaxf(hi, value);
    }
    lo_smem[tid] = lo;
    hi_smem[tid] = hi;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            lo_smem[tid] = fminf(lo_smem[tid], lo_smem[tid + stride]);
            hi_smem[tid] = fmaxf(hi_smem[tid], hi_smem[tid + stride]);
        }
        __syncthreads();
    }

    const float scale = hi_smem[0] > lo_smem[0]
        ? (hi_smem[0] - lo_smem[0]) * (1.0f / 15.0f) : 1.0f;
    const int zero = max(0, min(15, __float2int_rn(-lo_smem[0] / scale)));
    if (tid == 0) {
        scales[static_cast<std::size_t>(segment) * rows + row] = scale;
        zeros[static_cast<std::size_t>(segment) * rows + row] = zero;
    }
    for (int word_index = tid; word_index < words_per_segment; word_index += 256) {
        uint32_t word = 0;
#pragma unroll
        for (int i = 0; i < kPack; ++i) {
            const int col = k0 + word_index * kPack + i;
            const float channel = channel_scales ? __half2float(channel_scales[col]) : 1.0f;
            const float value = input[static_cast<std::size_t>(row) * K + col] * channel;
            const int code = max(0, min(15, __float2int_rn(value / scale) + zero));
            word = insert_u4(word, code, i);
        }
        packed[(static_cast<std::size_t>(segment) * rows + row) * words_per_segment + word_index] = word;
    }
#endif
}

template <int K>
__global__ void __launch_bounds__(256, 1) pack_swiglu_u4(
        const bf16_t * __restrict__ gate_up,
        uint32_t * __restrict__ packed,
        float * __restrict__ scales,
        int32_t * __restrict__ zeros,
        int rows) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    if (threadIdx.x == 0) __builtin_trap();
    return;
#else
    constexpr int segment_k = K / kSegments;
    constexpr int words_per_segment = segment_k / kPack;
    const int row = static_cast<int>(blockIdx.x);
    const int segment = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int k0 = segment * segment_k;
    __shared__ float lo_smem[256];
    __shared__ float hi_smem[256];

    float lo = FLT_MAX;
    float hi = -FLT_MAX;
    for (int k = tid; k < segment_k; k += 256) {
        const int col = k0 + k;
        const float gate = __bfloat162float(gate_up[static_cast<std::size_t>(row) * (2 * K) + col]);
        const float up = __bfloat162float(gate_up[static_cast<std::size_t>(row) * (2 * K) + K + col]);
        const float value = (gate / (1.0f + expf(-gate))) * up;
        lo = fminf(lo, value);
        hi = fmaxf(hi, value);
    }
    lo_smem[tid] = lo;
    hi_smem[tid] = hi;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            lo_smem[tid] = fminf(lo_smem[tid], lo_smem[tid + stride]);
            hi_smem[tid] = fmaxf(hi_smem[tid], hi_smem[tid + stride]);
        }
        __syncthreads();
    }

    const float scale = hi_smem[0] > lo_smem[0]
        ? (hi_smem[0] - lo_smem[0]) * (1.0f / 15.0f) : 1.0f;
    const int zero = max(0, min(15, __float2int_rn(-lo_smem[0] / scale)));
    if (tid == 0) {
        scales[static_cast<std::size_t>(segment) * rows + row] = scale;
        zeros[static_cast<std::size_t>(segment) * rows + row] = zero;
    }
    for (int word_index = tid; word_index < words_per_segment; word_index += 256) {
        uint32_t word = 0;
#pragma unroll
        for (int i = 0; i < kPack; ++i) {
            const int col = k0 + word_index * kPack + i;
            const float gate = __bfloat162float(gate_up[static_cast<std::size_t>(row) * (2 * K) + col]);
            const float up = __bfloat162float(gate_up[static_cast<std::size_t>(row) * (2 * K) + K + col]);
            const float value = (gate / (1.0f + expf(-gate))) * up;
            const int code = max(0, min(15, __float2int_rn(value / scale) + zero));
            word = insert_u4(word, code, i);
        }
        packed[(static_cast<std::size_t>(segment) * rows + row) * words_per_segment + word_index] = word;
    }
#endif
}

template <int K, uint32_t Seed>
__global__ void __launch_bounds__(256, 1) pack_input_u4_hadamard(
        const float * __restrict__ input,
        uint32_t * __restrict__ packed,
        float * __restrict__ scales,
        int32_t * __restrict__ zeros,
        int rows) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    if (threadIdx.x == 0) __builtin_trap();
    return;
#else
    static_assert((K % kHadamardBlock) == 0, "Hadamard input K must be block aligned");
    constexpr int kBlocks = K / kHadamardBlock;
    constexpr int kItemsPerThread = kHadamardBlock / 256;
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    __shared__ float block[kHadamardBlock];
    float transformed_items[kBlocks][kItemsPerThread];
    float lo = FLT_MAX;
    float hi = -FLT_MAX;
#pragma unroll
    for (int block_index = 0; block_index < kBlocks; ++block_index) {
        const int block0 = block_index * kHadamardBlock;
#pragma unroll
        for (int item = 0; item < kItemsPerThread; ++item) {
            const int local = tid + item * 256;
            const int col = block0 + local;
            block[local] = input[static_cast<std::size_t>(row) * K + col] *
                           hadamard_sign(col, Seed);
        }
        __syncthreads();
        for (int stride = 1; stride < kHadamardBlock; stride <<= 1) {
            for (int pair = tid; pair < kHadamardBlock / 2; pair += 256) {
                const int group = pair / stride;
                const int offset = pair - group * stride;
                const int i = group * (2 * stride) + offset;
                const int j = i + stride;
                const float a = block[i];
                const float b = block[j];
                block[i] = a + b;
                block[j] = a - b;
            }
            __syncthreads();
        }
#pragma unroll
        for (int item = 0; item < kItemsPerThread; ++item) {
            const int local = tid + item * 256;
            const float value = block[local] * (1.0f / 32.0f);
            transformed_items[block_index][item] = value;
            lo = fminf(lo, value);
            hi = fmaxf(hi, value);
        }
        __syncthreads();
    }
    block[tid] = lo;
    block[256 + tid] = hi;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            block[tid] = fminf(block[tid], block[tid + stride]);
            block[256 + tid] = fmaxf(block[256 + tid], block[256 + tid + stride]);
        }
        __syncthreads();
    }
    const float scale = block[256] > block[0]
        ? (block[256] - block[0]) * (1.0f / 15.0f) : 1.0f;
    const int zero = max(0, min(15, __float2int_rn(-block[0] / scale)));
    if (tid == 0) {
        scales[row] = scale;
        zeros[row] = zero;
    }
#pragma unroll
    for (int block_index = 0; block_index < kBlocks; ++block_index) {
#pragma unroll
        for (int item = 0; item < kItemsPerThread; ++item) {
            const int code = max(0, min(15,
                __float2int_rn(transformed_items[block_index][item] / scale) + zero));
            uint32_t word = 0;
#pragma unroll
            for (int nibble = 0; nibble < kPack; ++nibble) {
                const int peer = __shfl(code, nibble, kPack);
                if ((tid & (kPack - 1)) == 0) {
                    word = insert_u4(word, peer, nibble);
                }
            }
            if ((tid & (kPack - 1)) == 0) {
                const int word_index = block_index * (kHadamardBlock / kPack) +
                    item * (256 / kPack) + tid / kPack;
                packed[static_cast<std::size_t>(row) * (K / kPack) + word_index] = word;
            }
        }
    }
#endif
}

template <int K, uint32_t Seed>
__global__ void __launch_bounds__(256, 1) pack_swiglu_u4_hadamard(
        const bf16_t * __restrict__ gate_up,
        uint32_t * __restrict__ packed,
        float * __restrict__ scales,
        int32_t * __restrict__ zeros,
        int rows) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    if (threadIdx.x == 0) __builtin_trap();
    return;
#else
    static_assert((K % kHadamardBlock) == 0, "Hadamard SwiGLU K must be block aligned");
    constexpr int kBlocks = K / kHadamardBlock;
    constexpr int kItemsPerThread = kHadamardBlock / 256;
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    __shared__ float block[kHadamardBlock];
    float transformed_items[kBlocks][kItemsPerThread];
    float lo = FLT_MAX;
    float hi = -FLT_MAX;
#pragma unroll
    for (int block_index = 0; block_index < kBlocks; ++block_index) {
        const int block0 = block_index * kHadamardBlock;
#pragma unroll
        for (int item = 0; item < kItemsPerThread; ++item) {
            const int local = tid + item * 256;
            const int col = block0 + local;
            const float gate = __bfloat162float(gate_up[static_cast<std::size_t>(row) * (2 * K) + col]);
            const float up = __bfloat162float(gate_up[static_cast<std::size_t>(row) * (2 * K) + K + col]);
            block[local] = (gate / (1.0f + expf(-gate))) * up * hadamard_sign(col, Seed);
        }
        __syncthreads();
        for (int stride = 1; stride < kHadamardBlock; stride <<= 1) {
            for (int pair = tid; pair < kHadamardBlock / 2; pair += 256) {
                const int group = pair / stride;
                const int offset = pair - group * stride;
                const int i = group * (2 * stride) + offset;
                const int j = i + stride;
                const float a = block[i];
                const float b = block[j];
                block[i] = a + b;
                block[j] = a - b;
            }
            __syncthreads();
        }
#pragma unroll
        for (int item = 0; item < kItemsPerThread; ++item) {
            const int local = tid + item * 256;
            const float value = block[local] * (1.0f / 32.0f);
            transformed_items[block_index][item] = value;
            lo = fminf(lo, value);
            hi = fmaxf(hi, value);
        }
        __syncthreads();
    }
    block[tid] = lo;
    block[256 + tid] = hi;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            block[tid] = fminf(block[tid], block[tid + stride]);
            block[256 + tid] = fmaxf(block[256 + tid], block[256 + tid + stride]);
        }
        __syncthreads();
    }
    const float scale = block[256] > block[0]
        ? (block[256] - block[0]) * (1.0f / 15.0f) : 1.0f;
    const int zero = max(0, min(15, __float2int_rn(-block[0] / scale)));
    if (tid == 0) {
        scales[row] = scale;
        zeros[row] = zero;
    }
#pragma unroll
    for (int block_index = 0; block_index < kBlocks; ++block_index) {
#pragma unroll
        for (int item = 0; item < kItemsPerThread; ++item) {
            const int code = max(0, min(15,
                __float2int_rn(transformed_items[block_index][item] / scale) + zero));
            uint32_t word = 0;
#pragma unroll
            for (int nibble = 0; nibble < kPack; ++nibble) {
                const int peer = __shfl(code, nibble, kPack);
                if ((tid & (kPack - 1)) == 0) {
                    word = insert_u4(word, peer, nibble);
                }
            }
            if ((tid & (kPack - 1)) == 0) {
                const int word_index = block_index * (kHadamardBlock / kPack) +
                    item * (256 / kPack) + tid / kPack;
                packed[static_cast<std::size_t>(row) * (K / kPack) + word_index] = word;
            }
        }
    }
#endif
}

struct input_value_loader {
    const float * input;
    int stride;

    __device__ __forceinline__ float operator()(int row, int col) const {
        return input[static_cast<std::size_t>(row) * stride + col];
    }
};

struct swiglu_value_loader {
    const bf16_t * gate_up;
    int width;

    __device__ __forceinline__ float operator()(int row, int col) const {
        const float gate = __bfloat162float(
            gate_up[static_cast<std::size_t>(row) * (2 * width) + col]);
        const float up = __bfloat162float(
            gate_up[static_cast<std::size_t>(row) * (2 * width) + width + col]);
        return (gate / (1.0f + expf(-gate))) * up;
    }
};

template <int K, uint32_t Seed, typename Loader>
__global__ void __launch_bounds__(256, 1) pack_u4_hadamard_segmented(
        Loader loader,
        uint32_t * __restrict__ packed,
        float * __restrict__ scales,
        int32_t * __restrict__ zeros,
        int rows) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    if (threadIdx.x == 0) __builtin_trap();
    return;
#else
    static_assert((K % kHadamardBlock) == 0 &&
                  (kHadamardBlock % kQuantSegment) == 0,
                  "segmented Hadamard K must be block aligned");
    constexpr int words_per_segment = kQuantSegment / kPack;
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    __shared__ float block[kHadamardBlock];
    __shared__ float lo_smem[256];
    __shared__ float hi_smem[256];
    for (int block0 = 0; block0 < K; block0 += kHadamardBlock) {
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int local = tid + item * 256;
            const int col = block0 + local;
            block[local] = loader(row, col) * hadamard_sign(col, Seed);
        }
        __syncthreads();
        for (int stride = 1; stride < kHadamardBlock; stride <<= 1) {
            for (int pair = tid; pair < kHadamardBlock / 2; pair += 256) {
                const int group = pair / stride;
                const int offset = pair - group * stride;
                const int i = group * (2 * stride) + offset;
                const int j = i + stride;
                const float a = block[i];
                const float b = block[j];
                block[i] = a + b;
                block[j] = a - b;
            }
            __syncthreads();
        }
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int local = tid + item * 256;
            const float value = block[local] * (1.0f / 32.0f);
            block[local] = value;
        }
        __syncthreads();
        for (int subsegment = 0; subsegment < kHadamardBlock / kQuantSegment; ++subsegment) {
            const int local = subsegment * kQuantSegment + tid;
            lo_smem[tid] = block[local];
            hi_smem[tid] = block[local];
            __syncthreads();
            for (int stride = 128; stride; stride >>= 1) {
                if (tid < stride) {
                    lo_smem[tid] = fminf(lo_smem[tid], lo_smem[tid + stride]);
                    hi_smem[tid] = fmaxf(hi_smem[tid], hi_smem[tid + stride]);
                }
                __syncthreads();
            }
            const float scale = hi_smem[0] > lo_smem[0]
                ? (hi_smem[0] - lo_smem[0]) * (1.0f / 15.0f) : 1.0f;
            const int zero = max(0, min(15, __float2int_rn(-lo_smem[0] / scale)));
            const int segment = block0 / kQuantSegment + subsegment;
            if (tid == 0) {
                scales[static_cast<std::size_t>(segment) * rows + row] = scale;
                zeros[static_cast<std::size_t>(segment) * rows + row] = zero;
            }
            if (tid < words_per_segment) {
                uint32_t value = 0;
#pragma unroll
                for (int i = 0; i < kPack; ++i) {
                    const float item = block[subsegment * kQuantSegment + tid * kPack + i];
                    const int code = max(0, min(15, __float2int_rn(item / scale) + zero));
                    value = insert_u4(value, code, i);
                }
                packed[(static_cast<std::size_t>(segment) * rows + row) * words_per_segment + tid] = value;
            }
            __syncthreads();
        }
    }
#endif
}

template <int InputK, int KeeperK>
__global__ void __launch_bounds__(256, 1) pack_input_i8_indexed(
        const float * __restrict__ input,
        const uint16_t * __restrict__ indices,
        int8_t * __restrict__ packed,
        float * __restrict__ scales,
        int rows) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    if (threadIdx.x == 0) __builtin_trap();
    return;
#else
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    __shared__ float vmax_smem[256];
    float vmax = 0.0f;
    for (int k = tid; k < KeeperK; k += 256) {
        const float value = input[static_cast<std::size_t>(row) * InputK + indices[k]];
        vmax = fmaxf(vmax, fabsf(value));
    }
    vmax_smem[tid] = vmax;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (tid < stride) vmax_smem[tid] = fmaxf(vmax_smem[tid], vmax_smem[tid + stride]);
        __syncthreads();
    }
    const float scale = vmax_smem[0] > 0.0f ? vmax_smem[0] * (1.0f / 127.0f) : 1.0f;
    if (tid == 0) scales[row] = scale;
    for (int k = tid; k < KeeperK; k += 256) {
        const float value = input[static_cast<std::size_t>(row) * InputK + indices[k]];
        const int code = max(-127, min(127, __float2int_rn(value / scale)));
        packed[static_cast<std::size_t>(row) * KeeperK + k] = static_cast<int8_t>(code);
    }
#endif
}

template <int InputK, int KeeperK>
__global__ void __launch_bounds__(256, 1) pack_swiglu_i8_indexed(
        const bf16_t * __restrict__ gate_up,
        const uint16_t * __restrict__ indices,
        int8_t * __restrict__ packed,
        float * __restrict__ scales,
        int rows) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    if (threadIdx.x == 0) __builtin_trap();
    return;
#else
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    __shared__ float vmax_smem[256];
    float vmax = 0.0f;
    for (int k = tid; k < KeeperK; k += 256) {
        const int col = indices[k];
        const float gate = __bfloat162float(gate_up[static_cast<std::size_t>(row) * (2 * InputK) + col]);
        const float up = __bfloat162float(gate_up[static_cast<std::size_t>(row) * (2 * InputK) + InputK + col]);
        const float value = (gate / (1.0f + expf(-gate))) * up;
        vmax = fmaxf(vmax, fabsf(value));
    }
    vmax_smem[tid] = vmax;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (tid < stride) vmax_smem[tid] = fmaxf(vmax_smem[tid], vmax_smem[tid + stride]);
        __syncthreads();
    }
    const float scale = vmax_smem[0] > 0.0f ? vmax_smem[0] * (1.0f / 127.0f) : 1.0f;
    if (tid == 0) scales[row] = scale;
    for (int k = tid; k < KeeperK; k += 256) {
        const int col = indices[k];
        const float gate = __bfloat162float(gate_up[static_cast<std::size_t>(row) * (2 * InputK) + col]);
        const float up = __bfloat162float(gate_up[static_cast<std::size_t>(row) * (2 * InputK) + InputK + col]);
        const float value = (gate / (1.0f + expf(-gate))) * up;
        const int code = max(-127, min(127, __float2int_rn(value / scale)));
        packed[static_cast<std::size_t>(row) * KeeperK + k] = static_cast<int8_t>(code);
    }
#endif
}

template <int K>
__global__ void __launch_bounds__(kThreads, 1) gemm_i8_correction(
        const uint32_t * __restrict__ aq,
        const float * __restrict__ a_scales,
        const uint32_t * __restrict__ weights,
        const float * __restrict__ weight_scales,
        bf16_t * __restrict__ output,
        int M,
        int N) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    if (threadIdx.x == 0) __builtin_trap();
    return;
#else
    static_assert((K % 64) == 0, "keeper K must be a multiple of 64");
    constexpr int words_per_k = K / 4;
    constexpr int words_per_chunk = 16;
    const int tid = static_cast<int>(threadIdx.x);
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int lane_lo = lane & 15;
    const int ntile = static_cast<int>(blockIdx.x);
    const int block_m = static_cast<int>(blockIdx.y) * kTileM;
    const int block_n = ntile * kTileN;
    __shared__ uint32_t a_lds[kTileM][words_per_chunk + 1];
    __shared__ uint32_t w_lds[kTileN][words_per_chunk + 1];
    vint8 dots[kMTilesPerWave][kNSubtiles];
#pragma unroll
    for (int mt = 0; mt < kMTilesPerWave; ++mt)
#pragma unroll
        for (int ns = 0; ns < kNSubtiles; ++ns)
#pragma unroll
            for (int e = 0; e < 8; ++e) dots[mt][ns][e] = 0;

    for (int chunk = 0; chunk < words_per_k; chunk += words_per_chunk) {
        for (int linear = tid; linear < kTileM * words_per_chunk; linear += kThreads) {
            const int row_local = linear / words_per_chunk;
            const int word = linear - row_local * words_per_chunk;
            const int row = block_m + row_local;
            a_lds[row_local][word] = row < M
                ? aq[static_cast<std::size_t>(row) * words_per_k + chunk + word]
                : 0;
        }
        for (int linear = tid; linear < words_per_chunk * kTileN; linear += kThreads) {
            const int word = linear / kTileN;
            const int nlocal = linear - word * kTileN;
            const std::size_t index =
                (static_cast<std::size_t>(ntile) * words_per_k + chunk + word) * kTileN + nlocal;
            w_lds[nlocal][word] = weights[index];
        }
        __syncthreads();
#pragma unroll
        for (int mt = 0; mt < kMTilesPerWave; ++mt) {
            const int a_row = wave * (16 * kMTilesPerWave) + mt * 16 + lane_lo;
#pragma unroll
            for (int word = 0; word < words_per_chunk; word += 4) {
                const vint4 av = {
                    static_cast<int>(a_lds[a_row][word + 0]),
                    static_cast<int>(a_lds[a_row][word + 1]),
                    static_cast<int>(a_lds[a_row][word + 2]),
                    static_cast<int>(a_lds[a_row][word + 3])};
#pragma unroll
                for (int ns = 0; ns < kNSubtiles; ++ns) {
                    const int w_row = ns * 16 + lane_lo;
                    const vint4 wv = {
                        static_cast<int>(w_lds[w_row][word + 0]),
                        static_cast<int>(w_lds[w_row][word + 1]),
                        static_cast<int>(w_lds[w_row][word + 2]),
                        static_cast<int>(w_lds[w_row][word + 3])};
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1151__)
                    dots[mt][ns] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                        true, av, true, wv, dots[mt][ns], false);
#endif
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mt = 0; mt < kMTilesPerWave; ++mt) {
#pragma unroll
        for (int ns = 0; ns < kNSubtiles; ++ns) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row = block_m + wave * (16 * kMTilesPerWave) + mt * 16 + c_frag_row(lane, e);
                const int n = block_n + ns * 16 + c_frag_col(lane);
                if (row < M) {
                    const std::size_t index = static_cast<std::size_t>(row) * N + n;
                    const float correction = static_cast<float>(dots[mt][ns][e]) *
                        a_scales[row] * weight_scales[n];
                    output[index] = __float2bfloat16(__bfloat162float(output[index]) + correction);
                }
            }
        }
    }
#endif
}

template <typename T>
__device__ __forceinline__ T output_cast(float value);

template <>
__device__ __forceinline__ float output_cast<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ bf16_t output_cast<bf16_t>(float value) {
    return __float2bfloat16(value);
}

template <int K, typename Output>
__global__ void __launch_bounds__(kThreads, 1) gemm_u4s4(
        const uint32_t * __restrict__ aq,
        const float * __restrict__ a_scales,
        const int32_t * __restrict__ a_zeros,
        const uint32_t * __restrict__ weights,
        const float * __restrict__ weight_scales,
        const int32_t * __restrict__ weight_sums,
        Output * __restrict__ output,
        int M,
        int N) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    if (threadIdx.x == 0) __builtin_trap();
    return;
#else
    constexpr int segment_k = kQuantSegment;
    constexpr int segments = K / segment_k;
    static_assert((K % segment_k) == 0 &&
                  (segment_k % (kPack * kWordsPerChunk)) == 0,
                  "K must fit 1024-value staged segments");
    constexpr int words_per_segment = segment_k / kPack;
    const int tid = static_cast<int>(threadIdx.x);
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int lane_lo = lane & 15;
    const int ntile = static_cast<int>(blockIdx.x);
    const int block_m = static_cast<int>(blockIdx.y) * kTileM;
    const int ntiles = N / kTileN;
    const int block_n = ntile * kTileN;
    __shared__ uint32_t a_lds[kTileM][kWordsPerChunk + 1];
    __shared__ uint32_t w_lds[kTileN][kWordsPerChunk + 1];
    float result[kMTilesPerWave][kNSubtiles][8];
#pragma unroll
    for (int mt = 0; mt < kMTilesPerWave; ++mt)
#pragma unroll
        for (int ns = 0; ns < kNSubtiles; ++ns)
#pragma unroll
            for (int e = 0; e < 8; ++e) result[mt][ns][e] = 0.0f;

#pragma unroll
    for (int segment = 0; segment < segments; ++segment) {
        vint8 dots[kMTilesPerWave][kNSubtiles];
#pragma unroll
        for (int mt = 0; mt < kMTilesPerWave; ++mt)
#pragma unroll
            for (int ns = 0; ns < kNSubtiles; ++ns)
#pragma unroll
                for (int e = 0; e < 8; ++e) dots[mt][ns][e] = 0;

        for (int chunk = 0; chunk < words_per_segment; chunk += kWordsPerChunk) {
            for (int linear = tid; linear < kTileM * kWordsPerChunk; linear += kThreads) {
                const int row_local = linear / kWordsPerChunk;
                const int word = linear - row_local * kWordsPerChunk;
                const int row = block_m + row_local;
                a_lds[row_local][word] = row < M
                    ? aq[(static_cast<std::size_t>(segment) * M + row) * words_per_segment + chunk + word]
                    : 0;
            }
            for (int linear = tid; linear < kWordsPerChunk * kTileN; linear += kThreads) {
                const int word = linear / kTileN;
                const int nlocal = linear - word * kTileN;
                const std::size_t index =
                    (((static_cast<std::size_t>(segment) * ntiles + ntile) * words_per_segment +
                       chunk + word) * kTileN + nlocal);
                w_lds[nlocal][word] = weights[index];
            }
            __syncthreads();
#pragma unroll
            for (int mt = 0; mt < kMTilesPerWave; ++mt) {
                const int a_row = wave * (16 * kMTilesPerWave) + mt * 16 + lane_lo;
#pragma unroll
                for (int word = 0; word < kWordsPerChunk; word += 2) {
                    const vint2 av = {static_cast<int>(a_lds[a_row][word]),
                                      static_cast<int>(a_lds[a_row][word + 1])};
#pragma unroll
                    for (int ns = 0; ns < kNSubtiles; ++ns) {
                        const int w_row = ns * 16 + lane_lo;
                        const vint2 wv = {static_cast<int>(w_lds[w_row][word]),
                                          static_cast<int>(w_lds[w_row][word + 1])};
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1151__)
                        dots[mt][ns] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                            false, av, true, wv, dots[mt][ns], false);
#endif
                    }
                }
            }
            __syncthreads();
        }

#pragma unroll
        for (int mt = 0; mt < kMTilesPerWave; ++mt) {
#pragma unroll
            for (int ns = 0; ns < kNSubtiles; ++ns) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int row = block_m + wave * (16 * kMTilesPerWave) + mt * 16 + c_frag_row(lane, e);
                    const int n = block_n + ns * 16 + c_frag_col(lane);
                    if (row < M) {
                        const int zero = a_zeros[static_cast<std::size_t>(segment) * M + row];
                        const int corrected = dots[mt][ns][e] - zero * weight_sums[static_cast<std::size_t>(segment) * N + n];
                        result[mt][ns][e] += static_cast<float>(corrected) *
                            a_scales[static_cast<std::size_t>(segment) * M + row] *
                            weight_scales[static_cast<std::size_t>(segment) * N + n];
                    }
                }
            }
        }
    }

#pragma unroll
    for (int mt = 0; mt < kMTilesPerWave; ++mt) {
#pragma unroll
        for (int ns = 0; ns < kNSubtiles; ++ns) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row = block_m + wave * (16 * kMTilesPerWave) + mt * 16 + c_frag_row(lane, e);
                const int n = block_n + ns * 16 + c_frag_col(lane);
                if (row < M) {
                    output[static_cast<std::size_t>(row) * N + n] = output_cast<Output>(result[mt][ns][e]);
                }
            }
        }
    }
#endif
}

} // namespace detail

template <int K>
static inline hipError_t launch_input_pack(
        const float * input,
        const fp16_t * channel_scales,
        const packed_activations & activations,
        int rows,
        hipStream_t stream) {
    if (!input || !activations.values || !activations.scales || !activations.zeros ||
        rows < 1 || (K % (kSegments * kPack)) != 0) return hipErrorInvalidValue;
    detail::pack_input_u4<K><<<dim3(rows, kSegments), 256, 0, stream>>>(
        input, channel_scales, activations.values, activations.scales, activations.zeros, rows);
    return hipGetLastError();
}

template <int K>
static inline hipError_t launch_swiglu_pack(
        const bf16_t * gate_up,
        const packed_activations & activations,
        int rows,
        hipStream_t stream) {
    if (!gate_up || !activations.values || !activations.scales || !activations.zeros ||
        rows < 1 || (K % (kSegments * kPack)) != 0) return hipErrorInvalidValue;
    detail::pack_swiglu_u4<K><<<dim3(rows, kSegments), 256, 0, stream>>>(
        gate_up, activations.values, activations.scales, activations.zeros, rows);
    return hipGetLastError();
}

template <int K, uint32_t Seed>
static inline hipError_t launch_hadamard_input_pack(
        const float * input,
        const packed_activations & activations,
        int rows,
        hipStream_t stream) {
    if (!input || !activations.values || !activations.scales ||
        !activations.zeros || rows < 1) return hipErrorInvalidValue;
    detail::pack_input_u4_hadamard<K, Seed><<<rows, 256, 0, stream>>>(
        input, activations.values, activations.scales, activations.zeros, rows);
    return hipGetLastError();
}

template <int K, uint32_t Seed>
static inline hipError_t launch_hadamard_swiglu_pack(
        const bf16_t * gate_up,
        const packed_activations & activations,
        int rows,
        hipStream_t stream) {
    if (!gate_up || !activations.values || !activations.scales ||
        !activations.zeros || rows < 1) return hipErrorInvalidValue;
    detail::pack_swiglu_u4_hadamard<K, Seed><<<rows, 256, 0, stream>>>(
        gate_up, activations.values, activations.scales, activations.zeros, rows);
    return hipGetLastError();
}

template <int K, uint32_t Seed>
static inline hipError_t launch_segmented_hadamard_input_pack(
        const float * input,
        const packed_activations & activations,
        int rows,
        hipStream_t stream) {
    if (!input || !activations.values || !activations.scales || !activations.zeros ||
        rows < 1) return hipErrorInvalidValue;
    detail::pack_u4_hadamard_segmented<K, Seed><<<rows, 256, 0, stream>>>(
        detail::input_value_loader{input, K}, activations.values,
        activations.scales, activations.zeros, rows);
    return hipGetLastError();
}

template <int K, uint32_t Seed>
static inline hipError_t launch_segmented_hadamard_swiglu_pack(
        const bf16_t * gate_up,
        const packed_activations & activations,
        int rows,
        hipStream_t stream) {
    if (!gate_up || !activations.values || !activations.scales || !activations.zeros ||
        rows < 1) return hipErrorInvalidValue;
    detail::pack_u4_hadamard_segmented<K, Seed><<<rows, 256, 0, stream>>>(
        detail::swiglu_value_loader{gate_up, K}, activations.values,
        activations.scales, activations.zeros, rows);
    return hipGetLastError();
}

template <int InputK, int KeeperK>
static inline hipError_t launch_indexed_input_pack(
        const float * input,
        const uint16_t * indices,
        int8_t * packed,
        float * scales,
        int rows,
        hipStream_t stream) {
    if (!input || !indices || !packed || !scales || rows < 1) return hipErrorInvalidValue;
    detail::pack_input_i8_indexed<InputK, KeeperK><<<rows, 256, 0, stream>>>(
        input, indices, packed, scales, rows);
    return hipGetLastError();
}

template <int InputK, int KeeperK>
static inline hipError_t launch_indexed_swiglu_pack(
        const bf16_t * gate_up,
        const uint16_t * indices,
        int8_t * packed,
        float * scales,
        int rows,
        hipStream_t stream) {
    if (!gate_up || !indices || !packed || !scales || rows < 1) return hipErrorInvalidValue;
    detail::pack_swiglu_i8_indexed<InputK, KeeperK><<<rows, 256, 0, stream>>>(
        gate_up, indices, packed, scales, rows);
    return hipGetLastError();
}

template <int K>
static inline hipError_t launch_i8_correction(
        const int8_t * activations,
        const float * activation_scales,
        const uint32_t * weights,
        const float * weight_scales,
        bf16_t * output,
        int rows,
        int cols,
        hipStream_t stream) {
    if (!activations || !activation_scales || !weights || !weight_scales || !output ||
        rows < 1 || cols < 1 || (cols % kTileN) != 0) return hipErrorInvalidValue;
    detail::gemm_i8_correction<K><<<dim3(cols / kTileN, (rows + kTileM - 1) / kTileM),
                                    kThreads, 0, stream>>>(
        reinterpret_cast<const uint32_t *>(activations), activation_scales,
        weights, weight_scales, output, rows, cols);
    return hipGetLastError();
}

template <int K, typename Output>
static inline hipError_t launch_gemm(
        const packed_activations & activations,
        const packed_matrix & matrix,
        Output * output,
        int rows,
        int cols,
        hipStream_t stream) {
    if (!activations.values || !activations.scales || !activations.zeros ||
        !matrix.weights || !matrix.scales || !matrix.sums || !output || rows < 1 ||
        cols < 1 || (cols % kTileN) != 0) return hipErrorInvalidValue;
    detail::gemm_u4s4<K, Output><<<dim3(cols / kTileN, (rows + kTileM - 1) / kTileM),
                                  kThreads, 0, stream>>>(
        activations.values, activations.scales, activations.zeros,
        matrix.weights, matrix.scales, matrix.sums, output, rows, cols);
    return hipGetLastError();
}

} // namespace promptforge_iu4

#endif
