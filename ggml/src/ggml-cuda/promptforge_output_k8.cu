#include "promptforge_output_k8.cuh"

#include "common.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"

#include "ggml.h"

#include <hip/hip_runtime.h>

#include <cerrno>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr int PF_OUTPUT_HIDDEN = 5120;
constexpr int PF_OUTPUT_VOCAB = 248320;
constexpr int PF_OUTPUT_BLOCKS_PER_ROW = PF_OUTPUT_HIDDEN / 32;
constexpr int PF_OUTPUT_K = 8;
constexpr int PF_OUTPUT_TOP_THREADS = 256;
constexpr int PF_OUTPUT_STAGE1_BLOCKS = PF_OUTPUT_VOCAB / PF_OUTPUT_TOP_THREADS;
constexpr int PF_OUTPUT_STAGE1_COUNT = PF_OUTPUT_STAGE1_BLOCKS * PF_OUTPUT_K;
constexpr int PF_OUTPUT_STAGE2_BLOCKS =
    (PF_OUTPUT_STAGE1_COUNT + PF_OUTPUT_TOP_THREADS - 1) / PF_OUTPUT_TOP_THREADS;
constexpr int PF_OUTPUT_STAGE2_COUNT = PF_OUTPUT_STAGE2_BLOCKS * PF_OUTPUT_K;
constexpr size_t PF_OUTPUT_Q4_BYTES =
    size_t(PF_OUTPUT_VOCAB) * PF_OUTPUT_BLOCKS_PER_ROW * sizeof(block_rocmfp4);

static_assert(PF_OUTPUT_HIDDEN % 32 == 0, "output width must be block aligned");
static_assert(PF_OUTPUT_VOCAB % PF_OUTPUT_TOP_THREADS == 0,
              "vocabulary must be exactly chunkable");
static_assert(sizeof(block_rocmfp4) == 18, "Q4_0_ROCMFP4 layout drift");
static_assert(sizeof(block_rocmfp8) == 33, "Q8_0_ROCMFPX layout drift");

struct PFOutputK8State {
    std::mutex init_mutex;
    bool initialized = false;
    bool enabled = false;
    bool ready = false;
    int device = -1;

    block_rocmfp4 * q4_weights = nullptr;
    block_q8_1 * activation = nullptr;
    float * q4_logits = nullptr;
    float * stage1_values = nullptr;
    int32_t * stage1_ids = nullptr;
    float * stage2_values = nullptr;
    int32_t * stage2_ids = nullptr;
    float * top_values = nullptr;
    int32_t * top_ids = nullptr;
    float * gathered_q8 = nullptr;
};

PFOutputK8State & output_state() {
    static PFOutputK8State * singleton = new PFOutputK8State();
    return *singleton;
}

bool hip_ok(hipError_t status, const char * what) {
    if (status == hipSuccess) return true;
    std::fprintf(stderr, "promptforge_output_k8: %s failed: %s\n",
                 what, hipGetErrorString(status));
    return false;
}

template <typename T>
bool allocate_device(T ** ptr, size_t count, const char * what) {
    return hip_ok(hipMalloc(reinterpret_cast<void **>(ptr), count * sizeof(T)), what);
}

__launch_bounds__(64, 1)
__global__ void output_q4_full_mmvq(
        const block_rocmfp4 * __restrict__ weights,
        const block_q8_1 * __restrict__ activation,
        float * __restrict__ logits) {
    constexpr int warp_size = 32;
    constexpr int nwarps = 2;
    constexpr int blocks_per_iter =
        VDR_ROCMFP4_Q8_1_MMVQ * nwarps * warp_size / QI_ROCMFP4;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp_size * warp + lane;
    const int row = blockIdx.x;
    float sum = 0.0f;

    for (int kbx = tid / (QI_ROCMFP4 / VDR_ROCMFP4_Q8_1_MMVQ);
         kbx < PF_OUTPUT_BLOCKS_PER_ROW; kbx += blocks_per_iter) {
        const int iqs = VDR_ROCMFP4_Q8_1_MMVQ *
            (tid % (QI_ROCMFP4 / VDR_ROCMFP4_Q8_1_MMVQ));
        sum += vec_dot_rocmfp4_q8_1(
            weights, &activation[kbx], row * PF_OUTPUT_BLOCKS_PER_ROW + kbx, iqs);
    }

    __shared__ float other_warp[warp_size];
    if (warp == 1) other_warp[lane] = sum;
    __syncthreads();
    if (warp == 1) return;
    sum += other_warp[lane];
    sum = warp_reduce_sum<warp_size>(sum);
    if (lane == 0) logits[row] = sum;
}

__launch_bounds__(32, 1)
__global__ void output_q8_gather_mmvq(
        const block_rocmfp8 * __restrict__ weights,
        const block_q8_1 * __restrict__ activation,
        const int32_t * __restrict__ row_ids,
        float * __restrict__ logits) {
    constexpr int warp_size = 32;
    constexpr int blocks_per_iter =
        VDR_ROCMFP8_Q8_1_MMVQ * warp_size / QI_ROCMFP8;
    const int lane = threadIdx.x;
    const int output_row = blockIdx.x;
    const int weight_row = row_ids[output_row];
    float sum = 0.0f;

    for (int kbx = lane / (QI_ROCMFP8 / VDR_ROCMFP8_Q8_1_MMVQ);
         kbx < PF_OUTPUT_BLOCKS_PER_ROW; kbx += blocks_per_iter) {
        const int iqs = VDR_ROCMFP8_Q8_1_MMVQ *
            (lane % (QI_ROCMFP8 / VDR_ROCMFP8_Q8_1_MMVQ));
        sum += vec_dot_rocmfpx_fp8_q8_1(
            weights, &activation[kbx],
            weight_row * PF_OUTPUT_BLOCKS_PER_ROW + kbx, iqs);
    }

    sum = warp_reduce_sum<warp_size>(sum);
    if (lane == 0) logits[output_row] = sum;
}

__device__ __forceinline__ bool top_better(float av, int ai, float bv, int bi) {
    if (av != bv) return av > bv;
    return ai < bi;
}

template <bool ExplicitIds>
__launch_bounds__(PF_OUTPUT_TOP_THREADS, 1)
__global__ void reduce_top8(
        const float * __restrict__ input_values,
        const int32_t * __restrict__ input_ids,
        int count,
        float * __restrict__ output_values,
        int32_t * __restrict__ output_ids) {
    __shared__ float values[PF_OUTPUT_TOP_THREADS];
    __shared__ int32_t ids[PF_OUTPUT_TOP_THREADS];
    const int tid = threadIdx.x;
    const int index = blockIdx.x * PF_OUTPUT_TOP_THREADS + tid;
    if (index < count) {
        values[tid] = input_values[index];
        ids[tid] = ExplicitIds ? input_ids[index] : index;
    } else {
        values[tid] = -FLT_MAX;
        ids[tid] = INT32_MAX;
    }
    __syncthreads();

    for (int width = 2; width <= PF_OUTPUT_TOP_THREADS; width <<= 1) {
        for (int stride = width >> 1; stride > 0; stride >>= 1) {
            const int partner = tid ^ stride;
            if (partner > tid) {
                const bool descending = (tid & width) == 0;
                const float av = values[tid];
                const int ai = ids[tid];
                const float bv = values[partner];
                const int bi = ids[partner];
                const bool do_swap = descending ? top_better(bv, bi, av, ai)
                                                : top_better(av, ai, bv, bi);
                if (do_swap) {
                    values[tid] = bv;
                    ids[tid] = bi;
                    values[partner] = av;
                    ids[partner] = ai;
                }
            }
            __syncthreads();
        }
    }

    if (tid < PF_OUTPUT_K) {
        const int out = blockIdx.x * PF_OUTPUT_K + tid;
        output_values[out] = values[tid];
        output_ids[out] = ids[tid];
    }
}

__global__ void fill_negative_infinity(float * dst, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        reinterpret_cast<uint32_t *>(dst)[index] = 0xff800000u;
    }
}

__global__ void scatter_exact_candidates(
        float * dst, const int32_t * ids, const float * values) {
    const int index = threadIdx.x;
    if (index < PF_OUTPUT_K) dst[ids[index]] = values[index];
}

bool exact_output_shape(
        const ggml_tensor * weight,
        const ggml_tensor * input,
        const ggml_tensor * dst) {
    return weight && input && dst &&
        weight->type == GGML_TYPE_Q8_0_ROCMFPX &&
        input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        weight->ne[0] == PF_OUTPUT_HIDDEN && weight->ne[1] == PF_OUTPUT_VOCAB &&
        weight->ne[2] == 1 && weight->ne[3] == 1 &&
        input->ne[0] == PF_OUTPUT_HIDDEN && input->ne[1] == 1 &&
        input->ne[2] == 1 && input->ne[3] == 1 &&
        dst->ne[0] == PF_OUTPUT_VOCAB && dst->ne[1] == 1 &&
        dst->ne[2] == 1 && dst->ne[3] == 1 &&
        std::strcmp(weight->name, "output.weight") == 0 &&
        std::strcmp(dst->name, "result_output") == 0;
}

} // namespace

bool promptforge_output_k8_backend_init(int device) {
    PFOutputK8State & s = output_state();
    std::lock_guard<std::mutex> lock(s.init_mutex);
    if (s.initialized) {
        return !s.enabled || (s.ready && s.device == device);
    }
    s.initialized = true;

    const char * enabled = std::getenv("PROMPTFORGE_OUTPUT_K8_STRICT_GREEDY");
    s.enabled = enabled && std::strcmp(enabled, "1") == 0;
    if (!s.enabled) {
        s.ready = true;
        return true;
    }

    const char * path = std::getenv("PROMPTFORGE_OUTPUT_K8_SIDECAR");
    if (!path || !path[0]) {
        std::fprintf(stderr,
            "promptforge_output_k8: PROMPTFORGE_OUTPUT_K8_SIDECAR is required\n");
        return false;
    }
    const char * graph_opt = std::getenv("GGML_CUDA_GRAPH_OPT");
    if (graph_opt && std::strcmp(graph_opt, "1") == 0) {
        std::fprintf(stderr,
            "promptforge_output_k8: strict-greedy route requires GGML_CUDA_GRAPH_OPT=0\n");
        return false;
    }

    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        std::fprintf(stderr, "promptforge_output_k8: cannot open %s: %s\n",
                     path, std::strerror(errno));
        return false;
    }
    struct stat st {};
    if (fstat(fd, &st) != 0 || size_t(st.st_size) != PF_OUTPUT_Q4_BYTES) {
        std::fprintf(stderr,
            "promptforge_output_k8: Q4 sidecar must be exactly %zu bytes\n",
            PF_OUTPUT_Q4_BYTES);
        close(fd);
        return false;
    }
    void * mapping = mmap(nullptr, PF_OUTPUT_Q4_BYTES, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) {
        std::fprintf(stderr, "promptforge_output_k8: mmap failed\n");
        return false;
    }

    s.device = device;
    bool ok = hip_ok(hipSetDevice(device), "hipSetDevice") &&
        allocate_device(&s.q4_weights,
            size_t(PF_OUTPUT_VOCAB) * PF_OUTPUT_BLOCKS_PER_ROW, "hipMalloc(q4_weights)") &&
        allocate_device(&s.activation, PF_OUTPUT_BLOCKS_PER_ROW, "hipMalloc(activation)") &&
        allocate_device(&s.q4_logits, PF_OUTPUT_VOCAB, "hipMalloc(q4_logits)") &&
        allocate_device(&s.stage1_values, PF_OUTPUT_STAGE1_COUNT, "hipMalloc(stage1_values)") &&
        allocate_device(&s.stage1_ids, PF_OUTPUT_STAGE1_COUNT, "hipMalloc(stage1_ids)") &&
        allocate_device(&s.stage2_values, PF_OUTPUT_STAGE2_COUNT, "hipMalloc(stage2_values)") &&
        allocate_device(&s.stage2_ids, PF_OUTPUT_STAGE2_COUNT, "hipMalloc(stage2_ids)") &&
        allocate_device(&s.top_values, PF_OUTPUT_K, "hipMalloc(top_values)") &&
        allocate_device(&s.top_ids, PF_OUTPUT_K, "hipMalloc(top_ids)") &&
        allocate_device(&s.gathered_q8, PF_OUTPUT_K, "hipMalloc(gathered_q8)") &&
        hip_ok(hipMemcpy(s.q4_weights, mapping, PF_OUTPUT_Q4_BYTES,
                         hipMemcpyHostToDevice), "hipMemcpy(q4_weights)");
    munmap(mapping, PF_OUTPUT_Q4_BYTES);
    if (!ok) return false;

    s.ready = true;
    std::fprintf(stderr,
        "{\"record\":\"promptforge_output_k8_init\","
        "\"mode\":\"target_only_strict_greedy\",\"k\":8,"
        "\"q4_bytes\":%zu,\"synthetic_logits\":true}\n",
        PF_OUTPUT_Q4_BYTES);
    return true;
}

bool promptforge_try_output_k8_strict_greedy(
        ggml_backend_cuda_context * ctx,
        const ggml_tensor * weight,
        const ggml_tensor * input,
        ggml_tensor * dst) {
    PFOutputK8State & s = output_state();
    if (!ctx || !s.enabled || !s.ready || !exact_output_shape(weight, input, dst)) {
        return false;
    }

    hipStream_t stream = ctx->stream();
    quantize_row_q8_1_cuda(
        static_cast<const float *>(input->data), nullptr, s.activation,
        GGML_TYPE_Q8_0_ROCMFPX,
        PF_OUTPUT_HIDDEN, PF_OUTPUT_HIDDEN, PF_OUTPUT_HIDDEN, PF_OUTPUT_HIDDEN,
        PF_OUTPUT_HIDDEN, 1, 1, 1, stream);

    output_q4_full_mmvq<<<dim3(PF_OUTPUT_VOCAB), dim3(32, 2), 0, stream>>>(
        s.q4_weights, s.activation, s.q4_logits);
    reduce_top8<false><<<dim3(PF_OUTPUT_STAGE1_BLOCKS),
                         dim3(PF_OUTPUT_TOP_THREADS), 0, stream>>>(
        s.q4_logits, nullptr, PF_OUTPUT_VOCAB, s.stage1_values, s.stage1_ids);
    reduce_top8<true><<<dim3(PF_OUTPUT_STAGE2_BLOCKS),
                        dim3(PF_OUTPUT_TOP_THREADS), 0, stream>>>(
        s.stage1_values, s.stage1_ids, PF_OUTPUT_STAGE1_COUNT,
        s.stage2_values, s.stage2_ids);
    reduce_top8<true><<<dim3(1), dim3(PF_OUTPUT_TOP_THREADS), 0, stream>>>(
        s.stage2_values, s.stage2_ids, PF_OUTPUT_STAGE2_COUNT,
        s.top_values, s.top_ids);
    output_q8_gather_mmvq<<<dim3(PF_OUTPUT_K), dim3(32), 0, stream>>>(
        static_cast<const block_rocmfp8 *>(weight->data), s.activation,
        s.top_ids, s.gathered_q8);

    float * output = static_cast<float *>(dst->data);
    fill_negative_infinity<<<dim3((PF_OUTPUT_VOCAB + 255) / 256), dim3(256), 0, stream>>>(
        output, PF_OUTPUT_VOCAB);
    scatter_exact_candidates<<<dim3(1), dim3(32), 0, stream>>>(
        output, s.top_ids, s.gathered_q8);
    return true;
}
