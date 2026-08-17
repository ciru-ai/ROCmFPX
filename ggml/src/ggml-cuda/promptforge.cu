#include "promptforge.cuh"

#include "common.cuh"
#include "ggml.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm_multiply_multiply.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr int PF_LAYERS = 64;
constexpr int PF_M = 2048;
constexpr int PF_CHECKPOINT_M = 2044;
constexpr int PF_TAIL_M = 1476;
constexpr int PF_H = 5120;
constexpr int PF_I = 17408;
constexpr uint64_t PF_FILE_BYTES = 17123004416ULL;
constexpr uint32_t PF_ENTRY_COUNT = 256;
constexpr int PF_GDN_N = 16384;
constexpr int PF_GDN_QKV_N = 10240;
constexpr int PF_GDN_Z_N = 6144;
constexpr uint64_t PF_GDN_FILE_BYTES = 4029685760ULL;
constexpr uint32_t PF_GDN_ENTRY_COUNT = 96;
constexpr uint64_t PF_GDN_DATA_OFFSET = 8192;

constexpr uint16_t PF_GATE_W8 = 0;
constexpr uint16_t PF_DOWN_W8 = 1;
constexpr uint16_t PF_GATE_SCALE = 2;
constexpr uint16_t PF_DOWN_SCALE = 3;
constexpr uint16_t PF_GDN_QKVZ_W8 = 4;
constexpr uint16_t PF_GDN_QKVZ_SCALE = 5;
constexpr uint8_t PF_S8 = 1;
constexpr uint8_t PF_F32 = 2;

#pragma pack(push, 1)
struct PFHeader {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t entry_bytes;
    uint32_t entry_count;
    uint64_t table_offset;
    uint64_t table_bytes;
    uint64_t data_offset;
    uint64_t file_bytes;
    uint64_t reserved;
};

struct PFEntry {
    uint16_t layer;
    uint16_t kind;
    uint8_t dtype;
    uint8_t rank;
    uint16_t reserved0;
    uint32_t rows;
    uint32_t cols;
    uint64_t offset;
    uint64_t length;
    uint8_t reserved1[32];
};
#pragma pack(pop)

static_assert(sizeof(PFHeader) == 64, "PromptForge header size");
static_assert(sizeof(PFEntry) == 64, "PromptForge entry size");

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using MultiplyMultiply = ck::tensor_operation::element_wise::MultiplyMultiply;
using CKDeviceOp = ck::tensor_operation::device::DeviceGemmMultipleDSplitK<
    Row, Col, ck::Tuple<Row, Col>, Row,
    int8_t, int8_t, ck::Tuple<float, float>, ck::bhalf_t,
    PassThrough, PassThrough, MultiplyMultiply>;

constexpr int PF_GATE_CK_INDEX_M2048 = 0;
constexpr int PF_DOWN_CK_INDEX_M2048 = 1;
constexpr int PF_GDN_CK_INDEX_M2048 = 1;
constexpr int PF_GDN_CK_INDEX_M2044 = 1;
constexpr int PF_GDN_CK_INDEX_M1476 = 1;
// The padded route used by the 1,476-row tail also supports the checkpoint-shaped 2,044-row block.
constexpr int PF_GATE_CK_INDEX_M2044 = 20;
constexpr int PF_DOWN_CK_INDEX_M2044 = 20;
// Validated gfx1151 CK route indices for the 1,476-row prompt tail.
constexpr int PF_GATE_CK_INDEX_M1476_STAGED = 20;
constexpr int PF_DOWN_CK_INDEX_M1476_STAGED = 20;

enum class PFMode { disabled, resident, m2048, m2048_fused, m2048_fused_tail1476 };
enum class PFGDNKind { none, qkv, z };

struct PFState {
    std::mutex init_mutex;
    bool initialized = false;
    bool ready = false;
    int device = -1;
    PFMode mode = PFMode::disabled;
    uint8_t * device_file = nullptr;
    bool gdn_enabled = false;
    uint8_t * gdn_device_file = nullptr;

    std::array<uint64_t, PF_LAYERS> gate_weight{};
    std::array<uint64_t, PF_LAYERS> gate_scale{};
    std::array<uint64_t, PF_LAYERS> down_weight{};
    std::array<uint64_t, PF_LAYERS> down_scale{};
    std::array<uint64_t, PF_LAYERS> gdn_weight{};
    std::array<uint64_t, PF_LAYERS> gdn_scale{};

    int8_t * gate_a8 = nullptr;
    float * gate_a_scale = nullptr;
    __hip_bfloat16 * gate_out = nullptr;
    int8_t * down_a8 = nullptr;
    float * down_a_scale = nullptr;
    __hip_bfloat16 * down_out = nullptr;

    std::unique_ptr<CKDeviceOp> gate_op;
    std::unique_ptr<CKDeviceOp> down_op;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> gate_invoker;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> down_invoker;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> gate_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> down_args;
    std::unique_ptr<CKDeviceOp> gate_checkpoint2044_op;
    std::unique_ptr<CKDeviceOp> down_checkpoint2044_op;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> gate_checkpoint2044_invoker;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> down_checkpoint2044_invoker;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> gate_checkpoint2044_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> down_checkpoint2044_args;
    std::unique_ptr<CKDeviceOp> gate_tail1476_op;
    std::unique_ptr<CKDeviceOp> down_tail1476_op;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> gate_tail1476_invoker;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> down_tail1476_invoker;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> gate_tail1476_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> down_tail1476_args;
    std::unique_ptr<CKDeviceOp> gdn_op;
    std::unique_ptr<CKDeviceOp> gdn_checkpoint2044_op;
    std::unique_ptr<CKDeviceOp> gdn_tail1476_op;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> gdn_invoker;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> gdn_checkpoint2044_invoker;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> gdn_tail1476_invoker;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> gdn_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> gdn_checkpoint2044_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> gdn_tail1476_args;

    bool count_active = false;
    int gate_count = 0;
    int down_count = 0;
    int pack_count = 0;
    int fallback_count = 0;
    const ggml_tensor * pending_down_input = nullptr;
    int pending_down_layer = -1;
    int pending_down_rows = 0;
    int active_rows = 0;
    int fused_down_pack_count = 0;
    int gdn_qkvz_count = 0;
    int gdn_pack_count = 0;
    int gdn_qkv_write_count = 0;
    int gdn_z_write_count = 0;
    int gdn_pair_miss_count = 0;
    int gdn_fallback_count = 0;
    int pending_gdn_layer = -1;
    int pending_gdn_rows = 0;
    const void * pending_gdn_input = nullptr;
    PFGDNKind pending_gdn_kind = PFGDNKind::none;
};

PFState & state() {
    static PFState * singleton = new PFState();
    return *singleton;
}

bool hip_check(hipError_t status, const char * what) {
    if (status == hipSuccess) {
        return true;
    }
    std::fprintf(stderr, "promptforge: %s failed: %s\n", what, hipGetErrorString(status));
    return false;
}

bool exact_tensor(const ggml_tensor * tensor, ggml_type type, int64_t ne0, int64_t ne1) {
    return tensor && tensor->type == type && tensor->ne[0] == ne0 && tensor->ne[1] == ne1 &&
           tensor->ne[2] == 1 && tensor->ne[3] == 1 && ggml_is_contiguous(tensor);
}

bool exact_shape(const ggml_tensor * tensor, int64_t ne0, int64_t ne1) {
    return tensor && tensor->ne[0] == ne0 && tensor->ne[1] == ne1 &&
           tensor->ne[2] == 1 && tensor->ne[3] == 1 && ggml_is_contiguous(tensor);
}

int exact_layer(const ggml_tensor * weight, const char * suffix) {
    if (!weight) {
        return -1;
    }
    char expected[GGML_MAX_NAME];
    for (int layer = 0; layer < PF_LAYERS; ++layer) {
        std::snprintf(expected, sizeof(expected), "blk.%d.%s", layer, suffix);
        if (std::strcmp(weight->name, expected) == 0) {
            return layer;
        }
    }
    return -1;
}

void begin_request_if_needed(PFState & s, int rows) {
    if (!s.count_active) {
        if (s.pending_gdn_kind != PFGDNKind::none) {
            ++s.gdn_pair_miss_count;
            std::fprintf(stderr, "promptforge: stale GDN QKVZ pair at request boundary (layer %d)\n",
                         s.pending_gdn_layer);
            std::abort();
        }
        s.count_active = true;
        s.gate_count = s.down_count = s.pack_count = s.fallback_count = 0;
        s.fused_down_pack_count = 0;
        s.gdn_qkvz_count = s.gdn_pack_count = 0;
        s.gdn_qkv_write_count = s.gdn_z_write_count = 0;
        s.gdn_pair_miss_count = s.gdn_fallback_count = 0;
        s.active_rows = rows;
        return;
    }
    if (s.active_rows != rows) {
        std::fprintf(stderr, "promptforge: active request row mismatch (%d vs %d)\n", s.active_rows, rows);
        std::abort();
    }
}

void emit_request_telemetry(PFState & s) {
    if (s.pending_gdn_kind != PFGDNKind::none) {
        ++s.gdn_pair_miss_count;
        std::fprintf(stderr, "promptforge: stale GDN QKVZ pair at request summary (layer %d)\n",
                     s.pending_gdn_layer);
        std::abort();
    }
    std::fprintf(stderr,
        "{\"record\":\"promptforge_request\",\"gate_up\":%d,\"down\":%d,"
        "\"packs\":%d,\"fallback\":%d,\"fused_down_pack\":%d,"
        "\"gdn_qkvz\":%d,\"gdn_packs\":%d,\"gdn_qkv_writes\":%d,"
        "\"gdn_z_writes\":%d,\"gdn_pair_miss\":%d,\"gdn_fallback\":%d,"
        "\"gdn_projection\":\"qkvz_w8\",\"rows\":%d}\n",
        s.gate_count, s.down_count, s.pack_count, s.fallback_count,
        s.fused_down_pack_count, s.gdn_qkvz_count, s.gdn_pack_count,
        s.gdn_qkv_write_count, s.gdn_z_write_count, s.gdn_pair_miss_count,
        s.gdn_fallback_count, s.active_rows);
    std::fflush(stderr);
    s.count_active = false;
}

__global__ void dynamic_a8_pack(const float * src, int8_t * dst, float * row_scale, int M, int K) {
    const int row = blockIdx.x;
    if (row >= M) return;
    __shared__ float smem[256];
    float vmax = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        vmax = fmaxf(vmax, fabsf(src[size_t(row) * K + k]));
    }
    smem[threadIdx.x] = vmax;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        __syncthreads();
    }
    const float scale = smem[0] > 0.0f ? smem[0] / 127.0f : 1.0f;
    if (threadIdx.x == 0) row_scale[row] = scale;
    __syncthreads();
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        int q = __float2int_rn(src[size_t(row) * K + k] / scale);
        q = max(-127, min(127, q));
        dst[size_t(row) * K + k] = int8_t(q);
    }
}

__global__ void split_silu_multiply(const __hip_bfloat16 * src, float * dst, int M, int N) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = size_t(M) * N;
    if (i < count) {
        const int row = int(i / N);
        const int col = int(i - size_t(row) * N);
        const float gate = __bfloat162float(src[size_t(row) * (2 * N) + col]);
        const float up = __bfloat162float(src[size_t(row) * (2 * N) + N + col]);
        dst[i] = (gate / (1.0f + expf(-gate))) * up;
    }
}

constexpr int PF_FUSED_PACK_THREADS = 512;
constexpr int PF_FUSED_PACK_VALUES = 34;
static_assert(PF_FUSED_PACK_THREADS * PF_FUSED_PACK_VALUES == PF_I, "fused pack shape");

__global__ void swiglu_row_a8_pack(const __hip_bfloat16 * src, int8_t * dst, float * row_scale, int rows) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float smem[PF_FUSED_PACK_THREADS];
    float values[PF_FUSED_PACK_VALUES];
    float vmax = 0.0f;
#pragma unroll
    for (int value = 0; value < PF_FUSED_PACK_VALUES; ++value) {
        const int col = threadIdx.x + value * PF_FUSED_PACK_THREADS;
        const float gate = __bfloat162float(src[size_t(row) * (2 * PF_I) + col]);
        const float up = __bfloat162float(src[size_t(row) * (2 * PF_I) + PF_I + col]);
        values[value] = (gate / (1.0f + expf(-gate))) * up;
        vmax = fmaxf(vmax, fabsf(values[value]));
    }
    smem[threadIdx.x] = vmax;
    __syncthreads();
    for (int stride = PF_FUSED_PACK_THREADS / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        __syncthreads();
    }
    const float scale = smem[0] > 0.0f ? smem[0] / 127.0f : 1.0f;
    if (threadIdx.x == 0) row_scale[row] = scale;
    __syncthreads();
#pragma unroll
    for (int value = 0; value < PF_FUSED_PACK_VALUES; ++value) {
        const int col = threadIdx.x + value * PF_FUSED_PACK_THREADS;
        int q = __float2int_rn(values[value] / scale);
        q = max(-127, min(127, q));
        dst[size_t(row) * PF_I + col] = int8_t(q);
    }
}

__global__ void bf16_to_f32(const __hip_bfloat16 * src, float * dst, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = __bfloat162float(src[i]);
}

__global__ void bf16_slice_to_f32(const __hip_bfloat16 * src, float * dst,
                                  int rows, int src_stride, int src_col, int dst_cols) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = size_t(rows) * dst_cols;
    if (i < count) {
        const int row = int(i / dst_cols);
        const int col = int(i - size_t(row) * dst_cols);
        dst[i] = __bfloat162float(src[size_t(row) * src_stride + src_col + col]);
    }
}

bool validate_entry(const PFEntry & e, int layer, uint16_t kind, uint8_t dtype,
                    uint8_t rank, uint32_t rows, uint32_t cols, uint64_t length) {
    return e.layer == layer && e.kind == kind && e.dtype == dtype && e.rank == rank &&
           e.rows == rows && e.cols == cols && e.length == length &&
           e.offset <= PF_FILE_BYTES && e.length <= PF_FILE_BYTES - e.offset;
}

bool load_sidecar(PFState & s, const char * path) {
    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        std::fprintf(stderr, "promptforge: cannot open sidecar %s\n", path);
        return false;
    }
    struct stat st {};
    if (fstat(fd, &st) != 0 || uint64_t(st.st_size) != PF_FILE_BYTES) {
        std::fprintf(stderr, "promptforge: wrong sidecar size\n");
        close(fd);
        return false;
    }
    void * mapping = mmap(nullptr, PF_FILE_BYTES, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) {
        std::fprintf(stderr, "promptforge: sidecar mmap failed\n");
        close(fd);
        return false;
    }

    const auto * header = static_cast<const PFHeader *>(mapping);
    const char expected_magic[8] = {'P','F','S','I','D','E','1','\0'};
    bool valid = std::memcmp(header->magic, expected_magic, 8) == 0 && header->version == 1 &&
                 header->header_bytes == sizeof(PFHeader) && header->entry_bytes == sizeof(PFEntry) &&
                 header->entry_count == PF_ENTRY_COUNT && header->table_offset == sizeof(PFHeader) &&
                 header->table_bytes == uint64_t(PF_ENTRY_COUNT) * sizeof(PFEntry) &&
                 header->data_offset == 20480 && header->file_bytes == PF_FILE_BYTES;
    const auto * entries = reinterpret_cast<const PFEntry *>(static_cast<const uint8_t *>(mapping) + header->table_offset);
    uint64_t next_offset = header->data_offset;
    for (int layer = 0; valid && layer < PF_LAYERS; ++layer) {
        const PFEntry & gw = entries[layer * 4 + 0];
        const PFEntry & gs = entries[layer * 4 + 1];
        const PFEntry & dw = entries[layer * 4 + 2];
        const PFEntry & ds = entries[layer * 4 + 3];
        valid = validate_entry(gw, layer, PF_GATE_W8, PF_S8, 2, 2 * PF_I, PF_H, uint64_t(2 * PF_I) * PF_H) &&
                validate_entry(gs, layer, PF_GATE_SCALE, PF_F32, 1, 2 * PF_I, 1, uint64_t(2 * PF_I) * sizeof(float)) &&
                validate_entry(dw, layer, PF_DOWN_W8, PF_S8, 2, PF_H, PF_I, uint64_t(PF_H) * PF_I) &&
                validate_entry(ds, layer, PF_DOWN_SCALE, PF_F32, 1, PF_H, 1, uint64_t(PF_H) * sizeof(float));
        for (const PFEntry * e : {&gw, &gs, &dw, &ds}) {
            valid = valid && e->offset == next_offset;
            next_offset += e->length;
        }
        s.gate_weight[layer] = gw.offset;
        s.gate_scale[layer] = gs.offset;
        s.down_weight[layer] = dw.offset;
        s.down_scale[layer] = ds.offset;
    }
    valid = valid && next_offset == PF_FILE_BYTES;
    if (!valid) {
        std::fprintf(stderr, "promptforge: invalid PFSIDE1 table\n");
        munmap(mapping, PF_FILE_BYTES);
        close(fd);
        return false;
    }

    if (!hip_check(hipMalloc(&s.device_file, PF_FILE_BYTES), "hipMalloc(sidecar)") ||
        !hip_check(hipMemcpy(s.device_file, mapping, PF_FILE_BYTES, hipMemcpyHostToDevice), "hipMemcpy(sidecar)")) {
        munmap(mapping, PF_FILE_BYTES);
        close(fd);
        return false;
    }
    madvise(mapping, PF_FILE_BYTES, MADV_DONTNEED);
    munmap(mapping, PF_FILE_BYTES);
    posix_fadvise(fd, 0, PF_FILE_BYTES, POSIX_FADV_DONTNEED);
    close(fd);
    return true;
}

bool validate_gdn_entry(const PFEntry & e, int layer, uint16_t kind, uint8_t dtype,
                        uint8_t rank, uint32_t rows, uint32_t cols, uint64_t length) {
    return e.layer == layer && e.kind == kind && e.dtype == dtype && e.rank == rank &&
           e.rows == rows && e.cols == cols && e.length == length &&
           e.offset <= PF_GDN_FILE_BYTES && e.length <= PF_GDN_FILE_BYTES - e.offset;
}

bool load_gdn_sidecar(PFState & s, const char * path) {
    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        std::fprintf(stderr, "promptforge: cannot open GDN sidecar %s\n", path);
        return false;
    }
    struct stat st {};
    if (fstat(fd, &st) != 0 || uint64_t(st.st_size) != PF_GDN_FILE_BYTES) {
        std::fprintf(stderr, "promptforge: wrong GDN sidecar size\n");
        close(fd);
        return false;
    }

    std::array<uint8_t, PF_GDN_DATA_OFFSET> header_table{};
    if (pread(fd, header_table.data(), header_table.size(), 0) != ssize_t(header_table.size())) {
        std::fprintf(stderr, "promptforge: cannot read bounded GDN header/table\n");
        close(fd);
        return false;
    }
    PFHeader header {};
    std::memcpy(&header, header_table.data(), sizeof(header));
    const char expected_magic[8] = {'P','F','S','I','D','E','1','\0'};
    bool valid = std::memcmp(header.magic, expected_magic, 8) == 0 && header.version == 1 &&
                 header.header_bytes == sizeof(PFHeader) && header.entry_bytes == sizeof(PFEntry) &&
                 header.entry_count == PF_GDN_ENTRY_COUNT && header.table_offset == sizeof(PFHeader) &&
                 header.table_bytes == uint64_t(PF_GDN_ENTRY_COUNT) * sizeof(PFEntry) &&
                 header.data_offset == PF_GDN_DATA_OFFSET && header.file_bytes == PF_GDN_FILE_BYTES;
    uint64_t next_offset = PF_GDN_DATA_OFFSET;
    int entry_index = 0;
    for (int layer = 0; valid && layer < PF_LAYERS; ++layer) {
        if (layer % 4 == 3) {
            continue;
        }
        PFEntry weight {};
        PFEntry scale {};
        const size_t weight_pos = size_t(header.table_offset) + size_t(entry_index++) * sizeof(PFEntry);
        const size_t scale_pos = size_t(header.table_offset) + size_t(entry_index++) * sizeof(PFEntry);
        std::memcpy(&weight, header_table.data() + weight_pos, sizeof(weight));
        std::memcpy(&scale, header_table.data() + scale_pos, sizeof(scale));
        valid = validate_gdn_entry(weight, layer, PF_GDN_QKVZ_W8, PF_S8, 2,
                                   PF_GDN_N, PF_H, uint64_t(PF_GDN_N) * PF_H) &&
                validate_gdn_entry(scale, layer, PF_GDN_QKVZ_SCALE, PF_F32, 1,
                                   PF_GDN_N, 1, uint64_t(PF_GDN_N) * sizeof(float)) &&
                weight.offset == next_offset;
        next_offset += weight.length;
        valid = valid && scale.offset == next_offset;
        next_offset += scale.length;
        s.gdn_weight[layer] = weight.offset;
        s.gdn_scale[layer] = scale.offset;
    }
    valid = valid && entry_index == int(PF_GDN_ENTRY_COUNT) && next_offset == PF_GDN_FILE_BYTES;
    if (!valid) {
        std::fprintf(stderr, "promptforge: invalid bounded GDN PFSIDE1 header/table\n");
        close(fd);
        return false;
    }

    void * mapping = mmap(nullptr, PF_GDN_FILE_BYTES, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) {
        std::fprintf(stderr, "promptforge: GDN sidecar mmap failed\n");
        close(fd);
        return false;
    }
    if (!hip_check(hipMalloc(&s.gdn_device_file, PF_GDN_FILE_BYTES), "hipMalloc(GDN sidecar)") ||
        !hip_check(hipMemcpy(s.gdn_device_file, mapping, PF_GDN_FILE_BYTES, hipMemcpyHostToDevice),
                   "hipMemcpy(GDN sidecar)")) {
        munmap(mapping, PF_GDN_FILE_BYTES);
        close(fd);
        return false;
    }
    madvise(mapping, PF_GDN_FILE_BYTES, MADV_DONTNEED);
    munmap(mapping, PF_GDN_FILE_BYTES);
    posix_fadvise(fd, 0, PF_GDN_FILE_BYTES, POSIX_FADV_DONTNEED);
    close(fd);
    return true;
}

bool allocate_scratch(PFState & s) {
    return hip_check(hipMalloc(&s.gate_a8, size_t(PF_M) * PF_H), "hipMalloc(gate_a8)") &&
           hip_check(hipMalloc(&s.gate_a_scale, size_t(PF_M) * sizeof(float)), "hipMalloc(gate_scale)") &&
           hip_check(hipMalloc(&s.gate_out, size_t(PF_M) * (2 * PF_I) * sizeof(__hip_bfloat16)), "hipMalloc(gate_out)") &&
           hip_check(hipMalloc(&s.down_a8, size_t(PF_M) * PF_I), "hipMalloc(down_a8)") &&
           hip_check(hipMalloc(&s.down_a_scale, size_t(PF_M) * sizeof(float)), "hipMalloc(down_scale)") &&
           hip_check(hipMalloc(&s.down_out, size_t(PF_M) * PF_H * sizeof(__hip_bfloat16)), "hipMalloc(down_out)");
}

bool build_routes(PFState & s) {
    std::vector<std::unique_ptr<CKDeviceOp>> gate_ops;
    std::vector<std::unique_ptr<CKDeviceOp>> down_ops;
    ck::tensor_operation::device::instance::
        add_device_gemm_multiply_multiply_wmma_c_shuffle_i8_i8_bf16_km_nk_mn_instances(gate_ops);
    ck::tensor_operation::device::instance::
        add_device_gemm_multiply_multiply_wmma_c_shuffle_i8_i8_bf16_km_nk_mn_instances(down_ops);
    if (gate_ops.size() <= PF_GATE_CK_INDEX_M2048 || down_ops.size() <= PF_DOWN_CK_INDEX_M2048) {
        std::fprintf(stderr, "promptforge: required CK indices absent\n");
        return false;
    }
    s.gate_op = std::move(gate_ops[PF_GATE_CK_INDEX_M2048]);
    s.down_op = std::move(down_ops[PF_DOWN_CK_INDEX_M2048]);
    for (int layer = 0; layer < PF_LAYERS; ++layer) {
        auto gate_arg = s.gate_op->MakeArgumentPointer(
            s.gate_a8, s.device_file + s.gate_weight[layer],
            std::array<const void *, 2>{s.device_file + s.gate_scale[layer], s.gate_a_scale}, s.gate_out,
            PF_M, 2 * PF_I, PF_H, PF_H, PF_H,
            std::array<ck::index_t, 2>{0, 0}, 2 * PF_I, 1,
            PassThrough{}, PassThrough{}, MultiplyMultiply{});
        auto down_arg = s.down_op->MakeArgumentPointer(
            s.down_a8, s.device_file + s.down_weight[layer],
            std::array<const void *, 2>{s.device_file + s.down_scale[layer], s.down_a_scale}, s.down_out,
            PF_M, PF_H, PF_I, PF_I, PF_I,
            std::array<ck::index_t, 2>{0, 0}, PF_H, 1,
            PassThrough{}, PassThrough{}, MultiplyMultiply{});
        if (!s.gate_op->IsSupportedArgument(gate_arg.get()) || !s.down_op->IsSupportedArgument(down_arg.get())) {
            std::fprintf(stderr, "promptforge: CK argument unsupported for layer %d\n", layer);
            return false;
        }
        s.gate_args[layer] = std::move(gate_arg);
        s.down_args[layer] = std::move(down_arg);
    }
    s.gate_invoker = s.gate_op->MakeInvokerPointer();
    s.down_invoker = s.down_op->MakeInvokerPointer();

    if (s.mode == PFMode::m2048_fused_tail1476) {
        std::vector<std::unique_ptr<CKDeviceOp>> gate_checkpoint_ops;
        std::vector<std::unique_ptr<CKDeviceOp>> down_checkpoint_ops;
        std::vector<std::unique_ptr<CKDeviceOp>> gate_tail_ops;
        std::vector<std::unique_ptr<CKDeviceOp>> down_tail_ops;
        ck::tensor_operation::device::instance::
            add_device_gemm_multiply_multiply_wmma_c_shuffle_i8_i8_bf16_km_nk_mn_instances(gate_checkpoint_ops);
        ck::tensor_operation::device::instance::
            add_device_gemm_multiply_multiply_wmma_c_shuffle_i8_i8_bf16_km_nk_mn_instances(down_checkpoint_ops);
        ck::tensor_operation::device::instance::
            add_device_gemm_multiply_multiply_wmma_c_shuffle_i8_i8_bf16_km_nk_mn_instances(gate_tail_ops);
        ck::tensor_operation::device::instance::
            add_device_gemm_multiply_multiply_wmma_c_shuffle_i8_i8_bf16_km_nk_mn_instances(down_tail_ops);
        if (gate_checkpoint_ops.size() <= PF_GATE_CK_INDEX_M2044 ||
            down_checkpoint_ops.size() <= PF_DOWN_CK_INDEX_M2044 ||
            gate_tail_ops.size() <= PF_GATE_CK_INDEX_M1476_STAGED ||
            down_tail_ops.size() <= PF_DOWN_CK_INDEX_M1476_STAGED) {
            std::fprintf(stderr, "promptforge: checkpoint/tail CK indices absent\n");
            return false;
        }
        s.gate_checkpoint2044_op = std::move(gate_checkpoint_ops[PF_GATE_CK_INDEX_M2044]);
        s.down_checkpoint2044_op = std::move(down_checkpoint_ops[PF_DOWN_CK_INDEX_M2044]);
        s.gate_tail1476_op = std::move(gate_tail_ops[PF_GATE_CK_INDEX_M1476_STAGED]);
        s.down_tail1476_op = std::move(down_tail_ops[PF_DOWN_CK_INDEX_M1476_STAGED]);
        for (int layer = 0; layer < PF_LAYERS; ++layer) {
            auto checkpoint_gate_arg = s.gate_checkpoint2044_op->MakeArgumentPointer(
                s.gate_a8, s.device_file + s.gate_weight[layer],
                std::array<const void *, 2>{s.device_file + s.gate_scale[layer], s.gate_a_scale}, s.gate_out,
                PF_CHECKPOINT_M, 2 * PF_I, PF_H, PF_H, PF_H,
                std::array<ck::index_t, 2>{0, 0}, 2 * PF_I, 1,
                PassThrough{}, PassThrough{}, MultiplyMultiply{});
            auto checkpoint_down_arg = s.down_checkpoint2044_op->MakeArgumentPointer(
                s.down_a8, s.device_file + s.down_weight[layer],
                std::array<const void *, 2>{s.device_file + s.down_scale[layer], s.down_a_scale}, s.down_out,
                PF_CHECKPOINT_M, PF_H, PF_I, PF_I, PF_I,
                std::array<ck::index_t, 2>{0, 0}, PF_H, 1,
                PassThrough{}, PassThrough{}, MultiplyMultiply{});
            auto gate_arg = s.gate_tail1476_op->MakeArgumentPointer(
                s.gate_a8, s.device_file + s.gate_weight[layer],
                std::array<const void *, 2>{s.device_file + s.gate_scale[layer], s.gate_a_scale}, s.gate_out,
                PF_TAIL_M, 2 * PF_I, PF_H, PF_H, PF_H,
                std::array<ck::index_t, 2>{0, 0}, 2 * PF_I, 1,
                PassThrough{}, PassThrough{}, MultiplyMultiply{});
            auto down_arg = s.down_tail1476_op->MakeArgumentPointer(
                s.down_a8, s.device_file + s.down_weight[layer],
                std::array<const void *, 2>{s.device_file + s.down_scale[layer], s.down_a_scale}, s.down_out,
                PF_TAIL_M, PF_H, PF_I, PF_I, PF_I,
                std::array<ck::index_t, 2>{0, 0}, PF_H, 1,
                PassThrough{}, PassThrough{}, MultiplyMultiply{});
            if (!s.gate_checkpoint2044_op->IsSupportedArgument(checkpoint_gate_arg.get()) ||
                !s.down_checkpoint2044_op->IsSupportedArgument(checkpoint_down_arg.get()) ||
                !s.gate_tail1476_op->IsSupportedArgument(gate_arg.get()) ||
                !s.down_tail1476_op->IsSupportedArgument(down_arg.get())) {
                std::fprintf(stderr, "promptforge: checkpoint/tail CK argument unsupported for layer %d\n", layer);
                return false;
            }
            s.gate_checkpoint2044_args[layer] = std::move(checkpoint_gate_arg);
            s.down_checkpoint2044_args[layer] = std::move(checkpoint_down_arg);
            s.gate_tail1476_args[layer] = std::move(gate_arg);
            s.down_tail1476_args[layer] = std::move(down_arg);
        }
        s.gate_checkpoint2044_invoker = s.gate_checkpoint2044_op->MakeInvokerPointer();
        s.down_checkpoint2044_invoker = s.down_checkpoint2044_op->MakeInvokerPointer();
        s.gate_tail1476_invoker = s.gate_tail1476_op->MakeInvokerPointer();
        s.down_tail1476_invoker = s.down_tail1476_op->MakeInvokerPointer();
    }
    if (s.mode == PFMode::m2048_fused_tail1476) {
        std::fprintf(stderr,
            "{\"record\":\"promptforge_init\",\"mode\":\"m2048_fused_tail1476\","
            "\"device_bytes\":%llu,\"gate_ck_index\":%d,\"down_ck_index\":%d,"
            "\"gate_ck_index_m2044\":%d,\"down_ck_index_m2044\":%d,"
            "\"gate_ck_index_m1476\":%d,\"down_ck_index_m1476\":%d}\n",
            (unsigned long long) PF_FILE_BYTES,
            PF_GATE_CK_INDEX_M2048, PF_DOWN_CK_INDEX_M2048,
            PF_GATE_CK_INDEX_M2044, PF_DOWN_CK_INDEX_M2044,
            PF_GATE_CK_INDEX_M1476_STAGED, PF_DOWN_CK_INDEX_M1476_STAGED);
    } else {
        std::fprintf(stderr,
            "{\"record\":\"promptforge_init\",\"mode\":\"%s\",\"device_bytes\":%llu,"
            "\"gate_ck_index\":0,\"down_ck_index\":1}\n",
            s.mode == PFMode::m2048 ? "m2048" :
            s.mode == PFMode::m2048_fused ? "m2048_fused" : "resident",
            (unsigned long long) PF_FILE_BYTES);
    }
    return true;
}

bool build_gdn_routes(PFState & s) {
    std::vector<std::unique_ptr<CKDeviceOp>> gdn_ops;
    std::vector<std::unique_ptr<CKDeviceOp>> gdn_checkpoint2044_ops;
    std::vector<std::unique_ptr<CKDeviceOp>> gdn_tail1476_ops;
    ck::tensor_operation::device::instance::
        add_device_gemm_multiply_multiply_wmma_c_shuffle_i8_i8_bf16_km_nk_mn_instances(gdn_ops);
    ck::tensor_operation::device::instance::
        add_device_gemm_multiply_multiply_wmma_c_shuffle_i8_i8_bf16_km_nk_mn_instances(gdn_checkpoint2044_ops);
    ck::tensor_operation::device::instance::
        add_device_gemm_multiply_multiply_wmma_c_shuffle_i8_i8_bf16_km_nk_mn_instances(gdn_tail1476_ops);
    if (gdn_ops.size() <= PF_GDN_CK_INDEX_M2048 ||
        gdn_checkpoint2044_ops.size() <= PF_GDN_CK_INDEX_M2044 ||
        gdn_tail1476_ops.size() <= PF_GDN_CK_INDEX_M1476) {
        std::fprintf(stderr, "promptforge: required GDN CK indices absent\n");
        return false;
    }
    s.gdn_op = std::move(gdn_ops[PF_GDN_CK_INDEX_M2048]);
    s.gdn_checkpoint2044_op = std::move(gdn_checkpoint2044_ops[PF_GDN_CK_INDEX_M2044]);
    s.gdn_tail1476_op = std::move(gdn_tail1476_ops[PF_GDN_CK_INDEX_M1476]);
    for (int layer = 0; layer < PF_LAYERS; ++layer) {
        if (layer % 4 == 3) {
            continue;
        }
        auto arg = s.gdn_op->MakeArgumentPointer(
            s.gate_a8, s.gdn_device_file + s.gdn_weight[layer],
            std::array<const void *, 2>{s.gdn_device_file + s.gdn_scale[layer], s.gate_a_scale}, s.gate_out,
            PF_M, PF_GDN_N, PF_H, PF_H, PF_H,
            std::array<ck::index_t, 2>{0, 0}, PF_GDN_N, 1,
            PassThrough{}, PassThrough{}, MultiplyMultiply{});
        auto checkpoint_arg = s.gdn_checkpoint2044_op->MakeArgumentPointer(
            s.gate_a8, s.gdn_device_file + s.gdn_weight[layer],
            std::array<const void *, 2>{s.gdn_device_file + s.gdn_scale[layer], s.gate_a_scale}, s.gate_out,
            PF_CHECKPOINT_M, PF_GDN_N, PF_H, PF_H, PF_H,
            std::array<ck::index_t, 2>{0, 0}, PF_GDN_N, 1,
            PassThrough{}, PassThrough{}, MultiplyMultiply{});
        auto tail_arg = s.gdn_tail1476_op->MakeArgumentPointer(
            s.gate_a8, s.gdn_device_file + s.gdn_weight[layer],
            std::array<const void *, 2>{s.gdn_device_file + s.gdn_scale[layer], s.gate_a_scale}, s.gate_out,
            PF_TAIL_M, PF_GDN_N, PF_H, PF_H, PF_H,
            std::array<ck::index_t, 2>{0, 0}, PF_GDN_N, 1,
            PassThrough{}, PassThrough{}, MultiplyMultiply{});
        if (!s.gdn_op->IsSupportedArgument(arg.get()) ||
            !s.gdn_checkpoint2044_op->IsSupportedArgument(checkpoint_arg.get()) ||
            !s.gdn_tail1476_op->IsSupportedArgument(tail_arg.get())) {
            std::fprintf(stderr, "promptforge: GDN CK argument unsupported for layer %d\n", layer);
            return false;
        }
        s.gdn_args[layer] = std::move(arg);
        s.gdn_checkpoint2044_args[layer] = std::move(checkpoint_arg);
        s.gdn_tail1476_args[layer] = std::move(tail_arg);
    }
    s.gdn_invoker = s.gdn_op->MakeInvokerPointer();
    s.gdn_checkpoint2044_invoker = s.gdn_checkpoint2044_op->MakeInvokerPointer();
    s.gdn_tail1476_invoker = s.gdn_tail1476_op->MakeInvokerPointer();
    std::fprintf(stderr,
        "{\"record\":\"promptforge_gdn_init\",\"projection\":\"qkvz_w8\","
        "\"device_bytes\":%llu,\"layers\":48,\"n\":%d,\"k\":%d,"
        "\"ck_index_m2048\":%d,\"ck_index_m2044\":%d,\"ck_index_m1476\":%d}\n",
        (unsigned long long) PF_GDN_FILE_BYTES, PF_GDN_N, PF_H,
        PF_GDN_CK_INDEX_M2048, PF_GDN_CK_INDEX_M2044, PF_GDN_CK_INDEX_M1476);
    return true;
}

void run_gate(PFState & s, int layer, const float * input, float * output, hipStream_t stream) {
    hipLaunchKernelGGL(dynamic_a8_pack, dim3(PF_M), dim3(256), 0, stream,
                       input, s.gate_a8, s.gate_a_scale, PF_M, PF_H);
    s.gate_invoker->Run(s.gate_args[layer].get(), ::StreamConfig{stream, false});
    const size_t count = size_t(PF_M) * PF_I;
    hipLaunchKernelGGL(split_silu_multiply, dim3((count + 255) / 256), dim3(256), 0, stream,
                       s.gate_out, output, PF_M, PF_I);
}

void run_gate_fused(PFState & s, int layer, int rows, const float * input, hipStream_t stream) {
    hipLaunchKernelGGL(dynamic_a8_pack, dim3(rows), dim3(256), 0, stream,
                       input, s.gate_a8, s.gate_a_scale, rows, PF_H);
    if (rows == PF_TAIL_M) {
        s.gate_tail1476_invoker->Run(s.gate_tail1476_args[layer].get(), ::StreamConfig{stream, false});
    } else if (rows == PF_CHECKPOINT_M) {
        s.gate_checkpoint2044_invoker->Run(s.gate_checkpoint2044_args[layer].get(), ::StreamConfig{stream, false});
    } else {
        s.gate_invoker->Run(s.gate_args[layer].get(), ::StreamConfig{stream, false});
    }
    hipLaunchKernelGGL(swiglu_row_a8_pack, dim3(rows), dim3(PF_FUSED_PACK_THREADS), 0, stream,
                       s.gate_out, s.down_a8, s.down_a_scale, rows);
}

void run_down(PFState & s, int layer, const float * input, float * output, hipStream_t stream) {
    hipLaunchKernelGGL(dynamic_a8_pack, dim3(PF_M), dim3(256), 0, stream,
                       input, s.down_a8, s.down_a_scale, PF_M, PF_I);
    s.down_invoker->Run(s.down_args[layer].get(), ::StreamConfig{stream, false});
    const size_t count = size_t(PF_M) * PF_H;
    hipLaunchKernelGGL(bf16_to_f32, dim3((count + 255) / 256), dim3(256), 0, stream,
                       s.down_out, output, count);
}

void run_down_fused(PFState & s, int layer, int rows, float * output, hipStream_t stream) {
    if (rows == PF_TAIL_M) {
        s.down_tail1476_invoker->Run(s.down_tail1476_args[layer].get(), ::StreamConfig{stream, false});
    } else if (rows == PF_CHECKPOINT_M) {
        s.down_checkpoint2044_invoker->Run(s.down_checkpoint2044_args[layer].get(), ::StreamConfig{stream, false});
    } else {
        s.down_invoker->Run(s.down_args[layer].get(), ::StreamConfig{stream, false});
    }
    s.pending_down_input = nullptr;
    s.pending_down_layer = -1;
    s.pending_down_rows = 0;
    const size_t count = size_t(rows) * PF_H;
    hipLaunchKernelGGL(bf16_to_f32, dim3((count + 255) / 256), dim3(256), 0, stream,
                       s.down_out, output, count);
}

void write_gdn_slice(PFState & s, PFGDNKind kind, int rows, float * output, hipStream_t stream) {
    const int src_col = kind == PFGDNKind::qkv ? 0 : PF_GDN_QKV_N;
    const int dst_cols = kind == PFGDNKind::qkv ? PF_GDN_QKV_N : PF_GDN_Z_N;
    const size_t count = size_t(rows) * dst_cols;
    hipLaunchKernelGGL(bf16_slice_to_f32, dim3((count + 255) / 256), dim3(256), 0, stream,
                       s.gate_out, output, rows, PF_GDN_N, src_col, dst_cols);
}

void run_gdn_qkvz(PFState & s, int layer, int rows, const float * input, hipStream_t stream) {
    hipLaunchKernelGGL(dynamic_a8_pack, dim3(rows), dim3(256), 0, stream,
                       input, s.gate_a8, s.gate_a_scale, rows, PF_H);
    if (rows == PF_TAIL_M) {
        s.gdn_tail1476_invoker->Run(s.gdn_tail1476_args[layer].get(), ::StreamConfig{stream, false});
    } else if (rows == PF_CHECKPOINT_M) {
        s.gdn_checkpoint2044_invoker->Run(s.gdn_checkpoint2044_args[layer].get(), ::StreamConfig{stream, false});
    } else {
        s.gdn_invoker->Run(s.gdn_args[layer].get(), ::StreamConfig{stream, false});
    }
}

} // namespace

bool promptforge_backend_init(int device) {
    PFState & s = state();
    std::lock_guard<std::mutex> lock(s.init_mutex);
    if (s.initialized) {
        return s.ready && ((s.mode == PFMode::disabled && !s.gdn_enabled) || s.device == device);
    }
    s.initialized = true;
    const char * gdn_sidecar = std::getenv("PROMPTFORGE_GDN_SIDECAR");
    s.gdn_enabled = gdn_sidecar && gdn_sidecar[0];
    const char * graph_opt = std::getenv("GGML_CUDA_GRAPH_OPT");
    if (s.gdn_enabled && graph_opt && std::strcmp(graph_opt, "1") == 0) {
        std::fprintf(stderr, "promptforge: PROMPTFORGE_GDN_SIDECAR rejects GGML_CUDA_GRAPH_OPT=1\n");
        return false;
    }
    const char * mode = std::getenv("PROMPTFORGE_MODE");
    if (!mode || !mode[0]) {
        s.mode = PFMode::disabled;
        if (!s.gdn_enabled) {
            s.ready = true;
            return true;
        }
    } else if (std::strcmp(mode, "resident") == 0) {
        s.mode = PFMode::resident;
    } else if (std::strcmp(mode, "m2048") == 0) {
        s.mode = PFMode::m2048;
    } else if (std::strcmp(mode, "m2048_fused") == 0) {
        s.mode = PFMode::m2048_fused;
    } else if (std::strcmp(mode, "m2048_fused_tail1476") == 0) {
        s.mode = PFMode::m2048_fused_tail1476;
    } else {
        std::fprintf(stderr, "promptforge: PROMPTFORGE_MODE must be resident, m2048, m2048_fused, or m2048_fused_tail1476\n");
        return false;
    }
    const char * sidecar = std::getenv("PROMPTFORGE_SIDECAR");
    if (s.mode != PFMode::disabled && (!sidecar || !sidecar[0])) {
        std::fprintf(stderr, "promptforge: PROMPTFORGE_SIDECAR is required\n");
        return false;
    }
    s.device = device;
    if (!hip_check(hipSetDevice(device), "hipSetDevice") ||
        (s.mode != PFMode::disabled && !load_sidecar(s, sidecar)) ||
        !allocate_scratch(s) ||
        (s.mode != PFMode::disabled && !build_routes(s)) ||
        (s.gdn_enabled && (!load_gdn_sidecar(s, gdn_sidecar) || !build_gdn_routes(s)))) {
        return false;
    }
    s.ready = true;
    return true;
}

bool promptforge_try_gdn_qkvz(ggml_backend_cuda_context * ctx,
                              const ggml_tensor * weight, const ggml_tensor * input, ggml_tensor * dst) {
    PFState & s = state();
    if (!s.ready || !s.gdn_enabled || !ctx) {
        return false;
    }

    PFGDNKind kind = PFGDNKind::none;
    int output_cols = 0;
    int layer = exact_layer(weight, "attn_qkv.weight");
    if (layer >= 0) {
        kind = PFGDNKind::qkv;
        output_cols = PF_GDN_QKV_N;
    } else {
        layer = exact_layer(weight, "attn_gate.weight");
        if (layer >= 0) {
            kind = PFGDNKind::z;
            output_cols = PF_GDN_Z_N;
        }
    }
    if (layer < 0 || layer % 4 == 3) {
        return false;
    }

    const int rows = exact_tensor(input, GGML_TYPE_F32, PF_H, PF_M) &&
                     exact_tensor(dst, GGML_TYPE_F32, output_cols, PF_M) ? PF_M :
                     exact_tensor(input, GGML_TYPE_F32, PF_H, PF_CHECKPOINT_M) &&
                     exact_tensor(dst, GGML_TYPE_F32, output_cols, PF_CHECKPOINT_M) ? PF_CHECKPOINT_M :
                     exact_tensor(input, GGML_TYPE_F32, PF_H, PF_TAIL_M) &&
                     exact_tensor(dst, GGML_TYPE_F32, output_cols, PF_TAIL_M) ? PF_TAIL_M : 0;
    if (rows == 0 || !exact_shape(weight, PF_H, output_cols)) {
        if (s.count_active) {
            ++s.gdn_fallback_count;
        }
        return false;
    }
    begin_request_if_needed(s, rows);

    if (s.pending_gdn_kind == PFGDNKind::none) {
        run_gdn_qkvz(s, layer, rows, static_cast<const float *>(input->data), ctx->stream());
        ++s.gdn_pack_count;
        write_gdn_slice(s, kind, rows, static_cast<float *>(dst->data), ctx->stream());
        if (kind == PFGDNKind::qkv) {
            ++s.gdn_qkv_write_count;
        } else {
            ++s.gdn_z_write_count;
        }
        s.pending_gdn_layer = layer;
        s.pending_gdn_rows = rows;
        s.pending_gdn_input = input->data;
        s.pending_gdn_kind = kind;
        return true;
    }

    const bool matching_kind = s.pending_gdn_kind != kind;
    if (!matching_kind || s.pending_gdn_layer != layer || s.pending_gdn_rows != rows ||
        s.pending_gdn_input != input->data) {
        ++s.gdn_pair_miss_count;
        std::fprintf(stderr,
            "promptforge: GDN QKVZ pair mismatch (pending layer %d rows %d kind %d, requested layer %d rows %d kind %d)\n",
            s.pending_gdn_layer, s.pending_gdn_rows, int(s.pending_gdn_kind),
            layer, rows, int(kind));
        std::abort();
    }
    write_gdn_slice(s, kind, rows, static_cast<float *>(dst->data), ctx->stream());
    if (kind == PFGDNKind::qkv) {
        ++s.gdn_qkv_write_count;
    } else {
        ++s.gdn_z_write_count;
    }
    s.pending_gdn_layer = -1;
    s.pending_gdn_rows = 0;
    s.pending_gdn_input = nullptr;
    s.pending_gdn_kind = PFGDNKind::none;
    ++s.gdn_qkvz_count;
    return true;
}

bool promptforge_try_fuse_gate_up(ggml_backend_cuda_context * ctx,
                                  ggml_tensor * first, ggml_tensor * second, ggml_tensor * glu) {
    PFState & s = state();
    const bool fused = s.mode == PFMode::m2048_fused || s.mode == PFMode::m2048_fused_tail1476;
    if (!s.ready || (s.mode != PFMode::m2048 && !fused) || !ctx || !glu ||
        ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0) {
        return false;
    }
    const int rows = exact_tensor(glu, GGML_TYPE_F32, PF_I, PF_M) ? PF_M :
                     s.mode == PFMode::m2048_fused_tail1476 &&
                     exact_tensor(glu, GGML_TYPE_F32, PF_I, PF_CHECKPOINT_M) ? PF_CHECKPOINT_M :
                     s.mode == PFMode::m2048_fused_tail1476 &&
                     exact_tensor(glu, GGML_TYPE_F32, PF_I, PF_TAIL_M) ? PF_TAIL_M : 0;
    if (rows == 0) return false;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    int layer = exact_layer(first ? first->src[0] : nullptr, "ffn_gate.weight");
    if (layer >= 0 && exact_layer(second ? second->src[0] : nullptr, "ffn_up.weight") == layer) {
        gate = first;
        up = second;
    } else {
        layer = exact_layer(second ? second->src[0] : nullptr, "ffn_gate.weight");
        if (layer < 0 || exact_layer(first ? first->src[0] : nullptr, "ffn_up.weight") != layer) {
            return false;
        }
        gate = second;
        up = first;
    }
    if (!gate || !up || gate->op != GGML_OP_MUL_MAT || up->op != GGML_OP_MUL_MAT ||
        gate->src[1] != up->src[1] || !exact_tensor(gate->src[1], GGML_TYPE_F32, PF_H, rows) ||
        !exact_tensor(gate, GGML_TYPE_F32, PF_I, rows) || !exact_tensor(up, GGML_TYPE_F32, PF_I, rows) ||
        !((glu->src[0] == gate && glu->src[1] == up) || (glu->src[0] == up && glu->src[1] == gate))) {
        return false;
    }
    if (fused && s.pending_down_input) {
        std::fprintf(stderr, "promptforge: fused down handoff still pending for layer %d\n", s.pending_down_layer);
        std::abort();
    }
    if (s.pending_gdn_kind != PFGDNKind::none) {
        ++s.gdn_pair_miss_count;
        std::fprintf(stderr, "promptforge: stale GDN QKVZ pair before FFN layer %d (pending layer %d)\n",
                     layer, s.pending_gdn_layer);
        std::abort();
    }
    begin_request_if_needed(s, rows);
    if (fused) {
        run_gate_fused(s, layer, rows, static_cast<const float *>(gate->src[1]->data), ctx->stream());
        s.pending_down_input = glu;
        s.pending_down_layer = layer;
        s.pending_down_rows = rows;
    } else {
        run_gate(s, layer, static_cast<const float *>(gate->src[1]->data), static_cast<float *>(glu->data), ctx->stream());
    }
    if (s.count_active) {
        ++s.gate_count;
        if (fused) {
            s.pack_count += 2;
            ++s.fused_down_pack_count;
        } else {
            ++s.pack_count;
        }
    }
    return true;
}

bool promptforge_try_down(ggml_backend_cuda_context * ctx,
                          const ggml_tensor * weight, const ggml_tensor * input, ggml_tensor * dst) {
    PFState & s = state();
    const bool fused = s.mode == PFMode::m2048_fused || s.mode == PFMode::m2048_fused_tail1476;
    if (!s.ready || (s.mode != PFMode::m2048 && !fused)) {
        return false;
    }
    if (!ctx) {
        if (fused && s.pending_down_input) {
            std::fprintf(stderr, "promptforge: fused down handoff has no CUDA context\n");
            std::abort();
        }
        return false;
    }
    const int layer = exact_layer(weight, "ffn_down.weight");
    const int rows = exact_tensor(input, GGML_TYPE_F32, PF_I, PF_M) &&
                     exact_tensor(dst, GGML_TYPE_F32, PF_H, PF_M) ? PF_M :
                     s.mode == PFMode::m2048_fused_tail1476 &&
                     exact_tensor(input, GGML_TYPE_F32, PF_I, PF_CHECKPOINT_M) &&
                     exact_tensor(dst, GGML_TYPE_F32, PF_H, PF_CHECKPOINT_M) ? PF_CHECKPOINT_M :
                     s.mode == PFMode::m2048_fused_tail1476 &&
                     exact_tensor(input, GGML_TYPE_F32, PF_I, PF_TAIL_M) &&
                     exact_tensor(dst, GGML_TYPE_F32, PF_H, PF_TAIL_M) ? PF_TAIL_M : 0;
    const bool exact_down = layer >= 0 && rows != 0;
    if (fused) {
        if (!s.pending_down_input) {
            return false;
        }
        if (!exact_down || input != s.pending_down_input || layer != s.pending_down_layer ||
            rows != s.pending_down_rows) {
            std::fprintf(stderr,
                "promptforge: fused down handoff mismatch (pending layer %d, requested layer %d)\n",
                s.pending_down_layer, layer);
            std::abort();
        }
        run_down_fused(s, layer, rows, static_cast<float *>(dst->data), ctx->stream());
    } else if (!exact_down) {
        return false;
    } else {
        run_down(s, layer, static_cast<const float *>(input->data), static_cast<float *>(dst->data), ctx->stream());
    }
    if (s.count_active) {
        ++s.down_count;
        if (s.mode == PFMode::m2048) {
            ++s.pack_count;
        }
        if (layer == PF_LAYERS - 1) {
            emit_request_telemetry(s);
        }
    }
    return true;
}
