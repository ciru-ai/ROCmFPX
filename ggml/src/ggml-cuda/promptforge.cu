#include "promptforge.cuh"
#include "promptforge_iu4.cuh"

#include "common.cuh"
#include "ggml.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm_multiply_multiply.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_wmma_cshuffle_v3.hpp"
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
constexpr int PF_DECODE_M_MIN = 2;
constexpr int PF_DECODE_M_MAX = 5;
constexpr int PF_NGRAM_VERIFY_M = 65;
// Short/medium prompt-only route. M1 target decode and M2-M5 MTP verification
// remain on their qualified paths. Exact M65 ngram verification is separately gated.
constexpr int PF_SMALLM_MIN = 96;
constexpr int PF_SMALLM_MAX = 512;
constexpr int PF_SMALLM_BUCKETS = 3;
constexpr int PF_H = 5120;
constexpr int PF_I = 17408;
constexpr uint64_t PF_FILE_BYTES = 17123004416ULL;
constexpr uint32_t PF_ENTRY_COUNT = 256;
constexpr uint64_t PF_IU4_FILE_BYTES = 8576856064ULL;
constexpr uint32_t PF_IU4_ENTRY_COUNT = 384;
constexpr uint64_t PF_IU4_DATA_OFFSET = 28672;
constexpr uint64_t PF_IU4_SEGMENTED_FILE_BYTES = 9091182592ULL;
constexpr int PF_GATE_KEEPER_K = 384;
constexpr int PF_DOWN_KEEPER_K = 1280;
constexpr uint32_t PF_GATE_HADAMARD_SEED = 0xA511E9B3U;
constexpr uint32_t PF_DOWN_HADAMARD_SEED = 0x63D83595U;
constexpr uint64_t PF_FFN_CORRECTION_FILE_BYTES = 1285533696ULL;
constexpr uint32_t PF_FFN_CORRECTION_ENTRY_COUNT = 384;
constexpr uint64_t PF_FFN_CORRECTION_DATA_OFFSET = 28672;
constexpr int PF_GDN_N = 16384;
constexpr int PF_GDN_QKV_N = 10240;
constexpr int PF_GDN_Z_N = 6144;
constexpr uint64_t PF_GDN_FILE_BYTES = 4029685760ULL;
constexpr uint32_t PF_GDN_ENTRY_COUNT = 96;
constexpr uint64_t PF_GDN_DATA_OFFSET = 8192;
constexpr uint64_t PF_GDN_IU4_FILE_BYTES = 2019569664ULL;
constexpr uint32_t PF_GDN_IU4_ENTRY_COUNT = 144;
constexpr uint64_t PF_GDN_IU4_DATA_OFFSET = 12288;
constexpr int PF_ATTENTION_Q_N = 12288;
constexpr int PF_ATTENTION_K_N = 1024;
constexpr int PF_ATTENTION_V_N = 1024;
constexpr int PF_ATTENTION_QKV_N = PF_ATTENTION_Q_N + PF_ATTENTION_K_N + PF_ATTENTION_V_N;
constexpr int PF_ATTENTION_VALUE_N = 6144;
constexpr uint64_t PF_ATTENTION_IU4_FILE_BYTES = 841359360ULL;
constexpr uint32_t PF_ATTENTION_IU4_ENTRY_COUNT = 96;
constexpr uint64_t PF_ATTENTION_IU4_DATA_OFFSET = 8192;
constexpr uint64_t PF_GDN_OUTPUT_IU4_FILE_BYTES = 756953088ULL;
constexpr uint32_t PF_GDN_OUTPUT_IU4_ENTRY_COUNT = 144;
constexpr uint64_t PF_GDN_OUTPUT_IU4_DATA_OFFSET = 12288;

constexpr uint16_t PF_GATE_W8 = 0;
constexpr uint16_t PF_DOWN_W8 = 1;
constexpr uint16_t PF_GATE_SCALE = 2;
constexpr uint16_t PF_DOWN_SCALE = 3;
constexpr uint16_t PF_GDN_QKVZ_W8 = 4;
constexpr uint16_t PF_GDN_QKVZ_SCALE = 5;
constexpr uint16_t PF_GDN_QKVZ_W4 = 20;
constexpr uint16_t PF_GDN_QKVZ_W4_SCALE = 21;
constexpr uint16_t PF_GDN_QKVZ_W4_SUM = 22;
constexpr uint16_t PF_ATTENTION_QKV_W4 = 30;
constexpr uint16_t PF_ATTENTION_QKV_W4_SCALE = 31;
constexpr uint16_t PF_ATTENTION_QKV_W4_SUM = 32;
constexpr uint16_t PF_ATTENTION_OUTPUT_W4 = 33;
constexpr uint16_t PF_ATTENTION_OUTPUT_W4_SCALE = 34;
constexpr uint16_t PF_ATTENTION_OUTPUT_W4_SUM = 35;
constexpr uint16_t PF_GDN_OUTPUT_W4 = 40;
constexpr uint16_t PF_GDN_OUTPUT_W4_SCALE = 41;
constexpr uint16_t PF_GDN_OUTPUT_W4_SUM = 42;
constexpr uint16_t PF_GATE_W4 = 10;
constexpr uint16_t PF_GATE_W4_SCALE = 11;
constexpr uint16_t PF_GATE_W4_SUM = 12;
constexpr uint16_t PF_DOWN_W4 = 13;
constexpr uint16_t PF_DOWN_W4_SCALE = 14;
constexpr uint16_t PF_DOWN_W4_SUM = 15;
constexpr uint16_t PF_GATE_CORRECTION_INDEX = 50;
constexpr uint16_t PF_GATE_CORRECTION_W8 = 51;
constexpr uint16_t PF_GATE_CORRECTION_SCALE = 52;
constexpr uint16_t PF_DOWN_CORRECTION_INDEX = 53;
constexpr uint16_t PF_DOWN_CORRECTION_W8 = 54;
constexpr uint16_t PF_DOWN_CORRECTION_SCALE = 55;
constexpr uint8_t PF_S8 = 1;
constexpr uint8_t PF_F32 = 2;
constexpr uint8_t PF_I32 = 3;
constexpr uint8_t PF_S4 = 4;
constexpr uint8_t PF_U16 = 5;

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

template <ck::index_t... Is>
using PFSequence = ck::Sequence<Is...>;

struct IU4ScaleCorrect {
    template <typename E, typename C, typename D0, typename D1, typename D2, typename D3>
    __host__ __device__ constexpr void operator()(
            E & e, const C & c, const D0 & weight_scale, const D1 & row_scale,
            const D2 & weight_sum, const D3 & row_zero) const {
        const float corrected = ck::type_convert<float>(c) -
            ck::type_convert<float>(row_zero) * ck::type_convert<float>(weight_sum);
        e = ck::type_convert<E>(corrected * ck::type_convert<float>(weight_scale) *
                                ck::type_convert<float>(row_scale));
    }
};

template <ck::tensor_operation::device::GemmSpecialization Specialization>
using IU4DeviceOpT = ck::tensor_operation::device::DeviceGemmMultipleD_Wmma_CShuffleV3<
    Row, Col, ck::Tuple<Row, Col, Row, Col>, Row,
    uint8_t, int8_t, ck::Tuple<float, float, int32_t, int32_t>, ck::bhalf_t, int32_t, int32_t,
    PassThrough, PassThrough, IU4ScaleCorrect, Specialization,
    128, 128, 64, 64, 8, 8, 16, 16, 4, 2,
    PFSequence<4, 32, 1>, PFSequence<1, 0, 2>, PFSequence<1, 0, 2>, 2, 8, 8, 0,
    PFSequence<4, 32, 1>, PFSequence<1, 0, 2>, PFSequence<1, 0, 2>, 2, 8, 8, 0,
    1, 1, PFSequence<1, 32, 1, 4>, PFSequence<1, 1, 1, 1, 1>,
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v1,
    uint8_t, int8_t>;

template <ck::tensor_operation::device::GemmSpecialization Specialization>
using IU4DownDeviceOpT = ck::tensor_operation::device::DeviceGemmMultipleD_Wmma_CShuffleV3<
    Row, Col, ck::Tuple<Row, Col, Row, Col>, Row,
    uint8_t, int8_t, ck::Tuple<float, float, int32_t, int32_t>, ck::bhalf_t, int32_t, int32_t,
    PassThrough, PassThrough, IU4ScaleCorrect, Specialization,
    256, 128, 256, 64, 8, 8, 16, 16, 4, 4,
    PFSequence<4, 64, 1>, PFSequence<1, 0, 2>, PFSequence<1, 0, 2>, 2, 8, 8, 1,
    PFSequence<4, 64, 1>, PFSequence<1, 0, 2>, PFSequence<1, 0, 2>, 2, 8, 8, 1,
    1, 1, PFSequence<1, 32, 1, 8>, PFSequence<1, 1, 1, 1, 1>,
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v1,
    uint8_t, int8_t>;

template <ck::tensor_operation::device::GemmSpecialization Specialization>
using IU4PaddedDeviceOpT = ck::tensor_operation::device::DeviceGemmMultipleD_Wmma_CShuffleV3<
    Row, Col, ck::Tuple<Row, Col, Row, Col>, Row,
    uint8_t, int8_t, ck::Tuple<float, float, int32_t, int32_t>, ck::bhalf_t, int32_t, int32_t,
    PassThrough, PassThrough, IU4ScaleCorrect, Specialization,
    256, 128, 128, 64, 8, 8, 16, 16, 4, 2,
    PFSequence<4, 64, 1>, PFSequence<1, 0, 2>, PFSequence<1, 0, 2>, 2, 8, 8, 1,
    PFSequence<4, 64, 1>, PFSequence<1, 0, 2>, PFSequence<1, 0, 2>, 2, 8, 8, 1,
    1, 1, PFSequence<1, 32, 1, 8>, PFSequence<1, 1, 1, 1, 1>,
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v3,
    uint8_t, int8_t>;

// CK's smallest established WMMA tile. Unlike the prompt-tail route above,
// this pads speculative verification widths to M32 rather than M128.
template <ck::tensor_operation::device::GemmSpecialization Specialization>
using IU4DecodeDeviceOpT = ck::tensor_operation::device::DeviceGemmMultipleD_Wmma_CShuffleV3<
    Row, Col, ck::Tuple<Row, Col, Row, Col>, Row,
    uint8_t, int8_t, ck::Tuple<float, float, int32_t, int32_t>, ck::bhalf_t, int32_t, int32_t,
    PassThrough, PassThrough, IU4ScaleCorrect, Specialization,
    64, 32, 64, 64, 8, 8, 16, 16, 2, 2,
    PFSequence<4, 16, 1>, PFSequence<1, 0, 2>, PFSequence<1, 0, 2>, 2, 8, 8, 1,
    PFSequence<4, 16, 1>, PFSequence<1, 0, 2>, PFSequence<1, 0, 2>, 2, 8, 8, 1,
    1, 1, PFSequence<1, 16, 1, 4>, PFSequence<1, 1, 1, 1, 1>,
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v1,
    uint8_t, int8_t>;

using IU4DeviceOp = IU4DeviceOpT<ck::tensor_operation::device::GemmSpecialization::Default>;
using IU4DownDeviceOp = IU4DownDeviceOpT<ck::tensor_operation::device::GemmSpecialization::Default>;
using IU4SmallGateDeviceOp = IU4DeviceOpT<ck::tensor_operation::device::GemmSpecialization::MNKPadding>;
using IU4SmallDownDeviceOp = IU4DownDeviceOpT<ck::tensor_operation::device::GemmSpecialization::MNKPadding>;
using IU4PaddedDeviceOp = IU4PaddedDeviceOpT<ck::tensor_operation::device::GemmSpecialization::MNKPadding>;
using IU4DecodeDeviceOp = IU4DecodeDeviceOpT<ck::tensor_operation::device::GemmSpecialization::MNKPadding>;

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

enum class PFMode { disabled, resident, m2048, m2048_fused, m2048_fused_tail1476, iu4_ffn };
enum class PFGDNKind { none, qkv, z };
enum class PFAttentionKind { none, q, k, v };

struct PFRowPlan {
    int actual_rows = 0;
    int execution_rows = 0;
    int bucket = -1;
};

struct PFIU4RowRouteCache {
    int execution_rows = 0;
    bool ready = false;
    bool gdn_ready = false;
    uint64_t rebuilds = 0;
    uint64_t gdn_rebuilds = 0;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> gate_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> down_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> gdn_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> gdn_output_args;
};

struct PFState {
    std::mutex init_mutex;
    bool initialized = false;
    bool ready = false;
    int device = -1;
    PFMode mode = PFMode::disabled;
    uint8_t * device_file = nullptr;
    bool ffn_correction_enabled = false;
    uint8_t * ffn_correction_device_file = nullptr;
    bool ffn_hadamard = false;
    bool ffn_segmented = false;
    bool ffn_keep_late4 = false;
    bool ffn_keep_late6 = false;
    bool gdn_enabled = false;
    uint8_t * gdn_device_file = nullptr;
    bool gdn_w8 = false;
    bool gdn_hadamard = false;
    bool gdn_keep_late4 = false;
    bool gdn_keep_early4 = false;
    bool gdn_keep_early3 = false;
    bool attention_enabled = false;
    uint8_t * attention_device_file = nullptr;
    bool gdn_output_enabled = false;
    bool gdn_output_keep_v3_lateq6 = false;
    bool iu4_decode_enabled = false;
    bool smallm_iu4_enabled = false;
    bool smallm_gdn_iu4_enabled = false;
    bool smallm_gdn_output_iu4_enabled = false;
    bool ngram_m65_iu4_enabled = false;
    uint8_t * gdn_output_device_file = nullptr;
    std::mutex smallm_cache_mutex;

    std::array<uint64_t, PF_LAYERS> gate_weight{};
    std::array<uint64_t, PF_LAYERS> gate_scale{};
    std::array<uint64_t, PF_LAYERS> down_weight{};
    std::array<uint64_t, PF_LAYERS> down_scale{};
    std::array<uint64_t, PF_LAYERS> gate_sum{};
    std::array<uint64_t, PF_LAYERS> down_sum{};
    std::array<uint64_t, PF_LAYERS> gate_correction_index{};
    std::array<uint64_t, PF_LAYERS> gate_correction_weight{};
    std::array<uint64_t, PF_LAYERS> gate_correction_scale{};
    std::array<uint64_t, PF_LAYERS> down_correction_index{};
    std::array<uint64_t, PF_LAYERS> down_correction_weight{};
    std::array<uint64_t, PF_LAYERS> down_correction_scale{};
    std::array<uint64_t, PF_LAYERS> gdn_weight{};
    std::array<uint64_t, PF_LAYERS> gdn_scale{};
    std::array<uint64_t, PF_LAYERS> gdn_sum{};
    std::array<uint64_t, PF_LAYERS> attention_qkv_weight{};
    std::array<uint64_t, PF_LAYERS> attention_qkv_scale{};
    std::array<uint64_t, PF_LAYERS> attention_qkv_sum{};
    std::array<uint64_t, PF_LAYERS> attention_output_weight{};
    std::array<uint64_t, PF_LAYERS> attention_output_scale{};
    std::array<uint64_t, PF_LAYERS> attention_output_sum{};
    std::array<uint64_t, PF_LAYERS> gdn_output_weight{};
    std::array<uint64_t, PF_LAYERS> gdn_output_scale{};
    std::array<uint64_t, PF_LAYERS> gdn_output_sum{};

    int8_t * gate_a8 = nullptr;
    float * gate_a_scale = nullptr;
    __hip_bfloat16 * gate_out = nullptr;
    int8_t * down_a8 = nullptr;
    float * down_a_scale = nullptr;
    uint32_t * gate_a4 = nullptr;
    int32_t * gate_a_zero = nullptr;
    uint32_t * down_a4 = nullptr;
    int32_t * down_a_zero = nullptr;
    __hip_bfloat16 * down_out = nullptr;
    int8_t * gate_correction_a8 = nullptr;
    float * gate_correction_a_scale = nullptr;
    int8_t * down_correction_a8 = nullptr;
    float * down_correction_a_scale = nullptr;
    std::unique_ptr<CKDeviceOp> gate_op;
    std::unique_ptr<CKDeviceOp> down_op;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> gate_invoker;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> down_invoker;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> gate_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> down_args;
    std::unique_ptr<IU4DeviceOp> iu4_gate_op;
    std::unique_ptr<IU4DownDeviceOp> iu4_down_op;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> iu4_gate_invoker;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> iu4_down_invoker;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> iu4_gate_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> iu4_down_args;
    std::unique_ptr<CKDeviceOp> gate_checkpoint2044_op;
    std::unique_ptr<CKDeviceOp> down_checkpoint2044_op;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> gate_checkpoint2044_invoker;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> down_checkpoint2044_invoker;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> gate_checkpoint2044_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> down_checkpoint2044_args;
    std::unique_ptr<IU4PaddedDeviceOp> iu4_gate_padded_op;
    std::unique_ptr<IU4PaddedDeviceOp> iu4_down_padded_op;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> iu4_gate_padded_invoker;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> iu4_down_padded_invoker;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> iu4_gate_checkpoint2044_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> iu4_down_checkpoint2044_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> iu4_gate_tail1476_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> iu4_down_tail1476_args;
    std::unique_ptr<IU4DecodeDeviceOp> iu4_decode_op;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> iu4_decode_invoker;
    std::array<std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>,
                          PF_DECODE_M_MAX + 1>, PF_LAYERS> iu4_gate_decode_args;
    std::array<std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>,
                          PF_DECODE_M_MAX + 1>, PF_LAYERS> iu4_down_decode_args;
    std::unique_ptr<IU4SmallGateDeviceOp> iu4_smallm_gate_op;
    std::unique_ptr<IU4SmallDownDeviceOp> iu4_smallm_down_op;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> iu4_smallm_gate_invoker;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> iu4_smallm_down_invoker;
    std::array<PFIU4RowRouteCache, PF_SMALLM_BUCKETS> iu4_smallm_routes;
    std::unique_ptr<CKDeviceOp> gate_tail1476_op;
    std::unique_ptr<CKDeviceOp> down_tail1476_op;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> gate_tail1476_invoker;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> down_tail1476_invoker;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> gate_tail1476_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> down_tail1476_args;
    std::unique_ptr<CKDeviceOp> gdn_op;
    std::unique_ptr<CKDeviceOp> gdn_checkpoint2044_op;
    std::unique_ptr<CKDeviceOp> gdn_tail1476_op;
    std::unique_ptr<IU4PaddedDeviceOp> iu4_gdn_op;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> gdn_invoker;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> gdn_checkpoint2044_invoker;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> gdn_tail1476_invoker;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> gdn_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> gdn_checkpoint2044_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> gdn_tail1476_args;
    std::unique_ptr<IU4PaddedDeviceOp> attention_qkv_op;
    std::unique_ptr<IU4PaddedDeviceOp> attention_output_op;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> attention_qkv_invoker;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> attention_output_invoker;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> attention_qkv_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> attention_qkv_checkpoint2044_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> attention_qkv_tail1476_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> attention_output_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> attention_output_checkpoint2044_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> attention_output_tail1476_args;
    std::unique_ptr<IU4PaddedDeviceOp> gdn_output_op;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> gdn_output_invoker;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> gdn_output_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> gdn_output_checkpoint2044_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> gdn_output_tail1476_args;
    std::unique_ptr<IU4DecodeDeviceOp> gdn_output_decode_op;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker> gdn_output_decode_invoker;
    std::array<std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>,
                          PF_DECODE_M_MAX + 1>, PF_LAYERS> gdn_output_decode_args;

    bool count_active = false;
    int gate_count = 0;
    int down_count = 0;
    int pack_count = 0;
    int fallback_count = 0;
    int ffn_keeper_count = 0;
    const ggml_tensor * pending_down_input = nullptr;
    int pending_down_layer = -1;
    int pending_down_rows = 0;
    int pending_down_execution_rows = 0;
    int pending_down_smallm_bucket = -1;
    int active_rows = 0;
    int fused_down_pack_count = 0;
    int gdn_qkvz_count = 0;
    int gdn_pack_count = 0;
    int gdn_qkv_write_count = 0;
    int gdn_z_write_count = 0;
    int gdn_pair_miss_count = 0;
    int gdn_fallback_count = 0;
    int gdn_keeper_count = 0;
    int pending_gdn_layer = -1;
    int pending_gdn_rows = 0;
    const void * pending_gdn_input = nullptr;
    PFGDNKind pending_gdn_kind = PFGDNKind::none;
    int attention_qkv_count = 0;
    int attention_qkv_pack_count = 0;
    int attention_q_write_count = 0;
    int attention_k_write_count = 0;
    int attention_v_write_count = 0;
    int attention_output_count = 0;
    int attention_output_pack_count = 0;
    int attention_fallback_count = 0;
    int pending_attention_layer = -1;
    int pending_attention_rows = 0;
    const void * pending_attention_input = nullptr;
    uint8_t pending_attention_mask = 0;
    int gdn_output_count = 0;
    int gdn_output_pack_count = 0;
    int gdn_output_fallback_count = 0;
    bool smallm_iu4_request = false;
    int smallm_iu4_execution_rows = 0;
    int smallm_iu4_gate_count = 0;
    int smallm_iu4_down_count = 0;
    int smallm_iu4_fallback_count = 0;
    bool smallm_gdn_iu4_request = false;
    int smallm_gdn_iu4_execution_rows = 0;
    int smallm_gdn_iu4_qkvz_count = 0;
    int smallm_gdn_iu4_output_count = 0;
    int smallm_gdn_iu4_fallback_count = 0;
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

bool smallm_rows(int rows) {
    return rows >= PF_SMALLM_MIN && rows <= PF_SMALLM_MAX;
}

PFRowPlan smallm_row_plan(int rows) {
    PFRowPlan plan;
    plan.actual_rows = rows;
    if (!smallm_rows(rows) && rows != PF_NGRAM_VERIFY_M) {
        return plan;
    }
    if (rows <= 128) {
        plan.execution_rows = 128;
        plan.bucket = 0;
    } else if (rows <= 256) {
        plan.execution_rows = 256;
        plan.bucket = 1;
    } else {
        plan.execution_rows = 512;
        plan.bucket = 2;
    }
    return plan;
}

bool v3_lateq6_gdn_output_layer(int layer) {
    return layer == 48 || layer == 49 || layer == 50 ||
           layer == 52 || layer == 53 || layer == 54 ||
           layer == 56 || layer == 57 || layer == 58 ||
           layer == 60 || layer == 61 || layer == 62;
}

bool canonical_true_iu4_lead(const PFState & s) {
    return s.mode == PFMode::iu4_ffn && s.ffn_hadamard && !s.ffn_segmented &&
           !s.ffn_correction_enabled && !s.ffn_keep_late4 &&
           s.gdn_enabled && !s.gdn_w8 && s.gdn_hadamard &&
           !s.gdn_keep_late4 && !s.gdn_keep_early4 && !s.gdn_keep_early3 &&
           s.gdn_output_enabled;
}

bool smallm_iu4_route_enabled(const PFState & s, int rows) {
    const bool admitted = smallm_rows(rows) ||
        (s.ngram_m65_iu4_enabled && rows == PF_NGRAM_VERIFY_M);
    return s.smallm_iu4_enabled && canonical_true_iu4_lead(s) && admitted;
}

bool smallm_gdn_iu4_route_enabled(const PFState & s, int rows) {
    const bool admitted = smallm_rows(rows) ||
        (s.ngram_m65_iu4_enabled && rows == PF_NGRAM_VERIFY_M);
    return s.smallm_gdn_iu4_enabled && s.smallm_iu4_enabled &&
           canonical_true_iu4_lead(s) && admitted;
}

bool smallm_gdn_output_iu4_route_enabled(const PFState & s, int rows) {
    return s.smallm_gdn_output_iu4_enabled && smallm_gdn_iu4_route_enabled(s, rows);
}

int exact_smallm_iu4_rows(const PFState & s, const ggml_tensor * tensor, int64_t ne0) {
    if (!tensor) {
        return 0;
    }
    const int rows = int(tensor->ne[1]);
    return smallm_iu4_route_enabled(s, rows) && exact_tensor(tensor, GGML_TYPE_F32, ne0, rows)
        ? rows : 0;
}

int exact_smallm_gdn_iu4_rows(const PFState & s, const ggml_tensor * tensor, int64_t ne0) {
    if (!tensor) {
        return 0;
    }
    const int rows = int(tensor->ne[1]);
    return smallm_gdn_iu4_route_enabled(s, rows) && exact_tensor(tensor, GGML_TYPE_F32, ne0, rows)
        ? rows : 0;
}

int exact_smallm_gdn_output_iu4_rows(const PFState & s, const ggml_tensor * tensor, int64_t ne0) {
    if (!tensor || !smallm_rows(int(tensor->ne[1]))) {
        return 0;
    }
    const int rows = int(tensor->ne[1]);
    return smallm_gdn_output_iu4_route_enabled(s, rows) &&
           exact_tensor(tensor, GGML_TYPE_F32, ne0, rows) ? rows : 0;
}

void emit_smallm_iu4_fallback(PFState & s, int rows, const char * stage, const char * reason) {
    if (s.count_active && s.active_rows == rows) {
        ++s.fallback_count;
        ++s.smallm_iu4_fallback_count;
    }
    const PFRowPlan plan = smallm_row_plan(rows);
    std::fprintf(stderr,
        "{\"record\":\"promptforge_smallm_iu4_fallback\",\"rows\":%d,"
        "\"execution_rows\":%d,\"bucket\":%d,\"stage\":\"%s\",\"reason\":\"%s\"}\n",
        rows, plan.execution_rows, plan.bucket, stage, reason);
    std::fflush(stderr);
}

void emit_smallm_gdn_iu4_fallback(PFState & s, int rows, const char * stage, const char * reason) {
    if (s.count_active && s.active_rows == rows) {
        ++s.smallm_gdn_iu4_fallback_count;
    }
    const PFRowPlan plan = smallm_row_plan(rows);
    std::fprintf(stderr,
        "{\"record\":\"promptforge_smallm_gdn_iu4_fallback\",\"rows\":%d,"
        "\"execution_rows\":%d,\"bucket\":%d,\"stage\":\"%s\",\"reason\":\"%s\"}\n",
        rows, plan.execution_rows, plan.bucket, stage, reason);
    std::fflush(stderr);
}

bool iu4_decode_width_enabled(const PFState & s, int rows) {
    return rows >= PF_DECODE_M_MIN && rows <= PF_DECODE_M_MAX &&
           s.iu4_decode_enabled &&
           s.mode == PFMode::iu4_ffn && s.ffn_hadamard && !s.ffn_segmented &&
           s.gdn_enabled && s.gdn_w8 && s.gdn_output_enabled;
}

int exact_iu4_decode_rows(const PFState & s, const ggml_tensor * tensor, int64_t ne0) {
    if (!tensor || tensor->ne[1] < PF_DECODE_M_MIN || tensor->ne[1] > PF_DECODE_M_MAX) {
        return 0;
    }
    const int rows = int(tensor->ne[1]);
    return iu4_decode_width_enabled(s, rows) && exact_tensor(tensor, GGML_TYPE_F32, ne0, rows)
        ? rows : 0;
}

int exact_iu4_decode_rows(const PFState & s, const ggml_tensor * input,
                          const ggml_tensor * output, int64_t input_ne0, int64_t output_ne0) {
    const int rows = exact_iu4_decode_rows(s, input, input_ne0);
    return rows != 0 && exact_tensor(output, GGML_TYPE_F32, output_ne0, rows) ? rows : 0;
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
        if (s.pending_attention_mask != 0) {
            std::fprintf(stderr, "promptforge: stale attention QKV set at request boundary (layer %d mask %u)\n",
                         s.pending_attention_layer, unsigned(s.pending_attention_mask));
            std::abort();
        }
        s.count_active = true;
        s.gate_count = s.down_count = s.pack_count = s.fallback_count = 0;
        s.ffn_keeper_count = 0;
        s.fused_down_pack_count = 0;
        s.gdn_qkvz_count = s.gdn_pack_count = 0;
        s.gdn_qkv_write_count = s.gdn_z_write_count = 0;
        s.gdn_pair_miss_count = s.gdn_fallback_count = s.gdn_keeper_count = 0;
        s.attention_qkv_count = s.attention_qkv_pack_count = 0;
        s.attention_q_write_count = s.attention_k_write_count = s.attention_v_write_count = 0;
        s.attention_output_count = s.attention_output_pack_count = s.attention_fallback_count = 0;
        s.gdn_output_count = s.gdn_output_pack_count = s.gdn_output_fallback_count = 0;
        s.smallm_iu4_request = smallm_iu4_route_enabled(s, rows);
        s.smallm_iu4_execution_rows = s.smallm_iu4_request ? smallm_row_plan(rows).execution_rows : 0;
        s.smallm_iu4_gate_count = s.smallm_iu4_down_count = s.smallm_iu4_fallback_count = 0;
        s.smallm_gdn_iu4_request = smallm_gdn_iu4_route_enabled(s, rows);
        s.smallm_gdn_iu4_execution_rows =
            s.smallm_gdn_iu4_request ? smallm_row_plan(rows).execution_rows : 0;
        s.smallm_gdn_iu4_qkvz_count = s.smallm_gdn_iu4_output_count =
            s.smallm_gdn_iu4_fallback_count = 0;
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
    if (s.pending_attention_mask != 0) {
        std::fprintf(stderr, "promptforge: stale attention QKV set at request summary (layer %d mask %u)\n",
                     s.pending_attention_layer, unsigned(s.pending_attention_mask));
        std::abort();
    }
    uint64_t smallm_cache_rebuilds = 0;
    uint64_t smallm_gdn_cache_rebuilds = 0;
    for (const auto & route : s.iu4_smallm_routes) {
        smallm_cache_rebuilds += route.rebuilds;
        smallm_gdn_cache_rebuilds += route.gdn_rebuilds;
    }
    std::fprintf(stderr,
        "{\"record\":\"promptforge_request\",\"gate_up\":%d,\"down\":%d,"
        "\"packs\":%d,\"fallback\":%d,\"ffn_keepers\":%d,\"fused_down_pack\":%d,"
        "\"gdn_qkvz\":%d,\"gdn_packs\":%d,\"gdn_qkv_writes\":%d,"
        "\"gdn_z_writes\":%d,\"gdn_pair_miss\":%d,\"gdn_fallback\":%d,"
        "\"gdn_keepers\":%d,"
        "\"gdn_projection\":\"%s\","
        "\"attention_qkv\":%d,\"attention_qkv_packs\":%d,"
        "\"attention_q_writes\":%d,\"attention_k_writes\":%d,\"attention_v_writes\":%d,"
        "\"attention_output\":%d,\"attention_output_packs\":%d,"
        "\"attention_fallback\":%d,\"gdn_output\":%d,"
        "\"gdn_output_packs\":%d,\"gdn_output_fallback\":%d,\"rows\":%d,"
        "\"smallm_iu4\":%s,\"smallm_iu4_execution_rows\":%d,"
        "\"smallm_iu4_gate_up\":%d,\"smallm_iu4_down\":%d,"
        "\"smallm_iu4_fallback\":%d,\"smallm_iu4_cache_rebuilds\":%llu,"
        "\"smallm_gdn_iu4\":%s,\"smallm_gdn_iu4_execution_rows\":%d,"
        "\"smallm_gdn_iu4_qkvz\":%d,\"smallm_gdn_iu4_output\":%d,"
        "\"smallm_gdn_iu4_fallback\":%d,\"smallm_gdn_iu4_cache_rebuilds\":%llu,"
        "\"ngram_m65_iu4\":%s}\n",
        s.gate_count, s.down_count, s.pack_count, s.fallback_count,
        s.ffn_keeper_count, s.fused_down_pack_count, s.gdn_qkvz_count, s.gdn_pack_count,
        s.gdn_qkv_write_count, s.gdn_z_write_count, s.gdn_pair_miss_count,
        s.gdn_fallback_count, s.gdn_keeper_count,
        s.mode == PFMode::iu4_ffn && !s.gdn_w8 ?
            (s.gdn_hadamard ? "qkvz_iu4_hadamard" : "qkvz_iu4") : "qkvz_w8",
        s.attention_qkv_count, s.attention_qkv_pack_count,
        s.attention_q_write_count, s.attention_k_write_count, s.attention_v_write_count,
        s.attention_output_count, s.attention_output_pack_count, s.attention_fallback_count,
        s.gdn_output_count, s.gdn_output_pack_count, s.gdn_output_fallback_count,
        s.active_rows, s.smallm_iu4_request ? "true" : "false",
        s.smallm_iu4_execution_rows, s.smallm_iu4_gate_count, s.smallm_iu4_down_count,
        s.smallm_iu4_fallback_count, (unsigned long long) smallm_cache_rebuilds,
        s.smallm_gdn_iu4_request ? "true" : "false", s.smallm_gdn_iu4_execution_rows,
        s.smallm_gdn_iu4_qkvz_count, s.smallm_gdn_iu4_output_count,
        s.smallm_gdn_iu4_fallback_count, (unsigned long long) smallm_gdn_cache_rebuilds,
        s.ngram_m65_iu4_enabled && s.active_rows == PF_NGRAM_VERIFY_M &&
            s.smallm_iu4_request && s.smallm_gdn_iu4_request ? "true" : "false");
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
    const size_t group = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t i = group * 4;
    if (i + 3 < count) {
        reinterpret_cast<float4 *>(dst)[group] = make_float4(
            __bfloat162float(src[i + 0]), __bfloat162float(src[i + 1]),
            __bfloat162float(src[i + 2]), __bfloat162float(src[i + 3]));
    } else {
        for (size_t j = i; j < count; ++j) {
            dst[j] = __bfloat162float(src[j]);
        }
    }
}

__global__ void bf16_slice_to_f32(const __hip_bfloat16 * src, float * dst,
                                  int rows, int src_stride, int src_col, int dst_cols) {
    const size_t group = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int groups_per_row = dst_cols / 4;
    const size_t group_count = size_t(rows) * groups_per_row;
    if (group < group_count) {
        const int row = int(group / groups_per_row);
        const int col = int(group - size_t(row) * groups_per_row) * 4;
        const size_t src_i = size_t(row) * src_stride + src_col + col;
        reinterpret_cast<float4 *>(dst)[group] = make_float4(
            __bfloat162float(src[src_i + 0]), __bfloat162float(src[src_i + 1]),
            __bfloat162float(src[src_i + 2]), __bfloat162float(src[src_i + 3]));
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

bool validate_iu4_entry(const PFEntry & e, int layer, uint16_t kind, uint8_t dtype,
                        uint8_t rank, uint32_t rows, uint32_t cols, uint64_t length) {
    return e.layer == layer && e.kind == kind && e.dtype == dtype && e.rank == rank &&
           e.rows == rows && e.cols == cols && e.length == length &&
           e.offset <= PF_IU4_FILE_BYTES && e.length <= PF_IU4_FILE_BYTES - e.offset;
}

bool load_iu4_sidecar(PFState & s, const char * path) {
    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        std::fprintf(stderr, "promptforge: cannot open IU4 sidecar %s\n", path);
        return false;
    }
    struct stat st {};
    if (fstat(fd, &st) != 0 || uint64_t(st.st_size) != PF_IU4_FILE_BYTES) {
        std::fprintf(stderr, "promptforge: wrong IU4 sidecar size\n");
        close(fd);
        return false;
    }
    void * mapping = mmap(nullptr, PF_IU4_FILE_BYTES, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) {
        std::fprintf(stderr, "promptforge: IU4 sidecar mmap failed\n");
        close(fd);
        return false;
    }

    const auto * header = static_cast<const PFHeader *>(mapping);
    const char expected_magic[8] = {'P','F','S','I','U','4','F','\0'};
    bool valid = std::memcmp(header->magic, expected_magic, 8) == 0 && header->version == 1 &&
                 header->header_bytes == sizeof(PFHeader) && header->entry_bytes == sizeof(PFEntry) &&
                 header->entry_count == PF_IU4_ENTRY_COUNT && header->table_offset == sizeof(PFHeader) &&
                 header->table_bytes == uint64_t(PF_IU4_ENTRY_COUNT) * sizeof(PFEntry) &&
                 header->data_offset == PF_IU4_DATA_OFFSET && header->file_bytes == PF_IU4_FILE_BYTES;
    const auto * entries = reinterpret_cast<const PFEntry *>(
        static_cast<const uint8_t *>(mapping) + header->table_offset);
    uint64_t next_offset = header->data_offset;
    for (int layer = 0; valid && layer < PF_LAYERS; ++layer) {
        const PFEntry & gw = entries[layer * 6 + 0];
        const PFEntry & gs = entries[layer * 6 + 1];
        const PFEntry & gz = entries[layer * 6 + 2];
        const PFEntry & dw = entries[layer * 6 + 3];
        const PFEntry & ds = entries[layer * 6 + 4];
        const PFEntry & dz = entries[layer * 6 + 5];
        valid = validate_iu4_entry(gw, layer, PF_GATE_W4, PF_S4, 2, 2 * PF_I, PF_H,
                                   uint64_t(2 * PF_I) * PF_H / 2) &&
                validate_iu4_entry(gs, layer, PF_GATE_W4_SCALE, PF_F32, 1, 2 * PF_I, 1,
                                   uint64_t(2 * PF_I) * sizeof(float)) &&
                validate_iu4_entry(gz, layer, PF_GATE_W4_SUM, PF_I32, 1, 2 * PF_I, 1,
                                   uint64_t(2 * PF_I) * sizeof(int32_t)) &&
                validate_iu4_entry(dw, layer, PF_DOWN_W4, PF_S4, 2, PF_H, PF_I,
                                   uint64_t(PF_H) * PF_I / 2) &&
                validate_iu4_entry(ds, layer, PF_DOWN_W4_SCALE, PF_F32, 1, PF_H, 1,
                                   uint64_t(PF_H) * sizeof(float)) &&
                validate_iu4_entry(dz, layer, PF_DOWN_W4_SUM, PF_I32, 1, PF_H, 1,
                                   uint64_t(PF_H) * sizeof(int32_t));
        for (const PFEntry * e : {&gw, &gs, &gz, &dw, &ds, &dz}) {
            valid = valid && e->offset == next_offset;
            next_offset += e->length;
        }
        s.gate_weight[layer] = gw.offset;
        s.gate_scale[layer] = gs.offset;
        s.gate_sum[layer] = gz.offset;
        s.down_weight[layer] = dw.offset;
        s.down_scale[layer] = ds.offset;
        s.down_sum[layer] = dz.offset;
    }
    valid = valid && next_offset == PF_IU4_FILE_BYTES;
    if (!valid) {
        std::fprintf(stderr, "promptforge: invalid PFSIU4F table\n");
        munmap(mapping, PF_IU4_FILE_BYTES);
        close(fd);
        return false;
    }
    if (!hip_check(hipMalloc(&s.device_file, PF_IU4_FILE_BYTES), "hipMalloc(IU4 sidecar)") ||
        !hip_check(hipMemcpy(s.device_file, mapping, PF_IU4_FILE_BYTES, hipMemcpyHostToDevice),
                   "hipMemcpy(IU4 sidecar)")) {
        munmap(mapping, PF_IU4_FILE_BYTES);
        close(fd);
        return false;
    }
    madvise(mapping, PF_IU4_FILE_BYTES, MADV_DONTNEED);
    munmap(mapping, PF_IU4_FILE_BYTES);
    posix_fadvise(fd, 0, PF_IU4_FILE_BYTES, POSIX_FADV_DONTNEED);
    close(fd);
    return true;
}

bool load_iu4_segmented_sidecar(PFState & s, const char * path) {
    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        std::fprintf(stderr, "promptforge: cannot open segmented IU4 sidecar %s\n", path);
        return false;
    }
    struct stat st {};
    if (fstat(fd, &st) != 0 || uint64_t(st.st_size) != PF_IU4_SEGMENTED_FILE_BYTES) {
        std::fprintf(stderr, "promptforge: wrong segmented IU4 sidecar size\n");
        close(fd);
        return false;
    }
    void * mapping = mmap(nullptr, PF_IU4_SEGMENTED_FILE_BYTES, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) {
        std::fprintf(stderr, "promptforge: segmented IU4 sidecar mmap failed\n");
        close(fd);
        return false;
    }
    const auto * header = static_cast<const PFHeader *>(mapping);
    const char expected_magic[8] = {'P','F','S','I','U','4','S','\0'};
    bool valid = std::memcmp(header->magic, expected_magic, 8) == 0 && header->version == 1 &&
                 header->header_bytes == sizeof(PFHeader) && header->entry_bytes == sizeof(PFEntry) &&
                 header->entry_count == PF_IU4_ENTRY_COUNT && header->table_offset == sizeof(PFHeader) &&
                 header->table_bytes == uint64_t(PF_IU4_ENTRY_COUNT) * sizeof(PFEntry) &&
                 header->data_offset == PF_IU4_DATA_OFFSET &&
                 header->file_bytes == PF_IU4_SEGMENTED_FILE_BYTES;
    const auto * entries = reinterpret_cast<const PFEntry *>(
        static_cast<const uint8_t *>(mapping) + header->table_offset);
    uint64_t next_offset = header->data_offset;
    constexpr int gate_segments = PF_H / promptforge_iu4::kQuantSegment;
    constexpr int down_segments = PF_I / promptforge_iu4::kQuantSegment;
    auto valid_entry = [](const PFEntry & e, int layer, uint16_t kind, uint8_t dtype,
                          uint8_t rank, uint32_t rows, uint32_t cols, uint64_t length) {
        return e.layer == layer && e.kind == kind && e.dtype == dtype && e.rank == rank &&
               e.rows == rows && e.cols == cols && e.length == length &&
               e.offset <= PF_IU4_SEGMENTED_FILE_BYTES &&
               e.length <= PF_IU4_SEGMENTED_FILE_BYTES - e.offset;
    };
    for (int layer = 0; valid && layer < PF_LAYERS; ++layer) {
        const PFEntry & gw = entries[layer * 6 + 0];
        const PFEntry & gs = entries[layer * 6 + 1];
        const PFEntry & gz = entries[layer * 6 + 2];
        const PFEntry & dw = entries[layer * 6 + 3];
        const PFEntry & ds = entries[layer * 6 + 4];
        const PFEntry & dz = entries[layer * 6 + 5];
        valid = valid_entry(gw, layer, PF_GATE_W4, PF_S4, 2, 2 * PF_I, PF_H,
                            uint64_t(2 * PF_I) * PF_H / 2) &&
                valid_entry(gs, layer, PF_GATE_W4_SCALE, PF_F32, 2,
                            gate_segments, 2 * PF_I,
                            uint64_t(gate_segments) * 2 * PF_I * sizeof(float)) &&
                valid_entry(gz, layer, PF_GATE_W4_SUM, PF_I32, 2,
                            gate_segments, 2 * PF_I,
                            uint64_t(gate_segments) * 2 * PF_I * sizeof(int32_t)) &&
                valid_entry(dw, layer, PF_DOWN_W4, PF_S4, 2, PF_H, PF_I,
                            uint64_t(PF_H) * PF_I / 2) &&
                valid_entry(ds, layer, PF_DOWN_W4_SCALE, PF_F32, 2,
                            down_segments, PF_H,
                            uint64_t(down_segments) * PF_H * sizeof(float)) &&
                valid_entry(dz, layer, PF_DOWN_W4_SUM, PF_I32, 2,
                            down_segments, PF_H,
                            uint64_t(down_segments) * PF_H * sizeof(int32_t));
        for (const PFEntry * e : {&gw, &gs, &gz, &dw, &ds, &dz}) {
            valid = valid && e->offset == next_offset;
            next_offset += e->length;
        }
        s.gate_weight[layer] = gw.offset;
        s.gate_scale[layer] = gs.offset;
        s.gate_sum[layer] = gz.offset;
        s.down_weight[layer] = dw.offset;
        s.down_scale[layer] = ds.offset;
        s.down_sum[layer] = dz.offset;
    }
    valid = valid && next_offset == PF_IU4_SEGMENTED_FILE_BYTES;
    if (!valid) {
        std::fprintf(stderr, "promptforge: invalid PFSIU4S table\n");
        munmap(mapping, PF_IU4_SEGMENTED_FILE_BYTES);
        close(fd);
        return false;
    }
    if (!hip_check(hipMalloc(&s.device_file, PF_IU4_SEGMENTED_FILE_BYTES),
                   "hipMalloc(segmented IU4 sidecar)") ||
        !hip_check(hipMemcpy(s.device_file, mapping, PF_IU4_SEGMENTED_FILE_BYTES,
                            hipMemcpyHostToDevice),
                   "hipMemcpy(segmented IU4 sidecar)")) {
        munmap(mapping, PF_IU4_SEGMENTED_FILE_BYTES);
        close(fd);
        return false;
    }
    madvise(mapping, PF_IU4_SEGMENTED_FILE_BYTES, MADV_DONTNEED);
    munmap(mapping, PF_IU4_SEGMENTED_FILE_BYTES);
    posix_fadvise(fd, 0, PF_IU4_SEGMENTED_FILE_BYTES, POSIX_FADV_DONTNEED);
    close(fd);
    return true;
}

bool validate_ffn_correction_entry(const PFEntry & e, int layer, uint16_t kind,
                                   uint8_t dtype, uint8_t rank, uint32_t rows,
                                   uint32_t cols, uint64_t length) {
    return e.layer == layer && e.kind == kind && e.dtype == dtype && e.rank == rank &&
           e.rows == rows && e.cols == cols && e.length == length &&
           e.offset <= PF_FFN_CORRECTION_FILE_BYTES &&
           e.length <= PF_FFN_CORRECTION_FILE_BYTES - e.offset;
}

bool load_ffn_correction_sidecar(PFState & s, const char * path) {
    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        std::fprintf(stderr, "promptforge: cannot open FFN correction sidecar %s\n", path);
        return false;
    }
    struct stat st {};
    if (fstat(fd, &st) != 0 || uint64_t(st.st_size) != PF_FFN_CORRECTION_FILE_BYTES) {
        std::fprintf(stderr, "promptforge: wrong FFN correction sidecar size\n");
        close(fd);
        return false;
    }
    void * mapping = mmap(nullptr, PF_FFN_CORRECTION_FILE_BYTES, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) {
        std::fprintf(stderr, "promptforge: FFN correction sidecar mmap failed\n");
        close(fd);
        return false;
    }

    const auto * header = static_cast<const PFHeader *>(mapping);
    const char expected_magic[8] = {'P','F','S','I','U','4','K','\0'};
    bool valid = std::memcmp(header->magic, expected_magic, 8) == 0 && header->version == 1 &&
                 header->header_bytes == sizeof(PFHeader) && header->entry_bytes == sizeof(PFEntry) &&
                 header->entry_count == PF_FFN_CORRECTION_ENTRY_COUNT &&
                 header->table_offset == sizeof(PFHeader) &&
                 header->table_bytes == uint64_t(PF_FFN_CORRECTION_ENTRY_COUNT) * sizeof(PFEntry) &&
                 header->data_offset == PF_FFN_CORRECTION_DATA_OFFSET &&
                 header->file_bytes == PF_FFN_CORRECTION_FILE_BYTES;
    const auto * entries = reinterpret_cast<const PFEntry *>(
        static_cast<const uint8_t *>(mapping) + header->table_offset);
    uint64_t next_offset = header->data_offset;
    for (int layer = 0; valid && layer < PF_LAYERS; ++layer) {
        const PFEntry & gi = entries[layer * 6 + 0];
        const PFEntry & gw = entries[layer * 6 + 1];
        const PFEntry & gs = entries[layer * 6 + 2];
        const PFEntry & di = entries[layer * 6 + 3];
        const PFEntry & dw = entries[layer * 6 + 4];
        const PFEntry & ds = entries[layer * 6 + 5];
        valid = validate_ffn_correction_entry(
                    gi, layer, PF_GATE_CORRECTION_INDEX, PF_U16, 1,
                    PF_GATE_KEEPER_K, 1, uint64_t(PF_GATE_KEEPER_K) * sizeof(uint16_t)) &&
                validate_ffn_correction_entry(
                    gw, layer, PF_GATE_CORRECTION_W8, PF_S8, 2,
                    2 * PF_I, PF_GATE_KEEPER_K, uint64_t(2 * PF_I) * PF_GATE_KEEPER_K) &&
                validate_ffn_correction_entry(
                    gs, layer, PF_GATE_CORRECTION_SCALE, PF_F32, 1,
                    2 * PF_I, 1, uint64_t(2 * PF_I) * sizeof(float)) &&
                validate_ffn_correction_entry(
                    di, layer, PF_DOWN_CORRECTION_INDEX, PF_U16, 1,
                    PF_DOWN_KEEPER_K, 1, uint64_t(PF_DOWN_KEEPER_K) * sizeof(uint16_t)) &&
                validate_ffn_correction_entry(
                    dw, layer, PF_DOWN_CORRECTION_W8, PF_S8, 2,
                    PF_H, PF_DOWN_KEEPER_K, uint64_t(PF_H) * PF_DOWN_KEEPER_K) &&
                validate_ffn_correction_entry(
                    ds, layer, PF_DOWN_CORRECTION_SCALE, PF_F32, 1,
                    PF_H, 1, uint64_t(PF_H) * sizeof(float));
        for (const PFEntry * e : {&gi, &gw, &gs, &di, &dw, &ds}) {
            valid = valid && e->offset == next_offset;
            next_offset += e->length;
        }
        s.gate_correction_index[layer] = gi.offset;
        s.gate_correction_weight[layer] = gw.offset;
        s.gate_correction_scale[layer] = gs.offset;
        s.down_correction_index[layer] = di.offset;
        s.down_correction_weight[layer] = dw.offset;
        s.down_correction_scale[layer] = ds.offset;
    }
    valid = valid && next_offset == PF_FFN_CORRECTION_FILE_BYTES;
    if (!valid) {
        std::fprintf(stderr, "promptforge: invalid PFSIU4K table\n");
        munmap(mapping, PF_FFN_CORRECTION_FILE_BYTES);
        close(fd);
        return false;
    }
    if (!hip_check(hipMalloc(&s.ffn_correction_device_file, PF_FFN_CORRECTION_FILE_BYTES),
                   "hipMalloc(FFN correction sidecar)") ||
        !hip_check(hipMemcpy(s.ffn_correction_device_file, mapping,
                            PF_FFN_CORRECTION_FILE_BYTES, hipMemcpyHostToDevice),
                   "hipMemcpy(FFN correction sidecar)")) {
        munmap(mapping, PF_FFN_CORRECTION_FILE_BYTES);
        close(fd);
        return false;
    }
    madvise(mapping, PF_FFN_CORRECTION_FILE_BYTES, MADV_DONTNEED);
    munmap(mapping, PF_FFN_CORRECTION_FILE_BYTES);
    posix_fadvise(fd, 0, PF_FFN_CORRECTION_FILE_BYTES, POSIX_FADV_DONTNEED);
    close(fd);
    std::fprintf(stderr,
        "{\"record\":\"promptforge_ffn_correction_init\",\"device_bytes\":%llu,"
        "\"gate_keeper_k\":%d,\"down_keeper_k\":%d}\n",
        (unsigned long long) PF_FFN_CORRECTION_FILE_BYTES,
        PF_GATE_KEEPER_K, PF_DOWN_KEEPER_K);
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

bool validate_gdn_iu4_entry(const PFEntry & e, int layer, uint16_t kind, uint8_t dtype,
                            uint8_t rank, uint32_t rows, uint32_t cols, uint64_t length) {
    return e.layer == layer && e.kind == kind && e.dtype == dtype && e.rank == rank &&
           e.rows == rows && e.cols == cols && e.length == length &&
           e.offset <= PF_GDN_IU4_FILE_BYTES && e.length <= PF_GDN_IU4_FILE_BYTES - e.offset;
}

bool load_gdn_iu4_sidecar(PFState & s, const char * path) {
    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        std::fprintf(stderr, "promptforge: cannot open GDN IU4 sidecar %s\n", path);
        return false;
    }
    struct stat st {};
    if (fstat(fd, &st) != 0 || uint64_t(st.st_size) != PF_GDN_IU4_FILE_BYTES) {
        std::fprintf(stderr, "promptforge: wrong GDN IU4 sidecar size\n");
        close(fd);
        return false;
    }

    std::array<uint8_t, PF_GDN_IU4_DATA_OFFSET> header_table{};
    if (pread(fd, header_table.data(), header_table.size(), 0) != ssize_t(header_table.size())) {
        std::fprintf(stderr, "promptforge: cannot read GDN IU4 header/table\n");
        close(fd);
        return false;
    }
    PFHeader header {};
    std::memcpy(&header, header_table.data(), sizeof(header));
    const char expected_magic[8] = {'P','F','S','I','U','4','G','\0'};
    bool valid = std::memcmp(header.magic, expected_magic, 8) == 0 && header.version == 1 &&
                 header.header_bytes == sizeof(PFHeader) && header.entry_bytes == sizeof(PFEntry) &&
                 header.entry_count == PF_GDN_IU4_ENTRY_COUNT && header.table_offset == sizeof(PFHeader) &&
                 header.table_bytes == uint64_t(PF_GDN_IU4_ENTRY_COUNT) * sizeof(PFEntry) &&
                 header.data_offset == PF_GDN_IU4_DATA_OFFSET && header.file_bytes == PF_GDN_IU4_FILE_BYTES;
    uint64_t next_offset = PF_GDN_IU4_DATA_OFFSET;
    int entry_index = 0;
    for (int layer = 0; valid && layer < PF_LAYERS; ++layer) {
        if (layer % 4 == 3) {
            continue;
        }
        PFEntry weight {};
        PFEntry scale {};
        PFEntry sum {};
        for (PFEntry * entry : {&weight, &scale, &sum}) {
            const size_t pos = size_t(header.table_offset) + size_t(entry_index++) * sizeof(PFEntry);
            std::memcpy(entry, header_table.data() + pos, sizeof(*entry));
        }
        valid = validate_gdn_iu4_entry(weight, layer, PF_GDN_QKVZ_W4, PF_S4, 2,
                                       PF_GDN_N, PF_H, uint64_t(PF_GDN_N) * PF_H / 2) &&
                validate_gdn_iu4_entry(scale, layer, PF_GDN_QKVZ_W4_SCALE, PF_F32, 1,
                                       PF_GDN_N, 1, uint64_t(PF_GDN_N) * sizeof(float)) &&
                validate_gdn_iu4_entry(sum, layer, PF_GDN_QKVZ_W4_SUM, PF_I32, 1,
                                       PF_GDN_N, 1, uint64_t(PF_GDN_N) * sizeof(int32_t));
        for (const PFEntry * entry : {&weight, &scale, &sum}) {
            valid = valid && entry->offset == next_offset;
            next_offset += entry->length;
        }
        s.gdn_weight[layer] = weight.offset;
        s.gdn_scale[layer] = scale.offset;
        s.gdn_sum[layer] = sum.offset;
    }
    valid = valid && entry_index == int(PF_GDN_IU4_ENTRY_COUNT) &&
            next_offset == PF_GDN_IU4_FILE_BYTES;
    if (!valid) {
        std::fprintf(stderr, "promptforge: invalid PFSIU4G header/table\n");
        close(fd);
        return false;
    }

    void * mapping = mmap(nullptr, PF_GDN_IU4_FILE_BYTES, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) {
        std::fprintf(stderr, "promptforge: GDN IU4 sidecar mmap failed\n");
        close(fd);
        return false;
    }
    if (!hip_check(hipMalloc(&s.gdn_device_file, PF_GDN_IU4_FILE_BYTES), "hipMalloc(GDN IU4 sidecar)") ||
        !hip_check(hipMemcpy(s.gdn_device_file, mapping, PF_GDN_IU4_FILE_BYTES, hipMemcpyHostToDevice),
                   "hipMemcpy(GDN IU4 sidecar)")) {
        munmap(mapping, PF_GDN_IU4_FILE_BYTES);
        close(fd);
        return false;
    }
    madvise(mapping, PF_GDN_IU4_FILE_BYTES, MADV_DONTNEED);
    munmap(mapping, PF_GDN_IU4_FILE_BYTES);
    posix_fadvise(fd, 0, PF_GDN_IU4_FILE_BYTES, POSIX_FADV_DONTNEED);
    close(fd);
    return true;
}

bool validate_attention_iu4_entry(const PFEntry & e, int layer, uint16_t kind, uint8_t dtype,
                                  uint8_t rank, uint32_t rows, uint32_t cols, uint64_t length) {
    return e.layer == layer && e.kind == kind && e.dtype == dtype && e.rank == rank &&
           e.rows == rows && e.cols == cols && e.length == length &&
           e.offset <= PF_ATTENTION_IU4_FILE_BYTES &&
           e.length <= PF_ATTENTION_IU4_FILE_BYTES - e.offset;
}

bool load_attention_iu4_sidecar(PFState & s, const char * path) {
    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        std::fprintf(stderr, "promptforge: cannot open attention IU4 sidecar %s\n", path);
        return false;
    }
    struct stat st {};
    if (fstat(fd, &st) != 0 || uint64_t(st.st_size) != PF_ATTENTION_IU4_FILE_BYTES) {
        std::fprintf(stderr, "promptforge: wrong attention IU4 sidecar size\n");
        close(fd);
        return false;
    }
    std::array<uint8_t, PF_ATTENTION_IU4_DATA_OFFSET> header_table{};
    if (pread(fd, header_table.data(), header_table.size(), 0) != ssize_t(header_table.size())) {
        std::fprintf(stderr, "promptforge: cannot read attention IU4 header/table\n");
        close(fd);
        return false;
    }
    PFHeader header {};
    std::memcpy(&header, header_table.data(), sizeof(header));
    const char expected_magic[8] = {'P','F','S','I','U','4','A','\0'};
    bool valid = std::memcmp(header.magic, expected_magic, 8) == 0 && header.version == 1 &&
                 header.header_bytes == sizeof(PFHeader) && header.entry_bytes == sizeof(PFEntry) &&
                 header.entry_count == PF_ATTENTION_IU4_ENTRY_COUNT &&
                 header.table_offset == sizeof(PFHeader) &&
                 header.table_bytes == uint64_t(PF_ATTENTION_IU4_ENTRY_COUNT) * sizeof(PFEntry) &&
                 header.data_offset == PF_ATTENTION_IU4_DATA_OFFSET &&
                 header.file_bytes == PF_ATTENTION_IU4_FILE_BYTES;
    uint64_t next_offset = PF_ATTENTION_IU4_DATA_OFFSET;
    int entry_index = 0;
    for (int layer = 3; valid && layer < PF_LAYERS; layer += 4) {
        std::array<PFEntry, 6> layer_entries{};
        for (PFEntry & entry : layer_entries) {
            const size_t pos = size_t(header.table_offset) + size_t(entry_index++) * sizeof(PFEntry);
            std::memcpy(&entry, header_table.data() + pos, sizeof(entry));
        }
        const PFEntry & qw = layer_entries[0];
        const PFEntry & qs = layer_entries[1];
        const PFEntry & qz = layer_entries[2];
        const PFEntry & ow = layer_entries[3];
        const PFEntry & os = layer_entries[4];
        const PFEntry & oz = layer_entries[5];
        valid = validate_attention_iu4_entry(qw, layer, PF_ATTENTION_QKV_W4, PF_S4, 2,
                                             PF_ATTENTION_QKV_N, PF_H,
                                             uint64_t(PF_ATTENTION_QKV_N) * PF_H / 2) &&
                validate_attention_iu4_entry(qs, layer, PF_ATTENTION_QKV_W4_SCALE, PF_F32, 1,
                                             PF_ATTENTION_QKV_N, 1,
                                             uint64_t(PF_ATTENTION_QKV_N) * sizeof(float)) &&
                validate_attention_iu4_entry(qz, layer, PF_ATTENTION_QKV_W4_SUM, PF_I32, 1,
                                             PF_ATTENTION_QKV_N, 1,
                                             uint64_t(PF_ATTENTION_QKV_N) * sizeof(int32_t)) &&
                validate_attention_iu4_entry(ow, layer, PF_ATTENTION_OUTPUT_W4, PF_S4, 2,
                                             PF_H, PF_ATTENTION_VALUE_N,
                                             uint64_t(PF_H) * PF_ATTENTION_VALUE_N / 2) &&
                validate_attention_iu4_entry(os, layer, PF_ATTENTION_OUTPUT_W4_SCALE, PF_F32, 1,
                                             PF_H, 1, uint64_t(PF_H) * sizeof(float)) &&
                validate_attention_iu4_entry(oz, layer, PF_ATTENTION_OUTPUT_W4_SUM, PF_I32, 1,
                                             PF_H, 1, uint64_t(PF_H) * sizeof(int32_t));
        for (const PFEntry & entry : layer_entries) {
            valid = valid && entry.offset == next_offset;
            next_offset += entry.length;
        }
        s.attention_qkv_weight[layer] = qw.offset;
        s.attention_qkv_scale[layer] = qs.offset;
        s.attention_qkv_sum[layer] = qz.offset;
        s.attention_output_weight[layer] = ow.offset;
        s.attention_output_scale[layer] = os.offset;
        s.attention_output_sum[layer] = oz.offset;
    }
    valid = valid && entry_index == int(PF_ATTENTION_IU4_ENTRY_COUNT) &&
            next_offset == PF_ATTENTION_IU4_FILE_BYTES;
    if (!valid) {
        std::fprintf(stderr, "promptforge: invalid PFSIU4A header/table\n");
        close(fd);
        return false;
    }
    void * mapping = mmap(nullptr, PF_ATTENTION_IU4_FILE_BYTES, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) {
        std::fprintf(stderr, "promptforge: attention IU4 sidecar mmap failed\n");
        close(fd);
        return false;
    }
    if (!hip_check(hipMalloc(&s.attention_device_file, PF_ATTENTION_IU4_FILE_BYTES),
                   "hipMalloc(attention IU4 sidecar)") ||
        !hip_check(hipMemcpy(s.attention_device_file, mapping, PF_ATTENTION_IU4_FILE_BYTES,
                            hipMemcpyHostToDevice), "hipMemcpy(attention IU4 sidecar)")) {
        munmap(mapping, PF_ATTENTION_IU4_FILE_BYTES);
        close(fd);
        return false;
    }
    madvise(mapping, PF_ATTENTION_IU4_FILE_BYTES, MADV_DONTNEED);
    munmap(mapping, PF_ATTENTION_IU4_FILE_BYTES);
    posix_fadvise(fd, 0, PF_ATTENTION_IU4_FILE_BYTES, POSIX_FADV_DONTNEED);
    close(fd);
    return true;
}

bool load_gdn_output_iu4_sidecar(PFState & s, const char * path) {
    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        std::fprintf(stderr, "promptforge: cannot open GDN output IU4 sidecar %s\n", path);
        return false;
    }
    struct stat st {};
    if (fstat(fd, &st) != 0 || uint64_t(st.st_size) != PF_GDN_OUTPUT_IU4_FILE_BYTES) {
        std::fprintf(stderr, "promptforge: wrong GDN output IU4 sidecar size\n");
        close(fd);
        return false;
    }
    std::array<uint8_t, PF_GDN_OUTPUT_IU4_DATA_OFFSET> header_table{};
    if (pread(fd, header_table.data(), header_table.size(), 0) != ssize_t(header_table.size())) {
        std::fprintf(stderr, "promptforge: cannot read GDN output IU4 header/table\n");
        close(fd);
        return false;
    }
    PFHeader header {};
    std::memcpy(&header, header_table.data(), sizeof(header));
    const char expected_magic[8] = {'P','F','S','I','U','4','O','\0'};
    bool valid = std::memcmp(header.magic, expected_magic, 8) == 0 && header.version == 1 &&
                 header.header_bytes == sizeof(PFHeader) && header.entry_bytes == sizeof(PFEntry) &&
                 header.entry_count == PF_GDN_OUTPUT_IU4_ENTRY_COUNT &&
                 header.table_offset == sizeof(PFHeader) &&
                 header.table_bytes == uint64_t(PF_GDN_OUTPUT_IU4_ENTRY_COUNT) * sizeof(PFEntry) &&
                 header.data_offset == PF_GDN_OUTPUT_IU4_DATA_OFFSET &&
                 header.file_bytes == PF_GDN_OUTPUT_IU4_FILE_BYTES;
    uint64_t next_offset = PF_GDN_OUTPUT_IU4_DATA_OFFSET;
    int entry_index = 0;
    for (int layer = 0; valid && layer < PF_LAYERS; ++layer) {
        if (layer % 4 == 3) {
            continue;
        }
        std::array<PFEntry, 3> entries{};
        for (PFEntry & entry : entries) {
            const size_t pos = size_t(header.table_offset) + size_t(entry_index++) * sizeof(PFEntry);
            std::memcpy(&entry, header_table.data() + pos, sizeof(entry));
        }
        const PFEntry & weight = entries[0];
        const PFEntry & scale = entries[1];
        const PFEntry & sum = entries[2];
        valid = validate_attention_iu4_entry(weight, layer, PF_GDN_OUTPUT_W4, PF_S4, 2,
                                             PF_H, PF_ATTENTION_VALUE_N,
                                             uint64_t(PF_H) * PF_ATTENTION_VALUE_N / 2) &&
                validate_attention_iu4_entry(scale, layer, PF_GDN_OUTPUT_W4_SCALE, PF_F32, 1,
                                             PF_H, 1, uint64_t(PF_H) * sizeof(float)) &&
                validate_attention_iu4_entry(sum, layer, PF_GDN_OUTPUT_W4_SUM, PF_I32, 1,
                                             PF_H, 1, uint64_t(PF_H) * sizeof(int32_t));
        for (const PFEntry & entry : entries) {
            valid = valid && entry.offset == next_offset;
            next_offset += entry.length;
        }
        s.gdn_output_weight[layer] = weight.offset;
        s.gdn_output_scale[layer] = scale.offset;
        s.gdn_output_sum[layer] = sum.offset;
    }
    valid = valid && entry_index == int(PF_GDN_OUTPUT_IU4_ENTRY_COUNT) &&
            next_offset == PF_GDN_OUTPUT_IU4_FILE_BYTES;
    if (!valid) {
        std::fprintf(stderr, "promptforge: invalid PFSIU4O header/table\n");
        close(fd);
        return false;
    }
    void * mapping = mmap(nullptr, PF_GDN_OUTPUT_IU4_FILE_BYTES, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) {
        std::fprintf(stderr, "promptforge: GDN output IU4 sidecar mmap failed\n");
        close(fd);
        return false;
    }
    if (!hip_check(hipMalloc(&s.gdn_output_device_file, PF_GDN_OUTPUT_IU4_FILE_BYTES),
                   "hipMalloc(GDN output IU4 sidecar)") ||
        !hip_check(hipMemcpy(s.gdn_output_device_file, mapping, PF_GDN_OUTPUT_IU4_FILE_BYTES,
                            hipMemcpyHostToDevice), "hipMemcpy(GDN output IU4 sidecar)")) {
        munmap(mapping, PF_GDN_OUTPUT_IU4_FILE_BYTES);
        close(fd);
        return false;
    }
    madvise(mapping, PF_GDN_OUTPUT_IU4_FILE_BYTES, MADV_DONTNEED);
    munmap(mapping, PF_GDN_OUTPUT_IU4_FILE_BYTES);
    posix_fadvise(fd, 0, PF_GDN_OUTPUT_IU4_FILE_BYTES, POSIX_FADV_DONTNEED);
    close(fd);
    return true;
}

bool allocate_scratch(PFState & s) {
    const bool base_ok = hip_check(hipMalloc(&s.gate_a8, size_t(PF_M) * PF_H), "hipMalloc(gate_a8)") &&
           hip_check(hipMalloc(&s.gate_a4, size_t(PF_M) * PF_ATTENTION_VALUE_N / 2), "hipMalloc(gate_a4)") &&
           hip_check(hipMalloc(&s.gate_a_scale, size_t(PF_M) *
                               (s.ffn_segmented ? PF_H / promptforge_iu4::kQuantSegment : 1) *
                               sizeof(float)), "hipMalloc(gate_scale)") &&
           hip_check(hipMalloc(&s.gate_a_zero, size_t(PF_M) *
                               (s.ffn_segmented ? PF_H / promptforge_iu4::kQuantSegment : 1) *
                               sizeof(int32_t)), "hipMalloc(gate_zero)") &&
           hip_check(hipMalloc(&s.gate_out, size_t(PF_M) * (2 * PF_I) * sizeof(__hip_bfloat16)), "hipMalloc(gate_out)") &&
           hip_check(hipMalloc(&s.down_a8, size_t(PF_M) * PF_I), "hipMalloc(down_a8)") &&
           hip_check(hipMalloc(&s.down_a4, size_t(PF_M) * PF_I / 2), "hipMalloc(down_a4)") &&
           hip_check(hipMalloc(&s.down_a_scale, size_t(PF_M) *
                               (s.ffn_segmented ? PF_I / promptforge_iu4::kQuantSegment : 1) *
                               sizeof(float)), "hipMalloc(down_scale)") &&
           hip_check(hipMalloc(&s.down_a_zero, size_t(PF_M) *
                               (s.ffn_segmented ? PF_I / promptforge_iu4::kQuantSegment : 1) *
                               sizeof(int32_t)), "hipMalloc(down_zero)") &&
           hip_check(hipMalloc(&s.down_out, size_t(PF_M) * PF_H * sizeof(__hip_bfloat16)), "hipMalloc(down_out)");
    if (!base_ok) return false;
    if (s.ffn_correction_enabled &&
        (!hip_check(hipMalloc(&s.gate_correction_a8, size_t(PF_M) * PF_GATE_KEEPER_K),
                    "hipMalloc(gate_correction_a8)") ||
         !hip_check(hipMalloc(&s.gate_correction_a_scale, size_t(PF_M) * sizeof(float)),
                    "hipMalloc(gate_correction_scale)") ||
         !hip_check(hipMalloc(&s.down_correction_a8, size_t(PF_M) * PF_DOWN_KEEPER_K),
                    "hipMalloc(down_correction_a8)") ||
         !hip_check(hipMalloc(&s.down_correction_a_scale, size_t(PF_M) * sizeof(float)),
                    "hipMalloc(down_correction_scale)"))) {
        return false;
    }
    return true;
}

bool ensure_smallm_iu4_cache(PFState & s, int rows) {
    const PFRowPlan plan = smallm_row_plan(rows);
    if (!smallm_iu4_route_enabled(s, rows) || plan.bucket < 0 ||
        !s.iu4_smallm_gate_op || !s.iu4_smallm_down_op) {
        return false;
    }
    PFIU4RowRouteCache & route = s.iu4_smallm_routes[plan.bucket];
    if (route.ready && route.execution_rows == plan.execution_rows) {
        return true;
    }

    std::lock_guard<std::mutex> lock(s.smallm_cache_mutex);
    if (route.ready && route.execution_rows == plan.execution_rows) {
        return true;
    }

    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> gate_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> down_args;
    for (int layer = 0; layer < PF_LAYERS; ++layer) {
        const std::array<const void *, 4> gate_d{
            s.device_file + s.gate_scale[layer], s.gate_a_scale,
            s.device_file + s.gate_sum[layer], s.gate_a_zero};
        const std::array<const void *, 4> down_d{
            s.device_file + s.down_scale[layer], s.down_a_scale,
            s.device_file + s.down_sum[layer], s.down_a_zero};
        auto gate_arg = s.iu4_smallm_gate_op->MakeArgumentPointer(
            s.gate_a4, s.device_file + s.gate_weight[layer], gate_d, s.gate_out,
            plan.execution_rows, 2 * PF_I, PF_H / 2, PF_H / 2, PF_H / 2,
            std::array<ck::index_t, 4>{0, 0, 0, 0}, 2 * PF_I, 1,
            PassThrough{}, PassThrough{}, IU4ScaleCorrect{});
        auto down_arg = s.iu4_smallm_down_op->MakeArgumentPointer(
            s.down_a4, s.device_file + s.down_weight[layer], down_d, s.down_out,
            plan.execution_rows, PF_H, PF_I / 2, PF_I / 2, PF_I / 2,
            std::array<ck::index_t, 4>{0, 0, 0, 0}, PF_H, 1,
            PassThrough{}, PassThrough{}, IU4ScaleCorrect{});
        if (!s.iu4_smallm_gate_op->IsSupportedArgument(gate_arg.get()) ||
            !s.iu4_smallm_down_op->IsSupportedArgument(down_arg.get())) {
            std::fprintf(stderr,
                "{\"record\":\"promptforge_smallm_iu4_cache\",\"ready\":false,"
                "\"rows\":%d,\"execution_rows\":%d,\"bucket\":%d,\"layer\":%d}\n",
                rows, plan.execution_rows, plan.bucket, layer);
            std::fflush(stderr);
            return false;
        }
        gate_args[layer] = std::move(gate_arg);
        down_args[layer] = std::move(down_arg);
    }

    route.gate_args = std::move(gate_args);
    route.down_args = std::move(down_args);
    route.execution_rows = plan.execution_rows;
    route.ready = true;
    ++route.rebuilds;
    std::fprintf(stderr,
        "{\"record\":\"promptforge_smallm_iu4_cache\",\"ready\":true,"
        "\"rows\":%d,\"execution_rows\":%d,\"bucket\":%d,"
        "\"gate_args\":%d,\"down_args\":%d,\"rebuild\":%llu}\n",
        rows, plan.execution_rows, plan.bucket, PF_LAYERS, PF_LAYERS,
        (unsigned long long) route.rebuilds);
    std::fflush(stderr);
    return true;
}

bool ensure_smallm_gdn_iu4_cache(PFState & s, int rows) {
    const PFRowPlan plan = smallm_row_plan(rows);
    if (!smallm_gdn_iu4_route_enabled(s, rows) || plan.bucket < 0 ||
        !s.iu4_smallm_gate_op || !s.iu4_smallm_down_op ||
        !s.gdn_device_file || !s.gdn_output_device_file) {
        return false;
    }
    PFIU4RowRouteCache & route = s.iu4_smallm_routes[plan.bucket];
    if (route.gdn_ready && route.execution_rows == plan.execution_rows) {
        return true;
    }

    std::lock_guard<std::mutex> lock(s.smallm_cache_mutex);
    if (route.gdn_ready && route.execution_rows == plan.execution_rows) {
        return true;
    }
    if (route.execution_rows != 0 && route.execution_rows != plan.execution_rows) {
        return false;
    }

    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> gdn_args;
    std::array<std::unique_ptr<ck::tensor_operation::device::BaseArgument>, PF_LAYERS> output_args;
    for (int layer = 0; layer < PF_LAYERS; ++layer) {
        if (layer % 4 == 3) {
            continue;
        }
        const std::array<const void *, 4> gdn_d{
            s.gdn_device_file + s.gdn_scale[layer], s.gate_a_scale,
            s.gdn_device_file + s.gdn_sum[layer], s.gate_a_zero};
        const std::array<const void *, 4> output_d{
            s.gdn_output_device_file + s.gdn_output_scale[layer], s.gate_a_scale,
            s.gdn_output_device_file + s.gdn_output_sum[layer], s.gate_a_zero};
        auto gdn_arg = s.iu4_smallm_gate_op->MakeArgumentPointer(
            s.gate_a4, s.gdn_device_file + s.gdn_weight[layer], gdn_d, s.gate_out,
            plan.execution_rows, PF_GDN_N, PF_H / 2, PF_H / 2, PF_H / 2,
            std::array<ck::index_t, 4>{0, 0, 0, 0}, PF_GDN_N, 1,
            PassThrough{}, PassThrough{}, IU4ScaleCorrect{});
        auto output_arg = s.iu4_smallm_down_op->MakeArgumentPointer(
            s.gate_a4, s.gdn_output_device_file + s.gdn_output_weight[layer], output_d, s.down_out,
            plan.execution_rows, PF_H, PF_ATTENTION_VALUE_N / 2,
            PF_ATTENTION_VALUE_N / 2, PF_ATTENTION_VALUE_N / 2,
            std::array<ck::index_t, 4>{0, 0, 0, 0}, PF_H, 1,
            PassThrough{}, PassThrough{}, IU4ScaleCorrect{});
        if (!s.iu4_smallm_gate_op->IsSupportedArgument(gdn_arg.get()) ||
            !s.iu4_smallm_down_op->IsSupportedArgument(output_arg.get())) {
            std::fprintf(stderr,
                "{\"record\":\"promptforge_smallm_gdn_iu4_cache\",\"ready\":false,"
                "\"rows\":%d,\"execution_rows\":%d,\"bucket\":%d,\"layer\":%d}\n",
                rows, plan.execution_rows, plan.bucket, layer);
            std::fflush(stderr);
            return false;
        }
        gdn_args[layer] = std::move(gdn_arg);
        output_args[layer] = std::move(output_arg);
    }

    route.gdn_args = std::move(gdn_args);
    route.gdn_output_args = std::move(output_args);
    route.execution_rows = plan.execution_rows;
    route.gdn_ready = true;
    ++route.gdn_rebuilds;
    std::fprintf(stderr,
        "{\"record\":\"promptforge_smallm_gdn_iu4_cache\",\"ready\":true,"
        "\"rows\":%d,\"execution_rows\":%d,\"bucket\":%d,"
        "\"qkvz_args\":48,\"output_args\":48,\"rebuild\":%llu}\n",
        rows, plan.execution_rows, plan.bucket, (unsigned long long) route.gdn_rebuilds);
    std::fflush(stderr);
    return true;
}

bool build_iu4_routes(PFState & s) {
    s.iu4_gate_op = std::make_unique<IU4DeviceOp>();
    s.iu4_down_op = std::make_unique<IU4DownDeviceOp>();
    s.iu4_gate_padded_op = std::make_unique<IU4PaddedDeviceOp>();
    s.iu4_down_padded_op = std::make_unique<IU4PaddedDeviceOp>();
    s.iu4_decode_op = std::make_unique<IU4DecodeDeviceOp>();
    if (s.smallm_iu4_enabled) {
        s.iu4_smallm_gate_op = std::make_unique<IU4SmallGateDeviceOp>();
        s.iu4_smallm_down_op = std::make_unique<IU4SmallDownDeviceOp>();
    }
    for (int layer = 0; layer < PF_LAYERS; ++layer) {
        const std::array<const void *, 4> gate_d{
            s.device_file + s.gate_scale[layer], s.gate_a_scale,
            s.device_file + s.gate_sum[layer], s.gate_a_zero};
        const std::array<const void *, 4> down_d{
            s.device_file + s.down_scale[layer], s.down_a_scale,
            s.device_file + s.down_sum[layer], s.down_a_zero};
        auto gate_arg = s.iu4_gate_op->MakeArgumentPointer(
            s.gate_a4, s.device_file + s.gate_weight[layer], gate_d, s.gate_out,
            PF_M, 2 * PF_I, PF_H / 2, PF_H / 2, PF_H / 2,
            std::array<ck::index_t, 4>{0, 0, 0, 0}, 2 * PF_I, 1,
            PassThrough{}, PassThrough{}, IU4ScaleCorrect{});
        auto down_arg = s.iu4_down_op->MakeArgumentPointer(
            s.down_a4, s.device_file + s.down_weight[layer], down_d, s.down_out,
            PF_M, PF_H, PF_I / 2, PF_I / 2, PF_I / 2,
            std::array<ck::index_t, 4>{0, 0, 0, 0}, PF_H, 1,
            PassThrough{}, PassThrough{}, IU4ScaleCorrect{});
        auto checkpoint_gate_arg = s.iu4_gate_padded_op->MakeArgumentPointer(
            s.gate_a4, s.device_file + s.gate_weight[layer], gate_d, s.gate_out,
            PF_CHECKPOINT_M, 2 * PF_I, PF_H / 2, PF_H / 2, PF_H / 2,
            std::array<ck::index_t, 4>{0, 0, 0, 0}, 2 * PF_I, 1,
            PassThrough{}, PassThrough{}, IU4ScaleCorrect{});
        auto checkpoint_down_arg = s.iu4_down_padded_op->MakeArgumentPointer(
            s.down_a4, s.device_file + s.down_weight[layer], down_d, s.down_out,
            PF_CHECKPOINT_M, PF_H, PF_I / 2, PF_I / 2, PF_I / 2,
            std::array<ck::index_t, 4>{0, 0, 0, 0}, PF_H, 1,
            PassThrough{}, PassThrough{}, IU4ScaleCorrect{});
        auto tail_gate_arg = s.iu4_gate_padded_op->MakeArgumentPointer(
            s.gate_a4, s.device_file + s.gate_weight[layer], gate_d, s.gate_out,
            PF_TAIL_M, 2 * PF_I, PF_H / 2, PF_H / 2, PF_H / 2,
            std::array<ck::index_t, 4>{0, 0, 0, 0}, 2 * PF_I, 1,
            PassThrough{}, PassThrough{}, IU4ScaleCorrect{});
        auto tail_down_arg = s.iu4_down_padded_op->MakeArgumentPointer(
            s.down_a4, s.device_file + s.down_weight[layer], down_d, s.down_out,
            PF_TAIL_M, PF_H, PF_I / 2, PF_I / 2, PF_I / 2,
            std::array<ck::index_t, 4>{0, 0, 0, 0}, PF_H, 1,
            PassThrough{}, PassThrough{}, IU4ScaleCorrect{});
        if (!s.iu4_gate_op->IsSupportedArgument(gate_arg.get()) ||
            !s.iu4_down_op->IsSupportedArgument(down_arg.get()) ||
            !s.iu4_gate_padded_op->IsSupportedArgument(checkpoint_gate_arg.get()) ||
            !s.iu4_down_padded_op->IsSupportedArgument(checkpoint_down_arg.get()) ||
            !s.iu4_gate_padded_op->IsSupportedArgument(tail_gate_arg.get()) ||
            !s.iu4_down_padded_op->IsSupportedArgument(tail_down_arg.get())) {
            std::fprintf(stderr, "promptforge: IU4 CK argument unsupported for layer %d\n", layer);
            return false;
        }
        s.iu4_gate_args[layer] = std::move(gate_arg);
        s.iu4_down_args[layer] = std::move(down_arg);
        s.iu4_gate_checkpoint2044_args[layer] = std::move(checkpoint_gate_arg);
        s.iu4_down_checkpoint2044_args[layer] = std::move(checkpoint_down_arg);
        s.iu4_gate_tail1476_args[layer] = std::move(tail_gate_arg);
        s.iu4_down_tail1476_args[layer] = std::move(tail_down_arg);
        for (int rows = PF_DECODE_M_MIN; rows <= PF_DECODE_M_MAX; ++rows) {
            auto decode_gate_arg = s.iu4_decode_op->MakeArgumentPointer(
                s.gate_a4, s.device_file + s.gate_weight[layer], gate_d, s.gate_out,
                rows, 2 * PF_I, PF_H / 2, PF_H / 2, PF_H / 2,
                std::array<ck::index_t, 4>{0, 0, 0, 0}, 2 * PF_I, 1,
                PassThrough{}, PassThrough{}, IU4ScaleCorrect{});
            auto decode_down_arg = s.iu4_decode_op->MakeArgumentPointer(
                s.down_a4, s.device_file + s.down_weight[layer], down_d, s.down_out,
                rows, PF_H, PF_I / 2, PF_I / 2, PF_I / 2,
                std::array<ck::index_t, 4>{0, 0, 0, 0}, PF_H, 1,
                PassThrough{}, PassThrough{}, IU4ScaleCorrect{});
            if (!s.iu4_decode_op->IsSupportedArgument(decode_gate_arg.get()) ||
                !s.iu4_decode_op->IsSupportedArgument(decode_down_arg.get())) {
                std::fprintf(stderr,
                             "promptforge: IU4 decode CK argument unsupported for layer %d M%d\n",
                             layer, rows);
                return false;
            }
            s.iu4_gate_decode_args[layer][rows] = std::move(decode_gate_arg);
            s.iu4_down_decode_args[layer][rows] = std::move(decode_down_arg);
        }
    }
    s.iu4_gate_invoker = s.iu4_gate_op->MakeInvokerPointer();
    s.iu4_down_invoker = s.iu4_down_op->MakeInvokerPointer();
    s.iu4_gate_padded_invoker = s.iu4_gate_padded_op->MakeInvokerPointer();
    s.iu4_down_padded_invoker = s.iu4_down_padded_op->MakeInvokerPointer();
    s.iu4_decode_invoker = s.iu4_decode_op->MakeInvokerPointer();
    if (s.smallm_iu4_enabled) {
        s.iu4_smallm_gate_invoker = s.iu4_smallm_gate_op->MakeInvokerPointer();
        s.iu4_smallm_down_invoker = s.iu4_smallm_down_op->MakeInvokerPointer();
    }
    std::fprintf(stderr,
        "{\"record\":\"promptforge_init\",\"mode\":\"iu4_ffn\","
        "\"device_bytes\":%llu,\"wmma\":\"v_wmma_i32_16x16x16_iu4\","
        "\"layouts\":\"production-shaped\",\"transform\":\"%s\","
        "\"decode_rows\":[2,3,4,5],\"decode_tile_m\":32,"
        "\"rows\":[1476,2044,2048]}\n",
        (unsigned long long) PF_IU4_FILE_BYTES,
        s.ffn_hadamard ? "block_hadamard_1024" : "none");
    std::fprintf(stderr,
        "{\"record\":\"promptforge_smallm_iu4_init\",\"enabled\":%s,"
        "\"min_rows\":%d,\"max_rows\":%d,\"buckets\":[128,256,512],"
        "\"scope\":\"ffn_gate_up_down\",\"cache\":\"lazy_bucket\"}\n",
        s.smallm_iu4_enabled ? "true" : "false", PF_SMALLM_MIN, PF_SMALLM_MAX);
    std::fflush(stderr);
    return true;
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
    if (s.mode == PFMode::iu4_ffn && !s.gdn_w8) {
        s.iu4_gdn_op = std::make_unique<IU4PaddedDeviceOp>();
        for (int layer = 0; layer < PF_LAYERS; ++layer) {
            if (layer % 4 == 3) {
                continue;
            }
            const std::array<const void *, 4> d{
                s.gdn_device_file + s.gdn_scale[layer], s.gate_a_scale,
                s.gdn_device_file + s.gdn_sum[layer], s.gate_a_zero};
            auto make_arg = [&](int rows) {
                return s.iu4_gdn_op->MakeArgumentPointer(
                    s.gate_a4, s.gdn_device_file + s.gdn_weight[layer], d, s.gate_out,
                    rows, PF_GDN_N, PF_H / 2, PF_H / 2, PF_H / 2,
                    std::array<ck::index_t, 4>{0, 0, 0, 0}, PF_GDN_N, 1,
                    PassThrough{}, PassThrough{}, IU4ScaleCorrect{});
            };
            auto arg = make_arg(PF_M);
            auto checkpoint_arg = make_arg(PF_CHECKPOINT_M);
            auto tail_arg = make_arg(PF_TAIL_M);
            if (!s.iu4_gdn_op->IsSupportedArgument(arg.get()) ||
                !s.iu4_gdn_op->IsSupportedArgument(checkpoint_arg.get()) ||
                !s.iu4_gdn_op->IsSupportedArgument(tail_arg.get())) {
                std::fprintf(stderr, "promptforge: GDN IU4 CK argument unsupported for layer %d\n", layer);
                return false;
            }
            s.gdn_args[layer] = std::move(arg);
            s.gdn_checkpoint2044_args[layer] = std::move(checkpoint_arg);
            s.gdn_tail1476_args[layer] = std::move(tail_arg);
        }
        s.gdn_invoker = s.iu4_gdn_op->MakeInvokerPointer();
        s.gdn_checkpoint2044_invoker = s.iu4_gdn_op->MakeInvokerPointer();
        s.gdn_tail1476_invoker = s.iu4_gdn_op->MakeInvokerPointer();
        std::fprintf(stderr,
            "{\"record\":\"promptforge_gdn_init\",\"projection\":\"%s\","
            "\"device_bytes\":%llu,\"layers\":48,\"n\":%d,\"k\":%d,"
            "\"transform\":\"%s\"}\n",
            s.gdn_hadamard ? "qkvz_iu4_hadamard" : "qkvz_iu4",
            (unsigned long long) PF_GDN_IU4_FILE_BYTES, PF_GDN_N, PF_H,
            s.gdn_hadamard ? "block_hadamard_1024" : "none");
        return true;
    }
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

bool build_attention_routes(PFState & s) {
    s.attention_qkv_op = std::make_unique<IU4PaddedDeviceOp>();
    s.attention_output_op = std::make_unique<IU4PaddedDeviceOp>();
    for (int layer = 3; layer < PF_LAYERS; layer += 4) {
        const std::array<const void *, 4> qkv_d{
            s.attention_device_file + s.attention_qkv_scale[layer], s.gate_a_scale,
            s.attention_device_file + s.attention_qkv_sum[layer], s.gate_a_zero};
        const std::array<const void *, 4> output_d{
            s.attention_device_file + s.attention_output_scale[layer], s.gate_a_scale,
            s.attention_device_file + s.attention_output_sum[layer], s.gate_a_zero};
        auto make_qkv_arg = [&](int rows) {
            return s.attention_qkv_op->MakeArgumentPointer(
                s.gate_a4, s.attention_device_file + s.attention_qkv_weight[layer], qkv_d, s.gate_out,
                rows, PF_ATTENTION_QKV_N, PF_H / 2, PF_H / 2, PF_H / 2,
                std::array<ck::index_t, 4>{0, 0, 0, 0}, PF_ATTENTION_QKV_N, 1,
                PassThrough{}, PassThrough{}, IU4ScaleCorrect{});
        };
        auto make_output_arg = [&](int rows) {
            return s.attention_output_op->MakeArgumentPointer(
                s.gate_a4, s.attention_device_file + s.attention_output_weight[layer], output_d, s.down_out,
                rows, PF_H, PF_ATTENTION_VALUE_N / 2,
                PF_ATTENTION_VALUE_N / 2, PF_ATTENTION_VALUE_N / 2,
                std::array<ck::index_t, 4>{0, 0, 0, 0}, PF_H, 1,
                PassThrough{}, PassThrough{}, IU4ScaleCorrect{});
        };
        auto qkv = make_qkv_arg(PF_M);
        auto qkv_checkpoint = make_qkv_arg(PF_CHECKPOINT_M);
        auto qkv_tail = make_qkv_arg(PF_TAIL_M);
        auto output = make_output_arg(PF_M);
        auto output_checkpoint = make_output_arg(PF_CHECKPOINT_M);
        auto output_tail = make_output_arg(PF_TAIL_M);
        if (!s.attention_qkv_op->IsSupportedArgument(qkv.get()) ||
            !s.attention_qkv_op->IsSupportedArgument(qkv_checkpoint.get()) ||
            !s.attention_qkv_op->IsSupportedArgument(qkv_tail.get()) ||
            !s.attention_output_op->IsSupportedArgument(output.get()) ||
            !s.attention_output_op->IsSupportedArgument(output_checkpoint.get()) ||
            !s.attention_output_op->IsSupportedArgument(output_tail.get())) {
            std::fprintf(stderr, "promptforge: attention IU4 CK argument unsupported for layer %d\n", layer);
            return false;
        }
        s.attention_qkv_args[layer] = std::move(qkv);
        s.attention_qkv_checkpoint2044_args[layer] = std::move(qkv_checkpoint);
        s.attention_qkv_tail1476_args[layer] = std::move(qkv_tail);
        s.attention_output_args[layer] = std::move(output);
        s.attention_output_checkpoint2044_args[layer] = std::move(output_checkpoint);
        s.attention_output_tail1476_args[layer] = std::move(output_tail);
    }
    s.attention_qkv_invoker = s.attention_qkv_op->MakeInvokerPointer();
    s.attention_output_invoker = s.attention_output_op->MakeInvokerPointer();
    std::fprintf(stderr,
        "{\"record\":\"promptforge_attention_init\",\"projection\":\"qkv_output_iu4\","
        "\"device_bytes\":%llu,\"layers\":16,\"qkv_n\":%d,\"output_k\":%d}\n",
        (unsigned long long) PF_ATTENTION_IU4_FILE_BYTES,
        PF_ATTENTION_QKV_N, PF_ATTENTION_VALUE_N);
    return true;
}

bool build_gdn_output_routes(PFState & s) {
    s.gdn_output_op = std::make_unique<IU4PaddedDeviceOp>();
    s.gdn_output_decode_op = std::make_unique<IU4DecodeDeviceOp>();
    for (int layer = 0; layer < PF_LAYERS; ++layer) {
        if (layer % 4 == 3) {
            continue;
        }
        const std::array<const void *, 4> d{
            s.gdn_output_device_file + s.gdn_output_scale[layer], s.gate_a_scale,
            s.gdn_output_device_file + s.gdn_output_sum[layer], s.gate_a_zero};
        auto make_arg = [&](int rows) {
            return s.gdn_output_op->MakeArgumentPointer(
                s.gate_a4, s.gdn_output_device_file + s.gdn_output_weight[layer], d, s.down_out,
                rows, PF_H, PF_ATTENTION_VALUE_N / 2,
                PF_ATTENTION_VALUE_N / 2, PF_ATTENTION_VALUE_N / 2,
                std::array<ck::index_t, 4>{0, 0, 0, 0}, PF_H, 1,
                PassThrough{}, PassThrough{}, IU4ScaleCorrect{});
        };
        auto arg = make_arg(PF_M);
        auto checkpoint_arg = make_arg(PF_CHECKPOINT_M);
        auto tail_arg = make_arg(PF_TAIL_M);
        if (!s.gdn_output_op->IsSupportedArgument(arg.get()) ||
            !s.gdn_output_op->IsSupportedArgument(checkpoint_arg.get()) ||
            !s.gdn_output_op->IsSupportedArgument(tail_arg.get())) {
            std::fprintf(stderr, "promptforge: GDN output IU4 CK argument unsupported for layer %d\n", layer);
            return false;
        }
        s.gdn_output_args[layer] = std::move(arg);
        s.gdn_output_checkpoint2044_args[layer] = std::move(checkpoint_arg);
        s.gdn_output_tail1476_args[layer] = std::move(tail_arg);
        for (int rows = PF_DECODE_M_MIN; rows <= PF_DECODE_M_MAX; ++rows) {
            auto decode_arg = s.gdn_output_decode_op->MakeArgumentPointer(
                s.gate_a4, s.gdn_output_device_file + s.gdn_output_weight[layer], d, s.down_out,
                rows, PF_H, PF_ATTENTION_VALUE_N / 2,
                PF_ATTENTION_VALUE_N / 2, PF_ATTENTION_VALUE_N / 2,
                std::array<ck::index_t, 4>{0, 0, 0, 0}, PF_H, 1,
                PassThrough{}, PassThrough{}, IU4ScaleCorrect{});
            if (!s.gdn_output_decode_op->IsSupportedArgument(decode_arg.get())) {
                std::fprintf(stderr,
                             "promptforge: GDN output IU4 decode CK argument unsupported for layer %d M%d\n",
                             layer, rows);
                return false;
            }
            s.gdn_output_decode_args[layer][rows] = std::move(decode_arg);
        }
    }
    s.gdn_output_invoker = s.gdn_output_op->MakeInvokerPointer();
    s.gdn_output_decode_invoker = s.gdn_output_decode_op->MakeInvokerPointer();
    std::fprintf(stderr,
        "{\"record\":\"promptforge_gdn_output_init\",\"projection\":\"output_iu4\","
        "\"device_bytes\":%llu,\"layers\":48,\"n\":%d,\"k\":%d,"
        "\"decode_rows\":[2,3,4,5],\"decode_tile_m\":32}\n",
        (unsigned long long) PF_GDN_OUTPUT_IU4_FILE_BYTES, PF_H, PF_ATTENTION_VALUE_N);
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
    if (s.mode == PFMode::iu4_ffn) {
        const promptforge_iu4::packed_activations activations{
            s.gate_a4, s.gate_a_scale, s.gate_a_zero};
        if (s.ffn_segmented) {
            promptforge_iu4::launch_segmented_hadamard_input_pack<PF_H, PF_GATE_HADAMARD_SEED>(
                input, activations, rows, stream);
        } else if (s.ffn_hadamard) {
            promptforge_iu4::launch_hadamard_input_pack<PF_H, PF_GATE_HADAMARD_SEED>(
                input, activations, rows, stream);
        } else {
            promptforge_iu4::launch_input_pack<PF_H>(input, nullptr, activations, rows, stream);
        }
        if (s.ffn_correction_enabled) {
            promptforge_iu4::launch_indexed_input_pack<PF_H, PF_GATE_KEEPER_K>(
                input,
                reinterpret_cast<const uint16_t *>(
                    s.ffn_correction_device_file + s.gate_correction_index[layer]),
                s.gate_correction_a8, s.gate_correction_a_scale, rows, stream);
        }
        if (s.ffn_segmented) {
            const promptforge_iu4::packed_matrix matrix{
                reinterpret_cast<const uint32_t *>(s.device_file + s.gate_weight[layer]),
                reinterpret_cast<const float *>(s.device_file + s.gate_scale[layer]),
                reinterpret_cast<const int32_t *>(s.device_file + s.gate_sum[layer])};
            promptforge_iu4::launch_gemm<PF_H>(
                activations, matrix, s.gate_out, rows, 2 * PF_I, stream);
        } else if (smallm_iu4_route_enabled(s, rows)) {
            const PFRowPlan plan = smallm_row_plan(rows);
            s.iu4_smallm_gate_invoker->Run(
                s.iu4_smallm_routes[plan.bucket].gate_args[layer].get(),
                ::StreamConfig{stream, false});
        } else if (iu4_decode_width_enabled(s, rows)) {
            s.iu4_decode_invoker->Run(
                s.iu4_gate_decode_args[layer][rows].get(), ::StreamConfig{stream, false});
        } else if (rows == PF_M) {
            s.iu4_gate_invoker->Run(s.iu4_gate_args[layer].get(), ::StreamConfig{stream, false});
        } else if (rows == PF_CHECKPOINT_M) {
            s.iu4_gate_padded_invoker->Run(
                s.iu4_gate_checkpoint2044_args[layer].get(), ::StreamConfig{stream, false});
        } else {
            s.iu4_gate_padded_invoker->Run(
                s.iu4_gate_tail1476_args[layer].get(), ::StreamConfig{stream, false});
        }
        if (s.ffn_correction_enabled) {
            promptforge_iu4::launch_i8_correction<PF_GATE_KEEPER_K>(
                s.gate_correction_a8, s.gate_correction_a_scale,
                reinterpret_cast<const uint32_t *>(
                    s.ffn_correction_device_file + s.gate_correction_weight[layer]),
                reinterpret_cast<const float *>(
                    s.ffn_correction_device_file + s.gate_correction_scale[layer]),
                s.gate_out, rows, 2 * PF_I, stream);
        }
        const promptforge_iu4::packed_activations down_activations{
            s.down_a4, s.down_a_scale, s.down_a_zero};
        if (s.ffn_segmented) {
            promptforge_iu4::launch_segmented_hadamard_swiglu_pack<PF_I, PF_DOWN_HADAMARD_SEED>(
                s.gate_out, down_activations, rows, stream);
        } else if (s.ffn_hadamard) {
            promptforge_iu4::launch_hadamard_swiglu_pack<PF_I, PF_DOWN_HADAMARD_SEED>(
                s.gate_out, down_activations, rows, stream);
        } else {
            promptforge_iu4::launch_swiglu_pack<PF_I>(s.gate_out, down_activations, rows, stream);
        }
        if (s.ffn_correction_enabled) {
            promptforge_iu4::launch_indexed_swiglu_pack<PF_I, PF_DOWN_KEEPER_K>(
                s.gate_out,
                reinterpret_cast<const uint16_t *>(
                    s.ffn_correction_device_file + s.down_correction_index[layer]),
                s.down_correction_a8, s.down_correction_a_scale, rows, stream);
        }
        return;
    }
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
    hipLaunchKernelGGL(bf16_to_f32, dim3((count / 4 + 255) / 256), dim3(256), 0, stream,
                       s.down_out, output, count);
}

void run_down_fused(PFState & s, int layer, int rows, float * output, hipStream_t stream) {
    if (s.mode == PFMode::iu4_ffn) {
        if (s.ffn_segmented) {
            const promptforge_iu4::packed_activations activations{
                s.down_a4, s.down_a_scale, s.down_a_zero};
            const promptforge_iu4::packed_matrix matrix{
                reinterpret_cast<const uint32_t *>(s.device_file + s.down_weight[layer]),
                reinterpret_cast<const float *>(s.device_file + s.down_scale[layer]),
                reinterpret_cast<const int32_t *>(s.device_file + s.down_sum[layer])};
            promptforge_iu4::launch_gemm<PF_I>(
                activations, matrix, s.down_out, rows, PF_H, stream);
        } else if (s.pending_down_smallm_bucket >= 0) {
            s.iu4_smallm_down_invoker->Run(
                s.iu4_smallm_routes[s.pending_down_smallm_bucket].down_args[layer].get(),
                ::StreamConfig{stream, false});
        } else if (iu4_decode_width_enabled(s, rows)) {
            s.iu4_decode_invoker->Run(
                s.iu4_down_decode_args[layer][rows].get(), ::StreamConfig{stream, false});
        } else if (rows == PF_M) {
            s.iu4_down_invoker->Run(s.iu4_down_args[layer].get(), ::StreamConfig{stream, false});
        } else if (rows == PF_CHECKPOINT_M) {
            s.iu4_down_padded_invoker->Run(
                s.iu4_down_checkpoint2044_args[layer].get(), ::StreamConfig{stream, false});
        } else {
            s.iu4_down_padded_invoker->Run(
                s.iu4_down_tail1476_args[layer].get(), ::StreamConfig{stream, false});
        }
    } else if (rows == PF_TAIL_M) {
        s.down_tail1476_invoker->Run(s.down_tail1476_args[layer].get(), ::StreamConfig{stream, false});
    } else if (rows == PF_CHECKPOINT_M) {
        s.down_checkpoint2044_invoker->Run(s.down_checkpoint2044_args[layer].get(), ::StreamConfig{stream, false});
    } else {
        s.down_invoker->Run(s.down_args[layer].get(), ::StreamConfig{stream, false});
    }
    if (s.mode == PFMode::iu4_ffn && s.ffn_correction_enabled) {
        promptforge_iu4::launch_i8_correction<PF_DOWN_KEEPER_K>(
            s.down_correction_a8, s.down_correction_a_scale,
            reinterpret_cast<const uint32_t *>(
                s.ffn_correction_device_file + s.down_correction_weight[layer]),
            reinterpret_cast<const float *>(
                s.ffn_correction_device_file + s.down_correction_scale[layer]),
            s.down_out, rows, PF_H, stream);
    }
    s.pending_down_input = nullptr;
    s.pending_down_layer = -1;
    s.pending_down_rows = 0;
    s.pending_down_execution_rows = 0;
    s.pending_down_smallm_bucket = -1;
    const size_t count = size_t(rows) * PF_H;
    hipLaunchKernelGGL(bf16_to_f32, dim3((count / 4 + 255) / 256), dim3(256), 0, stream,
                       s.down_out, output, count);
}

void write_gdn_slice(PFState & s, PFGDNKind kind, int rows, float * output, hipStream_t stream) {
    const int src_col = kind == PFGDNKind::qkv ? 0 : PF_GDN_QKV_N;
    const int dst_cols = kind == PFGDNKind::qkv ? PF_GDN_QKV_N : PF_GDN_Z_N;
    const size_t count = size_t(rows) * dst_cols;
    hipLaunchKernelGGL(bf16_slice_to_f32, dim3((count / 4 + 255) / 256), dim3(256), 0, stream,
                       s.gate_out, output, rows, PF_GDN_N, src_col, dst_cols);
}

void run_gdn_qkvz(PFState & s, int layer, int rows, const float * input, hipStream_t stream) {
    if (s.mode == PFMode::iu4_ffn && !s.gdn_w8) {
        const promptforge_iu4::packed_activations activations{
            s.gate_a4, s.gate_a_scale, s.gate_a_zero};
        if (s.gdn_hadamard) {
            promptforge_iu4::launch_hadamard_input_pack<PF_H, PF_GATE_HADAMARD_SEED>(
                input, activations, rows, stream);
        } else {
            promptforge_iu4::launch_input_pack<PF_H>(input, nullptr, activations, rows, stream);
        }
    } else {
        hipLaunchKernelGGL(dynamic_a8_pack, dim3(rows), dim3(256), 0, stream,
                           input, s.gate_a8, s.gate_a_scale, rows, PF_H);
    }
    if (smallm_gdn_iu4_route_enabled(s, rows)) {
        const PFRowPlan plan = smallm_row_plan(rows);
        s.iu4_smallm_gate_invoker->Run(
            s.iu4_smallm_routes[plan.bucket].gdn_args[layer].get(),
            ::StreamConfig{stream, false});
    } else if (rows == PF_TAIL_M) {
        s.gdn_tail1476_invoker->Run(s.gdn_tail1476_args[layer].get(), ::StreamConfig{stream, false});
    } else if (rows == PF_CHECKPOINT_M) {
        s.gdn_checkpoint2044_invoker->Run(s.gdn_checkpoint2044_args[layer].get(), ::StreamConfig{stream, false});
    } else {
        s.gdn_invoker->Run(s.gdn_args[layer].get(), ::StreamConfig{stream, false});
    }
}

void run_attention_qkv(PFState & s, int layer, int rows, const float * input, hipStream_t stream) {
    const promptforge_iu4::packed_activations activations{
        s.gate_a4, s.gate_a_scale, s.gate_a_zero};
    promptforge_iu4::launch_input_pack<PF_H>(input, nullptr, activations, rows, stream);
    if (rows == PF_TAIL_M) {
        s.attention_qkv_invoker->Run(
            s.attention_qkv_tail1476_args[layer].get(), ::StreamConfig{stream, false});
    } else if (rows == PF_CHECKPOINT_M) {
        s.attention_qkv_invoker->Run(
            s.attention_qkv_checkpoint2044_args[layer].get(), ::StreamConfig{stream, false});
    } else {
        s.attention_qkv_invoker->Run(
            s.attention_qkv_args[layer].get(), ::StreamConfig{stream, false});
    }
}

void write_attention_qkv_slice(PFState & s, PFAttentionKind kind, int rows,
                               float * output, hipStream_t stream) {
    const int src_col = kind == PFAttentionKind::q ? 0 :
                        kind == PFAttentionKind::k ? PF_ATTENTION_Q_N :
                                                    PF_ATTENTION_Q_N + PF_ATTENTION_K_N;
    const int dst_cols = kind == PFAttentionKind::q ? PF_ATTENTION_Q_N :
                                                      PF_ATTENTION_K_N;
    const size_t count = size_t(rows) * dst_cols;
    hipLaunchKernelGGL(bf16_slice_to_f32, dim3((count / 4 + 255) / 256), dim3(256), 0, stream,
                       s.gate_out, output, rows, PF_ATTENTION_QKV_N, src_col, dst_cols);
}

void run_attention_output(PFState & s, int layer, int rows, const float * input,
                          float * output, hipStream_t stream) {
    const promptforge_iu4::packed_activations activations{
        s.gate_a4, s.gate_a_scale, s.gate_a_zero};
    promptforge_iu4::launch_input_pack<PF_ATTENTION_VALUE_N>(
        input, nullptr, activations, rows, stream);
    if (rows == PF_TAIL_M) {
        s.attention_output_invoker->Run(
            s.attention_output_tail1476_args[layer].get(), ::StreamConfig{stream, false});
    } else if (rows == PF_CHECKPOINT_M) {
        s.attention_output_invoker->Run(
            s.attention_output_checkpoint2044_args[layer].get(), ::StreamConfig{stream, false});
    } else {
        s.attention_output_invoker->Run(
            s.attention_output_args[layer].get(), ::StreamConfig{stream, false});
    }
    const size_t count = size_t(rows) * PF_H;
    hipLaunchKernelGGL(bf16_to_f32, dim3((count / 4 + 255) / 256), dim3(256), 0, stream,
                       s.down_out, output, count);
}

void run_gdn_output(PFState & s, int layer, int rows, const float * input,
                    float * output, hipStream_t stream) {
    const promptforge_iu4::packed_activations activations{
        s.gate_a4, s.gate_a_scale, s.gate_a_zero};
    promptforge_iu4::launch_input_pack<PF_ATTENTION_VALUE_N>(
        input, nullptr, activations, rows, stream);
    if (smallm_gdn_output_iu4_route_enabled(s, rows)) {
        const PFRowPlan plan = smallm_row_plan(rows);
        s.iu4_smallm_down_invoker->Run(
            s.iu4_smallm_routes[plan.bucket].gdn_output_args[layer].get(),
            ::StreamConfig{stream, false});
    } else if (iu4_decode_width_enabled(s, rows)) {
        s.gdn_output_decode_invoker->Run(
            s.gdn_output_decode_args[layer][rows].get(), ::StreamConfig{stream, false});
    } else if (rows == PF_TAIL_M) {
        s.gdn_output_invoker->Run(
            s.gdn_output_tail1476_args[layer].get(), ::StreamConfig{stream, false});
    } else if (rows == PF_CHECKPOINT_M) {
        s.gdn_output_invoker->Run(
            s.gdn_output_checkpoint2044_args[layer].get(), ::StreamConfig{stream, false});
    } else {
        s.gdn_output_invoker->Run(
            s.gdn_output_args[layer].get(), ::StreamConfig{stream, false});
    }
    const size_t count = size_t(rows) * PF_H;
    hipLaunchKernelGGL(bf16_to_f32, dim3((count / 4 + 255) / 256), dim3(256), 0, stream,
                       s.down_out, output, count);
}

} // namespace

bool promptforge_backend_init(int device) {
    PFState & s = state();
    std::lock_guard<std::mutex> lock(s.init_mutex);
    if (s.initialized) {
        return s.ready &&
               ((s.mode == PFMode::disabled && !s.gdn_enabled && !s.attention_enabled &&
                 !s.gdn_output_enabled) ||
                s.device == device);
    }
    s.initialized = true;
    const char * ffn_correction_sidecar = std::getenv("PROMPTFORGE_IU4_CORRECTION_SIDECAR");
    s.ffn_correction_enabled = ffn_correction_sidecar && ffn_correction_sidecar[0];
    const char * ffn_hadamard = std::getenv("PROMPTFORGE_IU4_HADAMARD");
    s.ffn_hadamard = ffn_hadamard && std::strcmp(ffn_hadamard, "1") == 0;
    const char * ffn_segmented = std::getenv("PROMPTFORGE_IU4_SEGMENTED");
    s.ffn_segmented = ffn_segmented && std::strcmp(ffn_segmented, "1") == 0;
    if (s.ffn_segmented) s.ffn_hadamard = true;
    const char * ffn_keepers = std::getenv("PROMPTFORGE_FFN_KEEPERS");
    s.ffn_keep_late4 = ffn_keepers && std::strcmp(ffn_keepers, "late4") == 0;
    s.ffn_keep_late6 = ffn_keepers && std::strcmp(ffn_keepers, "late6") == 0;
    if (ffn_keepers && ffn_keepers[0] &&
        !s.ffn_keep_late4 && !s.ffn_keep_late6) {
        std::fprintf(stderr,
            "promptforge: PROMPTFORGE_FFN_KEEPERS must be late4 or late6\n");
        return false;
    }
    const char * gdn_sidecar = std::getenv("PROMPTFORGE_GDN_SIDECAR");
    s.gdn_enabled = gdn_sidecar && gdn_sidecar[0];
    const char * gdn_precision = std::getenv("PROMPTFORGE_GDN_PRECISION");
    s.gdn_w8 = gdn_precision && std::strcmp(gdn_precision, "w8") == 0;
    const char * gdn_hadamard = std::getenv("PROMPTFORGE_GDN_IU4_HADAMARD");
    s.gdn_hadamard = gdn_hadamard && std::strcmp(gdn_hadamard, "1") == 0;
    const char * gdn_keepers = std::getenv("PROMPTFORGE_GDN_KEEPERS");
    s.gdn_keep_late4 = gdn_keepers && std::strcmp(gdn_keepers, "late4") == 0;
    s.gdn_keep_early4 = gdn_keepers && std::strcmp(gdn_keepers, "early4") == 0;
    s.gdn_keep_early3 = gdn_keepers && std::strcmp(gdn_keepers, "early3") == 0;
    const char * attention_sidecar = std::getenv("PROMPTFORGE_ATTENTION_SIDECAR");
    s.attention_enabled = attention_sidecar && attention_sidecar[0];
    const char * gdn_output_sidecar = std::getenv("PROMPTFORGE_GDN_OUTPUT_SIDECAR");
    s.gdn_output_enabled = gdn_output_sidecar && gdn_output_sidecar[0];
    const char * gdn_output_keepers = std::getenv("PROMPTFORGE_GDN_OUTPUT_KEEPERS");
    s.gdn_output_keep_v3_lateq6 = gdn_output_keepers &&
        std::strcmp(gdn_output_keepers, "v3_lateq6") == 0;
    if (gdn_output_keepers && gdn_output_keepers[0] &&
        !s.gdn_output_keep_v3_lateq6) {
        std::fprintf(stderr,
            "promptforge: PROMPTFORGE_GDN_OUTPUT_KEEPERS must be v3_lateq6\n");
        return false;
    }
    const char * iu4_decode = std::getenv("PROMPTFORGE_ENABLE_IU4_DECODE_M2_M5");
    s.iu4_decode_enabled = iu4_decode && std::strcmp(iu4_decode, "1") == 0;
    const char * smallm_iu4 = std::getenv("PROMPTFORGE_ENABLE_SMALLM_IU4");
    if (smallm_iu4 && smallm_iu4[0] && std::strcmp(smallm_iu4, "0") != 0 &&
        std::strcmp(smallm_iu4, "1") != 0) {
        std::fprintf(stderr, "promptforge: PROMPTFORGE_ENABLE_SMALLM_IU4 must be 0 or 1\n");
        return false;
    }
    s.smallm_iu4_enabled = smallm_iu4 && std::strcmp(smallm_iu4, "1") == 0;
    const char * smallm_gdn_iu4 = std::getenv("PROMPTFORGE_ENABLE_SMALLM_GDN_IU4");
    if (smallm_gdn_iu4 && smallm_gdn_iu4[0] &&
        std::strcmp(smallm_gdn_iu4, "0") != 0 && std::strcmp(smallm_gdn_iu4, "1") != 0) {
        std::fprintf(stderr, "promptforge: PROMPTFORGE_ENABLE_SMALLM_GDN_IU4 must be 0 or 1\n");
        return false;
    }
    s.smallm_gdn_iu4_enabled =
        smallm_gdn_iu4 && std::strcmp(smallm_gdn_iu4, "1") == 0;
    const char * smallm_gdn_output_iu4 =
        std::getenv("PROMPTFORGE_ENABLE_SMALLM_GDN_OUTPUT_IU4");
    if (smallm_gdn_output_iu4 && smallm_gdn_output_iu4[0] &&
        std::strcmp(smallm_gdn_output_iu4, "0") != 0 &&
        std::strcmp(smallm_gdn_output_iu4, "1") != 0) {
        std::fprintf(stderr,
            "promptforge: PROMPTFORGE_ENABLE_SMALLM_GDN_OUTPUT_IU4 must be 0 or 1\n");
        return false;
    }
    s.smallm_gdn_output_iu4_enabled = s.smallm_gdn_iu4_enabled &&
        (!smallm_gdn_output_iu4 || std::strcmp(smallm_gdn_output_iu4, "0") != 0);
    const char * ngram_m65_iu4 = std::getenv("PROMPTFORGE_ENABLE_NGRAM_M65_IU4");
    if (ngram_m65_iu4 && ngram_m65_iu4[0] &&
        std::strcmp(ngram_m65_iu4, "0") != 0 && std::strcmp(ngram_m65_iu4, "1") != 0) {
        std::fprintf(stderr, "promptforge: PROMPTFORGE_ENABLE_NGRAM_M65_IU4 must be 0 or 1\n");
        return false;
    }
    s.ngram_m65_iu4_enabled = ngram_m65_iu4 && std::strcmp(ngram_m65_iu4, "1") == 0;
    const char * graph_opt = std::getenv("GGML_CUDA_GRAPH_OPT");
    if ((s.ffn_correction_enabled || s.ffn_hadamard || s.gdn_enabled ||
         s.attention_enabled || s.gdn_output_enabled) &&
        graph_opt && std::strcmp(graph_opt, "1") == 0) {
        std::fprintf(stderr, "promptforge: projection sidecars reject GGML_CUDA_GRAPH_OPT=1\n");
        return false;
    }
    const char * mode = std::getenv("PROMPTFORGE_MODE");
    if (!mode || !mode[0]) {
        s.mode = PFMode::disabled;
        if (!s.ffn_correction_enabled && !s.ffn_hadamard && !s.gdn_enabled && !s.attention_enabled &&
            !s.gdn_output_enabled && !s.smallm_iu4_enabled && !s.smallm_gdn_iu4_enabled &&
            !s.ngram_m65_iu4_enabled) {
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
    } else if (std::strcmp(mode, "iu4_ffn") == 0) {
        s.mode = PFMode::iu4_ffn;
    } else {
        std::fprintf(stderr, "promptforge: unsupported PROMPTFORGE_MODE\n");
        return false;
    }
    const char * sidecar = std::getenv("PROMPTFORGE_SIDECAR");
    if (s.mode != PFMode::disabled && (!sidecar || !sidecar[0])) {
        std::fprintf(stderr, "promptforge: PROMPTFORGE_SIDECAR is required\n");
        return false;
    }
    if (s.ffn_correction_enabled && s.mode != PFMode::iu4_ffn) {
        std::fprintf(stderr, "promptforge: FFN correction sidecar requires iu4_ffn mode\n");
        return false;
    }
    if (s.ffn_hadamard && s.mode != PFMode::iu4_ffn) {
        std::fprintf(stderr, "promptforge: FFN Hadamard transform requires iu4_ffn mode\n");
        return false;
    }
    if (s.ffn_hadamard && s.ffn_correction_enabled) {
        std::fprintf(stderr, "promptforge: FFN Hadamard and residual correction are separate candidates\n");
        return false;
    }
    if (s.gdn_hadamard &&
        (s.mode != PFMode::iu4_ffn || !s.gdn_enabled || s.gdn_w8)) {
        std::fprintf(stderr,
            "promptforge: GDN IU4 Hadamard requires native-IU4 GDN in iu4_ffn mode\n");
        return false;
    }
    if (s.smallm_iu4_enabled && !canonical_true_iu4_lead(s)) {
        std::fprintf(stderr,
            "promptforge: PROMPTFORGE_ENABLE_SMALLM_IU4=1 requires the nonsegmented "
            "Hadamard-IU4 FFN + Hadamard-IU4 GDN + native-IU4 GDN-output lead\n");
        return false;
    }
    if (s.smallm_gdn_iu4_enabled &&
        (!s.smallm_iu4_enabled || !canonical_true_iu4_lead(s))) {
        std::fprintf(stderr,
            "promptforge: PROMPTFORGE_ENABLE_SMALLM_GDN_IU4=1 requires the promoted "
            "small-M FFN route on the canonical true-IU4 lead\n");
        return false;
    }
    if (s.ngram_m65_iu4_enabled &&
        (!s.smallm_iu4_enabled || !s.smallm_gdn_iu4_enabled ||
         !canonical_true_iu4_lead(s))) {
        std::fprintf(stderr,
            "promptforge: PROMPTFORGE_ENABLE_NGRAM_M65_IU4=1 requires the promoted "
            "small-M FFN and recurrent QKVZ routes on the canonical true-IU4 lead\n");
        return false;
    }
    s.device = device;
    if (!hip_check(hipSetDevice(device), "hipSetDevice") ||
        (s.mode == PFMode::iu4_ffn ?
         (s.ffn_segmented ? !load_iu4_segmented_sidecar(s, sidecar) :
                            !load_iu4_sidecar(s, sidecar)) :
         s.mode != PFMode::disabled && !load_sidecar(s, sidecar)) ||
        (s.ffn_correction_enabled &&
         !load_ffn_correction_sidecar(s, ffn_correction_sidecar)) ||
        !allocate_scratch(s) ||
        (s.mode == PFMode::iu4_ffn ? (!s.ffn_segmented && !build_iu4_routes(s)) :
         s.mode != PFMode::disabled && !build_routes(s)) ||
        (s.gdn_enabled &&
         ((s.mode == PFMode::iu4_ffn && !s.gdn_w8 ? !load_gdn_iu4_sidecar(s, gdn_sidecar) :
                                       !load_gdn_sidecar(s, gdn_sidecar)) ||
          !build_gdn_routes(s))) ||
        (s.attention_enabled &&
         (s.mode != PFMode::iu4_ffn ||
          !load_attention_iu4_sidecar(s, attention_sidecar) ||
          !build_attention_routes(s))) ||
        (s.gdn_output_enabled &&
         (s.mode != PFMode::iu4_ffn ||
          !load_gdn_output_iu4_sidecar(s, gdn_output_sidecar) ||
          !build_gdn_output_routes(s)))) {
        return false;
    }
    if (s.ffn_segmented) {
        std::fprintf(stderr,
            "{\"record\":\"promptforge_init\",\"mode\":\"iu4_ffn\","
            "\"device_bytes\":%llu,\"wmma\":\"v_wmma_i32_16x16x16_iu4\","
            "\"layouts\":\"segment256_panel64\","
            "\"transform\":\"block_hadamard_1024\","
            "\"gate_segments\":20,\"down_segments\":68,"
            "\"rows\":[1476,2044,2048]}\n",
            (unsigned long long) PF_IU4_SEGMENTED_FILE_BYTES);
    }
    std::fprintf(stderr,
        "{\"record\":\"promptforge_smallm_gdn_iu4_init\",\"enabled\":%s,"
        "\"min_rows\":%d,\"max_rows\":%d,\"buckets\":[128,256,512],"
        "\"qkvz_enabled\":%s,\"output_enabled\":%s,"
        "\"scope\":\"recurrent_qkvz_and_output\",\"cache\":\"shared_lazy_bucket\"}\n",
        s.smallm_gdn_iu4_enabled ? "true" : "false", PF_SMALLM_MIN, PF_SMALLM_MAX,
        s.smallm_gdn_iu4_enabled ? "true" : "false",
        s.smallm_gdn_output_iu4_enabled ? "true" : "false");
    std::fprintf(stderr,
        "{\"record\":\"promptforge_ngram_m65_iu4_init\",\"enabled\":%s,"
        "\"physical_rows\":65,\"execution_rows\":128,"
        "\"ffn\":%s,\"gdn_qkvz\":%s,\"gdn_output\":false,"
        "\"attention\":false,\"m2_m5_unchanged\":true}\n",
        s.ngram_m65_iu4_enabled ? "true" : "false",
        s.ngram_m65_iu4_enabled ? "true" : "false",
        s.ngram_m65_iu4_enabled ? "true" : "false");
    std::fprintf(stderr,
        "{\"record\":\"promptforge_v3_keeper_init\","
        "\"ffn_late6\":%s,\"gdn_output_lateq6\":%s}\n",
        s.ffn_keep_late6 ? "true" : "false",
        s.gdn_output_keep_v3_lateq6 ? "true" : "false");
    std::fflush(stderr);
    s.ready = true;
    return true;
}

bool promptforge_try_attention_qkv(ggml_backend_cuda_context * ctx,
                                   const ggml_tensor * weight, const ggml_tensor * input,
                                   ggml_tensor * dst) {
    PFState & s = state();
    if (!s.ready || !s.attention_enabled || !ctx) {
        return false;
    }
    PFAttentionKind kind = PFAttentionKind::none;
    int output_cols = 0;
    uint8_t bit = 0;
    int layer = exact_layer(weight, "attn_q.weight");
    if (layer >= 0) {
        kind = PFAttentionKind::q;
        output_cols = PF_ATTENTION_Q_N;
        bit = 1;
    } else if ((layer = exact_layer(weight, "attn_k.weight")) >= 0) {
        kind = PFAttentionKind::k;
        output_cols = PF_ATTENTION_K_N;
        bit = 2;
    } else if ((layer = exact_layer(weight, "attn_v.weight")) >= 0) {
        kind = PFAttentionKind::v;
        output_cols = PF_ATTENTION_V_N;
        bit = 4;
    } else {
        return false;
    }
    if (layer % 4 != 3) {
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
            ++s.attention_fallback_count;
        }
        return false;
    }
    begin_request_if_needed(s, rows);
    if (s.pending_attention_mask == 0) {
        run_attention_qkv(s, layer, rows, static_cast<const float *>(input->data), ctx->stream());
        ++s.attention_qkv_pack_count;
        s.pending_attention_layer = layer;
        s.pending_attention_rows = rows;
        s.pending_attention_input = input->data;
    } else if (s.pending_attention_layer != layer || s.pending_attention_rows != rows ||
               s.pending_attention_input != input->data || (s.pending_attention_mask & bit) != 0) {
        std::fprintf(stderr,
            "promptforge: attention QKV set mismatch (pending layer %d rows %d mask %u, "
            "requested layer %d rows %d bit %u)\n",
            s.pending_attention_layer, s.pending_attention_rows, unsigned(s.pending_attention_mask),
            layer, rows, unsigned(bit));
        std::abort();
    }
    write_attention_qkv_slice(s, kind, rows, static_cast<float *>(dst->data), ctx->stream());
    if (kind == PFAttentionKind::q) {
        ++s.attention_q_write_count;
    } else if (kind == PFAttentionKind::k) {
        ++s.attention_k_write_count;
    } else {
        ++s.attention_v_write_count;
    }
    s.pending_attention_mask |= bit;
    if (s.pending_attention_mask == 7) {
        s.pending_attention_layer = -1;
        s.pending_attention_rows = 0;
        s.pending_attention_input = nullptr;
        s.pending_attention_mask = 0;
        ++s.attention_qkv_count;
    }
    return true;
}

bool promptforge_try_attention_output(ggml_backend_cuda_context * ctx,
                                      const ggml_tensor * weight, const ggml_tensor * input,
                                      ggml_tensor * dst) {
    PFState & s = state();
    if (!s.ready || !s.attention_enabled || !ctx) {
        return false;
    }
    const int layer = exact_layer(weight, "attn_output.weight");
    if (layer < 0 || layer % 4 != 3) {
        return false;
    }
    const int rows = exact_tensor(input, GGML_TYPE_F32, PF_ATTENTION_VALUE_N, PF_M) &&
                     exact_tensor(dst, GGML_TYPE_F32, PF_H, PF_M) ? PF_M :
                     exact_tensor(input, GGML_TYPE_F32, PF_ATTENTION_VALUE_N, PF_CHECKPOINT_M) &&
                     exact_tensor(dst, GGML_TYPE_F32, PF_H, PF_CHECKPOINT_M) ? PF_CHECKPOINT_M :
                     exact_tensor(input, GGML_TYPE_F32, PF_ATTENTION_VALUE_N, PF_TAIL_M) &&
                     exact_tensor(dst, GGML_TYPE_F32, PF_H, PF_TAIL_M) ? PF_TAIL_M : 0;
    if (rows == 0 || !exact_shape(weight, PF_ATTENTION_VALUE_N, PF_H)) {
        if (s.count_active) {
            ++s.attention_fallback_count;
        }
        return false;
    }
    if (s.pending_attention_mask != 0) {
        std::fprintf(stderr, "promptforge: incomplete attention QKV set before output layer %d\n", layer);
        std::abort();
    }
    if (s.pending_attention_mask != 0) {
        std::fprintf(stderr,
            "promptforge: incomplete attention QKV set before FFN layer %d (pending layer %d mask %u)\n",
            layer, s.pending_attention_layer, unsigned(s.pending_attention_mask));
        std::abort();
    }
    begin_request_if_needed(s, rows);
    run_attention_output(s, layer, rows, static_cast<const float *>(input->data),
                         static_cast<float *>(dst->data), ctx->stream());
    ++s.attention_output_count;
    ++s.attention_output_pack_count;
    return true;
}

bool promptforge_try_gdn_output(ggml_backend_cuda_context * ctx,
                                const ggml_tensor * weight, const ggml_tensor * input,
                                ggml_tensor * dst) {
    PFState & s = state();
    if (!s.ready || !s.gdn_output_enabled || !ctx) {
        return false;
    }
    const int layer = exact_layer(weight, "ssm_out.weight");
    if (layer < 0 || layer % 4 == 3) {
        return false;
    }
    if (s.gdn_output_keep_v3_lateq6 && v3_lateq6_gdn_output_layer(layer)) {
        if (s.count_active) {
            ++s.gdn_output_fallback_count;
        }
        return false;
    }
    const int dynamic_smallm_rows =
        exact_smallm_gdn_output_iu4_rows(s, input, PF_ATTENTION_VALUE_N);
    const int rows = exact_tensor(input, GGML_TYPE_F32, PF_ATTENTION_VALUE_N, PF_M) &&
                     exact_tensor(dst, GGML_TYPE_F32, PF_H, PF_M) ? PF_M :
                     exact_tensor(input, GGML_TYPE_F32, PF_ATTENTION_VALUE_N, PF_CHECKPOINT_M) &&
                     exact_tensor(dst, GGML_TYPE_F32, PF_H, PF_CHECKPOINT_M) ? PF_CHECKPOINT_M :
                     exact_tensor(input, GGML_TYPE_F32, PF_ATTENTION_VALUE_N, PF_TAIL_M) &&
                     exact_tensor(dst, GGML_TYPE_F32, PF_H, PF_TAIL_M) ? PF_TAIL_M :
                     dynamic_smallm_rows != 0 &&
                     exact_tensor(dst, GGML_TYPE_F32, PF_H, dynamic_smallm_rows) ? dynamic_smallm_rows :
                     exact_iu4_decode_rows(s, input, dst, PF_ATTENTION_VALUE_N, PF_H);
    if (rows == 0 || !exact_shape(weight, PF_ATTENTION_VALUE_N, PF_H)) {
        if (s.count_active) {
            ++s.gdn_output_fallback_count;
        }
        return false;
    }
    const bool use_smallm = dynamic_smallm_rows != 0 && rows == dynamic_smallm_rows;
    if (use_smallm && !ensure_smallm_gdn_iu4_cache(s, rows)) {
        emit_smallm_gdn_iu4_fallback(s, rows, "gdn_output", "cache_unavailable");
        return false;
    }
    begin_request_if_needed(s, rows);
    run_gdn_output(s, layer, rows, static_cast<const float *>(input->data),
                   static_cast<float *>(dst->data), ctx->stream());
    ++s.gdn_output_count;
    ++s.gdn_output_pack_count;
    if (use_smallm) {
        ++s.smallm_gdn_iu4_output_count;
    }
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
    const bool gdn_keeper =
        (s.gdn_keep_late4 && (layer == 58 || layer == 60 || layer == 61 || layer == 62)) ||
        (s.gdn_keep_early4 && (layer == 0 || layer == 1 || layer == 2 || layer == 4)) ||
        (s.gdn_keep_early3 && (layer == 0 || layer == 1 || layer == 2));
    if (s.mode == PFMode::iu4_ffn && gdn_keeper) {
        if (s.count_active && kind == PFGDNKind::qkv) {
            ++s.gdn_keeper_count;
        }
        return false;
    }

    const int dynamic_smallm_rows = exact_smallm_gdn_iu4_rows(s, input, PF_H);
    const int rows = exact_tensor(input, GGML_TYPE_F32, PF_H, PF_M) &&
                     exact_tensor(dst, GGML_TYPE_F32, output_cols, PF_M) ? PF_M :
                     exact_tensor(input, GGML_TYPE_F32, PF_H, PF_CHECKPOINT_M) &&
                     exact_tensor(dst, GGML_TYPE_F32, output_cols, PF_CHECKPOINT_M) ? PF_CHECKPOINT_M :
                     exact_tensor(input, GGML_TYPE_F32, PF_H, PF_TAIL_M) &&
                     exact_tensor(dst, GGML_TYPE_F32, output_cols, PF_TAIL_M) ? PF_TAIL_M :
                     dynamic_smallm_rows != 0 &&
                     exact_tensor(dst, GGML_TYPE_F32, output_cols, dynamic_smallm_rows) ? dynamic_smallm_rows : 0;
    if (rows == 0 || !exact_shape(weight, PF_H, output_cols)) {
        if (s.count_active) {
            ++s.gdn_fallback_count;
        }
        return false;
    }
    const bool use_smallm = dynamic_smallm_rows != 0 && rows == dynamic_smallm_rows;
    if (use_smallm && !ensure_smallm_gdn_iu4_cache(s, rows)) {
        emit_smallm_gdn_iu4_fallback(s, rows, "gdn_qkvz", "cache_unavailable");
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
    if (use_smallm) {
        ++s.smallm_gdn_iu4_qkvz_count;
    }
    return true;
}

bool promptforge_try_fuse_gate_up(ggml_backend_cuda_context * ctx,
                                  ggml_tensor * first, ggml_tensor * second, ggml_tensor * glu) {
    PFState & s = state();
    const bool fused = s.mode == PFMode::m2048_fused ||
                       s.mode == PFMode::m2048_fused_tail1476 ||
                       s.mode == PFMode::iu4_ffn;
    if (!s.ready || (s.mode != PFMode::m2048 && !fused) || !ctx || !glu ||
        ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0) {
        return false;
    }
    const int dynamic_smallm_rows = exact_smallm_iu4_rows(s, glu, PF_I);
    const int rows = exact_tensor(glu, GGML_TYPE_F32, PF_I, PF_M) ? PF_M :
                     (s.mode == PFMode::m2048_fused_tail1476 || s.mode == PFMode::iu4_ffn) &&
                     exact_tensor(glu, GGML_TYPE_F32, PF_I, PF_CHECKPOINT_M) ? PF_CHECKPOINT_M :
                     (s.mode == PFMode::m2048_fused_tail1476 || s.mode == PFMode::iu4_ffn) &&
                     exact_tensor(glu, GGML_TYPE_F32, PF_I, PF_TAIL_M) ? PF_TAIL_M :
                     dynamic_smallm_rows != 0 ? dynamic_smallm_rows :
                     exact_iu4_decode_rows(s, glu, PF_I);
    if (rows == 0) return false;
    const bool use_smallm = dynamic_smallm_rows != 0 && rows == dynamic_smallm_rows;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    int layer = exact_layer(first ? first->src[0] : nullptr, "ffn_gate.weight");
    if (layer >= 0 && exact_layer(second ? second->src[0] : nullptr, "ffn_up.weight") == layer) {
        gate = first;
        up = second;
    } else {
        layer = exact_layer(second ? second->src[0] : nullptr, "ffn_gate.weight");
        if (layer < 0 || exact_layer(first ? first->src[0] : nullptr, "ffn_up.weight") != layer) {
            if (use_smallm) emit_smallm_iu4_fallback(s, rows, "gate_up", "weight_pair");
            return false;
        }
        gate = second;
        up = first;
    }
    if (!gate || !up || gate->op != GGML_OP_MUL_MAT || up->op != GGML_OP_MUL_MAT ||
        gate->src[1] != up->src[1] || !exact_tensor(gate->src[1], GGML_TYPE_F32, PF_H, rows) ||
        !exact_tensor(gate, GGML_TYPE_F32, PF_I, rows) || !exact_tensor(up, GGML_TYPE_F32, PF_I, rows) ||
         !((glu->src[0] == gate && glu->src[1] == up) || (glu->src[0] == up && glu->src[1] == gate))) {
        if (use_smallm) emit_smallm_iu4_fallback(s, rows, "gate_up", "tensor_shape");
        return false;
    }
    if (use_smallm && !ensure_smallm_iu4_cache(s, rows)) {
        emit_smallm_iu4_fallback(s, rows, "gate_up", "cache_unavailable");
        return false;
    }
    const bool ffn_keeper = (s.ffn_keep_late4 && layer >= 60) ||
                            (s.ffn_keep_late6 && layer >= 58);
    if (s.mode == PFMode::iu4_ffn && ffn_keeper) {
        if (s.count_active) {
            ++s.ffn_keeper_count;
        }
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
        if (use_smallm) {
            const PFRowPlan plan = smallm_row_plan(rows);
            s.pending_down_execution_rows = plan.execution_rows;
            s.pending_down_smallm_bucket = plan.bucket;
        } else {
            s.pending_down_execution_rows = rows;
            s.pending_down_smallm_bucket = -1;
        }
    } else {
        run_gate(s, layer, static_cast<const float *>(gate->src[1]->data), static_cast<float *>(glu->data), ctx->stream());
    }
    if (s.count_active) {
        ++s.gate_count;
        if (use_smallm) {
            ++s.smallm_iu4_gate_count;
        }
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
    const bool fused = s.mode == PFMode::m2048_fused ||
                       s.mode == PFMode::m2048_fused_tail1476 ||
                       s.mode == PFMode::iu4_ffn;
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
    const int dynamic_smallm_rows = exact_smallm_iu4_rows(s, input, PF_I);
    const int rows = exact_tensor(input, GGML_TYPE_F32, PF_I, PF_M) &&
                     exact_tensor(dst, GGML_TYPE_F32, PF_H, PF_M) ? PF_M :
                     (s.mode == PFMode::m2048_fused_tail1476 || s.mode == PFMode::iu4_ffn) &&
                     exact_tensor(input, GGML_TYPE_F32, PF_I, PF_CHECKPOINT_M) &&
                     exact_tensor(dst, GGML_TYPE_F32, PF_H, PF_CHECKPOINT_M) ? PF_CHECKPOINT_M :
                     (s.mode == PFMode::m2048_fused_tail1476 || s.mode == PFMode::iu4_ffn) &&
                     exact_tensor(input, GGML_TYPE_F32, PF_I, PF_TAIL_M) &&
                     exact_tensor(dst, GGML_TYPE_F32, PF_H, PF_TAIL_M) ? PF_TAIL_M :
                     dynamic_smallm_rows != 0 &&
                     exact_tensor(dst, GGML_TYPE_F32, PF_H, dynamic_smallm_rows) ? dynamic_smallm_rows :
                     exact_iu4_decode_rows(s, input, dst, PF_I, PF_H);
    const bool exact_down = layer >= 0 && rows != 0;
    const bool ffn_keeper = (s.ffn_keep_late4 && layer >= 60) ||
                            (s.ffn_keep_late6 && layer >= 58);
    if (s.mode == PFMode::iu4_ffn && ffn_keeper) {
        if (s.count_active && layer == PF_LAYERS - 1) {
            emit_request_telemetry(s);
        }
        return false;
    }
    if (fused) {
        if (!s.pending_down_input) {
            return false;
        }
        if (!exact_down || input != s.pending_down_input || layer != s.pending_down_layer ||
            rows != s.pending_down_rows) {
            if (s.pending_down_smallm_bucket >= 0) {
                emit_smallm_iu4_fallback(s, s.pending_down_rows, "down", "handoff_mismatch");
            }
            std::fprintf(stderr,
                "promptforge: fused down handoff mismatch (pending layer %d, requested layer %d)\n",
                s.pending_down_layer, layer);
            std::abort();
        }
        if (s.pending_down_smallm_bucket >= 0) {
            const PFRowPlan plan = smallm_row_plan(rows);
            const PFIU4RowRouteCache & route = s.iu4_smallm_routes[s.pending_down_smallm_bucket];
            if (plan.bucket != s.pending_down_smallm_bucket ||
                plan.execution_rows != s.pending_down_execution_rows ||
                !route.ready || route.execution_rows != plan.execution_rows ||
                !route.gate_args[layer] || !route.down_args[layer]) {
                emit_smallm_iu4_fallback(s, rows, "down", "cache_changed_after_gate");
                std::abort();
            }
        }
        run_down_fused(s, layer, rows, static_cast<float *>(dst->data), ctx->stream());
    } else if (!exact_down) {
        return false;
    } else {
        run_down(s, layer, static_cast<const float *>(input->data), static_cast<float *>(dst->data), ctx->stream());
    }
    if (s.count_active) {
        ++s.down_count;
        if (smallm_iu4_route_enabled(s, rows)) {
            ++s.smallm_iu4_down_count;
        }
        if (s.mode == PFMode::m2048) {
            ++s.pack_count;
        }
        if (layer == PF_LAYERS - 1) {
            emit_request_telemetry(s);
        }
    }
    return true;
}
