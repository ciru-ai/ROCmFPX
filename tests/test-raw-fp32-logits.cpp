#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static void print_usage(int /* argc */, char ** argv) {
    LOG(
        "usage: %s -m MODEL -f PROMPT -n N --save-logits "
        "--logits-output-dir DIR [common llama options]\n"
        "\n"
        "Evaluate the first N prompt tokens with batch.logits[i] set for every "
        "row and save the contiguous [N, n_vocab] host-visible float32 output.\n"
        "N must be at least 16 so the diagnostic exercises an MMQ LM head.\n",
        argv[0]);
}

static std::string json_escape(const std::string & value) {
    std::string result;
    result.reserve(value.size() + 16);
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b";  break;
            case '\f': result += "\\f";  break;
            case '\n': result += "\\n";  break;
            case '\r': result += "\\r";  break;
            case '\t': result += "\\t";  break;
            default:
                if (ch < 0x20) {
                    constexpr char hex[] = "0123456789abcdef";
                    result += "\\u00";
                    result += hex[(ch >> 4) & 0x0f];
                    result += hex[ch & 0x0f];
                } else {
                    result += static_cast<char>(ch);
                }
                break;
        }
    }
    return result;
}

static void require_new_file(const std::filesystem::path & path) {
    if (std::filesystem::exists(path)) {
        throw std::runtime_error(
            "refusing to overwrite existing evidence file: " + path.string());
    }
}

static void write_binary(
        const std::filesystem::path & path,
        const void * data,
        const size_t size) {
    require_new_file(path);
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open " + path.string());
    }
    stream.write(static_cast<const char *>(data), size);
    if (!stream) {
        throw std::runtime_error("failed to write " + path.string());
    }
}

static void write_metadata(
        const std::filesystem::path & path,
        const common_params & params,
        const size_t n_prompt_tokens,
        const size_t n_rows,
        const int32_t n_vocab,
        const llama_context * ctx) {
    require_new_file(path);
    std::ofstream stream(path);
    if (!stream) {
        throw std::runtime_error("failed to open " + path.string());
    }
    stream
        << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"capture\": \"all requested token logits\",\n"
        << "  \"source_api\": \"llama_get_logits\",\n"
        << "  \"batch_logits_all_true\": true,\n"
        << "  \"dtype\": \"float32\",\n"
        << "  \"endianness\": \"little\",\n"
        << "  \"layout\": \"row-major [prompt_token, vocabulary]\",\n"
        << "  \"shape\": [" << n_rows << ", " << n_vocab << "],\n"
        << "  \"n_elements\": " << n_rows * static_cast<size_t>(n_vocab) << ",\n"
        << "  \"n_bytes\": "
        << n_rows * static_cast<size_t>(n_vocab) * sizeof(float) << ",\n"
        << "  \"n_prompt_tokens_available\": " << n_prompt_tokens << ",\n"
        << "  \"n_evaluated_tokens\": " << n_rows << ",\n"
        << "  \"semantic_rows\": \"prompt token positions 0 through "
        << n_rows - 1 << "\",\n"
        << "  \"n_ctx\": " << llama_n_ctx(ctx) << ",\n"
        << "  \"n_batch\": " << llama_n_batch(ctx) << ",\n"
        << "  \"n_ubatch\": " << llama_n_ubatch(ctx) << ",\n"
        << "  \"model_path\": \"" << json_escape(params.model.path) << "\",\n"
        << "  \"prompt_file\": \"" << json_escape(params.prompt_file) << "\",\n"
        << "  \"logits_file\": \"raw-fp32-all-logits.bin\",\n"
        << "  \"tokens_file\": \"raw-fp32-tokens.bin\"\n"
        << "}\n";
    if (!stream) {
        throw std::runtime_error("failed to write " + path.string());
    }
}

static int run(common_params & params) {
    static_assert(sizeof(float) == 4, "diagnostic requires 32-bit float");
    static_assert(sizeof(llama_token) == 4, "diagnostic requires 32-bit tokens");
    if (!std::numeric_limits<float>::is_iec559) {
        throw std::runtime_error("diagnostic requires IEEE-754 float32");
    }
    const uint32_t endian_probe = 1;
    if (*reinterpret_cast<const uint8_t *>(&endian_probe) != 1) {
        throw std::runtime_error("diagnostic currently requires little endian");
    }
    if (!params.save_logits) {
        throw std::runtime_error("--save-logits is required");
    }
    if (params.n_predict < 16) {
        throw std::runtime_error("-n/--predict must be at least 16");
    }
    if (params.embedding) {
        throw std::runtime_error("embedding mode is not supported");
    }

    auto llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        throw std::runtime_error("failed to initialize model/context");
    }
    if (llama_model_has_encoder(model) || !llama_model_has_decoder(model)) {
        throw std::runtime_error("diagnostic requires a decoder-only model");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens = common_tokenize(
        ctx,
        params.prompt,
        llama_vocab_get_add_bos(vocab));
    const size_t n_prompt_tokens = tokens.size();
    const size_t n_rows = static_cast<size_t>(params.n_predict);
    if (tokens.size() < n_rows) {
        throw std::runtime_error(
            "prompt tokenized to fewer tokens than requested by -n");
    }
    if (n_rows > llama_n_ctx(ctx) || n_rows > llama_n_batch(ctx)) {
        throw std::runtime_error(
            "requested rows exceed the effective context or batch size");
    }
    tokens.resize(n_rows);

    llama_batch batch = llama_batch_init(static_cast<int32_t>(n_rows), 0, 1);
    batch.n_tokens = static_cast<int32_t>(n_rows);
    for (size_t i = 0; i < n_rows; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }

    const int decode_result = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (decode_result != 0) {
        throw std::runtime_error(
            "llama_decode failed with status " +
            std::to_string(decode_result));
    }

    llama_synchronize(ctx);
    const float * logits = llama_get_logits(ctx);
    if (logits == nullptr) {
        throw std::runtime_error("llama_get_logits returned null");
    }
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
        throw std::runtime_error("model reported an invalid vocabulary size");
    }
    if (n_rows > std::numeric_limits<size_t>::max() /
            static_cast<size_t>(n_vocab) / sizeof(float)) {
        throw std::runtime_error("raw logits byte count overflow");
    }

    const std::filesystem::path output_dir(params.logits_output_dir);
    std::filesystem::create_directories(output_dir);
    if (!std::filesystem::is_directory(output_dir)) {
        throw std::runtime_error(
            "logits output path is not a directory: " + output_dir.string());
    }
    const size_t logits_bytes =
        n_rows * static_cast<size_t>(n_vocab) * sizeof(float);
    write_binary(
        output_dir / "raw-fp32-all-logits.bin",
        logits,
        logits_bytes);
    write_binary(
        output_dir / "raw-fp32-tokens.bin",
        tokens.data(),
        tokens.size() * sizeof(tokens[0]));
    write_metadata(
        output_dir / "metadata.json",
        params,
        n_prompt_tokens,
        n_rows,
        n_vocab,
        ctx);

    LOG(
        "raw_fp32_all_logits: rows=%zu cols=%d elements=%zu bytes=%zu\n",
        n_rows,
        n_vocab,
        n_rows * static_cast<size_t>(n_vocab),
        logits_bytes);
    llama_perf_context_print(ctx);
    return 0;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(
            argc,
            argv,
            params,
            LLAMA_EXAMPLE_DEBUG,
            print_usage)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);
    try {
        const int result = run(params);
        llama_backend_free();
        return result;
    } catch (const std::exception & error) {
        LOG_ERR("raw FP32 logits diagnostic failed: %s\n", error.what());
        llama_backend_free();
        return 1;
    }
}
