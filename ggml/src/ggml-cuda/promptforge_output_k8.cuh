#pragma once

struct ggml_backend_cuda_context;
struct ggml_tensor;

// Experimental target-only strict-greedy output route. It intentionally emits
// synthetic logits (-inf outside the exact Q8 rerank set), so callers must not
// use it for stochastic sampling, logprobs, penalties, or speculative decoding.
bool promptforge_output_k8_backend_init(int device);
bool promptforge_try_output_k8_strict_greedy(
    ggml_backend_cuda_context * ctx,
    const ggml_tensor * weight,
    const ggml_tensor * input,
    ggml_tensor * dst);
