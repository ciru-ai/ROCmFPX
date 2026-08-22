#pragma once

struct ggml_backend_cuda_context;
struct ggml_tensor;

bool promptforge_backend_init(int device);
bool promptforge_try_gdn_qkvz(ggml_backend_cuda_context * ctx,
                              const ggml_tensor * weight, const ggml_tensor * input, ggml_tensor * dst);
bool promptforge_try_attention_qkv(ggml_backend_cuda_context * ctx,
                                   const ggml_tensor * weight, const ggml_tensor * input, ggml_tensor * dst);
bool promptforge_try_attention_output(ggml_backend_cuda_context * ctx,
                                      const ggml_tensor * weight, const ggml_tensor * input, ggml_tensor * dst);
bool promptforge_try_gdn_output(ggml_backend_cuda_context * ctx,
                                const ggml_tensor * weight, const ggml_tensor * input, ggml_tensor * dst);
bool promptforge_try_fuse_gate_up(ggml_backend_cuda_context * ctx,
                                  ggml_tensor * first, ggml_tensor * second, ggml_tensor * glu);
bool promptforge_try_down(ggml_backend_cuda_context * ctx,
                          const ggml_tensor * weight, const ggml_tensor * input, ggml_tensor * dst);
