#!/usr/bin/env bash
set -euo pipefail

readonly server="${LLAMA_SERVER:-./build-kairic/bin/llama-server}"
readonly model="${MODEL_PATH:?set MODEL_PATH to Qwen3.8-27B-IU4-Kairic-Edge.gguf}"
readonly ffn="${KAIRIC_FFN_SIDECAR:?set KAIRIC_FFN_SIDECAR}"
readonly gdn="${KAIRIC_GDN_SIDECAR:?set KAIRIC_GDN_SIDECAR}"
readonly gdn_output="${KAIRIC_GDN_OUTPUT_SIDECAR:?set KAIRIC_GDN_OUTPUT_SIDECAR}"
readonly rocm="${ROCM_PATH:-/opt/rocm}"

readonly host="${HOST:-127.0.0.1}"
readonly port="${PORT:-8080}"
readonly context="${CONTEXT:-262144}"
readonly cache_ram="${CACHE_RAM:-8192}"
readonly alias_name="${MODEL_ALIAS:-main}"

for required in "$server" "$model" "$ffn" "$gdn" "$gdn_output"; do
  [[ -r "$required" ]] || {
    echo "missing or unreadable required artifact: $required" >&2
    exit 1
  }
done

[[ "$port" =~ ^[0-9]+$ ]] || {
  echo "PORT must be numeric: $port" >&2
  exit 2
}

readonly server_dir="$(cd -- "$(dirname -- "$server")" && pwd)"
readonly library_path="$server_dir:$rocm/lib:$rocm/lib/rocm_sysdeps/lib:$rocm/lib/llvm/lib:$rocm/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec /usr/bin/env \
  ROCM_PATH="$rocm" \
  HIP_VISIBLE_DEVICES=0 \
  HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  LD_LIBRARY_PATH="$library_path" \
  GGML_CUDA_GRAPH_OPT=0 \
  LLAMA_TARGET_GREEDY_ARGMAX_FASTPATH=1 \
  LLAMA_MTP_CPU_ARGMAX_FASTPATH=1 \
  PROMPTFORGE_TARGET_ONLY=0 \
  PROMPTFORGE_MODE=iu4_ffn \
  PROMPTFORGE_IU4_SIDECAR="$ffn" \
  PROMPTFORGE_SIDECAR="$ffn" \
  PROMPTFORGE_IU4_HADAMARD=1 \
  PROMPTFORGE_IU4_SEGMENTED=0 \
  PROMPTFORGE_ENABLE_FFN_KEEPERS=0 \
  PROMPTFORGE_FFN_KEEPERS=late6 \
  PROMPTFORGE_ENABLE_GDN=1 \
  PROMPTFORGE_ENABLE_GDN_W8=0 \
  PROMPTFORGE_GDN_SIDECAR_OVERRIDE="$gdn" \
  PROMPTFORGE_GDN_SIDECAR="$gdn" \
  PROMPTFORGE_GDN_IU4_HADAMARD=1 \
  PROMPTFORGE_ENABLE_GDN_KEEPERS=0 \
  PROMPTFORGE_ENABLE_GDN_OUTPUT=1 \
  PROMPTFORGE_GDN_OUTPUT_SIDECAR="$gdn_output" \
  PROMPTFORGE_GDN_OUTPUT_KEEPERS=v3_lateq6 \
  PROMPTFORGE_ENABLE_SMALLM_IU4=1 \
  PROMPTFORGE_ENABLE_SMALLM_GDN_IU4=1 \
  PROMPTFORGE_ENABLE_SMALLM_GDN_OUTPUT_IU4=0 \
  PROMPTFORGE_ENABLE_IU4_DECODE_M2_M5=0 \
  PROMPTFORGE_OUTPUT_K8_STRICT_GREEDY=0 \
  "$server" -m "$model" --alias "$alias_name" \
    --host "$host" --port "$port" --jinja -dev ROCm0 -ngl 999 \
    -c "$context" -b 2048 -ub 512 -fa on -ctk f16 -ctv f16 \
    -t 16 -tb 32 -np 1 -ctxcp 32 --cache-ram "$cache_ram" \
    --cache-prompt --cache-idle-slots --metrics \
    --kairic-edge \
    --spec-type draft-mtp \
    --spec-draft-device ROCm0 --spec-draft-ngl 999 \
    --spec-draft-type-k f16 --spec-draft-type-v f16 \
    --spec-draft-n-max 4 --spec-draft-n-min 0 \
    --spec-draft-p-min 0.0 --spec-draft-p-split 0.10 \
    --spec-draft-backend-sampling \
    --temp 0 --top-p 1.0 --top-k 0 --min-p 0.0 \
    --reasoning off --reasoning-format none --reasoning-budget -1
