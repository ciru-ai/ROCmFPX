#!/usr/bin/env bash
# Collect a support-safe Vulkan/RADV crash bundle after a Laguna test failure.

set -euo pipefail

OUTPUT_DIR="${1:-}"
SINCE="${SINCE:-15 minutes ago}"
CAPTURE_DEVCOREDUMP="${CAPTURE_DEVCOREDUMP:-0}"
RUN_METADATA="${RUN_METADATA:-not supplied}"

if [[ -z "$OUTPUT_DIR" || "$OUTPUT_DIR" != /* ]]; then
    echo "usage: $0 /absolute/output/directory" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"

{
    date --iso-8601=seconds
    uname -a
    printf 'run metadata: %s\n' "$RUN_METADATA"
    command -v nixos-version >/dev/null && nixos-version || true
    command -v lspci >/dev/null && lspci -nnk 2>&1 | sed -n '/VGA compatible controller/,+3p;/Display controller/,+3p' || true
    command -v modinfo >/dev/null && modinfo amdgpu | grep -E '^(filename|version|firmware):' || true
    printf '\nVulkan summary:\n'
    vulkaninfo --summary 2>&1 || true
} > "$OUTPUT_DIR/system.txt"

journalctl -k --since "$SINCE" --no-pager > "$OUTPUT_DIR/kernel.log" 2>&1 || true
systemctl --user status qwen-main.service --no-pager \
    > "$OUTPUT_DIR/qwen-main-service.txt" 2>&1 || true
journalctl --user -u qwen-main.service --since "$SINCE" --no-pager \
    > "$OUTPUT_DIR/qwen-main-journal.log" 2>&1 || true
pgrep -a llama-server > "$OUTPUT_DIR/llama-processes.txt" 2>&1 || true

: > "$OUTPUT_DIR/devcoredump-paths.txt"
for coredump_path in /sys/class/drm/card[0-9]*/device/devcoredump/data; do
    if [[ -e "$coredump_path" ]]; then
        printf '%s\n' "$coredump_path" >> "$OUTPUT_DIR/devcoredump-paths.txt"
    fi
done

if [[ "$CAPTURE_DEVCOREDUMP" == "1" ]]; then
    coredump_index=0
    while IFS= read -r coredump_path; do
        if [[ -r "$coredump_path" ]]; then
            coredump_index=$((coredump_index + 1))
            cp -- "$coredump_path" "$OUTPUT_DIR/amdgpu-devcoredump-$coredump_index.txt"
        else
            printf 'not readable as uid %s: %s\n' "$(id -u)" "$coredump_path" \
                >> "$OUTPUT_DIR/devcoredump-unreadable.txt"
        fi
    done < "$OUTPUT_DIR/devcoredump-paths.txt"
fi

grep -En 'DeviceLost|context is lost|ring .* timeout|GPU reset|device wedged|Not enough memory|ttm_' \
    "$OUTPUT_DIR/kernel.log" > "$OUTPUT_DIR/kernel-relevant.log" || true

grep -En 'DeviceLost|context is lost|ErrorDeviceLost|terminate called|decode\(\) failed' \
    "$OUTPUT_DIR/qwen-main-journal.log" > "$OUTPUT_DIR/server-relevant.log" || true

echo "diagnostics written to $OUTPUT_DIR"
