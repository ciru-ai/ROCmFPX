#!/usr/bin/env bash
# Supervise the Laguna Vulkan backend. systemd owns bounded restart/backoff;
# this wrapper detects DeviceLost, preserves diagnostics, and exits nonzero.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${RUNNER:-$ROOT/scripts/run-laguna-s21-rocmfp4-v4.sh}"
DEFAULT_STATE_ROOT="${XDG_STATE_HOME:-${HOME:-$ROOT}/.local/state}"
DIAGNOSTIC_ROOT="${DIAGNOSTIC_ROOT:-$DEFAULT_STATE_ROOT/rocmfpx/laguna-vulkan-failures}"
PREFLIGHT_TIMEOUT="${PREFLIGHT_TIMEOUT:-20}"
DEVICE_LOST_GRACE_SECONDS="${DEVICE_LOST_GRACE_SECONDS:-3}"
RESTART_WINDOW_SECONDS="${RESTART_WINDOW_SECONDS:-600}"
RESTART_BACKOFF_BASE_SECONDS="${RESTART_BACKOFF_BASE_SECONDS:-15}"
RESTART_BACKOFF_MAX_SECONDS="${RESTART_BACKOFF_MAX_SECONDS:-240}"
RESTART_LIMIT="${RESTART_LIMIT:-5}"
restart_state="$DIAGNOSTIC_ROOT/.restart-state"

if [[ "$#" -ne 1 && "$#" -ne 6 ]]; then
    echo "usage: $0 MODEL" >&2
    echo "   or: $0 MODEL ALIAS HOST PORT CTX BIN" >&2
    exit 2
fi

if [[ ! -x "$RUNNER" ]]; then
    echo "Laguna runner is not executable: $RUNNER" >&2
    exit 2
fi

mkdir -p "$DIAGNOSTIC_ROOT"

for numeric_setting in RESTART_WINDOW_SECONDS RESTART_BACKOFF_BASE_SECONDS \
    RESTART_BACKOFF_MAX_SECONDS RESTART_LIMIT; do
    numeric_value="${!numeric_setting}"
    if [[ ! "$numeric_value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$numeric_setting must be a positive integer, got: $numeric_value" >&2
        exit 2
    fi
done

now_epoch="$(date +%s)"
previous_failure_epoch=0
failure_count=0
if [[ -r "$restart_state" ]]; then
    read -r previous_failure_epoch failure_count < "$restart_state" || true
fi
if [[ ! "$previous_failure_epoch" =~ ^[0-9]+$ ||
      ! "$failure_count" =~ ^[0-9]+$ ||
      "$previous_failure_epoch" -gt "$now_epoch" ||
      "$((now_epoch - previous_failure_epoch))" -gt "$RESTART_WINDOW_SECONDS" ]]; then
    previous_failure_epoch=0
    failure_count=0
fi

if (( failure_count >= RESTART_LIMIT )); then
    echo "refusing Vulkan restart: $failure_count DeviceLost failures occurred within ${RESTART_WINDOW_SECONDS}s" >&2
    exit 75
fi

if (( failure_count > 0 )); then
    backoff_seconds="$RESTART_BACKOFF_BASE_SECONDS"
    for ((i = 1; i < failure_count; i++)); do
        if (( backoff_seconds >= RESTART_BACKOFF_MAX_SECONDS / 2 )); then
            backoff_seconds="$RESTART_BACKOFF_MAX_SECONDS"
            break
        fi
        backoff_seconds=$((backoff_seconds * 2))
    done
    (( backoff_seconds > RESTART_BACKOFF_MAX_SECONDS )) &&
        backoff_seconds="$RESTART_BACKOFF_MAX_SECONDS"
    echo "waiting ${backoff_seconds}s before Vulkan restart after $failure_count recent DeviceLost failure(s)" >&2
    sleep "$backoff_seconds"
fi

run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="$DIAGNOSTIC_ROOT/$run_stamp"
mkdir -p "$run_dir"
run_log="$run_dir/backend.log"
device_lost_marker="$run_dir/device-lost"
start_time="$(date --iso-8601=seconds)"

if ps -eo stat=,comm= | awk '$1 ~ /^D/ && $2 ~ /^llama-/ { found=1 } END { exit !found }'; then
    echo "refusing Vulkan restart: a llama process is stuck in D state" >&2
    exit 75
fi

if ! timeout "$PREFLIGHT_TIMEOUT" vulkaninfo --summary > "$run_dir/vulkan-preflight.txt" 2>&1; then
    echo "refusing Vulkan restart: vulkaninfo preflight failed" >&2
    exit 75
fi

kernel_recent="$(journalctl -k --since '10 minutes ago' --no-pager 2>/dev/null || true)"
timeout_line="$(printf '%s\n' "$kernel_recent" | grep -En 'ring comp_.* timeout|GPU reset begin|device wedged' | tail -1 | cut -d: -f1 || true)"
recovery_line="$(printf '%s\n' "$kernel_recent" | grep -En 'ring comp_.* reset succeeded|GPU reset succeeded' | tail -1 | cut -d: -f1 || true)"
if [[ -n "$timeout_line" && ( -z "$recovery_line" || "$recovery_line" -le "$timeout_line" ) ]]; then
    echo "refusing Vulkan restart: the latest compute-ring failure has no later recovery record" >&2
    exit 75
fi

"$RUNNER" "$@" > >(tee -a "$run_log") 2>&1 &
backend_pid=$!

forward_signal() {
    kill -TERM "$backend_pid" 2>/dev/null || true
}
trap forward_signal INT TERM HUP

(
    tail --pid="$backend_pid" -n 0 -F "$run_log" 2>/dev/null |
        while IFS= read -r line; do
            if [[ "$line" =~ DeviceLost|ErrorDeviceLost|context\ is\ lost ]]; then
                : > "$device_lost_marker"
                sleep "$DEVICE_LOST_GRACE_SECONDS"
                kill -TERM "$backend_pid" 2>/dev/null || true
                break
            fi
        done
) &
watcher_pid=$!

backend_status=0
wait "$backend_pid" || backend_status=$?
kill "$watcher_pid" 2>/dev/null || true
wait "$watcher_pid" 2>/dev/null || true

if [[ -e "$device_lost_marker" ]] ||
   grep -Eq 'DeviceLost|ErrorDeviceLost|context is lost' "$run_log"; then
    : > "$device_lost_marker"
    printf '%s %s\n' "$(date +%s)" "$((failure_count + 1))" > "$restart_state"
    RUN_METADATA="Laguna supervised backend DeviceLost; exit=$backend_status; argv=$*" \
    SINCE="$start_time" \
    CAPTURE_DEVCOREDUMP=1 \
        "$ROOT/scripts/collect-laguna-vulkan-diagnostics.sh" "$run_dir/diagnostics" || true
    echo "Laguna Vulkan device loss captured in $run_dir; requesting bounded service restart" >&2
    exit 70
fi

rm -f "$restart_state"
exit "$backend_status"
