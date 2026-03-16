#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 10 ]]; then
  echo "Usage: $0 <python_bin> <eval_script> <gpu_id> <start_idx> <max_users> <eval_run_root> <bundle_output> <global_db_path> <history_db_path> <extra_args_json>"
  exit 1
fi

PYTHON_BIN="$1"
EVAL_SCRIPT="$2"
GPU_ID="$3"
START_IDX="$4"
MAX_USERS="$5"
EVAL_RUN_ROOT="$6"
BUNDLE_OUTPUT="$7"
GLOBAL_DB_PATH="$8"
HISTORY_DB_PATH="$9"
EXTRA_ARGS_JSON="${10}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Parse JSON array into argv safely.
mapfile -t EXTRA_ARGS < <("$PYTHON_BIN" - <<'PY' "$EXTRA_ARGS_JSON"
import json, sys
for x in json.loads(sys.argv[1]):
    print(x)
PY
)

"$PYTHON_BIN" "$EVAL_SCRIPT" \
  --start-user-index "$START_IDX" \
  --max-users "$MAX_USERS" \
  --eval-run-root "$EVAL_RUN_ROOT" \
  --bundle-output "$BUNDLE_OUTPUT" \
  --global-db-path "$GLOBAL_DB_PATH" \
  --shared-history-db-path "$HISTORY_DB_PATH" \
  "${EXTRA_ARGS[@]}"
