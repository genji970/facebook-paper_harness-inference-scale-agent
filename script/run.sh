#!/usr/bin/env bash
set -e

# Always run from project root.
cd "$(dirname "$0")/.."

# Load .env if it exists.
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

mkdir -p outputs
mkdir -p logs
mkdir -p data_download/cache
mkdir -p evaluation_results

SWEBENCH_ARGS=()

if [ "${SWEBENCH_ENABLED:-false}" = "true" ]; then
  SWEBENCH_ARGS+=(--swebench_enabled)
fi

if [ "${SWEBENCH_CLEAN:-false}" = "true" ]; then
  SWEBENCH_ARGS+=(--swebench_clean)
fi

if [ "${SWEBENCH_MODAL:-false}" = "true" ]; then
  SWEBENCH_ARGS+=(--swebench_modal)
fi

if [ "${SWEBENCH_DRY_RUN:-false}" = "true" ]; then
  SWEBENCH_ARGS+=(--swebench_dry_run)
fi

if [ "${SWEBENCH_PREPARE_WORKSPACE:-false}" = "true" ]; then
  SWEBENCH_ARGS+=(--swebench_prepare_workspace)
fi

if [ "${SWEBENCH_FORCE_RECREATE_WORKSPACE:-false}" = "true" ]; then
  SWEBENCH_ARGS+=(--swebench_force_recreate_workspace)
fi

if [ "${SWEBENCH_KEEP_WORKSPACES:-false}" = "true" ]; then
  SWEBENCH_ARGS+=(--swebench_keep_workspaces)
fi

python experiment/run.py \
  --model_name "${MODEL_NAME:-gpt-4o-mini}" \
  --provider "${PROVIDER:-gemini}" \
  --gemini_api_key "${GEMINI_API_KEY:-}" \
  --endpoint_url "${OPENAI_BASE_URL:-https://api.openai.com/v1}" \
  --api_key "${OPENAI_API_KEY:-}" \
  --dataset_name "${DATASET_NAME:-princeton-nlp/SWE-bench_Verified}" \
  --dataset_config "${DATASET_CONFIG:-}" \
  --train_split "${TRAIN_SPLIT:-test}" \
  --data_cache_dir "${DATA_CACHE_DIR:-data_download/cache}" \
  --problem_field "${PROBLEM_FIELD:-problem_statement}" \
  --instance_id_field "${INSTANCE_ID_FIELD:-instance_id}" \
  --repo_field "${REPO_FIELD:-repo}" \
  --base_commit_field "${BASE_COMMIT_FIELD:-base_commit}" \
  --max_dataset_samples "${MAX_DATASET_SAMPLES:-1}" \
  --start_sample_index "${START_SAMPLE_INDEX:-0}" \
  --end_sample_index "${END_SAMPLE_INDEX:--1}" \
  --num_rollouts "${NUM_ROLLOUTS:-2}" \
  --top_k "${TOP_K:-1}" \
  --group_size "${GROUP_SIZE:-2}" \
  --vote_count "${VOTE_COUNT:-1}" \
  --num_iterations "${NUM_ITERATIONS:-2}" \
  --max_steps_per_rollout "${MAX_STEPS_PER_ROLLOUT:-50}" \
  --agent_max_steps "${AGENT_MAX_STEPS:-1}" \
  --executor_cwd "${EXECUTOR_CWD:-.}" \
  --executor_timeout "${EXECUTOR_TIMEOUT:-30}" \
  --executor_max_output_chars "${EXECUTOR_MAX_OUTPUT_CHARS:-12000}" \
  --executor_shell "${EXECUTOR_SHELL:-/bin/bash}" \
  --pdr_max_summary_chars "${PDR_MAX_SUMMARY_CHARS:-4000}" \
  --pdr_max_context_chars "${PDR_MAX_CONTEXT_CHARS:-16000}" \
  --experiment_output_path "${EXPERIMENT_OUTPUT_PATH:-outputs/pdr_rtv_results.jsonl}" \
  --experiment_summary_path "${EXPERIMENT_SUMMARY_PATH:-outputs/pdr_rtv_summary.json}" \
  --pipeline_output_path "${PIPELINE_OUTPUT_PATH:-outputs/pipeline_result.json}" \
  --swebench_dataset_name "${SWEBENCH_DATASET_NAME:-princeton-nlp/SWE-bench_Verified}" \
  --swebench_split "${SWEBENCH_SPLIT:-test}" \
  --swebench_predictions_path "${SWEBENCH_PREDICTIONS_PATH:-outputs/swebench_predictions.jsonl}" \
  --swebench_run_id "${SWEBENCH_RUN_ID:-pdr_rtv_eval}" \
  --swebench_max_workers "${SWEBENCH_MAX_WORKERS:-1}" \
  --swebench_cache_level "${SWEBENCH_CACHE_LEVEL:-env}" \
  --swebench_repo_dir "${SWEBENCH_REPO_DIR:-}" \
  --swebench_results_dir "${SWEBENCH_RESULTS_DIR:-evaluation_results}" \
  --continue_on_error \
  --swebench_workspace_root "${SWEBENCH_WORKSPACE_ROOT:-data_download/swebench_workspaces}" \
  --swebench_git_timeout "${SWEBENCH_GIT_TIMEOUT:-600}" \
  "${SWEBENCH_ARGS[@]}"