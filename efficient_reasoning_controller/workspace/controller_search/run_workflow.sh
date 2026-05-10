#!/usr/bin/env bash
set -euo pipefail

# directory location: the script is located in controller_search/, the project root is its parent directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ===== configuration area =====

WORKFLOW_WORKDIR_DEFAULT="${PROJECT_ROOT}"
WORKFLOW_METHOD_FILE_DEFAULT="code_base/method.py"
WORKFLOW_HISTORY_DIR_DEFAULT="code_base/history"
WORKFLOW_CODEX_LOG_PARENT_DEFAULT="${PROJECT_ROOT}/.workflow_logs"
# If you want to manually specify the result directory (e.g. matrix_results_Qwen3-8B-128), fill in the absolute path here; if left blank, use the proposer output_dir
WORKFLOW_RESULT_DIR_DEFAULT="${PROJECT_ROOT}/code_base/training_results"

# prompt configuration
WORKFLOW_PROMPT_PATH_DEFAULT="${PROJECT_ROOT}/controller_search/prompts/proposer_prompt_acc_and_cost.txt"
WORKFLOW_CRITIC_PROMPT_PATH_DEFAULT="${PROJECT_ROOT}/controller_search/prompts/critic_prompt.txt"

# resume: automatically continue the maximum number of rounds based on the directory name in code_base/history (export WORKFLOW_RESUME=1)

# rounds and evaluation
WORKFLOW_ROUNDS_DEFAULT="5"
WORKFLOW_EVAL_CMD_DEFAULT="python code_base/eval.py"
WORKFLOW_EVAL_CWD_DEFAULT="${PROJECT_ROOT}"
WORKFLOW_EVAL_TIMEOUT_SEC_DEFAULT="3600"

# Codex 
CODEX_BIN_DEFAULT="codex"
# proposer backend: codex or claude
WORKFLOW_PROPOSER_BACKEND_DEFAULT="claude"
# CODEX_MODEL_DEFAULT="gpt-5-codex"
# CODEX_EXEC_TIMEOUT_SEC_DEFAULT="1800"
# CODEX_EXTRA_ARGS_DEFAULT="--config /path/to/config.toml"

# ===== export (allow external export to override) =====
export WORKFLOW_RESUME=1
export WORKFLOW_WORKDIR="${WORKFLOW_WORKDIR:-${WORKFLOW_WORKDIR_DEFAULT}}"
export WORKFLOW_METHOD_FILE="${WORKFLOW_METHOD_FILE:-${WORKFLOW_METHOD_FILE_DEFAULT}}"
export WORKFLOW_HISTORY_DIR="${WORKFLOW_HISTORY_DIR:-${WORKFLOW_HISTORY_DIR_DEFAULT}}"
export WORKFLOW_CODEX_LOG_PARENT="${WORKFLOW_CODEX_LOG_PARENT:-${WORKFLOW_CODEX_LOG_PARENT_DEFAULT}}"
export WORKFLOW_RESULT_DIR="${WORKFLOW_RESULT_DIR:-${WORKFLOW_RESULT_DIR_DEFAULT}}"

export WORKFLOW_PROMPT_PATH="${WORKFLOW_PROMPT_PATH:-${WORKFLOW_PROMPT_PATH_DEFAULT}}"
export WORKFLOW_CRITIC_PROMPT_PATH="${WORKFLOW_CRITIC_PROMPT_PATH:-${WORKFLOW_CRITIC_PROMPT_PATH_DEFAULT}}"

export WORKFLOW_ROUNDS="${WORKFLOW_ROUNDS:-${WORKFLOW_ROUNDS_DEFAULT}}"
export WORKFLOW_EVAL_CMD="${WORKFLOW_EVAL_CMD:-${WORKFLOW_EVAL_CMD_DEFAULT}}"
export WORKFLOW_EVAL_CWD="${WORKFLOW_EVAL_CWD:-${WORKFLOW_EVAL_CWD_DEFAULT}}"
export WORKFLOW_EVAL_TIMEOUT_SEC="${WORKFLOW_EVAL_TIMEOUT_SEC:-${WORKFLOW_EVAL_TIMEOUT_SEC_DEFAULT}}"

export CODEX_BIN="${CODEX_BIN:-${CODEX_BIN_DEFAULT}}"
export WORKFLOW_PROPOSER_BACKEND="${WORKFLOW_PROPOSER_BACKEND:-${WORKFLOW_PROPOSER_BACKEND_DEFAULT}}"
# export CODEX_MODEL="${CODEX_MODEL:-${CODEX_MODEL_DEFAULT}}"
# export CODEX_EXEC_TIMEOUT_SEC="${CODEX_EXEC_TIMEOUT_SEC:-${CODEX_EXEC_TIMEOUT_SEC_DEFAULT}}"
# export CODEX_EXTRA_ARGS="${CODEX_EXTRA_ARGS:-${CODEX_EXTRA_ARGS_DEFAULT}}"

echo "[run_workflow] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[run_workflow] WORKFLOW_WORKDIR=${WORKFLOW_WORKDIR}"
echo "[run_workflow] WORKFLOW_METHOD_FILE=${WORKFLOW_METHOD_FILE}"
echo "[run_workflow] WORKFLOW_RESULT_DIR=${WORKFLOW_RESULT_DIR}"
echo "[run_workflow] WORKFLOW_PROMPT_PATH=${WORKFLOW_PROMPT_PATH}"
echo "[run_workflow] WORKFLOW_CRITIC_PROMPT_PATH=${WORKFLOW_CRITIC_PROMPT_PATH}"
echo "[run_workflow] WORKFLOW_PROPOSER_BACKEND=${WORKFLOW_PROPOSER_BACKEND}"
echo "[run_workflow] WORKFLOW_EVAL_CMD=${WORKFLOW_EVAL_CMD}"
echo "[run_workflow] WORKFLOW_ROUNDS=${WORKFLOW_ROUNDS}"

cd "${PROJECT_ROOT}"
python controller_search/workflow_propose_critic.py
