#!/usr/bin/env bash
set -euo pipefail

# 单节点 4 GPU 的 Pipeline 验证入口。30B 正式训练需要根据显存实测调整并行度。
: "${ECHO_ROOT:?set ECHO_ROOT to the local ECHO source tree}"
: "${MODEL_PATH:?set MODEL_PATH}"
: "${TRAIN_FILE:?set TRAIN_FILE to train.esr.parquet}"
: "${VAL_FILE:?set VAL_FILE to validation.esr.parquet}"
: "${ESR_RETRIEVAL_URL:?set ESR_RETRIEVAL_URL, for example http://127.0.0.1:8000}"
: "${ESR_VERIFIER_BASE_URL:?set ESR_VERIFIER_BASE_URL}"
: "${ESR_VERIFIER_MODEL:?set ESR_VERIFIER_MODEL}"

ESR_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TOOL_CONFIG="${ESR_ROOT}/configs/echo/esr_tools.yaml"
REWARD_FILE="${ESR_ROOT}/src/esr_grpo/integrations/reward.py"
export PYTHONPATH="${ESR_ROOT}/src:${ECHO_ROOT}:${PYTHONPATH:-}"
export ESR_STORE_DIR="${ESR_STORE_DIR:-${ESR_ROOT}/outputs/exp3/stores}"

python "${ESR_ROOT}/scripts/common/check_echo_compat.py" --echo-root "${ECHO_ROOT}"
if [[ "${ESR_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  python "${ESR_ROOT}/scripts/experiment3_training/preflight_training.py" \
    --echo-root "${ECHO_ROOT}" \
    --model-path "${MODEL_PATH}" \
    --train-file "${TRAIN_FILE}" \
    --val-file "${VAL_FILE}" \
    --tool-config "${TOOL_CONFIG}" \
    --output-dir "${CHECKPOINT_DIR:-${ESR_ROOT}/outputs/exp3/checkpoints}" \
    --required-gpus 4 \
    --strict
fi

python -m esr_grpo.integrations.echo_entrypoint \
  --config-path="${ECHO_ROOT}/examples/sglang_multiturn/config" \
  --config-name=bcp_multiturn_megatron_grpo \
  algorithm.adv_estimator=esr_grpo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-8}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-4096}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-16384}" \
  data.filter_overlong_prompts=False \
  data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.enable_thinking=True \
  +data.tool_config_path="${TOOL_CONFIG}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr="${LEARNING_RATE:-1e-6}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-4}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${ACTOR_TP:-1}" \
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${ACTOR_PP:-1}" \
  actor_rollout_ref.actor.megatron.context_parallel_size="${ACTOR_CP:-1}" \
  actor_rollout_ref.actor.megatron.param_offload=True \
  actor_rollout_ref.actor.megatron.grad_offload=True \
  actor_rollout_ref.actor.megatron.optimizer_offload=True \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-1}" \
  actor_rollout_ref.rollout.n="${N_RESP:-8}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN:-24576}" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=5 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=16000 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
  +actor_rollout_ref.rollout.multi_turn.context_compression_method=truncate \
  +actor_rollout_ref.rollout.multi_turn.enable_summarization=True \
  +actor_rollout_ref.rollout.multi_turn.max_summary_rounds=5 \
  +actor_rollout_ref.rollout.multi_turn.working_context_length="${WORKING_CONTEXT_LENGTH:-16384}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${REF_TP:-1}" \
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="${REF_PP:-1}" \
  actor_rollout_ref.ref.megatron.context_parallel_size="${REF_CP:-1}" \
  actor_rollout_ref.ref.megatron.param_offload=True \
  algorithm.use_kl_in_reward=False \
  +reward.custom_reward_function.path="${REWARD_FILE}" \
  reward.custom_reward_function.name=compute_score \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.project_name=esr-grpo \
  trainer.experiment_name="${EXPERIMENT_NAME:-qwen4b-bcp-esr-pipeline}" \
  trainer.default_local_dir="${CHECKPOINT_DIR:-${ESR_ROOT}/outputs/exp3/checkpoints}" \
  trainer.logger="${TRAINER_LOGGER:-['console']}" \
  trainer.save_freq="${SAVE_FREQ:-50}" \
  trainer.test_freq="${TEST_FREQ:-10}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-50}"
