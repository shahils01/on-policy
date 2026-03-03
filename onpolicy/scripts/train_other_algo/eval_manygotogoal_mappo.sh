#!/bin/sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}" || exit 1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

env="mujoco"
scenario="ManyAgentGoToGoalEnv-v0"
num_agents=5
agent_obsk=0
faulty_node=-1
eval_faulty_node=-1
algo="mappo"
exp="mlp"
seed=1

CUDA_VISIBLE_DEVICES=0 python -m onpolicy.scripts.train.eval_manygotogoal \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --scenario ${scenario} \
    --num_agents ${num_agents} \
    --agent_obsk ${agent_obsk} \
    --seed ${seed} \
    --faulty_node ${faulty_node} \
    --eval_faulty_node ${eval_faulty_node} \
    --eval_loops 100 \
    --eval_episodes 100 \
    --n_training_threads 32 \
    --n_rollout_threads 1 \
    --n_eval_rollout_threads 1 \
    --episode_length 200 \
    --num_env_steps 200000000 \
    --use_eval \
    --add_center_xy \
    --use_state_agent \
    --share_policy \
    --model_dir "CHANGE_ME_MODEL_DIR" \
    --use_wandb \
    --wandb_name "xxx" \
    --user_name "shahil-shaik7-clemson-university"
