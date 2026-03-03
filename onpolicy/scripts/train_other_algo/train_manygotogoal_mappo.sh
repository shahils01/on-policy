#!/bin/sh

# SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# cd "${REPO_ROOT}" || exit 1
# export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

env="mujoco"
scenario="ManyAgentGoToGoalEnv-v0"
num_agents=15
agent_obsk=0
algo="mappo"
exp="mlp"
running_max=5

user_name="shahil-shaik7-clemson-university"

for number in `seq ${running_max}`;
do
    echo "run ${number}/${running_max}"
    CUDA_VISIBLE_DEVICES=1 python ../train/train_manygotogoal.py \
        --env_name ${env} \
        --algorithm_name ${algo} \
        --experiment_name ${exp} \
        --scenario ${scenario} \
        --num_agents ${num_agents} \
        --agent_obsk ${agent_obsk} \
        --seed ${number} \
        --lr 4e-4 \
        --critic_lr 4e-4 \
        --clip_param 0.2 \
        --n_training_threads 32 \
        --n_rollout_threads 32 \
        --num_mini_batch 1 \
        --episode_length 200 \
        --num_env_steps 200000000 \
        --ppo_epoch 10 \
        --use_value_active_masks \
        --use_eval \
        --add_center_xy \
        --use_state_agent \
        --share_policy \
        --wandb_name "xxx" \
        --user_name "${user_name}"
done
