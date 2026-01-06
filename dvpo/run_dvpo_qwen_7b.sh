#!/bin/bash
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DVPO training script for Qwen-7B model

set -x

# Model and data paths
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
DATA_PATH="path/to/your/training/data"

# Training configuration
EXPERIMENT_NAME="dvpo_qwen7b_test"
PROJECT_NAME="verl-dvpo"

# Launch training
python verl-recipe/dvpo/main_dvpo.py \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=~/verl_experiments/${EXPERIMENT_NAME} \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=1 \
    trainer.test_freq=10 \
    trainer.save_freq=50 \
    trainer.logger=['console','wandb'] \
    trainer.critic_warmup=0 \
    data.train_files=${DATA_PATH} \
    data.val_files=${DATA_PATH} \
    data.train_batch_size=32 \
    data.max_response_length=2048 \
    data.max_prompt_length=1024 \
    data.shuffle=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.0 \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    critic.strategy=fsdp \
    critic.ppo_epochs=1 \
    critic.ppo_mini_batch_size=64 \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.fsdp_config.param_offload=false \
    critic.model.fsdp_config.optimizer_offload=false \
    critic.optim.lr=5e-6 \
    critic.optim.lr_scheduler_type=cosine \
    critic.optim.lr_warmup_steps_ratio=0.1 \
    critic.optim.min_lr_ratio=0.0 \
    critic.optim.weight_decay=0.0 \
    algorithm.adv_estimator=gae \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward_model.enable=true \
    reward_model.strategy=fsdp \
    reward_model.model.path=${MODEL_PATH} \
    reward_model.model.fsdp_config.param_offload=false \
    reward_model.model.fsdp_config.optimizer_offload=false \
    reward_model.reward_manager=dvpo \
    reward_model.overlong_buffer.enable=false \
    ray_kwargs.ray_init.num_cpus=16 \
    ray_kwargs.ray_init.num_gpus=8 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1
