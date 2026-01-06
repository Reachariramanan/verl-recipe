#!/bin/bash
# DVPO Ablation Study Script
# Tests different DVPO loss components by disabling them

set -x

MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
DATA_PATH="path/to/your/training/data"
EXPERIMENT_NAME="dvpo_ablation_test"

# Test 1: Full DVPO (baseline)
python verl-recipe/dvpo/main_dvpo.py \
    trainer.experiment_name=${EXPERIMENT_NAME}_full \
    data.train_files=${DATA_PATH} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    algorithm.dvpo_loss_weights.ablate_risk=false \
    algorithm.dvpo_loss_weights.ablate_cvar=false \
    algorithm.dvpo_loss_weights.ablate_gain=false \
    algorithm.dvpo_loss_weights.ablate_shift=false \
    algorithm.dvpo_loss_weights.ablate_shape=false \
    algorithm.dvpo_loss_weights.ablate_curv=false \
    algorithm.dvpo_loss_weights.ablate_consist=false

# Test 2: Ablate upper-tail gain (most important component)
python verl-recipe/dvpo/main_dvpo.py \
    trainer.experiment_name=${EXPERIMENT_NAME}_no_gain \
    data.train_files=${DATA_PATH} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    algorithm.dvpo_loss_weights.ablate_gain=true

# Test 3: Ablate CVaR (robustness component)
python verl-recipe/dvpo/main_dvpo.py \
    trainer.experiment_name=${EXPERIMENT_NAME}_no_cvar \
    data.train_files=${DATA_PATH} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    algorithm.dvpo_loss_weights.ablate_cvar=true

# Test 4: Ablate consistency (ensemble component)
python verl-recipe/dvpo/main_dvpo.py \
    trainer.experiment_name=${EXPERIMENT_NAME}_no_consist \
    data.train_files=${DATA_PATH} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    algorithm.dvpo_loss_weights.ablate_consist=true
