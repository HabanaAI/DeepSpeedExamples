#!/bin/bash
# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company

# -----------------------------------------------------------------------
# RLHF step3 reference training script for LLaMAv2-7B model
# -----------------------------------------------------------------------

set -ex

actor_model_path=${HL_ACTOR_MODEL}
critic_model_path=${HL_CRITIC_MODEL}
tag=${HL_TAG:-default_tag}
base_out_path=${HL_BASE_OUT_PATH:-/root/logs}
n_nodes=${HL_NUM_NODES:-1}
n_devices_per_node=${HL_DEVICES_PER_NODE:-8}
actor_zero_stage=${HL_ACTOR_ZERO_STAGE:-1}
critic_zero_stage=${HL_CRITIC_ZERO_STAGE:-1}
actor_ckp_act=${HL_ACTOR_CP_ACT:-0}
critic_ckp_act=${HL_CRITIC_CP_ACT:-0}
seed=${HL_SEED:-1}
mbs=${HL_MBS:-4}
gbs=${HL_GBS:-32}
tensorboard_path=${HL_TENSORBOARD_PATH:-}
log_file=${HL_LOG_FILE:-}
checkpoint_path=${HL_CHECKPOINT_PATH:-}
master_port=${HL_MASTER_PORT:-29500}
hybrid_engine=${HL_HYBRID_ENGINE:-0}
dataset_path=${HL_DATASET_PATH:-}
actor_learning_rate=${HL_ACTOR_LEARNING_RATE:-9e-6}
critic_learning_rate=${HL_CRITIC_LEARNING_RATE:-9e-6}
actor_weight_decay=${HL_ACTOR_WEIGHT_DECAY:-0.0}
critic_weight_decay=${HL_CRITIC_WEIGHT_DECAY:-0.0}
actor_dropout=${HL_ACTOR_DROPOUT:-0.0}
critic_dropout=${HL_CRITIC_DROPOUT:-0.0}
lora_actor_learning_rate=${HL_LORA_ACTOR_LR:-5e-4}
lora_critic_learning_rate=${HL_LORA_CRITIC_LR:-5e-4}
only_optimize_lora=${HL_ONLY_OPTIMIZE_LORA:-0}
lora_dim=${HL_LORA_DIM:-64}
epochs=${HL_EPOCHS:-1}
num_warmup_steps=${HL_NUM_WARMUP_STEPS:-100}
print_answers_interval=${HL_PRINT_ANSWERS_INTERVAL:-0}
hpz_enabled_actor=${HL_HPZ_ENABLED_ACTOR:-0}
hpz_enabled_critic=${HL_HPZ_ENABLED_CRITIC:-0}
hpu_graphs=${HL_ENABLE_HPU_GRAPHS:-1}
test_stop_step=${HL_RUN_STEPS:-0}

# Calculate GAS given global batch, n_nodes, n_devices_per_node
total_devices=$(($n_nodes*$n_devices_per_node))
per_device_batch=$(($gbs/$total_devices))
gas=$(($per_device_batch/$mbs))

# set gradient checkpointing arguments
ckp_act_args=""
if [ "$actor_ckp_act" -eq "1" ]; then
  ckp_act_args="--actor_gradient_checkpointing "
fi
if [ "$critic_ckp_act" -eq "1" ]; then
  ckp_act_args="$ckp_act_args --critic_gradient_checkpointing "
fi

# enable hybrid engine
hybrid_engine_args=""
if [ "$hybrid_engine" -eq "1" ]; then
  hybrid_engine_args="--enable_hybrid_engine "
fi

hpu_graphs_args=""
if [ "$hpu_graphs" -eq "1" ]; then
  hpu_graphs_args="--enable_hpu_graphs "
fi

# setup checkpoint, tensorboard and log path
prefix_name=${tag}/llamav2-7b/step3
run_name=gb_${gbs}_mbs_${mbs}_ep_${epochs}_act_lr_${actor_learning_rate}_do_${actor_dropout}_wd_${actor_weight_decay}_cri_lr_${critic_learning_rate}_do_${critic_dropout}_wd_${critic_weight_decay}

lora_args=""
if [ "$lora_dim" -ne "0" ]; then
  lora_args=" --actor_lora_dim ${lora_dim} --actor_lora_module_name layers. \
              --actor_lora_learning_rate ${lora_actor_learning_rate} \
              --critic_lora_dim ${lora_dim}  --critic_lora_module_name layers. \
              --critic_lora_learning_rate ${lora_critic_learning_rate} "
  run_name="${run_name}_lora_act_lr_${lora_actor_learning_rate}_lora_cri_lr_${lora_critic_learning_rate}"
  if [ "$only_optimize_lora" -ne "0" ]; then
    lora_args="${lora_args} --only_optimize_lora "
    run_name="${run_name}_only_lora"
  fi
fi

hpz_args=""
if [ "$hpz_enabled_actor" -ne "0" ]; then
  hpz_args="--zero_hpz_enabled_actor "
fi
if [ "$hpz_enabled_critic" -ne "0" ]; then
  hpz_args="${hpz_args} --zero_hpz_enabled_critic "
fi

stop_step_args=""
if [ "$test_stop_step" -ne "0" ]; then
  stop_step_args="--enable_test_mode --test_stop_step ${test_stop_step} "
fi

if [ -z "$tensorboard_path" ]; then
  tensorboard_path=${base_out_path}/tensorboard/${prefix_name}
fi

if [ -z "$log_file" ]; then
  log_file=${base_out_path}/logs/${prefix_name}/${run_name}.txt
fi

if [ -z "$checkpoint_path" ]; then
  checkpoint_path=${base_out_path}/checkpoints/${prefix_name}/${run_name}
fi

# configure print answers settings
print_answers_args=""
if [ "$print_answers_interval" -ne "0" ]; then
  print_answers_args="--print_answers --print_answers_interval ${print_answers_interval}"
fi

if [ "$n_nodes" -ne "1" -a -f "$HOSTSFILE" ]
then
    MULTINODE_CMD="--hostfile=$HOSTSFILE \
                   --master_addr $(head -n 1 $HOSTSFILE | sed -n s/[[:space:]]slots.*//p) "
fi

# create required paths
# if log-file/tb-path provided, caller should make sure directories exist
mkdir -p ${base_out_path}/logs/${prefix_name}

# RUN
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
training_dir=$( realpath $script_dir/../../training)

CMD="${training_dir}/step3_rlhf_finetuning/main.py \
        --dtype bf16 \
        --actor_model_name_or_path ${actor_model_path} \
        --critic_model_name_or_path ${critic_model_path} \
        --data_path ${dataset_path} \
        --num_padding_at_beginning 0 \
        --num_train_epochs ${epochs} \
        --gradient_accumulation_steps ${gas} \
        --per_device_generation_batch_size ${mbs} \
        --per_device_training_batch_size ${mbs} \
        --generation_batches 1 \
        --ppo_epochs 1 \
        --max_answer_seq_len 256 \
        --max_prompt_seq_len 256 \
        --lr_scheduler_type cosine \
        --actor_learning_rate ${actor_learning_rate} \
        --critic_learning_rate ${critic_learning_rate} \
        --num_warmup_steps ${num_warmup_steps} \
        --actor_weight_decay ${actor_weight_decay} \
        --critic_weight_decay ${critic_weight_decay} \
        --actor_dropout ${actor_dropout} \
        --critic_dropout ${critic_dropout} \
        ${lora_args} \
        ${ckp_act_args} \
        ${hybrid_engine_args} \
        --compute_fp32_loss \
        --actor_zero_stage ${actor_zero_stage} \
        --critic_zero_stage ${critic_zero_stage} \
        --seed ${seed} \
        --deepspeed \
        --output_dir ${checkpoint_path} \
        --enable_tensorboard \
        --tensorboard_path ${tensorboard_path} \
        ${print_answers_args} \
        --no_fused_kernels \
        ${hpu_graphs_args} \
        ${hpz_args}\
        ${stop_step_args}\
        "

deepspeed --num_nodes ${n_nodes} \
          --num_gpus ${n_devices_per_node} \
          --master_port ${master_port} \
          $MULTINODE_CMD \
          $CMD   |& tee ${log_file}
exit $PIPESTATUS
