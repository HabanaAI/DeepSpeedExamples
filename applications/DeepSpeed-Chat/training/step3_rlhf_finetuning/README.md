# Reinforcement Learning from human feedback (RLHF) finetuning
For more background you can visit [README from Microsoft/DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning).
This page will provide instructions on how the run step-3 RLHF script using multicard setups for either llamav2 and bloom.

## Example Script
The example bash script to launch step-3 training is located in:
1. bloom-1b1\bloom-560m: `DeepSpeedExamples/applications/DeepSpeed-Chat/example_scripts/bloom/train_step3_bloom_1.1b_560m.sh`
2. llamav2-7b: `DeepSpeedExamples/applications/DeepSpeed-Chat/example_scripts/llamav2-7b/train_step3.sh`

The execution of the above scripts is pretty similar, the instruction below are common.
Both scripts are launching training on a server with x8 cards.

## Controlling environment variables
The scripts accept the below environment variables to control the training.
Most of them are optional and their default values matches our experiments.

1. Mandatory:
- `HL_TAG`: tag name added to the artifacts of this run (string).
- `HL_BASE_OUT_PATH`: base path for artifacts.
- `HL_ACTOR_MODEL_PATH`: path for actor model checkpoint from step1.
- `HL_CRITIC_MODEL_PATH` - path for critic model checkpoint from step2.
- `HL_DATASET_PATH` - HF like dataset path or list of paths.
2. Optional:
- `HL_NUM_NODES` - Number of "boxes" (servers) participating in the training process, default is `1`.
- `HL_DEVICES_PER_NODE` - number of HPU cards per node, default is 8.
- `HL_ACTOR_CP_ACT` - whether to use activation-checkpointing memory optimization for actor model, default is set to 0 (false).
- `HL_CRITIC_CP_ACT` - whether to use activation-checkpointing memory optimization for critic model, default is set to 0 (false).
- `HL_SEED` - base seed that will be used to system initialization, default set to 10 for bloom and 1 for llamav2.
- `HL_MBS` - the micro-bs that each card will use during the training, default is 2 for bloom and 4 for llamav2.
- `HL_GBS` - the global-bs will be used for training, defauls is 64 for bloom and 32 for llamav2.
- `HL_TENSORBOARD_PATH` - tensorboard path - Optional, empty string for default.
- `HL_LOG_FILE` - log full filename- Optional, empty string for default
- `HL_MASTER_PORT` - deepspeed runner master_port - Optional, default is 29500
- `HL_HYBRID_ENGINE` - whether to use DeepSpeed HybridEngine, default is 0 (false), as it is not fully validated.
- `HL_ACTOR_LEARNING_RATE` - LR for actor model training, default is `1e-5` for bloom and `9e-6` for llamav2.
- `HL_ACTOR_LEARNING_RATE` - LR for critic model training, default is `6e-6` for bloom and `9e-6` for llamav2.
- `HL_ACTOR_WEIGHT_DECAY` - actor model weight decay factor, default is `0.1` for bloom and `0` for llamav2.
- `HL_CRITIC_WEIGHT_DECAY` - critic model weight decay factor, default is `0.1` for bloom and `0` for llamav2.
- `HL_ACTOR_DROPOUT` - actor model dropout rate, default is 0.
- `HL_CRITIC_DROPOUT` - critic model dropout rate, default is 0.
- `HL_LORA_ACTOR_LR` - LR for actor model LoRA parameterss training, default is `4e-4` for bloom and `5e-4` for llamav2.
- `HL_LORA_CRITIC_LR` - LR for actor model LoRA parameterss training, default is `6e-4` for bloom and `5e-4` for llamav2.
- `HL_ONLY_OPTIMIZE_LORA` - if set, will optimize only LoRA params, default is 0 (false) and relevant only to llamav2.
- `HL_LORA_DIM` - Which LoRA dimension will take place, 0 means no LoRA. default is `0` for bloom and `64` for llamav2.
- `HL_EPOCHS` - How many training epochs will be used, default is 1.
- `HL_NUM_WARMUP_STEPS` - How mant warm-up steps will be used, default is 100.
- `HL_PRINT_ANSWERS_INTERVAL` - Allows to print the generated answers during the training, default is 0 (no print).

## Launching the scripts
The above script can be called with the below template for bloom:
  ```bash
  cd DeepSpeedExamples/applications/DeepSpeed-Chat/example_scripts/bloom
  export HL_TAG=<tag>
  export HL_BASE_OUT_PATH=<base_out_path>
  export HL_ACTOR_MODEL_PATH=<act_model_path>
  export HL_CRITIC_MODEL_PATH=<cri_model_path>
  export HL_DATASET_PATH=<path_to_data_set_or_list>
  ...
  ...
  ./train_step3_bloom_1.1b_560m.sh
  ```
Or for llamav2-7b:
  ```bash
  cd DeepSpeedExamples/applications/DeepSpeed-Chat/example_scripts/llamav2-7b
  export HL_TAG=<tag>
  export HL_BASE_OUT_PATH=<base_out_path>
  export HL_ACTOR_MODEL_PATH=<act_model_path>
  export HL_CRITIC_MODEL_PATH=<cri_model_path>
  export HL_DATASET_PATH=<path_to_data_set_or_list>
  ...
  ...
  ./train_step3.sh
  ```
