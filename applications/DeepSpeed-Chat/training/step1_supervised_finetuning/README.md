# Supervised finetuning (SFT)
For more background you can visit also the [README from Microsoft/DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/README.md).
This page will provide instructions on how the run step-1 Finetuning script using multicard setups for either llamav2-7b and bloom-1b1.

## Example Script
The example bash script to launch step-1 training is located in:
1. bloom-1b1: `DeepSpeedExamples/applications/DeepSpeed-Chat/example_scripts/bloom/train_step1_bloom_1.1b.sh`
2. llamav2-7b: `DeepSpeedExamples/applications/DeepSpeed-Chat/example_scripts/llamav2-7b/train_step1.sh`
The execution of the above scripts is pretty similar, the instruction below are common.
Both scripts are launching training on a server with x8 cards.

## Controlling environment variables
The scripts accept the below environment variables to control the training. 
Most of them are optional and their default values matches our experiments.
1. Mandatory:
* `HL_BASE_OUT_PATH`: base path for artifacts.
* `HL_TAG`: tag name added to the artifacts of this run (string).
- `HL_DATASET_PATH` - HF like dataset path or list of paths.
2. Optional:
- `HL_NUM_NODES`: Number of "boxes" (servers) participating in the training process, default is 1 HLS2 server.
- `HL_DEVICES_PER_NODE`: number of HPU accelarator cards per node, default is 8 cards.
- `HL_ACTOR_ZERO_STAGE`: The zero stage DeepSpeed will use, default is ZeRO stage 1.
- `HL_ACTOR_CP_ACT`: whether to use activation-checkpointing memory optimization, default is set to 0 (false).
- `HL_SEED`: base seed that will be used to system initialization, default set to 10 for bloom and 1 for llamav2-7b.
- `HL_MBS`: the micro-bs that each card will use during the training, default is 8.
- `HL_GBS`: the blobal batch-size for the training, default is 128 for bloom-1b1 and 64 for llamav2-7b
- `HL_TENSORBOARD_PATH`: tensorboard path - default is empty string.
- `HL_TENSORBOARD_ENABLED`: Whether to use tensorboard logging, default is 0 (false), relevant only to bloom-1b1.
- `HL_LOG_FILE`: log full filename, default is empty string.
- `HL_ACTOR_MODEL`: The path from which to load the pretrained model, can be also HF based path. default is `bigscience/bloom-1b1` for bloom, and `meta-llama/Llama-2-7b-hf` for llamav2-7b
- `HL_MASTER_PORT` - deepspeed runner master_port, default is 29500
- `HL_CHECKPOINT_PATH` - A custom path to save the finetuned model checkpoint to, set to `${HL_BASE_OUT_PATH}/checkpoints`
- `HL_EPOCHS` - How many epochs to run, default is 4.
- `HL_MAX_SEQ_LEN` - the max sequence length for the training, default is 512. Relevant only to llamav2.
- `HL_DROPOUT` - sets the Dropout ratio, default is `0.1`.
- `HL_LEARNING_RATE` - LR for the training, default is `2e-5` for bloom and `9e-6` for llamav2.
- `HL_LORA_LEARNING_RATE` - LR for training the LORA params, default is `2e-5` for bloom and `5e-4` for llamav2.
- `HL_WEIGHT_DECAY` - weight decay scalar, default is `0` for bloom and `0.1` for llamav2.
- `HL_LORA_DIM` - Set whether to use LoRA in the training process and which dim size, default is 0 and means LoRA is off.
- `HL_ONLY_OPTIMIZE_LORA` - When using LoRA, if to tune only LoRA params, default is 0 (false). Relevant only to llamav2.

## Launching the scripts
The above script can be called with the below template for bloom:
  ```bash
  cd DeepSpeedExamples/applications/DeepSpeed-Chat/example_scripts/bloom
  export HL_TAG=<tag>
  export HL_BASE_OUT_PATH=<base_out_path>
  export HL_DATASET_PATH=<path_to_data_set_or_list>
  ...
  ...
  ./train_step1_bloom_1.1b.sh
  ```
Or for llamav2-7b:
  ```bash
  cd DeepSpeedExamples/applications/DeepSpeed-Chat/example_scripts/llamav2-7b
  export HL_TAG=<tag>
  export HL_BASE_OUT_PATH=<base_out_path>
  export HL_DATASET_PATH=<path_to_data_set_or_list>
  ...
  ...
  ./train_step1.sh
  ```
