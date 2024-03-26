# DeepSpeed-Chat: RLHF Training of Chat Models

This directory provides scripts for the 3 stages RLHF training of LM to a chat model, via two examples:
1. Basic functional example - bloom-1b1 as an actor model, and bloom-560m as a critic model.
2. Practical example - llamav2-7b both as actor and critic model.

## Table of Contents
* [Model-References](../../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Training and Examples](#training-and-examples)
* [Supported Configuration](#supported-configuration)
* [Changelog](#changelog)
* [Known Issues](#knownissues)

## Model Overview
The training process involves 3 stages as described in [Microsoft/DeepSpeedExample README](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat#-demonstration-individual-step-fine-tuning).
The below model are used:
1. Step 1 - Supervised Fine-Tuning.
2. Step 2 - Reward Model.
3. Step 3 - Reinforcement Learning with Human Feedback: Used the fined tuned models from Step-1 and Step-2.

## Setup
Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html)
to set up the environment including the `$PYTHON` environment variable. To achieve the best performance, please follow the methods outlined in the [Optimizing Training Platform guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).
The guides will walk you through the process of setting up your system to run the model on Gaudi2.

### Install Habana DeepSpeed-fork
Please follow the instruction in [DeepSpeed User Guide](https://docs.habana.ai/en/master/PyTorch/DeepSpeed/DeepSpeed_User_Guide/DeepSpeed_User_Guide.html)

### Clone Habana DeepSpeedExamples
In the docker container, clone this repository and switch to the branch that matches your SynapseAI version.
You can run the [`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version.
```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/DeepSpeedExamples
```

```
export DEEPSPEED_EXAMPLES_ROOT=/path/to/DeepSpeedExamples
export PYTHONPATH=/path/to/DeepSpeedExamples:$PYTHONPATH

```

### Install Model Requirements
* In the docker container, go to DeepSpeed-Chat directory:
  ```bash
  cd DeepSpeedExamples/applications/DeepSpeed-Chat/
  ```

* Install the required packages using pip:
  ```bash
  pip install -r requirements.txt
  ```

## Training and Examples
Example bash scripts for steps 1, 2 and 3 on multi-card setup are available under `DeepSpeedExamples/applications/DeepSpeed-Chat/example_scripts`, either in bloom or llamav2-7b subdirectories. 
The below READMEs further explain each bash script:
* [Step 1](training/step1_supervised_finetuning/README.md)
* [Step 2](training/step2_reward_model_finetuning/README.md)
* [Step 3](training/step3_rlhf_finetuning/README.md)

## Supported Configuration
| Validated on  | SynapseAI Version | PyTorch Version | Mode |
|---------|-------------------|-----------------|-------------|
| Gaudi2  | 1.15.0            | 2.2.0           | Training |


## Changelog
### 1.14.0
* Introduce this training script.

### Script Modifications
- Upstreamed fixes:
  - Fixed weight decay configuration.
  - Fixed Reward Model and step-2 to train v_head when only optimizing LoRA.
  - Fixed chat application.
  - Fixed BF16 step-2 accuracy for bloom-560m.
  - Fixed step-1 ppl calculation.
  - Fixed rw_eval.
  - Added end-of-text special token.
  - Added average loss prints in step-1 and step-2.
  - Added print of reward ema in step-3.
  - Handled too short generation in step-3.
- Added support for native pytorch adamw optimizer.
- Added HPU support:
  - mark_step calls.
  - imports of habana_frameworks.
  - optimum-habana usages.
- Optimized reward calculation for RewardModel.
- Added padding of generated sequence to avoid dynamic shapes.
- Added deepspeed.zero.init() context for RewardModel in step-2.
- Fixed LayerNorm reset in step-3 when using Zero3.
- Added support for calculating loss in FP32 for bloom model.
- Added example bash scripts for bloom and llamav2 under example_scripts.
