# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company

import os
import torch
from deepspeed.accelerator import get_accelerator



def is_hpu():
    return get_accelerator().device_name() == "hpu"


def hpu_mark_step():
    if is_hpu():
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()


def accelerator_model_preparation(tp_size_for_inference=1):
    if is_hpu():
        import optimum.habana.transformers
        import optimum.habana.transformers.gradient_checkpointing
        from  optimum.habana.transformers.models.bloom.modeling_bloom import GaudiBloomForCausalLM
        import optimum.habana.transformers.modeling_utils

        # Configure optimum-habana for tp size
        GaudiBloomForCausalLM.set_tp_for_inference(tp_size_for_inference)

        optimum.habana.transformers.modeling_utils.adapt_transformers_to_gaudi()
        torch.utils.checkpoint.checkpoint = optimum.habana.transformers.gradient_checkpointing.checkpoint
