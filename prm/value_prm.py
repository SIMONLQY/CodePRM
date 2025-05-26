# Process directory
import sys
import os
import warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from datasets import load_dataset
import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
from flash_attn.utils.distributed import all_gather
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
from prm.ring_attn_utils import convert_ring_attn_params
from transformers.integrations.deepspeed import HfDeepSpeedConfig
import argparse
import os
from utils import *
from datetime import datetime

from peft import PeftModel
from peft import get_peft_model, LoraConfig, TaskType


# Construct transformer with a value head for sequence classification.
# https://github.com/huggingface/transformers/blob/405b56269812056d9593869e22b7b264d806cb1e/src/transformers/models/llama/modeling_llama.py#L1254
def get_llm_for_sequence_regression(
    base_model_name_or_path:str,
    model_name_or_path: str,
    model_type: str,
    *,
    bf16=True,
    lora_config=None,
    normalize_reward=False,
    use_flash_attention_2=False,
    ds_config: dict = None,
    init_value_head: bool = False,
    value_head_prefix="value_head",
    device_map=None,
    packing_samples=False,
    **kwargs,
) -> nn.Module:
    """Get transformer with a sequence classification head on top (linear layer).

    Args:
        model_name_or_path (str): Path to pretrained model.
        model_type (str): Either "reward" or "critic.
        bf16 (bool, optional): Whether enable bfloat16. Defaults to True.
        normalize_reward (bool, optional): Whether normalize reward. Defaults to False.
        use_flash_attention_2 (bool, optional): Whether use Flash Attention 2.0. Defaults to False.
        ds_config (dict, optional): Deepspeed config, used to automatically splitting the model onto
            multiple gpus during from_pretrained when ZeRO-3 enabled. Defaults to None.

    Returns:
        nn.Module: pretrained transformer model.
    """
    assert (
        model_type == "critic" or model_type == "reward"
    ), f"invalid model_type: {model_type}, should be critic or reward."

    config = AutoConfig.from_pretrained(base_model_name_or_path, trust_remote_code=True)  # need original model name or path to get base class
    config.normalize_reward = normalize_reward
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

    base_class = AutoModel._model_mapping[type(config)]
    base_pretrained_class = base_class.__base__
    if model_type == "reward":
        cls_class = _get_reward_model(base_pretrained_class, base_class, value_head_prefix, packing_samples)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    if lora_config:
        assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
        nf4_config = None
        # nf4_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        # )
    else:
        nf4_config = None

    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        quantization_config=nf4_config,
        device_map=device_map,
        **kwargs,
    )

    # LoRA
    if lora_config:
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)

        # if load_in_4bit:
        #     for name, module in model.named_modules():
        #         if isinstance(module, LoraLayer):
        #             module = module.to(torch.bfloat16)
        #         if "norm" in name:
        #             module = module.to(torch.float32)
        #         if value_head_prefix in name or "embed_tokens" in name:
        #             if hasattr(module, "weight"):
        #                 module = module.to(torch.bfloat16)

    # MoE - balancing loss
    model_config = model.config.to_dict()
    if "output_router_logits" in model_config:
        print("[MoE] set output_router_logits as True")
        model.config.output_router_logits = True

    # https://github.com/huggingface/transformers/issues/26877
    model.config.use_cache = False

    # NOTE: For reward model training only, intialize value_head manually
    # because deepspeed.zero.Init() will not intialize them.
    if init_value_head:
        value_head = getattr(model, value_head_prefix)
        if dschf is not None:
            with deepspeed.zero.GatheredParameters([value_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
            value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))

    return model



def _get_reward_model(base_pretrained_model, base_llm_model, value_head_prefix="value_head", packing_samples=False):
    class RewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.packing_samples = packing_samples

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            ring_attn_group=None,
            packed_seq_lens=None,
            labels: Optional[torch.LongTensor] = None,  # 这里必须加上，否则trainer会自动在compute loss里去掉forward没用到的input
            inputs_embeds: Optional[torch.Tensor]=None,
            **kwargs,
        ) -> torch.Tensor:
            if not self.packing_samples:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
            else:
                if ring_attn_group is not None:
                    input_ids, attention_mask, position_ids = convert_ring_attn_params(
                        input_ids, attention_mask, packed_seq_lens, ring_attn_group
                    )
                else:
                    # reset the positions for packed samples
                    position_ids = reset_position_ids(attention_mask)
            position_ids.masked_fill_(attention_mask == 0, 1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)

            if self.packing_samples:
                if ring_attn_group is not None:
                    reward = all_gather(values, ring_attn_group).reshape(1, -1)
                else:
                    reward = values
                packed_seq_lens = torch.tensor(packed_seq_lens, device=values.device)
                eos_indices = packed_seq_lens.cumsum(dim=0) - 1
                reward = reward.squeeze(0).gather(dim=0, index=eos_indices)
            else:
                # eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                # reward = values.gather(dim=1, index=eos_indices).squeeze(1)
                reward = values

            if not self.training and self.normalize_reward:
                reward = (reward - self.mean) / self.std

            return (reward, outputs) if return_output else reward

    return RewardModel

# Reset positions for packed samples
# For example
# Input: attention_mask = torch.tensor([[1, 1, 1, 2, 2, 2, 3, 3, 0]])
# Output: position_ids  = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 0]])
def reset_position_ids(attention_mask):
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
    for i in range(attention_mask.size(0)):
        mask = attention_mask[i]
        seq_num = mask.max().item()
        for index in range(1, seq_num + 1):
            sample_mask = mask == index
            sample_length = sample_mask.sum().item()
            position_ids[i, sample_mask] = torch.arange(sample_length, device=mask.device)
    return position_ids



class ValuePRM(object):
    def __init__(self, args):
        self.args = args

        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = '\n\n\n\n\n'

        # load model
        self.model_name = args.prm_model_name
        self.model_load_path = f"{get_proj_path()}/dataProcess/ft_results/models/{self.model_name}"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_load_path)
        self.step_tag_id = self.tokenizer.encode(f" {self.step_tag}")[-1]  # 76325
        self.candidate_tokens = self.tokenizer.encode(f" {self.good_token} {self.bad_token}")  # [488, 481]
        self.tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
        self.tokenizer.padding_side = "left"  # Allow batched inference
        self.model = args.prm_local_model_actor

    def form_process_data(self, question, process_list, code=None, feedback=None):
        reasoning_before_process = ''
        for step_data in process_list:
            reasoning_before_process += f"{step_data.strip()}\n"

        process = ''
        for step_data in process_list:
            process += f"{step_data.strip()} {self.step_tag}"

        if 'priv-True' in self.model_name:
            input_for_prm = f"{question} \n{reasoning_before_process} \n{code} \n{feedback} \n{process}"
        else:
            input_for_prm = f"{question} \n{process}"
        return input_for_prm

    def process_judge(self, question, process_list, code=None, feedback=None):
        input_for_prm = self.form_process_data(question, process_list, code, feedback)
        input_id = torch.tensor([self.tokenizer.encode(input_for_prm)])
        attention_mask = torch.ones(input_id.shape, dtype=torch.long)
        # print(input_id)

        with torch.no_grad():
            rewards = self.model.get_output_rewards(input_id, attention_mask)

            step_scores = rewards[input_id == self.step_tag_id].view(-1).tolist()
            # 将step_scores中大于0.5的设置为1，否则设置为0， 然后返回一个list，按顺序排列结果
            return [1 if score > 0.5 else 0 for score in step_scores], step_scores