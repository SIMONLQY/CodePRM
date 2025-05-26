# -*- coding:utf-8 _*-
import torch
from typing import List, Union, Optional, Literal
import dataclasses
import transformers
from utils import get_raw_data_path, extract_list_from_text, get_proj_path
from openai import OpenAI
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from prm import get_llm_for_sequence_regression
import json


class LocalModelActor:
    def __init__(self, model_name, vllm_mode=False):
        self.vllm_mode = vllm_mode
        if 'ft' in model_name:
            model_load_path = f"{get_proj_path()}/dataProcess/ft_results/models/{model_name}"
        else:
            model_load_path = f"{get_raw_data_path()}/LLMWeights/{model_name}/"

        if vllm_mode:
            # 暂时不support
            self.model = LLM(
                model=model_load_path,
                gpu_memory_utilization=0.9,
                trust_remote_code=True,
                tensor_parallel_size=1,
                max_model_len=8192,
            )
        else:
            if 'value' in model_name:
                # 加载model_load_path中的adapter_config.json文件，获得base_model_name_or_path
                adapter_config = json.load(open(f"{model_load_path}/adapter_config.json"))
                base_model_name_or_path = adapter_config["base_model_name_or_path"]
                self.model = get_llm_for_sequence_regression(base_model_name_or_path,
                                                             model_load_path,
                                                             model_type='reward',
                                                             lora_config=None,  # set to none as not for training
                                                             init_value_head=False,
                                                             value_head_prefix="value_head",
                                                             packing_samples=False
                                                             )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_load_path,
                    # load_in_8bit=True,   # Enables 8-bit quantization
                    # device_map="auto",   # Automatically assigns the model to available GPUs/CPUs
                    # torch_dtype=torch.float16,  # Mixed precision for faster inference
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                )
            self.model.to('cuda')

    def get_output_logits(self, input_id):
        input_id = input_id.to(self.model.device)
        return self.model(input_id).logits.cpu()

    def get_output_rewards(self, input_id, attention_mask):
        input_id = input_id.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        rewards, outputs = self.model(input_ids=input_id,
                            attention_mask=attention_mask,
                            return_output=True)
        rewards = torch.sigmoid(rewards)

        return rewards.cpu()

    def vllm_call(self, prompt, sampling_params:SamplingParams):
        return self.model.generate(prompt, sampling_params)

    def vllm_get_tokenizer(self):
        return self.model.get_tokenizer()



