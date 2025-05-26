# Process directory
import sys
import os
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path[:cur_path.find('CodePRM')] + 'CodePRM')  # 这里要改为你自己的项目的主目录
import warnings
warnings.filterwarnings('ignore')
import shutil
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from datasets import load_dataset
import argparse
import os
from utils import *

from peft import PeftModel
from peft import get_peft_model, LoraConfig, TaskType
# Ensure bitsandbytes is available for 8-bit quantization
# import bitsandbytes as bnb
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

from torch.nn import BCEWithLogitsLoss
from transformers import DataCollatorWithPadding
from datasets import concatenate_datasets

import random


class TokenPRM(object):
    def __init__(self, args):
        self.args = args

        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = '\n\n\n\n\n'

        # load model
        self.model_name = args.prm_model_name
        if 'ft' in self.model_name:
            self.model_load_path = f"{get_proj_path()}/dataProcess/ft_results/models/{self.model_name}"
        else:
            self.model_load_path = f"{get_raw_data_path()}/LLMWeights/{self.model_name}/"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_load_path)
        self.step_tag_id = self.tokenizer.encode(f" {self.step_tag}")[-1]  # 76325
        self.candidate_tokens = self.tokenizer.encode(f" {self.good_token} {self.bad_token}")  # [488, 481]
        self.tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
        self.tokenizer.padding_side = "left"  # Allow batched inference
        self.model = args.prm_local_model_actor

    def form_process_data(self, question, process_list, code=None, feedback=None):
        reasoning_before_process = ''
        for step_data in process_list:
            reasoning_before_process += f"\n{step_data.strip()}"

        process = ''
        for step_data in process_list:
            process += f"{step_data.strip()} {self.step_tag}"

        if 'priv-True' in self.model_name:
            input_for_prm = f"{question}\n{reasoning_before_process}\n{code}\n{feedback}\n{process}"
        else:
            input_for_prm = f"{question}\n{process}"
        return input_for_prm


    def process_judge(self, question, process_list, code=None, feedback=None):
        input_for_prm = self.form_process_data(question, process_list, code, feedback)
        input_id = torch.tensor([self.tokenizer.encode(input_for_prm)])
        # print(input_id)

        with torch.no_grad():
            logits = self.model.get_output_logits(input_id)[:, :, self.candidate_tokens]
            # print(logits)
            scores = logits.softmax(dim=-1)[:, :, 0]
            # print(scores)
            step_scores = scores[input_id == self.step_tag_id].view(-1).cpu().tolist()
            # 将step_scores中大于0.5的设置为1，否则设置为0， 然后返回一个list，按顺序排列结果
            return [1 if score > 0.5 else 0 for score in step_scores], step_scores