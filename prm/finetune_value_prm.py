# Process directory
import sys
import os
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path[:cur_path.find('CodePRM')] + 'CodePRM')  # 这里要改为你自己的项目的主目录
import warnings
warnings.filterwarnings('ignore')
import shutil
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# CUDA devices
# os.environ['CUDA_VISIBLE_DEVICES'] = '3, 4'  # 必须放在import各种python的包之前运行
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
# torch.multiprocessing.set_start_method('spawn')
import random

from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
                          AutoConfig, AutoModel, BitsAndBytesConfig)
import torch
from datasets import load_dataset, concatenate_datasets, load_from_disk
from value_prm import get_llm_for_sequence_regression
from utils import *
from datetime import datetime

from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from transformers import DataCollatorWithPadding
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.nn as nn
import argparse

# Define a custom metric function (e.g., accuracy for binary classification)
def compute_metrics(eval_pred):
    pre, _ = eval_pred
    auc = roc_auc_score(pre[1], pre[0])
    ll = log_loss(pre[1], pre[0])
    acc = accuracy_score(pre[1], pre[0] > 0.5)
    result = {
        'auc': auc,
        'll': ll,
        'acc': acc,
    }
    # print(result)
    return result

def compute_metrics_regression(eval_pred):
    preds, _ = eval_pred
    mse = mean_squared_error(preds[1], preds[0])
    mae = mean_absolute_error(preds[1], preds[0])
    r2 = r2_score(preds[1], preds[0])
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
    }


class PRMTrainer(Trainer):
    def __init__(self,
                 model=None,
                 huggingface_args=None,
                 aux_args=None,
                 data_collator=None,
                 train_dataset=None,
                 eval_dataset=None,
                 tokenizer=None,
                 compute_metrics=None,
                 preprocess_logits_for_metrics=None):
        super().__init__(model=model,
                         args=huggingface_args,
                         data_collator=data_collator,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         tokenizer=tokenizer,
                         compute_metrics=compute_metrics,
                         preprocess_logits_for_metrics=preprocess_logits_for_metrics)
        self.aux_args = aux_args
        self.loss_type = aux_args.loss_type
        if self.loss_type in ['nce', 'orm']:
            self.loss_fn = nn.BCELoss(reduction='none')
        elif self.loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')

    def compute_loss(self, model, inputs, return_outputs=False):
        rewards, outputs = model(input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            return_output=True)

        # 找到标签不为 -100 的位置
        valid_mask = inputs['labels'] != -100  # 假设 'labels' 是用于训练的标签
        valid_indices = valid_mask.nonzero(as_tuple=True)

        # 提取有效的奖励值和标签
        valid_rewards = rewards[valid_indices]
        valid_labels = inputs['labels'][valid_indices].to(dtype=torch.bfloat16)

        if self.loss_type in ['nce', 'mse']:
            # 如果是二分类任务，确保奖励值经过 sigmoid
            rewards_sigmoid = torch.sigmoid(valid_rewards)

            if self.loss_type == 'nce':
                # 二元交叉熵损失
                loss = self.loss_fn(rewards_sigmoid, valid_labels)
            elif self.loss_type == 'mse':
                # 均方误差损失
                loss = self.loss_fn(rewards_sigmoid, valid_labels)
            else:
                raise ValueError('loss type not supported')
            # 对有效标签的损失取平均
            loss = (loss * valid_mask[valid_indices]).sum() / valid_mask.sum()
        else:
            raise ValueError('loss type not supported')
        return (loss, (loss, rewards)) if return_outputs else loss


class LLMFinetuner(object):
    def __init__(self, args):
        self.args = args
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.per_device_eval_batch_size = args.per_device_eval_batch_size
        self.total_batch_size = args.total_batch_size
        self.learning_rate = args.learning_rate
        self.server = args.server

        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = '\n\n\n\n\n'

        self.trunc_sample_num = 0

        self.pub_test_pass_trace_num = 0
        self.but_failed_code_num = 0

        now = datetime.now() # 获取当前日期和时间
        # 格式化日期为四位数字字符串（MMDD）
        self.datetime_str = now.strftime("%m%d%H%M")

        # hyper params
        self.model_name = args.model_name
        self.model_path = f"{get_raw_data_path()}/LLMWeights/{args.model_name}/"
        self.data_path = f"{get_proj_path()}/dataProcess/ft_data/all/mc_prm_data/"
        self.model_save_path = f"{get_proj_path()}/dataProcess/ft_results/models/{args.model_name}-value-ft-priv-{self.args.privileged}-{self.datetime_str}"
        self.tokenizer_save_path = self.model_save_path

        # training init
        if self.args.test_only:
            self.__model_init__(model_load_path=self.model_save_path, mode='test')
            self.__data_init__()
            self.__auto_trainer_init__()
            self.sample_test()
            eval_result = self.trainer.evaluate()
            print(eval_result)
        else:
            self.__model_init__(model_load_path=self.model_path)
            self.__data_init__()
            self.__auto_trainer_init__()

            self.sample_test()
            # print(f"Before fine tune:")
            # eval_result = self.trainer.evaluate()
            # print(eval_result)
            # print(f'-----------------')
            self.auto_train()
            print(f'After fine tune:')
            self.__model_init__(model_load_path=self.model_save_path, mode='test')
            self.sample_test()

    def __model_init__(self, model_load_path, mode='train'):
        # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_load_path)
        self.step_tag_id = self.tokenizer.encode(f" {self.step_tag}")[-1]  # 76325
        self.candidate_tokens = self.tokenizer.encode(f" {self.good_token} {self.bad_token}") # [488, 481]
        print(f"candidate_tokens: {self.candidate_tokens}")
        self.tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
        self.tokenizer.padding_side = "left"  # Allow batched inference

        if mode == 'train':
            self.lora_config = LoraConfig(
                # task_type=TaskType.CAUSAL_LM,  # LoRA for causal language modeling task
                r=8,  # Rank of LoRA
                lora_alpha=32,  # Alpha scaling factor for LoRA
                lora_dropout=0.1,  # Dropout rate for LoRA layers
                target_modules=["q_proj", "v_proj"],  # Apply LoRA to specific layers
                modules_to_save=["value_head"],  # 对于增加了value_head的模型，进行lora fine tune想要保存的模块
            )
            self.model = get_llm_for_sequence_regression(self.model_path,
                                                         model_load_path,
                                                         model_type='reward',
                                                         lora_config=self.lora_config,
                                                         init_value_head=True,
                                                         value_head_prefix="value_head",
                                                         packing_samples=False,
                                                         use_flash_attention_2=True)
            # 打印 LoRA 层的参数
            # for name, param in self.model.named_parameters():
            #     if 'lora' in name:
            #         print(f"LoRA parameter: {name} - {param.shape}")
        else:
            self.model = get_llm_for_sequence_regression(self.model_path,
                                                         model_load_path,
                                                         model_type='reward',
                                                         lora_config=None,
                                                         init_value_head=False,
                                                         value_head_prefix="value_head",
                                                         packing_samples=False)
        self.model.to('cuda')
        print(self.model.device)

    def __data_init__(self):
        # 加载数据
        DATA_PATH = {
            "test": os.path.join(self.data_path, 'test/data_list.json'),
            "train": os.path.join(self.data_path, "train/data_list.json"),
            # "train": os.path.join(self.data_path, "train/data_list_little.json"),
        }

        self.dataset = load_dataset('json', data_files=DATA_PATH)

        # 过滤 test 数据集中的样本，选择 question_id 在 3000-3999 范围内的样本
        # def filter_samples(example):
        #     if example['feedback'] == 'All public test cases passed!':
        #         return True
        #     else:
        #         return False
        # self.dataset['train'] = self.dataset['train'].filter(filter_samples)
        # self.dataset['test'] = self.dataset['test'].filter(filter_samples)

        # test只选一部分数据
        if self.args.test_only:
            self.dataset['train'] = self.dataset['train'].select(range(200))  # 只在一部分训练
        self.dataset['train'] = self.dataset['train'].shuffle(seed=42)
        self.dataset['test'] = self.dataset['test'].shuffle(seed=42).select(range(500)) # 随机选一部分做测试

        print('start processing')
        self.tokenized_datasets = self.dataset.map(self.preprocess_function)
        self.tokenized_datasets['train'] = self.tokenized_datasets['train'].remove_columns(['question', 'process', 'label'])
        self.tokenized_datasets['test'] = self.tokenized_datasets['test'].remove_columns(['question', 'process', 'label'])
        print(self.tokenized_datasets['train'])
        print('dataset processed')
        print(f"pub pass but total failed ratio: {self.but_failed_code_num} / {self.pub_test_pass_trace_num} = "
              f"{self.but_failed_code_num / self.pub_test_pass_trace_num}")
        # print(tokenized_datasets['train']['input_ids'])
        # print(len(tokenized_datasets['train']['input_ids'][0]))

        # Data collator for padding inputs dynamically
        self.data_collator = DataCollatorWithPadding(self.tokenizer)

    def __auto_trainer_init__(self):
        # 训练器加载
        BATCH_SIZE = args.total_batch_size
        GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // args.per_device_train_batch_size
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp = world_size != 1
        if ddp:
            GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

        print(f"world_size: {world_size}")
        print(f"ddp: {ddp}")

        fp = f'bs_{args.total_batch_size}_lr_{args.learning_rate}'
        output_path = f'{get_proj_path()}/dataProcess/ft_results/ckpts/prm_ft_ckpt_{self.datetime_str}_value_priv_{self.args.privileged}/{fp}'

        # Training arguments
        self.training_args = TrainingArguments(
            output_dir=output_path,
            evaluation_strategy="steps",  # Evaluate at the end of each epoch
            save_strategy="steps",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            num_train_epochs=1,
            weight_decay=0.01,
            logging_strategy="steps",  # 确保日志策略为 "steps"
            logging_steps=50,  # 如果eval strategy是 steps，且eval steps没有被额外指定，则保存间隔为logging_steps
            save_steps=50,  # 如果save strategy是 steps，那么这个参数就是保存的间隔（不设置的话default是500）
            save_total_limit=5,
            # fp16=True,  # Enable mixed precision for better performance on supported hardware
            bf16=True,
            report_to=self.args.report_to,  # wandb/none
            logging_dir="./logs",  # 日志目录
            dataloader_num_workers=4,
            deepspeed=None,
            ddp_find_unused_parameters=False,
            load_best_model_at_end=True,  # 在结束时加载最佳模型
            metric_for_best_model="mse",  # 选择 AUC 作为最佳模型的评价指标
            greater_is_better=False,  # 对于 AUC，值越大越好
            label_names=["labels"],
            # remove_unused_columns=False,  # 默认为true，这会导致compute loss时自动删除forward没有用到的列
        )
        # Initialize the Trainer
        self.trainer = PRMTrainer(
            model=self.model,
            huggingface_args=self.training_args,
            aux_args=self.args,
            train_dataset=self.tokenized_datasets['train'],
            eval_dataset=self.tokenized_datasets['test'],  # Replace with a validation set if available
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
            compute_metrics=compute_metrics_regression,
        )

    def auto_train(self):
        # 训练前
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name} - mean before training: {param.data.mean()}")
        self.model.train()
        self.trainer.train()
        # self.trainer.evaluate()
        # 训练后
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name} - mean after training: {param.data.mean()}")

        self.model.save_pretrained(self.model_save_path)
        self.tokenizer.save_pretrained(self.tokenizer_save_path)

    def sample_test(self):
        input_ids = self.tokenized_datasets['test'][2]['input_ids']
        labels = self.tokenized_datasets['test'][2]['labels']
        attention_mask = self.tokenized_datasets['test'][2]['attention_mask']
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.model.device)
        labels = torch.tensor(labels).unsqueeze(0).to(self.model.device)
        attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(self.model.device)
        with torch.no_grad():
            rewards, outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,return_output=True)

            # 找到标签不为 -100 的位置
            valid_mask = labels != -100  # 假设 'labels' 是用于训练的标签
            valid_indices = valid_mask.nonzero(as_tuple=True)

            # 提取有效的奖励值和标签
            valid_rewards = rewards[valid_indices]
            rewards_sigmoid = torch.sigmoid(valid_rewards)
            valid_labels = labels[valid_indices].to(dtype=torch.bfloat16)

            print(f"step_scores: {rewards_sigmoid}")
            print(f"labels: {valid_labels}")


    def preprocess_function(self, example):
        # instruction ???
        if self.args.privileged:
            input = f"{example['question']} \n{example['process_before_code']} \n{example['code']} \n{example['feedback']} \n{example['process']}"
        else:
            input = f"{example['question']} \n{example['process']}"
        tokenized_inputs = self.tokenizer(
            input,
            truncation=True,
            padding='max_length',
            # padding=True,
            max_length=3500,
        )

        if example['feedback'] == 'All public test cases passed!':
            self.pub_test_pass_trace_num += 1
            if example['label'][-1] == 0:
                self.but_failed_code_num += 1

        def find_all_indices(lst, element):
            return [i for i, x in enumerate(lst) if x == element]

        length = len(tokenized_inputs['input_ids'])
        # print(length)
        indices = find_all_indices(tokenized_inputs['input_ids'], self.step_tag_id)

        if len(indices) != len(example['label']):
            print(f"Not equal label and step tags for {example['question_id']}: {len(indices)} - {len(example['label'])}")
            print('---------')
            print(f"ori input length: {len(self.tokenizer.encode(input))}")
            # 这里privileged版本会出现，因为会截断
            example['label'] = example['label'][:len(indices)]
            example['saved_flag'] = example['saved_flag'][:len(indices)]
            self.trunc_sample_num += 1 # 当这样的案例较少时，可以允许手动将最后一个位置设置为可以回传loss，这样虽然存在重复loss，但是个数较少，可以使得loss正常显示计算
            # if self.trunc_sample_num < 100:
            #     example['saved_flag'][-1] = False

        assert len(indices) == len(example['label'])

        tokenized_inputs['labels'] = [-100.0] * length
        with_valid_label_flag = False
        # tokenized_inputs['attention_mask'] = [1] *length
        # print(len(indices))
        for i in range(len(indices)):
            if not example['saved_flag'][i]:  # 在preprocess时已经去掉了没有saved flag为False的trace，所以这里如果不行肯定是因为长度超出被截掉了
                with_valid_label_flag = True
                if 0.0 <= example['label'][i] <= 1.0:
                    tokenized_inputs['labels'][indices[i]] = float(example['label'][i])
                else:
                    raise ValueError('label is wrong')
            # tokenized_inputs['attention_mask'][indices[i]] = 0
        # tokenized_inputs['labels'] = [-100] *(length-1) + tokenized_inputs['input_ids'][length-1:]

        # 如果labels都是-100，则此样本不会回传loss，但是实际trainer会将loss设置为nan，这种样本应当在数据集形成时丢弃掉，现在似乎没有办法，记录一下
        if not with_valid_label_flag:
            print(f"all -100 labels for {example['question_id']}")
            print('---------')
            print(f"ori input length: {len(self.tokenizer.encode(input))}")
            print(f"input length: {len(tokenized_inputs['input_ids'])}")
            print(f"labels: {example['label']}")
            print(f"saved_flag: {example['saved_flag']}")
            print('---------')
        tokenized_inputs['step_labels'] = tokenized_inputs['labels']
        return tokenized_inputs

    def preprocess_logits_for_metrics(self, rewards, labels):
        # 这里拿到的第一个元素，是compute loss在return outputs为True的情况下第二个元素的[1:]
        # 找到标签不为 -100 的位置
        valid_mask = labels != -100  # 假设 'labels' 是用于训练的标签
        valid_indices = valid_mask.nonzero(as_tuple=True)

        # 提取有效的奖励值和标签
        valid_rewards = rewards[valid_indices]
        rewards_sigmoid = torch.sigmoid(valid_rewards)
        valid_labels = labels[valid_indices].float()
        return rewards_sigmoid, valid_labels


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--model_path", type=str, default=f"")
    parser.add_argument("--data_path", type=str, default=f"")
    parser.add_argument("--loss_type", type=str, default='mse')
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--total_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--server", type=str, default='1')
    parser.add_argument("--privileged", type=str2bool, default=False)
    parser.add_argument("--report_to", type=str, default='none', choices=['none', 'wandb'])
    parser.add_argument("--test_only", type=str2bool, default=False)
    args = parser.parse_args()

    finetuner = LLMFinetuner(args)
