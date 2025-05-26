# -*- coding:utf-8 _*-
from torch.utils.data import DataLoader, SequentialSampler
from utils import *
import time
from models import *
from torcheval.metrics import HitRate, ReciprocalRank
import torchmetrics
from dataSet import *
from tqdm import tqdm
from math import sqrt, log
from executors import AppsExecutor, HumanevalExecutor
from ChatModels import GPT35Chat, GPT2Chat, KimiChat

class BS:
    def __init__(self, args):
        self.args = args
        self.sample_times = 0
        self.gamma = 0.9
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        if 'kimi' in args.arch:
            self.generator = KimiChat(args.arch, self.tokenizer, args)
        elif 'gpt2' in args.arch or 'neo' in args.arch:  # gpt2, gpt-neo
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(f'{get_raw_data_path()}/LLMWeights/GPT2/models/gpt2tokenizer/')
            self.generator = GPT2Chat(args.arch, self.tokenizer, args)
        else:
            self.generator = GPT35Chat(args.generator_local_model_actor, args.arch, self.tokenizer, args, api_port=args.generator_api_port)

        if args.dataset in ['apps', 'codeforces']:
            self.executor = AppsExecutor(args)
        elif args.dataset == 'humaneval':
            self.executor = HumanevalExecutor(args)
        else:
            raise ValueError(f"Unknown dataset {args.dataset}")

        self.cached_reward = {}
        self.cur_prob_instance = None
        self.sample_times = []
        self.st = time.time()


    def generate(self, problem_instance):
        self.st = time.time()
        self.cur_prob_instance = problem_instance
        raw_prompt = problem_instance['prompt']
        initial_state = self.tokenizer.encode(raw_prompt)
        if len(initial_state) >= self.args.horizon:
            return None

        generated_codes = self.get_samples(initial_state, self.args.rollout)
        for code in generated_codes:
            code_id = self.tokenizer.encode(code)
            estimate = self.get_reward(code_id)
            self.sample_times.append(time.time() - self.st)

        complete_programs_ids = list(map(lambda x: list(x), self.cached_reward.keys()))
        if complete_programs_ids is None or len(complete_programs_ids) == 0:
            return None
        complete_programs = [self.convert_state_to_program(s) for s in complete_programs_ids]

        train_rewards = [self.cached_reward[tuple(s)] for s in complete_programs_ids]
        test_rewards = [self.get_reward(s, mode='test') for s in complete_programs_ids]
        best_idx = np.argmax(train_rewards)

        output_dict = {}
        output_dict['final_program'] = complete_programs[best_idx]
        output_dict['train_reward'] = train_rewards[best_idx]
        output_dict['test_reward'] = test_rewards[best_idx]

        output_dict['all_programs'] = complete_programs
        output_dict['all_train_rewards'] = train_rewards
        output_dict['all_test_rewards'] = test_rewards
        output_dict['avg_sample_time'] = np.mean(np.array(self.sample_times))

        self.cached_reward = {}
        self.generator.clean_cache()
        self.sample_times = []
        return output_dict

    def get_samples(self, cur_state, n_generate_samples, stop=None):
        with_instru_input_prompt = f"""
Complete the Python program to solve the problem. Remember to contain the complete program including all the imports and function header in your response.
Generate the code ONLY. No other explanation or words attached!
{self.tokenizer.decode(cur_state)}
"""
        outputs = []
        while n_generate_samples > 0:
            n = min(n_generate_samples, 5)
            n_generate_samples -= n
            messages, _ = self.generator.generate_response_api(with_instru_input_prompt,
                                                               top_k=None,
                                                               max_length=min(n * 1024, 10000),
                                                               temperature=0.7,
                                                               n=n,
                                                               stop=stop)
            if n == 1:
                outputs.extend([messages])
            else:
                outputs.extend(messages)
        return outputs

    def get_reward(self, s, mode='train'):
        if tuple(s) in self.cached_reward.keys() and mode == 'train':
            # cache rewards for training
            return self.cached_reward[tuple(s)]

        # 转换成文本
        output_str = self.convert_state_to_program(s)
        # 计算pass rate
        try:
            curr_res = self.executor.check_correctness(self.cur_prob_instance, output_str, mode)
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            # if not np.all(curr_res):
            #     print(f"Results were not all True: {curr_res}")
        except Exception as e:
            print(f"test framework exception = {repr(e)}{e}\n")
            curr_res = []
        # How to read results [-2] = compile error, [-1] = runtime error [False] = failed test case [True] = passed test case")
        assert isinstance(curr_res, list)
        pass_rate = np.mean(np.asarray(curr_res) > 0) if len(curr_res) > 0 else 0
        reward = pass_rate
        if mode == 'train':
            self.cached_reward[tuple(s)] = reward
        return reward

    def convert_state_to_program(self, s):
        s = self.tokenizer.decode(s)
        if "ANSWER:" in s:
            s = s.split("ANSWER:\n")[1]
        s = s.replace("<|endoftext|>", "")
        return s