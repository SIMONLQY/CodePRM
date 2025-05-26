# -*- coding:utf-8 _*-
from torch.utils.data import DataLoader, SequentialSampler
from utils import *
import time
import itertools
from models import *
from torcheval.metrics import HitRate, ReciprocalRank
import torchmetrics
from dataSet import *
from tqdm import tqdm
from math import sqrt, log
from executors import AppsExecutor, HumanevalExecutor
from ChatModels import GPT35Chat, GPT2Chat
import re
import json
import math
from prm import *


class BestOfNStep:
    def __init__(self, args):
        self.args = args
        self.sample_nums = 0
        self.gamma = 0.9
        self.save_mid_json = []
        self.stops = ['Code:', None]

        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = '\n\n\n\n\n'

        # generator模型
        if 'gpt2' in args.arch or 'neo' in args.arch:  # gpt2, gpt-neo
            raise ValueError("MCTSAg only supports GPT3.5")
        else:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.generator = GPT35Chat(args.generator_local_model_actor, args.arch, self.tokenizer, args, self.save_mid_json, api_port=args.generator_api_port)
        # 数据集
        if args.dataset in ['apps', 'codeforces']:
            self.executor = AppsExecutor(args)
        elif args.dataset == 'humaneval':
            self.executor = HumanevalExecutor(args)
        else:
            raise ValueError(f"Dataset {args.dataset} is not supported.")
        # prm模型
        if self.args.prm_model_name != 'gpt4o-mini' and 'ft' not in self.args.prm_model_name:
            self.llm_verifier = GPT35Chat(args.prm_local_model_actor, args.prm_model_name, self.tokenizer, args, call_mode='api', api_port=args.prm_api_port)
        elif 'ft' in self.args.prm_model_name:
            if 'value' in self.args.prm_model_name:
                self.llm_verifier = ValuePRM(args)
            else:
                self.llm_verifier = TokenPRM(args)
        else:
            self.llm_verifier = self.generator

        self.term_cond = lambda: self.sample_nums > args.max_sample_times

        self.search_depth = 4
        self.root = None
        self.cached_reward = {}
        self.cached_value = {}
        self.cached_verbal_feedback = {}
        self.cur_prob_instance = None
        self.sample_times = []
        self.st = time.time()

    def generate(self, problem_instance):
        self.st = time.time()
        self.cur_prob_instance = problem_instance
        raw_prompt = problem_instance['prompt']
        done = False
        initial_state = self.tokenizer.encode(raw_prompt)
        if len(initial_state) >= self.args.horizon:
            return None

        self.best_of_n_procedure(initial_state, done)

        if len(self.cached_value) == 0:
            state = self.generator.get_rationale_predicted_sequence(initial_state)
            complete_prog_score = self.get_reward(state)

        complete_programs_ids = list(map(lambda x: list(x), self.cached_value.keys()))
        if complete_programs_ids is None or len(complete_programs_ids) == 0:
            return None
        complete_programs = [self.convert_state_to_program(s) for s in complete_programs_ids]

        train_rewards = [self.cached_value[tuple(s)] for s in complete_programs_ids]
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
        self.cached_value = {}
        self.cached_verbal_feedback = {}
        self.generator.clean_cache()
        self.sample_nums = 0
        self.save_mid_json = []

        return output_dict

    def best_of_n_procedure(self, initial_state, done):
        ys = [MineState(self.cur_prob_instance['prompt'],
                        self.cur_prob_instance['prompt'],
                        [], [], '', '', 0.0)]
        # 如果rollouts=1，产生的程序存在cached_rewards里面的只有一个完整程序，其实select那一步就已经选了走哪个完整程序了
        print("Performing rollouts.")
        for step in range(self.search_depth):
            if step == 0:
                n_gen = self.args.rollout
            else:
                n_gen = 1

            new_ys = []
            for y in ys:
                new_ys.extend(self.get_samples(y,
                                               n_generate_samples=n_gen,
                                               step=step))

            ids = [id for id in range(len(new_ys))]
            assert len(ids) == self.args.rollout
            ys = new_ys[:]

        assert len(new_ys) == self.args.rollout
        ids = [i for i in range(len(new_ys))]
        values = []
        codes = []
        for s in new_ys:
            code, complete_prog_score = self.get_sample_code_score(s)
            trace_value = self.get_evaluations(s)
            codes.append(code)
            values.append(trace_value)

        # 对于n条里面最终代码出错的trace，重新思考
        for i in range(self.args.max_think_times - 1):
            for j, ys in enumerate(new_ys):
                if ys.complete_program_score < 1.0:
                    wrong_id = self.locate_wrong_thought(ys)
                    self.refine_thoughts(ys, wrong_id)
                    _, __ = self.get_sample_code_score(ys)
                    new_value = self.get_evaluations(ys)
                    values[j] = new_value
                    codes[j] = ys.code

                    code_ids = self.tokenizer.encode(ys.code)
                    if tuple(code_ids) not in self.cached_value.keys():
                        self.cached_value[tuple(code_ids)] = new_value
                else:
                    continue

        # value assignment
        for i, code in enumerate(codes):
            code_ids = self.tokenizer.encode(code)
            if tuple(code_ids) not in self.cached_value.keys():
                self.cached_value[tuple(code_ids)] = values[i]

    def refine_thoughts(self, s, wrong_id):
        if isinstance(wrong_id, list):
            to_refine_ids = wrong_id
        else:
            to_refine_ids = [wrong_id]

        for refine_id in to_refine_ids:
            # 将错误的thought进行修正
            system_msg = f"You are an expert programmer."
            input_prompt = f"""
{s.generator_state}\n{s.code}\n{s.pure_feedback}
Above is the combination of problem, thoughts and code. 
Each thought is bounded with an id number at the beginning of the thought.
* Revise and enhance the {refine_id + 1}-Thought above in the thoughts by providing a improved new thought to replace it.
* Remember that you only need to provide the new thought in one or two sentences, not the code.
            """
            print('\n-----------14renew action input prompt')
            print(input_prompt)
            response, _ = self.generator.generate_response_api(input_prompt, top_k=1, max_length=1024,
                                                          system_message=system_msg)
            if 'Thought:' not in response:
                response = f'{refine_id + 1}-Thought: ' + response


            print('\n-----------15renew action response')
            print(response)

            s.generator_state = s.generator_state.replace(s.thought_steps_list[refine_id], response)
            s.verifier_state = s.verifier_state.replace(s.thought_steps_list[refine_id], response)
            s.thought_steps_list[refine_id] = response

    def locate_wrong_thought(self, final_state):
        # 定位错误thought
        if self.args.verify_pos_mode == 'llm':
            # 让LLM判断几号出错了
            wrong_id = self.llm_locate_wrong_thought(final_state)
        elif self.args.verify_pos_mode == 'random':
            wrong_id = np.random.choice(len(final_state.thought_steps_list))
        elif self.args.verify_pos_mode == 'last':
            wrong_id = len(final_state.thought_steps_list) - 1
        elif self.args.verify_pos_mode in ['llm_score', 'llm_score_thresh'] :
            # 获得step scores
            if final_state.thought_steps_scores_list:  # 说明可以在evaluation那一步获得step score，这里直接用，用完在主代码中就会更新
                step_scores = final_state.thought_steps_scores_list
            else:
                if 'ft' not in self.args.prm_model_name:
                    # each state list
                    state_list = []
                    tmp_state = self.cur_prob_instance['prompt']
                    for thought in final_state.thought_steps_list:
                        tmp_state = tmp_state + '\n' + thought
                        state_list.append(tmp_state)

                    # 让LLM一次性给所有thought打分
                    step_scores = self.llm_score_thought(state_list[-1], final_state.code, final_state.pure_feedback)
                    if len(step_scores) >= len(final_state.thought_steps_list):
                        step_scores = step_scores[:len(final_state.thought_steps_list)]
                    else:
                        step_scores = [np.random.rand() for _ in range(len(final_state.thought_steps_list))]

                    # 分别给每个state打分
                    # step_scores = []
                    # for i, each_state in enumerate(state_list):
                    #     score = self.llm_score_thought(each_state, final_state.code, final_state.pure_feedback)
                    #     step_scores.append(score)
                else:
                    _, step_scores = self.llm_verifier.process_judge(self.cur_prob_instance['prompt'],
                                                                    final_state.thought_steps_list,
                                                                    code=final_state.code,
                                                                    feedback=final_state.pure_feedback)
            # 设置wrong_id 为得分最低的 / 得分低于某个threshold的
            if self.args.verify_pos_mode == 'llm_score_thresh':
                wrong_id = [i for i, score in enumerate(step_scores) if score < self.args.prm_verify_thresh]
                if len(wrong_id) == 0:
                    wrong_id = [step_scores.index(min(step_scores))]
            else:
                wrong_id = step_scores.index(min(step_scores))
        else:
            raise ValueError("Unknown verify_pos_mode")
        return wrong_id

    def llm_score_thought(self, state, code, pure_feedback):
        system_msg = f"You are an expert programmer."
        if self.args.with_execution_enhance:
            input_info = f"{state}\n{code}\n{pure_feedback}\nAbove is the combination of problem, thoughts and corresponding code and feedback."
        else:
            input_info = f"{state}\nAbove is a problem and the thoughts to solve it."
        input_prompt = f"""
{input_info}
Each thought is bounded with an id number at the beginning of the thought.

* Please analysis and judge the correctness of each thought.
* Return the correctness scores in range [0, 1]. 
* Evaluate the correctness of EACH thought and give all the evaluation scores. 
* Please wrap your response into a JSON object in ```json ... ```, which contains keys `i-Thought` with i as the number of your thought, and value is between 0~1 as the score of the thought.
* The JSON should be a **dict**, the dict items are split with comma ',' and the object should be including in {{}}.
* The score should be between 0-1. For example, 
    score 0.1 means the thought provides a wrong strategy or algorithm for the problem that would lead to the wrong code.
    score 0.3 means the thought is erroneous for not paying attention to the dict writing problem.
    score 0.8 means the thought is correct but miss some tip for implementation.
    score 1.0 means the thought is totally correct.
* Reply with the JSON object only, with no other words or explanation attached.

Example Answer:
```json
{{
    "1-Thought": 0.3,
    "2-Thought": 0.8,
    "3-Thought": 1.0,
    "4-Thought": 0.5,
    ...
}}
```
                """
        print('\n--------------9 score thought input prompt')
        print(input_prompt)

        response, _ = self.llm_verifier.generate_response_api(input_prompt, top_k=1, max_length=1024,
                                                              system_message=system_msg)

        print('\n--------------10 score thought response')
        print(response)

        try:
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            response = json.loads(response)
            step_scores = []
            for key, value in response.items():
                step_scores.append(float(value))
        except Exception as e1:
            try:
                float_pattern = re.compile(r'-?\d+\.?\d*')
                initial_scores = [float(match) for match in float_pattern.findall(response)]
                step_scores = [score for i, score in enumerate(initial_scores) if i % 2 == 1]
            except Exception as e2:
                step_scores = []

        return step_scores

    def llm_locate_wrong_thought(self, final_state):
        thoughts = final_state.thought_steps_list
        code = final_state.code
        pure_feedback = final_state.pure_feedback
        state = final_state.generator_state

        system_msg = f"You are an expert programmer."
        input_prompt = f"""
Below is the combination of problem, thoughts, code and execution feedback.
Each thought is bounded with an id number at the beginning of the thought.
* Please analysis and find which thought is flawed that lead to the wrong code.
* Remember that you only need to find the wrong thought, not the code. 
* Please notice that you should analysis the thoughts, not the execution analysis which also have ids.
Example Answers:
{{"problematic id": 2,  "explanation": "[Your explanation of why the second thought is wrong]"}} 
{{"problematic id": 1,  "explanation": "[Your explanation of why the first thought is wrong]"}} 
{{"problematic id": 3,  "explanation": "[Your explanation of why the third thought is wrong]"}} \n

Here are the thoughts and code:
{state}\n{code}\n{pure_feedback}\n

Begin your response:
        """

        print('\n--------------9 find wrong thought input prompt')
        print(input_prompt)

        response, _ = self.generator.generate_response_api(input_prompt, top_k=1, max_length=1024,
                                                           system_message=system_msg)

        print('\n--------------10 find wrong thought response')
        print(response)

        try:
            float_pattern = re.compile(r'-?\d+\.?\d*')
            response_id = [float(match) for match in float_pattern.findall(response)]
            wrong_id = int(response_id[0]) - 1
        except Exception as e:
            print(f"Error in parsing evaluation response: {repr(e)}{e}")
            wrong_id = len(thoughts) - 1  # 修改最近的一个
        if wrong_id <= 0 or wrong_id > len(thoughts):
            wrong_id = len(thoughts) - 1
        return wrong_id

    def convert_state_to_program(self, s):
        s = self.tokenizer.decode(s)
        if "ANSWER:" in s:
            s = s.split("ANSWER:\n")[1]
        s = s.replace("<|endoftext|>", "")
        return s

    def get_reward(self, s, mode='train', with_verbal=False):
        if tuple(s) in self.cached_reward.keys() and mode == 'train':
            # cache rewards for training
            if with_verbal:
                return [self.cached_reward[tuple(s)], self.cached_verbal_feedback[tuple(s)]]
            else:
                return self.cached_reward[tuple(s)]

        # 转换成文本
        output_str = self.convert_state_to_program(s)

        # 计算pass rate
        try:
            curr_res = self.executor.check_correctness(self.cur_prob_instance, output_str, mode, with_verbal=with_verbal)  # with_verbal: curr_res=[[True/False, feedback_dict]]
            fixed = []
            verbal_feedbacks = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                if with_verbal:
                    verbal_feedbacks.append(e[1])
                    e = e[0]
                fixed.append(e)

            curr_res = fixed
            # if not np.all(curr_res):
            #     print(f"Results were not all True: {curr_res}")
        except Exception as e:
            print(f"test framework exception = {repr(e)}{e}\n")
            with open("error.log", "a") as f:
                f.write(f"test framework exception = {repr(e)}{e}\n")
            if with_verbal:
                feedback_dict = {
                    'error': '',
                    'output': f"# Input:\nUnknown\n# Ground Truth Output:\n:\nUnknown\n\n# Current Execution Output: \nThe code executes failed! No result got!"
                }
                curr_res = [False]
                verbal_feedbacks = [feedback_dict]
            else:
                curr_res = [-1]

        # How to read results [-2] = compile error, [-1] = runtime error [False] = failed test case [True] = passed test case")
        assert isinstance(curr_res, list)
        pass_rate = np.mean(np.asarray(curr_res) > 0) if len(curr_res) > 0 else 0
        reward = pass_rate

        # 添加到cached reward
        if mode == 'train':
            self.cached_reward[tuple(s)] = reward
            if with_verbal:
                self.cached_verbal_feedback[tuple(s)] = verbal_feedbacks

        if with_verbal:
            return [reward, verbal_feedbacks]
        else:
            return reward

    def transition(self, s, a):
        if isinstance(a, list):
            next_state = s + a
        else:
            next_state = s + [a]
        if self.generator.terminal_token in a or len(next_state) == self.args.horizon:
            # either the program finishes, or the state reaches the maximum length
            done = True
        else:
            done = False

        if done:
            reward = self.get_reward(next_state)
            if tuple(next_state) not in self.cached_value.keys():
                self.cached_value[tuple(next_state)] = reward
        else:
            reward = 0  # no intermediate reward
        return next_state, reward, done


    def get_samples(self, y, n_generate_samples, stop=None, step=0):
        prompt = f"""
{y.generator_state}
* Above is a problem to be solved by Python program.
* I need you to analyze this problem and provide reasoning and thoughts.
* If no previous thoughts provided, please provide your one thought step on how to solve the problem in one or two sentences.
* If there are previous thoughts provided, please follow them and offer ONE more further thought step in one or two sentences, as a detailed thinking or enhancement for previous ones.
* Remember to output ONE reasoning step but following previous thoughts if there are.
* Your response should be in 1~2 sentences, not too long.
        """
        outputs = []
        while n_generate_samples > 0:
            n = min(n_generate_samples, 20)
            n_generate_samples -= n
            messages, _ = self.generator.generate_response_api(prompt,
                                                               top_k=None,
                                                               max_length=min(n * 1024, 10000),
                                                               temperature=0.7,
                                                               n=n,
                                                               stop=stop)
            if n ==1:
                outputs.extend([messages])
            else:
                outputs.extend(messages)
        return [MineState(generator_state=y.generator_state + '\n' + f'{len(y.thought_steps_list) + 1}-Thought:' + response,
                          verifier_state=y.verifier_state + f'{len(y.thought_steps_list) + 1}-Thought:' + response + self.step_tag,
                          thought_steps_list=y.thought_steps_list + [f'{len(y.thought_steps_list) + 1}-Thought:' + response],
                          thought_steps_scores_list=y.thought_steps_scores_list[:]) for response in outputs]

    def get_sample_code_score(self, s):
        code_id = self.generator.get_rationale_predicted_sequence(self.tokenizer.encode(s.generator_state))
        code = self.tokenizer.decode(code_id)
        full_result = self.get_reward(code_id, with_verbal=True)
        complete_prog_score, verbal_feedbacks = full_result[0], full_result[1]
        failed_tests, verbalFeedback = self.gather_feedback(verbal_feedbacks,
                                                            self.tokenizer.encode(
                                                                f"{self.cur_prob_instance['prompt']}\n" + '\n'.join(s.thought_steps_list) + '\n'),
                                                            code)
        pure_feedback = failed_tests.strip()
        if pure_feedback == '':
            pure_feedback = 'All public test cases passed!'
        s.code = code
        s.pure_feedback = pure_feedback
        s.complete_program_score = complete_prog_score
        return code, complete_prog_score

    def get_evaluations(self, s):
        question_text = self.cur_prob_instance['prompt']
        thought_steps = s.thought_steps_list
        code = s.code
        code_score = s.complete_program_score
        pure_feedback = s.pure_feedback

        # value got
        if self.args.reward_mode == 'pub_test':
            value = code_score
        elif self.args.reward_mode == 'all_test':
            private_test_full_result = self.get_reward(self.tokenizer.encode(code), mode='test', with_verbal=True)
            private_score = private_test_full_result[0]
            value = 0.5 * code_score + 0.5 * private_score
        elif self.args.reward_mode == 'random':
            value = np.random.rand()
        elif self.args.reward_mode in ['pub_prm', 'prm']:
            if 'ft' not in self.args.prm_model_name:
                # each state list
                state_list = []
                tmp_state = question_text
                for thought in thought_steps:
                    tmp_state = tmp_state + '\n' + thought
                    state_list.append(tmp_state)

                # 让LLM一次性给所有thought打分
                thought_score_list = self.llm_score_thought(state_list[-1], code, pure_feedback)
                if len(thought_score_list) >= len(thought_steps):
                    thought_score_list = thought_score_list[:len(thought_steps)]
                else:
                    thought_score_list = [np.random.rand() for _ in range(len(thought_steps))]

                # # 让LLM给每个state打分
                # thought_score_list = []
                # for i, each_state in enumerate(state_list):
                #     score = self.llm_score_thought(each_state, code, pure_feedback)
                #     thought_score_list.append(score)
            else:
                _, thought_score_list = self.llm_verifier.process_judge(question_text, thought_steps, code, pure_feedback)

            s.thought_steps_scores_list = thought_score_list[:]
            prm_score = agg_prm_scores(self.args.prm_agg_mode, thought_score_list)
            if self.args.reward_mode == 'pub_prm':
                if code_score < 1.0:
                    value = code_score
                else:
                    value = 1.0 * code_score + 1.0 * prm_score
            elif self.args.reward_mode == 'prm':
                value = prm_score
            else:
                raise ValueError("Invalid reward mode for prm")
        elif self.args.reward_mode == 'pub_llm':
            raise ValueError("pub_llm mode is not supported for BestofN")
        return value

    def gather_feedback(self, verbal_feedbacks, state, code):
        failed_tests = ''
        verbalFeedback = ''
        failed_test_list = []

        # 删除code_exe_data['execution_feedback']中不是dict的那些元素
        false_feedback = [item for item in verbal_feedbacks if isinstance(item, dict)]
        ori_length = len(false_feedback)
        false_feedback = [
            {
                'error': shorten_text_no_tokenizer(item['error'], total_length=10000),
                'output': shorten_text_no_tokenizer(item['output'], total_length=10000)
            }
            for item in false_feedback
        ]
        tmp_feedback = [item for item in false_feedback if len(self.tokenizer.encode(item['output'])) < 512]
        tmp_feedback = tmp_feedback[:5]
        # 5个都很长，就只保留第一个
        if len(tmp_feedback) == 0 and ori_length > 0:
            false_feedback = false_feedback[:5]
            tmp_feedback = [
                {
                    'error': item['error'],
                    'output': shorten_text_block(item['output'], self.tokenizer, total_length=512,
                                                 section_length=512)
                }
                for item in false_feedback
            ]
        false_feedback = tmp_feedback

        for k, feedback in enumerate(false_feedback):
            failed_tests += f"\n\n## Failed test {k + 1}: \n{feedback['output']}"
            failed_test_list.append(feedback['output'])

        if failed_tests != '':
            verbalFeedback = f"""
{self.tokenizer.decode(state)}\n\n{code}
Above is the combination of problem, thoughts and code. The code is generated following the thoughts to solve the problem.
However, the code generated following the thoughts doesn't pass some test cases. Here are the test cases the code doesn't pass:
{failed_tests}\n
"""
            self.args.verbal_length_check_num += 1
            if len(self.tokenizer.encode(verbalFeedback, allowed_special={'<|endoftext|>'})) > 12000:
                self.args.verbal_length_exd_num += 1
                tmp_shorter = self.tokenizer.encode(verbalFeedback, allowed_special={'<|endoftext|>'})[:8000]
                verbalFeedback = self.tokenizer.decode(tmp_shorter)

        return failed_tests, verbalFeedback

class MineState(object):
    def __init__(self,
                 generator_state='',
                 verifier_state='',
                 thought_steps_list=[],
                 thought_steps_scores_list=[],
                 code='',
                 pure_feedback='',
                 complete_program_score=0.0):
        self.generator_state = generator_state
        self.verifier_state = verifier_state
        self.thought_steps_list = thought_steps_list
        self.thought_steps_scores_list = thought_steps_scores_list
        self.code = code
        self.pure_feedback = pure_feedback
        self.complete_program_score = complete_program_score

