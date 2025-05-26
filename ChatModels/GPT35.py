# -*- coding:utf-8 _*-
import random

import torch
from typing import List, Union, Optional, Literal
import dataclasses
import transformers
from utils import get_raw_data_path, extract_list_from_text
from openai import OpenAI
import openai
import tiktoken
import time
from .cache import GPTTopKCache, GPTSeqCache
import math
import re
import json
import os
from time import sleep
from vllm import LLM, SamplingParams
import numpy as np

def change_messages(tokenizer, messages, max_len):
    if isinstance(messages, str):
        message_lines = messages.split("\n")
        acc_msg_len = 0
        new_messages = ""
        for l in reversed(message_lines):
            acc_msg_len += len(tokenizer.encode(l, allowed_special={'<|endoftext|>'}))
            if acc_msg_len < max_len:
                new_messages = l + "\n" + new_messages
            else:
                break
        new_messages = new_messages.strip()
        return new_messages
    else:
        original_messages = messages
        new_messages = messages[:1]
        total_msg_len = len(tokenizer.encode(messages[0].content, allowed_special={'<|endoftext|>'}))
        rest_messages = []
        for msg in reversed(messages[1:]):
            msg_len = len(tokenizer.encode(msg.content, allowed_special={'<|endoftext|>'}))
            if msg_len + total_msg_len < max_len:
                rest_messages = [msg] + rest_messages
                total_msg_len += msg_len
            else:
                break
        messages = new_messages + rest_messages
    return messages


def prepare_prompt(messages):
    prompt = ""
    for i, message in enumerate(messages):
        if message.role == 'user':
            prompt += "User: "
        elif message.role == 'assistant':
            prompt += "Assistant: "
        prompt += message.content + "\n"
        if i == len(messages) - 1:
            prompt += "\n"
    return prompt

class GPT35Chat:
    def __init__(self, local_model_actor, model_name, tokenizer, args, save_mid_json=[], call_mode='api', api_port=0):
        if local_model_actor is not None:
            call_mode = 'local'
        self.name = model_name
        self.is_chat = True
        self.call_mode = call_mode
        if model_name in ['gpt3.5', 'kimi', 'gpt4', 'gpt4o-mini', 'gpt4o', 'o1-preview', 'o1-mini', 'gpt2', 'gpt-neo', ]:
            self.call_mode = 'api'
        self.args = args
        self.tokenizer = tokenizer
        self.device = args.device
        self.time_stamps = []
        self.ts_mode = args.ts_mode
        self.horizon = args.horizon
        self.client = OpenAI()

        if self.name in ['Qwen2.5-Coder-7B-Instruct', 'Qwen2.5-Coder-1.5B-Instruct', 'deepseek-coder-33b-instruct',
        'DeepSeek-Coder-V2-Lite-Instruct']:
            if self.call_mode == 'local':
                self.local_model_actor = local_model_actor
                self.local_model_tokenizer = local_model_actor.vllm_get_tokenizer()
                # self.terminal_token = self.llm_tokenizer.eos_token # actually eos token id is 32021
            else:
                self.client = OpenAI(
                    base_url=f"http://localhost:{api_port}/v1",
                    api_key="token-abc123"
                )
        self.log_prob_provided = False
        if self.args.model == 'MCTSToken':
            self.log_prob_provided = True
        if self.name == 'gpt3.5':
            self.model_name = 'gpt-3.5-turbo-0125'
        elif self.name == 'gpt4':
            self.model_name = 'gpt-4-turbo-2024-04-09'
        elif self.name == 'deepseek-reasoner':
            self.model_name = 'deepseek-reasoner'
        elif self.name == 'gpt4o-mini':
            self.model_name = 'gpt-4o-mini'
        elif self.name == 'gpt4o':
            self.model_name = 'gpt-4o'
        elif self.name == 'o1-preview':
            self.model_name = 'o1-preview'
        elif self.name == 'o1-mini':
            self.model_name = 'o1-mini'
        elif self.name == 'Qwen2.5-Coder-1.5B-Instruct':
            self.model_name = 'Qwen2.5-1.5B-Instruct'
        elif self.name == 'Qwen2.5-Coder-7B-Instruct':
            self.model_name = 'Qwen2.5-Coder-7B-Instruct'
        elif self.name == 'deepseek-coder-33b-instruct':
            self.model_name = 'deepseek-coder-33b-instruct'
        elif self.name == 'DeepSeek-Coder-V2-Lite-Instruct':
            self.model_name = 'DeepSeek-Coder-V2-Lite-Instruct'
        else:
            print(f'Model {self.name} not implemented error!')
            assert 0
        self.terminal_token = self.tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]
        self.width = args.width
        self.top_k_cache_steps = args.top_k_cache_steps
        self.top_k_cache = GPTTopKCache(args.width, cache_steps=args.top_k_cache_steps, tokenizer=tokenizer, args=self.args)
        self.seq_cache = GPTSeqCache(args=self.args)
        self.save_mid_json = save_mid_json

    def generate_chat(self, messages, stop, max_tokens: int = 1024, temperature: float = 0.0, num_comps: int = 1):
        for ti in range(10):  # try multiple times
            sleep_interval = 7
            if self.call_mode == 'api':
                try:
                    new_messages = change_messages(self.tokenizer, messages, 7000)
                    messages = new_messages
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[dataclasses.asdict(message) for message in messages],
                        temperature=temperature,
                        top_p=1,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        n=num_comps,
                        stop=stop
                    )
                except Exception as e:
                    print("GPT Error:", str(e))
                    if "context_length_exceeded" in str(e) or "context length" in str(e):
                        messages = change_messages(self.tokenizer, messages, 7000)
                        continue
                    else:
                        sleep_t = sleep_interval * (ti + 1)
                        print(f"get {ti +1}, error: {e}, sleep {sleep_t} seconds")
                        with open("error.log", "a") as f:
                            f.write(f"gpt failed multiple times with: {str(e)}\n")
                        sleep(sleep_t)
                        continue

                input_token_num = 0
                for msg in messages:
                    input_token_num += len(self.tokenizer.encode(msg.content, allowed_special={'<|endoftext|>'}))
                output_token_num = len(self.tokenizer.encode(response.choices[0].message.content, allowed_special={'<|endoftext|>'}))
                self.args.total_input_token_num += input_token_num
                self.args.total_output_token_num += output_token_num
                return response.choices[0].message.content  # type: ignore
            else:
                new_messages = change_messages(self.tokenizer, messages, 7000)
                messages = new_messages
                prompt = self.local_model_tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
                if self.name == 'deepseek-coder-33b-instruct':
                    stop_token_ids = [self.local_model_tokenizer.eos_token_id, 32021]  # 32021 is <EOT>
                else:
                    stop_token_ids = [self.local_model_tokenizer.eos_token_id]

                outputs = self.local_model_actor.vllm_call(
                    prompt=prompt,
                    sampling_params=SamplingParams(
                        n=num_comps,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_k=1,
                        stop_token_ids=stop_token_ids,
                    )
                )

                response_text = outputs[0].outputs[0].text

                input_token_num = len(self.tokenizer.encode(prompt, allowed_special={'<|endoftext|>'}))
                output_token_num = len(self.tokenizer.encode(response_text, allowed_special={'<|endoftext|>'}))
                self.args.total_input_token_num += input_token_num
                self.args.total_output_token_num += output_token_num
                return response_text

        else:
            print(f'try failure with multiple times')
            assert False


    def generate_response_api(self, prompt, top_k, max_length=1024, system_message=None, temperature=0.0, n=1, stop=None):
        sys_msg = "You are a helpful code generator that generate code to complete the given problem."
        if system_message:
            sys_msg = system_message
        for ti in range(20):
            sleep_interval = 7
            try:
                if self.call_mode == 'api':
                    if not self.log_prob_provided:
                        top_k = None
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}],
                        max_tokens=max_length,  # 调整生成文本的长度
                        temperature=temperature,
                        frequency_penalty=0,
                        presence_penalty=0,
                        logprobs=self.log_prob_provided,
                        top_logprobs=top_k,
                        n=n,
                        stop=stop
                    )
                    if n == 1:
                        message = response.choices[0].message.content
                        if self.log_prob_provided:
                            log_prob = response.choices[0].logprobs.content  # 是一个length等于top k的list，每个位置是一个list{token: .., logprob:.., bytes:..}
                        else:
                            log_prob = []
                    else:
                        message = [choice.message.content for choice in response.choices]
                        log_prob = []
                else:
                    messages = [{"role": "system", "content": f'{sys_msg}\n'},
                                {"role": "user", "content": prompt}]
                    prompt = self.local_model_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    if self.name == 'deepseek-coder-33b-instruct':
                        stop_token_ids = [self.local_model_tokenizer.eos_token_id, 32021] # 32021 is <EOT>
                    else:
                        stop_token_ids = [self.local_model_tokenizer.eos_token_id]
                    if stop:
                        stop_token_ids.extend(self.local_model_tokenizer.encode(stop))

                    outputs = self.local_model_actor.vllm_call(
                        prompt=prompt,
                        sampling_params=SamplingParams(
                            temperature=temperature,
                            max_tokens=max_length,
                            top_k=1,
                            stop_token_ids=stop_token_ids,
                            n=n
                        )
                    )
                    if n == 1:
                        message = outputs[0].outputs[0].text
                        log_prob = []
                    else:
                        message = [output.text for output in outputs[0].outputs]
                        log_prob = []

                input_token_num = len(self.tokenizer.encode(prompt, allowed_special={'<|endoftext|>'}))
                if isinstance(message, str):
                    output_token_num = len(self.tokenizer.encode(message, allowed_special={'<|endoftext|>'}))
                else:
                    output_token_num = 0
                    for msg in message:
                        output_token_num += len(self.tokenizer.encode(msg))
                self.args.total_input_token_num += input_token_num
                self.args.total_output_token_num += output_token_num
            except Exception as e:
                if "context_length_exceeded" in str(e) or "context length" in str(e):
                    prompt = self.tokenizer.decode(self.tokenizer.encode(prompt)[:7000])
                    print(f'GPT Error: context_length_exceeded, cut to 7000!')
                    continue
                else:
                    print("GPT Error:", str(e))
                    # if isinstance(message, list):  # 中转网站不稳定，采样16个会常常输出None，多试几次或许可以好
                    #     print(f"len message list: {len(message)}")
                    #     print(f"type message item: {type(message[0])}")
                    #     print(f"message: {message}")
                    sleep_t = sleep_interval * (ti + 1)
                    print(f"get {ti +1}, error: {e}, sleep {sleep_t} seconds")
                    with open("error.log", "a") as f:
                        f.write(f"gpt failed multiple times with: {str(e)}\n")
                    sleep(sleep_t)
                    continue
            return message, log_prob
        else:
            print(f'try failure with multiple times')
            assert False

    def generate_token_code_answer(self, state, top_k=3, max_length=1024, max_new_tokens=None, temperature=0.0):
        input_prompt = self.tokenizer.decode(state)

        with_instru_input_prompt = (f"Complete the Python program to solve the problem. Remember to contain the complete program including all the imports and function header in your response.\n"
                                    f"Generate the code ONLY. No other explanation or words attached!\n") + input_prompt

        # print('\n-----------------1')
        # print(with_instru_input_prompt)

        if max_new_tokens:
            max_length = max_new_tokens
        response_text, log_probs = self.generate_response_api(with_instru_input_prompt, top_k, max_length, temperature=temperature)

        # print('\n-----------------2')
        # print(response_text)

        # 这里整个gpt3.5生成的sequence用gpt2的tokenizer会出现token数量不一致,但是不能一个token一个token的encode，否则decode结果会不一样
        sequences = self.tokenizer.encode(response_text, allowed_special={'<|endoftext|>'})

        if len(log_probs) == 0:  # 之前已经生成了完整的程序,所以gpt判断不再需要token在后面
            log_probs = [{'<|endoftext|>': 1.0}]

        tmp_return = WithProbReturn(sequences=torch.tensor(sequences).unsqueeze(0),
                                    scores=log_probs,
                                    attentions=None,
                                    hidden_states=None,
                                    beam_indices=None)

        return tmp_return

    def get_token_predict_sequence(self, state, horizon=None, temperature=0.0):
        """
        Args:
            horizon: return a new sequence with this extra length
        Returns:
            Get the most likely sequence starting from state.
        """
        with torch.no_grad():
            # use_seq_cache:
            output_ids = self.seq_cache.get(state)
            if output_ids is not None:
                return output_ids

            model_output = self.generate_token_code_answer(
                state,
                top_k=self.width,
                max_length=1024,
                temperature=temperature
            )

            if self.top_k_cache_steps > 0:
                if hasattr(model_output, 'beam_indices'):
                    # beam search output
                    self.top_k_cache.add(state, model_output.sequences, model_output.scores, beam_indices=model_output.beam_indices)
                else:
                    self.top_k_cache.add(state, model_output.sequences, model_output.scores)

            output_ids_list = model_output.sequences.tolist()

            output_ids = output_ids_list[0]

            # use_seq_cache
            self.seq_cache.add(state, output_ids)

            self.time_stamps.append(time.time())
            return output_ids

    def get_top_k_token_predict(self, state):
        with torch.no_grad():
            if self.top_k_cache_steps > 0:
                top_k_info = self.top_k_cache.get(state)
                if top_k_info is not None:
                    return top_k_info

            model_output = self.generate_token_code_answer(
                state,
                top_k=self.width,
                max_new_tokens=1
            )

            if self.name not in ['gpt2', 'gpt-neo', 'kimi']:  # gpt3.5
                top_scores = []
                top_tokens = []
                for token_tops in model_output.scores:
                    top_scores.append([])
                    top_tokens.append([])
                    for token_probs in token_tops.top_logprobs:
                        top_scores[-1].append(math.exp(token_probs.logprob))
                        top_tokens[-1].append(self.tokenizer.encode(token_probs.token, allowed_special={'<|endoftext|>'})[0])
                return top_tokens[0], top_scores[0]
            else:
                raise ValueError('wrong arch!')

    def get_top_k_line_predict_2(self, state):
        with torch.no_grad():
            # 生成下面的line，以及line level的概率
            input_prompt = self.tokenizer.decode(state)

            with_instru_input_prompt = (f"{input_prompt} \n "
                                        f"Here is a problem to be solved by Python program. The program is now incomplete. \n"
                                        f"I need you predict the next line of the program, including line breaks and spaces. "
                                        f"For the next adapting search algorithms, i need you to output {self.width} possible next lines. Remember each only contain the next ONE line of the code, nothing else.\n"
                                        f"Note that do not rush to solve the problem in this one line, generate the next line is ok.\n"
                                        f"Please wrap your response into a JSON object that contains keys `line` with the name of each line, and key `possibility` with the possibility of each line. \n"
                                        f"Example Answers:\n")
            with_instru_input_prompt += """
[
    {"line":"    print('Hello World')", "possibility": 0.9},
    {"line":"    print('Hello')", "possibility": 0.05},
    {"line":"    print('Hi')", "possibility": 0.05}
]
"""

            # print('\n-----------------1')
            # print(with_instru_input_prompt)

            response_text, _ = self.generate_response_api(with_instru_input_prompt, top_k=1, max_length=2048, temperature=0.0)
            # print('\n-----------------2')
            # print(response_text)

            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]

            try:
                response_text = json.loads(response_text)
                top_scores = []
                top_lines = []
                top_lines_text = []
                for ele in response_text:
                    top_scores.append(ele['possibility'])
                    ele['line'] = '\n' + ele['line']
                    top_lines.append(self.tokenizer.encode(ele['line'], allowed_special={'<|endoftext|>'}))
                    top_lines_text.append(ele['line'])
            except Exception as e:
                top_lines = [self.tokenizer.encode('\n', allowed_special={'<|endoftext|>'}) for i in range(self.width)]
                top_scores = [1.0 for i in range(self.width)]

            return top_lines, top_scores

    def get_line_predict_sequence(self, state, horizon=None):
        with torch.no_grad():
            # use_seq_cache:
            output_ids = self.seq_cache.get(state)
            if output_ids is not None:
                return output_ids

            # generate code answer
            input_prompt = self.tokenizer.decode(state)
            with_instru_input_prompt = (f"Complete the Python program to solve the problem. Remember to contain the complete program including all the imports and function header in your response.\n"
                                        f"Response with the code ONLY. No other explanation or words attached!\n") + input_prompt
            response_text, _ = self.generate_response_api(with_instru_input_prompt, top_k=self.width, max_length=2048)
            sequences = self.tokenizer.encode(response_text, allowed_special={'<|endoftext|>'})

            model_output = WithProbReturn(sequences=torch.tensor(sequences).unsqueeze(0),
                                          scores=[],
                                          attentions=None,
                                          hidden_states=None,
                                          beam_indices=None)

            output_ids_list = model_output.sequences.tolist()
            output_ids = output_ids_list[0]

            # seq_cache and time_stamps
            self.seq_cache.add(state, output_ids)
            self.time_stamps.append(time.time())
            return output_ids

    def get_top_k_rationale_predict(self, state, with_verbal=False):
        with torch.no_grad():
            # 生成下面的line，以及line level的概率
            input_prompt = self.tokenizer.decode(state)
            # input_ids = torch.LongTensor(state).unsqueeze(0).to(self.device)

            if not with_verbal:
                tmp_instruct = f""
            else:
                tmp_instruct = f"\n* The goal is that the thoughts could lead to the code that not only avoids the current error but also solve the problem in a way that handles other potential test cases that we haven't encountered yet."

            with_instru_input_prompt = f"""
{input_prompt}
* I need you to analyze and provide new thoughts that can lead to the correct solution code. {tmp_instruct}
* If there are previous thoughts provided, please follow them and offer more detailed and further insights, as a detailed thinking or enhancement for previous ones.
* I need you to output {self.width} possible thoughts. Remember each only contain one possible distinct reasoning but all following previous thoughts if there are.
* Please wrap your response into a JSON object that contains keys `Thought-i` with i as the number of your thought, and key `Reasonableness` with the Reasonableness of each thought, which should between 0~1 and the sum should be 1.
* The JSON should be a **list of dicts**, the dicts are split with comma ','.

Example Answers:
[
    {{"Thought-1":" We could use the print function to finish the task in one line: print(2 + 3)", "Reasonableness": 0.7}},
    {{"Thought-2":" We should calculate the problem by setting a=2+3, and then print(a)", "Reasonableness": 0.29}},
    {{"Thought-3":" The problem can't be solved by Python.", "Reasonableness": 0.01}}
]
"""
            print('\n-----------------1')
            print(with_instru_input_prompt)

            response_text, _ = self.generate_response_api(with_instru_input_prompt, top_k=1, max_length=1024, temperature=0.0)
            print('\n-----------------2')
            print(response_text)


            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]

            try:
                self.args.all_json_num += 1
                # first try of extract thoughts:
                extracted_text = extract_list_from_text(response_text)
                if not extracted_text:
                    if response_text.strip()[0] != '[':
                        response_text = '[' + response_text + ']'
                else:
                    response_text = extracted_text

                response_text = json.loads(response_text)
                top_scores = []
                top_lines = []
                top_lines_text = []
                for i, ele in enumerate(response_text):
                    top_scores.append(ele['Reasonableness'])
                    ele[f'Thought-{i + 1}'] = '\nThought: ' + ele[f'Thought-{i + 1}']
                    top_lines.append(self.tokenizer.encode(ele[f'Thought-{i + 1}'], allowed_special={'<|endoftext|>'}))
                    top_lines_text.append(ele[f'Thought-{i + 1}'])
            except Exception as e:
                self.args.failed_json_num += 1
                top_lines, top_scores = self.get_top_k_rationale_predict_sample(state, with_verbal=with_verbal, temperature=0.7)

            return top_lines, top_scores

    def get_top_k_rationale_predict_sample(self, state, with_verbal=False, temperature=0.7):
        # 生成下面的line，以及line level的概率
        input_prompt = self.tokenizer.decode(state)
        if not with_verbal:
            tmp_instruct = f""
        else:
            tmp_instruct = f"\n* The goal is that the thoughts could lead to the code that not only avoids the current error but also solve the problem in a way that handles other potential test cases that we haven't encountered yet."

        with_instru_input_prompt = f"""
{input_prompt}
* I need you to analyze and provide a new thought that can lead to the correct solution code. {tmp_instruct}
* If there are previous thoughts provided, please follow them and offer one more detailed and further insight, as a detailed thinking or enhancement for previous ones.
* Please wrap your response into a JSON object that contains the key `Thought` as the thought key, and key `Reasonableness` with the Reasonableness of the thought, which should between 0~1.
Example Answer:
{{"Thought": " We could use the print function to finish the task in one line: print(2 + 3)", "Reasonableness": 0.7}},
    """
        outputs = []
        messages, _ = self.generate_response_api(with_instru_input_prompt, top_k=None, max_length=min(self.width * 1024, 10000), temperature=temperature, n=self.width)

        if self.width == 1:
            outputs.extend([messages])
        else:
            outputs.extend(messages)

        try:
            top_scores = []
            top_thoughts = []
            top_thought_text = []
            for i, ele in enumerate(outputs):
                # 提取代码并确保代码被```python包裹
                if '```json' in ele:
                    ele = ele.split('```json')[1].split('```')[0]
                response_dict = json.loads(ele)
                top_scores.append(response_dict['Reasonableness'])
                top_thoughts.append(self.tokenizer.encode(response_dict['Thought']))
                top_thought_text.append(response_dict['Thought'])
        except Exception as e:
            top_thoughts = [self.tokenizer.encode('skip one thought step') for _ in range(self.width)]
            top_scores = [1.0 for _ in range(self.width)]

        return top_thoughts, top_scores


    def get_rationale_predicted_sequence(self, state, temperature=0.0):
        with torch.no_grad():
            # use_seq_cache:
            output_ids = self.seq_cache.get(state)
            # input_ids = torch.LongTensor(state).unsqueeze(0).to(self.device)
            if output_ids is not None:
                return output_ids
            input_prompt = self.tokenizer.decode(state)

            with_instru_input_prompt = (f"Complete the Python program to solve the problem. Remember to contain the complete program including all the imports and function header in your response.\n"
                                        f"Also some thoughts are included that you can refer to and build upon when writing the code. "
                                        f"Answer with the code ONLY. No other explanation or words attached!\n") + input_prompt

            print('\n-----------------3')
            print(with_instru_input_prompt)

            response_text, _ = self.generate_response_api(with_instru_input_prompt,
                                                                  top_k=1,
                                                                  max_length=1024,
                                                                  temperature=temperature)

            print('\n-----------------4')
            print(response_text)

            # 这里整个gpt3.5生成的sequence用gpt2的tokenizer会出现token数量不一致,但是不能一个token一个token的encode，否则decode结果会不一样
            output_ids = self.tokenizer.encode(response_text, allowed_special={'<|endoftext|>'})
            # use_seq_cache
            self.seq_cache.add(state, output_ids)

            self.time_stamps.append(time.time())
            return output_ids

    def get_top_k_codes_predict(self, state, with_verbal=False):
        with torch.no_grad():
            # 生成下面的line，以及line level的概率
            input_prompt = self.tokenizer.decode(state)
            with_instru_input_prompt = (f"{input_prompt} \n\n"
                                        f"Above is your previous implementation of a function, a series of unit tests results, and your self-reflection on your previous implementation. \n"
                                        f"I need you analyze this problem and provide a solution code."
                                        f"Please wrap each solution into ```python ... ``` format\n"
                                        f"The solution should contain the complete program including all the imports and function header in your response.\n"
                                        f"Generate the code ONLY. No other explanation or words attached!\n"
                                        f"Example Answer:\n")
            with_instru_input_prompt += """
```python
def add(a, b):
return a + b
```
"""
            outputs = []
            messages, _ = self.generate_response_api(with_instru_input_prompt, top_k=None, max_length=min(self.width * 1024, 10000), temperature=0.7, n=self.width)

            if self.width == 1:
                outputs.extend([messages])
            else:
                outputs.extend(messages)

            try:
                top_scores = []
                top_codes = []
                top_code_text = []
                for i, ele in enumerate(outputs):
                    # 提取代码并确保代码被```python包裹
                    if '```python' in ele:
                        output = ele.split('```python')[1].split('```')[0]
                    output = '```python\n' + output + '\n```'
                    top_scores.append(1.0)
                    top_codes.append(self.tokenizer.encode(output))
                    top_code_text.append(output)
            except Exception as e:
                top_codes = [self.tokenizer.encode('\n', allowed_special={'<|endoftext|>'}) for i in range(self.width)]
                top_scores = [1.0 for i in range(self.width)]

            return top_codes, top_scores

    def get_code_predicted_sequence(self, state, horizon=None):
        trajectory = self.tokenizer.decode(state)
        code_text = extract_python_code(trajectory)[-1]

        print('\n-----------------3 extract code input')
        print(trajectory)

        print('\n-----------------4 extract code output')
        print(code_text)

        # 这里整个gpt3.5生成的sequence用gpt2的tokenizer会出现token数量不一致,但是不能一个token一个token的encode，否则decode结果会不一样
        sequences = self.tokenizer.encode(code_text, allowed_special={'<|endoftext|>'})
        self.time_stamps.append(time.time())
        return sequences

    def clean_cache(self):
        self.top_k_cache = GPTTopKCache(self.args.width, cache_steps=self.args.top_k_cache_steps, tokenizer=self.tokenizer, args=self.args)
        self.seq_cache = GPTSeqCache(args=self.args)
        self.time_stamps = []


class WithProbReturn:
    def __init__(self, sequences, scores, attentions, hidden_states, beam_indices=None, top_tokens=None):
        self.sequences = sequences
        self.scores = scores
        self.attentions = attentions
        self.hidden_states = hidden_states
        self.beam_indices = beam_indices
        self.top_tokens = top_tokens


def extract_python_code(text):
    pattern = r'```python(.*?)```'
    code_blocks = re.findall(pattern, text, re.DOTALL)
    for i in range(len(code_blocks)):
        code_blocks[i] = code_blocks[i].strip()
        code_blocks[i] = '\n```python \n' + code_blocks[i] + '\n```\n'
    return code_blocks