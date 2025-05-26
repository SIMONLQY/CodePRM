# -*- coding:utf-8 _*-
import torch
import transformers
from utils import get_raw_data_path
from openai import OpenAI
import openai
import tiktoken
import time
from .cache import GPTTopKCache, GPTSeqCache


class KimiChat:
    def __init__(self, model_name, tokenizer, args):
        self.name = model_name
        self.is_chat = True
        self.args = args
        self.tokenizer = tokenizer
        self.device = args.device
        self.time_stamps = []
        self.ts_mode = args.ts_mode
        self.horizon = args.horizon
        self.client = OpenAI(
            api_key="xxx",
            base_url="https://api.moonshot.cn/v1",
        )
        self.terminal_token = self.tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]
        self.width = args.width
        self.top_k_cache_steps = args.top_k_cache_steps
        self.top_k_cache = GPTTopKCache(args.width, cache_steps=args.top_k_cache_steps, tokenizer=tokenizer, args=self.args)
        self.seq_cache = GPTSeqCache(args=self.args)
        self.one_min_request_count = 0

    def generate_response_api(self, prompt, top_k, max_length=1024, system_message=None):
        sys_msg = "You are a helpful code generator that generate code to complete the given problem."

        if system_message:
            sys_msg = system_message

        try:
            response = self.client.chat.completions.create(
                model='moonshot-v1-8k',
                messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}],
                max_tokens=max_length,  # 调整生成文本的长度
                temperature=0.0,
                # top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                logprobs=True,
                top_logprobs=top_k
            )
            message = response.choices[0].message.content.strip()
            self.one_min_request_count += 1
            print('-------------')
            print(f"kimi requestion one minute nums: {self.one_min_request_count}")
            if self.one_min_request_count == 2:
                time.sleep(60)
                self.one_min_request_count = 0
        except Exception as e:
            print("GPT Error:", str(e))
            assert False, "GPT API error: " + str(e)
        return message, None

    def generate_code_answer(self, input_ids, top_k=3, max_length=1024, max_new_tokens=None):
        input_prompt = self.tokenizer.decode(input_ids[0].tolist())

        with_instru_input_prompt = (f"Complete the Python program to solve the problem. Remember to contain the complete program including all the imports and function header in your response.\n"
                                    f"Generate the code ONLY. No other explanation or words attached! \n") + input_prompt

        if max_new_tokens:
            max_length = max_new_tokens
        response_text, log_probs = self.generate_response_api(with_instru_input_prompt, top_k, max_length)

        # 这里整个gpt3.5生成的sequence用gpt2的tokenizer会出现token数量不一致,但是不能一个token一个token的encode，否则decode结果会不一样
        sequences = self.tokenizer.encode(response_text, allowed_special={'<|endoftext|>'})

        tmp_return = WithProbReturn(sequences=torch.tensor(sequences).unsqueeze(0).to(self.device),
                                    scores=log_probs,
                                    attentions=None,
                                    hidden_states=None,
                                    beam_indices=None)

        return tmp_return

    def get_token_predict_sequence(self, state, horizon=None):
        """
        Args:
            horizon: return a new sequence with this extra length
        Returns:
            Get the most likely sequence starting from state.
        """
        with torch.no_grad():
            encoded_ids = state  # as a list
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)

            # use_seq_cache:
            output_ids = self.seq_cache.get(encoded_ids)
            if output_ids is not None:
                return output_ids

            model_output = self.generate_code_answer(
                input_ids,
                top_k=self.width,
                max_length=1024,
            )

            output_ids_list = model_output.sequences.tolist()

            output_ids = output_ids_list[0]

            # use_seq_cache
            self.seq_cache.add(encoded_ids, output_ids)

            self.time_stamps.append(time.time())
            return output_ids

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
