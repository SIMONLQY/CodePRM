# -*- coding:utf-8 _*-
import torch
import transformers
from utils import get_raw_data_path
from openai import OpenAI
import openai
import tiktoken
import time
from .cache import GPTTopKCache, GPTSeqCache


class GPT2Chat:
    def __init__(self, model_name, tokenizer, args):
        self.name = model_name
        self.is_chat = True
        self.args = args
        self.tokenizer = tokenizer
        self.device = args.device
        self.time_stamps = []
        self.ts_mode = args.ts_mode
        self.horizon = args.horizon
        self.client = OpenAI()
        self.terminal_token = self.tokenizer.encode('<|endoftext|>')[0]
        self.width = args.width
        self.top_k_cache_steps = args.top_k_cache_steps
        self.top_k_cache = GPTTopKCache(args.width, cache_steps=args.top_k_cache_steps, tokenizer=tokenizer, args=self.args)
        self.seq_cache = GPTSeqCache(args=self.args)
        load_model_dir = f"{get_raw_data_path()}/LLMWeights/GPT2/models/1.5B"
        self.lm_model = transformers.AutoModelForCausalLM.from_pretrained(load_model_dir, pad_token_id=self.tokenizer.eos_token_id).to(self.device)
        if hasattr(self.lm_model, 'parallelize'):
            self.lm_model.parallelize()

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

            model_output = self.lm_model.generate(
                input_ids,
                top_k=self.args.width,
                num_beams=(1 if self.args.ts_mode == 'sample' else self.args.num_beams),  # if sampling enabled, beam should always be 1
                num_return_sequences=self.args.num_beams,
                do_sample=(self.args.ts_mode == 'sample'),
                early_stopping=True,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=True,
                max_length=1024,
                use_cache=True  # huggingface default cache is always enabled
            )

            if self.top_k_cache_steps > 0:
                if hasattr(model_output, 'beam_indices'):
                    # beam search output
                    self.top_k_cache.add(input_ids, model_output.sequences, model_output.scores, beam_indices=model_output.beam_indices)
                else:
                    self.top_k_cache.add(input_ids, model_output.sequences, model_output.scores)

            output_ids_list = model_output.sequences.tolist()

            output_ids = output_ids_list[0]

            # use_seq_cache
            self.seq_cache.add(encoded_ids, output_ids)

            self.time_stamps.append(time.time())
            return output_ids

    def get_top_k_token_predict(self, state):
        with torch.no_grad():
            if self.top_k_cache_steps > 0:
                top_k_info = self.top_k_cache.get(state)
                if top_k_info is not None:
                    return top_k_info

            encoded_ids = state
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)

            model_output = self.lm_model.generate(
                input_ids,
                top_k=self.args.width,
                num_beams=self.args.num_beams,  # if sampling enabled, beam should always be 1
                early_stopping=True,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
                num_return_sequences=self.args.num_beams,
                do_sample=(self.args.ts_mode == 'sample'),
            )

            top_k_scores, top_k_tokens = torch.topk(model_output.scores[0][0], k=self.args.width, sorted=True)
            top_k_scores = torch.softmax(top_k_scores, dim=-1)
            return top_k_tokens.tolist(), top_k_scores.tolist()

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
