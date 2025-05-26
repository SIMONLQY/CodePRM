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
import json
import math
from prm import *
from typing import List, Tuple, Dict, Any, Optional
import heapq
import datetime


class Step:
    def __init__(self, thought_text, saved_flag=False):
        self.thought_text = thought_text
        self.saved_flag = saved_flag

        self.total_pass_1_list = []
        self.total_pass_rate_list = []
        self.avg_mc_pass_rate = 0.0
        self.avg_pass_1 = 0.0

class Rollout:
    def __init__(self, new_thought_steps, code):
        self.new_thought_steps = new_thought_steps
        self.code = code
        new_thought_steps_text = [step.thought_text for step in new_thought_steps]
        self.text = '\n'.join(new_thought_steps_text) + '\n' + code

    def update_all_step_values(self, parent_state, code_pass_rate):
        for step in parent_state.all_steps + self.new_thought_steps:
            step.avg_mc_pass_rate = (step.avg_mc_pass_rate * len(step.total_pass_rate_list) + code_pass_rate) / (len(step.total_pass_rate_list) + 1)
            step.avg_pass_1 = (step.avg_pass_1 * len(step.total_pass_1_list) + float(int(code_pass_rate == 1.0))) / (len(step.total_pass_1_list) + 1)
            step.total_pass_1_list.append(float(int(code_pass_rate == 1.0)))
            step.total_pass_rate_list.append(code_pass_rate)

# Define the State class
class State:
    def __init__(self, solution_prefix: str, parent: Optional['State'] = None, all_steps=[]):
        self.solution_prefix = solution_prefix  # Solution prefix as a single string
        self.parent = parent  # Reference to the parent state
        self.N = 0  # Visit count (number of times selected)
        self.total_rollouts = 0  # Total number of rollouts generated from this state
        self.correct_rollouts = 0  # Number of correct rollouts
        self.MC: Optional[float] = None  # Monte Carlo estimation (c/k)
        self.Q: Dict[str, float] = {}  # Q(s, r): estimated value for each rollout
        self.R: List[Rollout] = []  # Set of all rollouts from this state
        self.incorrect_rollouts: List[Rollout] = []  # List of incorrect rollouts
        self.children: List['State'] = []  # List of child states
        self.all_steps = all_steps

    def add_rollout(self, rollout: Rollout):
        self.R.append(rollout)

    def add_incorrect_rollout(self, rollout: Rollout):
        if rollout not in self.incorrect_rollouts:
            self.incorrect_rollouts.append(rollout)

    def get_full_solution(self) -> str:
        # Return the complete solution from the root to this state
        if self.parent:
            return self.parent.get_full_solution() + '\n\n' + self.solution_prefix
        else:
            return self.solution_prefix

    def get_new_text(self) -> str:
        """
        Return the new text added at this node compared to the parent.
        """
        if self.parent:
            parent_text = self.parent.solution_prefix
            new_text = self.solution_prefix[len(parent_text):].strip()
            return new_text
        else:
            # Root node (the question)
            return self.solution_prefix.strip()

    def get_text_with_labels(self) -> Dict[str, Any]:
        """
        Return a nested dictionary where each node contains:
        - 'text': The new text at this node.
        - 'mc_value': The MC value at this node.
        - 'children': A list of child nodes with the same structure.
        """
        data = {
            'text': self.get_new_text(),
            'mc_value': self.MC,
            'children': [child.get_text_with_labels() for child in self.children]
        }
        return data

# Define the Candidate Pool as a priority queue with update capability
class CandidatePool:
    def __init__(self):
        self.heap: List[Tuple[float, int]] = []  # Heap of (-priority, unique_id)
        self.entry_finder: Dict[int, Tuple[float, int]] = {}  # Maps unique_id to (-priority, unique_id)
        self.counter = itertools.count()  # Unique sequence count
        self.id_to_rollout: Dict[int, Tuple[State, Rollout]] = {}  # Maps unique_id to (state, rollout)
        self.latest_id_per_rollout: Dict[Tuple[int, str], int] = {}  # Maps (state_id, rollout) to unique_id

    def add_or_update(self, state: State, rollout: Rollout, priority: float):
        """
        Add a new rollout or update the priority of an existing rollout.

        Parameters:
        - state (State): The state associated with the rollout.
        - rollout (str): The rollout string.
        - priority (float): The new priority score.
        """
        rollout_text = rollout.text
        state_id = id(state)  # Unique identifier for the state object
        rollout_key = (state_id, rollout_text)

        # Check if the rollout already exists in the pool
        if rollout_key in self.latest_id_per_rollout:
            # Previous unique_id exists; it is now outdated
            old_unique_id = self.latest_id_per_rollout[rollout_key]
            # Mark the old entry as invalid by removing it from entry_finder
            if old_unique_id in self.entry_finder:
                del self.entry_finder[old_unique_id]
                del self.id_to_rollout[old_unique_id]

        # Assign a new unique_id for the updated rollout
        unique_id = next(self.counter)
        self.latest_id_per_rollout[rollout_key] = unique_id

        # Add the new entry to the heap and mappings
        heapq.heappush(self.heap, (-priority, unique_id))  # Max-heap using negative priority
        self.entry_finder[unique_id] = (-priority, unique_id)
        self.id_to_rollout[unique_id] = (state, rollout)

    def pop(self) -> Tuple[Optional[State], Optional[Rollout]]:
        """
        Pop the rollout with the highest priority.

        Returns:
        - Tuple[Optional[State], Optional[str]]: The state and rollout string, or (None, None) if empty.
        """
        while self.heap:
            neg_priority, unique_id = heapq.heappop(self.heap)
            # Check if this unique_id is still valid
            if unique_id in self.entry_finder:
                # Valid entry
                state, rollout = self.id_to_rollout.pop(unique_id)
                del self.entry_finder[unique_id]
                # Remove from latest_id_per_rollout
                state_id = id(state)
                rollout_key = (state_id, rollout.text)
                if self.latest_id_per_rollout.get(rollout_key) == unique_id:
                    del self.latest_id_per_rollout[rollout_key]
                return state, rollout
            # Else, outdated entry; skip
        return None, None

    def is_empty(self) -> bool:
        return not self.entry_finder


# Define the Search Tree class
class SearchTree:
    def __init__(self):
        self.root: Optional[State] = None
        self.nodes: List[State] = []  # List of all states

    def add_state(self, state: State):
        self.nodes.append(state)


class TreeDataCollect:
    def __init__(self, args):
        self.args = args
        self.sample_nums = 0
        self.gamma = 0.9
        self.save_mid_json = []
        self.stops = ['Code:', None]

        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = '\n\n\n\n\n'

        # data saving
        self.save_mc_process_scores = {}

        # generator模型
        if 'gpt3.5' in args.arch or 'gpt4' in args.arch or 'o1' in args.arch:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.generator = GPT35Chat(args.generator_local_model_actor, args.arch, self.tokenizer, args, self.save_mid_json, api_port=args.generator_api_port)
        elif 'gpt2' in args.arch or 'neo' in args.arch:  # gpt2, gpt-neo
            raise ValueError("MCTSAg only supports GPT3.5")
        # 数据集
        if args.dataset in ['apps', 'codeforces']:
            self.executor = AppsExecutor(args)
        elif args.dataset == 'humaneval':
            self.executor = HumanevalExecutor(args)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        # prm模型
        if self.args.prm_model_name != 'gpt4o-mini' and 'ft' not in self.args.prm_model_name:
            self.llm_verifier = GPT35Chat(args.prm_local_model_actor, args.prm_model_name, self.tokenizer, args, call_mode='api', api_port=args.prm_api_port)
        elif 'ft' in self.args.prm_model_name:
            if self.args.prm_model_name in ['Skywork-o1-Open-PRM-Qwen-2.5-7B-ft', 'Skywork-o1-Open-PRM-Qwen-2.5-1.5B-ft']:
                self.llm_verifier = SkyworkPRM(args)
            elif self.args.prm_model_name == 'math-shepherd-mistral-7b-prm-ft':
                self.llm_verifier = MathShepherdPRM(args)
            elif 'value' in self.args.prm_model_name:
                self.llm_verifier = ValuePRM(args)
            else:
                self.llm_verifier = TokenPRM(args)
        else:
            self.llm_verifier = self.generator

        self.term_cond = lambda: self.sample_nums > args.max_sample_times

        self.root = None
        self.cached_reward = {}
        self.cached_value = {}
        self.cached_verbal_feedback = {}
        self.cur_prob_instance = None
        self.sample_times = []
        self.st = time.time()

        # TreeDataCollect parameters
        self.save_data_tree = False
        self.rollout_new_thought_num = 2
        self.alpha = self.args.alpha  # Weight for MC(s).
        self.beta = self.args.beta # Length penalty.
        self.c_puct = self.args.c_puct  # Exploration constant.
        self.L = self.args.L  # Maximum solution length.
        self.k = self.args.mc_nums  # Number of rollouts for Monte Carlo estimation.

        self.T = SearchTree()
        self.C = CandidatePool()
        self.selected_rollout_text = []  # 确保每个rollout只被选中一次
        self.all_state_monte_carlo_scores = {}

        self.search_count = 0
        self.total_rollouts = 0


    def generate(self, problem_instance):
        self.st = time.time()
        self.cur_prob_instance = problem_instance
        raw_prompt = problem_instance['prompt']
        done = False
        initial_state = self.tokenizer.encode(raw_prompt)
        if len(initial_state) >= self.args.horizon:
            return None

        if self.args.dataset == 'humaneval':
            self.save_mc_process_scores['ques_id'] = problem_instance['task_id'].split('/')[-1]
            self.save_mc_process_scores['question'] = problem_instance['prompt']
            # self.save_mc_process_scores['given_tests'] = problem_instance['given_tests']
            # self.save_mc_process_scores['tests'] = problem_instance['test']
        elif self.args.dataset in ['apps', 'codeforces']:
            self.save_mc_process_scores['ques_id'] = problem_instance['index']
            self.save_mc_process_scores['question'] = problem_instance['prompt']
            # self.save_mc_process_scores['given_tests'] = problem_instance['train_in_outs']
            # self.save_mc_process_scores['tests'] = problem_instance['test_in_outs']
        self.save_mc_process_scores['label'] = {}

        self.treed_data_procedure()

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

        # 保存omegamc结果
        if self.args.dataset == 'humaneval':
            problem_id = problem_instance['task_id'].split('/')[-1]
        elif self.args.dataset in ['apps', 'codeforces']:
            problem_id = problem_instance['index']
        self.save_mc_process_scores['final_program'] = complete_programs[best_idx]
        self.save_mc_process_scores['test_reward'] = test_rewards[best_idx]
        self.save_mc_process_scores['all_state_monte_carlo_scores'] = self.all_state_monte_carlo_scores
        result_loc = f"{get_proj_path()}/results/{self.args.dataset}/Experiment_{self.args.experiment_idx}/{self.args.repeat_id}/{self.args.APPDifficulty}/mc_labels/"
        os.makedirs(result_loc, exist_ok=True)
        result_loc = os.path.join(result_loc, f"{problem_id}.json")
        with open(result_loc, "w") as f:
            json.dump(self.save_mc_process_scores, f)

        # 保存题目结果
        output_dict = {}
        output_dict['final_program'] = complete_programs[best_idx]
        output_dict['train_reward'] = train_rewards[best_idx]
        output_dict['test_reward'] = test_rewards[best_idx]
        output_dict['all_programs'] = complete_programs
        output_dict['all_train_rewards'] = train_rewards
        output_dict['all_test_rewards'] = test_rewards
        output_dict['avg_sample_time'] = np.mean(np.array(self.sample_times))

        # 重制相关参数
        self.cached_reward = {}
        self.cached_value = {}
        self.cached_verbal_feedback = {}
        self.generator.clean_cache()
        self.sample_nums = 0

        self.T = SearchTree()  # Reset search tree
        self.C = CandidatePool()  # Reset candidate pool
        self.selected_rollout_text = []
        self.search_count = 0
        self.total_rollouts = 0
        self.save_mc_process_scores = {}

        return output_dict

    def treed_data_procedure(self):
        """
        Execute the tree data collect algorithm.

        Parameters:
        - question (str): The question to generate solutions for.

        Returns:
        - Collected data: List of dictionaries.
        """
        print(f"Running TreeDataCollect for the question ... ")
        question = self.cur_prob_instance['prompt']
        # Initialization
        initial_state = State(solution_prefix=question, parent=None, all_steps=[])
        self.T.root = initial_state
        self.search_count = 0

        # Monte Carlo Estimation for initial_state
        self.monte_carlo_estimation(initial_state)

        # Main loop
        while self.search_count < self.args.rollout and not self.C.is_empty():
            print(f"----------------------------")
            print(f"Search Count: {self.search_count} / {self.args.rollout}")
            # Selection Phase
            selected_state, selected_rollout = self.selection_phase()
            if selected_state is None or selected_rollout is None:
                # print("No more candidates to explore. Terminating search.\n")
                break

            self.selected_rollout_text.append(selected_rollout.text)

            self.binary_search_incorrect_step(selected_state, selected_rollout)

            # Maintenance Phase
            self.maintenance_phase(selected_state)

            # Increment search count
            self.search_count += 1

        # if self.save_data_tree:
        #     data = self.collect_tree_structure()
        # else:
        #     data = self.collect_solution_prefixes()


    def monte_carlo_estimation(self, state: State):
        """
        Perform Monte Carlo estimation for state by generating k rollouts
        and computing MC(s) = c / k, where c is the number of correct rollouts.
        """
        c = 0  # Correct rollouts count
        incorrect_rollouts = []
        correct_rollouts = []
        new_rollouts = self.get_rollouts(state, self.k)

        # Increment visit count of selected state
        state.N += 1

        for i, rollout in enumerate(new_rollouts):
            # Increment number of total rollouts
            self.total_rollouts += 1

            # Generate rollout r_i
            state.add_rollout(rollout)

            # Evaluate correctness of final answer in rollout
            code = rollout.code
            code_id = self.tokenizer.encode(code)
            full_result_train = self.get_reward(code_id, with_verbal=True, mode='train')
            full_result_test = self.get_reward(code_id, with_verbal=True, mode='test')
            full_result = [(full_result_train[0] + full_result_test[0]) / 2,
                           full_result_train[1] + full_result_test[1]]
            complete_prog_score, verbal_feedbacks = full_result[0], full_result[1]

            rollout.update_all_step_values(parent_state=state, code_pass_rate=complete_prog_score)

            is_correct = complete_prog_score == 1.0

            self.cached_value[tuple(code_id)] = complete_prog_score


            # print(f"Rollout {i + 1} Correctness: {'Correct' if is_correct else 'Incorrect'}\n")

            if is_correct:
                c += 1
                correct_rollouts.append(rollout)
            else:
                incorrect_rollouts.append(rollout)
                state.add_incorrect_rollout(rollout)  # Track incorrect rollouts

        # Update total rollouts and correct rollouts
        state.total_rollouts += self.k
        state.correct_rollouts += c
        state.MC = state.correct_rollouts / state.total_rollouts if state.total_rollouts > 0 else 0

        self.all_state_monte_carlo_scores[state.solution_prefix] = state.MC

        # print(f"Monte Carlo Estimation for State ID {self.T.nodes.index(state)}: MC = {state.MC:.2f}, Total Rollouts = {state.total_rollouts}, Correct Rollouts = {state.correct_rollouts}\n")

        self.T.add_state(state)  # self.T的主要作用是用来compute_U，其次就是存储了所有拥有MC分数的state，包括正确的rollout

        # Add correct rollouts to the tree
        if len(correct_rollouts) > 0:
            for rollout in correct_rollouts:
                self.add_correct_rollout_to_tree(state, rollout)
                self.save_labeled_rollout_to_db(state, rollout.new_thought_steps, rollout.code, wrong_id=-1)

        if len(incorrect_rollouts) > 0:
            for rollout in incorrect_rollouts:
                self.save_labeled_rollout_to_db(state, rollout.new_thought_steps, rollout.code, wrong_id=-2)

        # Add state is correct but with wrong rollouts to the candidate pool
        if 0 < state.MC < 1.0 or state.parent is None:  # 根节点由于没有thought，所以所有错误的rollout都应当加入到candidate pool当中
            # Add incorrect rollouts to candidate pool with updated priorities
            for rollout in incorrect_rollouts:
                priority = self.compute_selection_score(state, rollout)
                self.C.add_or_update(state, rollout, priority)

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

    def gather_saved_feedback(self, verbal_feedbacks):
        saved_feedback = [item for item in verbal_feedbacks if isinstance(item, dict)]
        ori_length = len(verbal_feedbacks)
        saved_feedback = [
            {
                'error': shorten_text_no_tokenizer(item['error'], total_length=10000),
                'output': shorten_text_no_tokenizer(item['output'], total_length=10000)
            }
            for item in saved_feedback
        ]
        tmp_save = [item for item in saved_feedback if len(self.tokenizer.encode(item['output'])) < 512]
        tmp_save = tmp_save[:5]
        if len(tmp_save) == 0 and ori_length > 0:
            saved_feedback = saved_feedback[:5]
            tmp_save = [
                {
                    'error': item['error'],
                    'output': shorten_text_block(item['output'], self.tokenizer, total_length=512, section_length=512)
                }
                for item in saved_feedback
            ]
        saved_feedback = tmp_save
        return saved_feedback

    def get_rollouts(self, state, n_generate_rollouts):
        new_thought_num = self.rollout_new_thought_num
        if state.parent is None:
            new_thought_num = 4

        ys = [state]
        for i in range(new_thought_num):
            if i == 0:
                n_gen = n_generate_rollouts  # 总共产生多少个rollout
            else:
                n_gen = 1  # 第一步之后每次跟着产生一个即可

            new_ys = []
            for y in ys:
                new_ys.extend(self.get_samples(y, n_gen))

            ys = new_ys[:]
        new_rollouts = self.get_rollout_thought_codes(state, ys)
        return new_rollouts

    def get_samples(self, y:State, n_gen:int):
        prompt = f"""
* Below is a problem to be solved by Python program.
* I need you to analyze this problem and provide reasoning and thoughts.
* If no previous thoughts provided, please provide your ONE thought step on how to solve the problem in one or two sentences.
* If there are previous thoughts provided, please follow them and offer ONE more further thought step in one or two sentences, as a detailed thinking or enhancement for previous ones.
* Remember to output ONE reasoning step but following previous thoughts if there are.
* Your response should be in 1~2 sentences, not too long.       

Here is the question and previous thoughts:
{y.solution_prefix}

Begin your response now.
"""
        outputs = []
        while n_gen > 0:
            n = min(n_gen, 20)
            n_gen -= n
            messages, _ = self.generator.generate_response_api(prompt,
                                                               top_k=None,
                                                               max_length=min(n * 1024, 10000),
                                                               temperature=0.7,
                                                               n=n,
                                                               stop=None)
            if n == 1:
                outputs.extend([messages])
            else:
                outputs.extend(messages)
        formated_responses = []
        for response in outputs:
            if '-Thought' not in response:
                formated_responses.append(f'{len(y.all_steps) + 1}-Thought:' + response)
            else:
                formated_responses.append(response)
        return [State(solution_prefix=y.solution_prefix + '\n' + response,
                      parent=y,
                      all_steps=y.all_steps + [Step(response, saved_flag=False)],
                      ) for response in formated_responses]

    def get_rollout_thought_codes(self, parent_state:State, ys:List[State]):
        new_rollouts = []
        for s in ys:
            code_id = self.generator.get_rationale_predicted_sequence(self.tokenizer.encode(s.solution_prefix))
            code = self.tokenizer.decode(code_id)
            additional_steps = [step for i, step in enumerate(s.all_steps) if i >= len(parent_state.all_steps)]
            additional_rollout = Rollout(additional_steps, code)
            new_rollouts.append(additional_rollout)
        return new_rollouts

    def selection_phase(self) -> Tuple[Optional[State], Optional[Rollout]]:
        """
        Select (state, rollout) with the highest score from candidate pool C.
        """
        selected_state, selected_rollout = self.C.pop()
        return selected_state, selected_rollout

    def maintenance_phase(self, state: State):
        """
        Update statistics and candidate pool for all incorrect rollouts associated with the state.

        Parameters:
        - state (State): The state whose incorrect rollouts need to be updated.
        """
        state.N += 1
        # Iterate through all incorrect rollouts of the state
        for rollout in state.incorrect_rollouts:
            if rollout.text in self.selected_rollout_text:  # 被selected rollout已经被pop，不应当再被加入到candidate pool
                continue
            # Since we've already determined these rollouts are incorrect, no need to re-evaluate correctness
            priority = self.compute_selection_score(state, rollout)
            # Update the candidate pool with the new priority
            self.C.add_or_update(state, rollout, priority)  # 这里主要是为了update,一下priority
            # print(f"Updated Incorrect Rollout: '{rollout}' with new priority: {priority:.4f}")

        # print("Maintenance Phase Completed.\n")

    def collect_tree_structure(self) -> Dict[str, Any]:
        """
        Collect the tree structure starting from the root.

        Returns:
            Dict[str, Any]: A nested dictionary representing the tree structure.
        """
        if self.T.root:
            tree_data = self.T.root.get_text_with_labels()
            return tree_data
        return {}

    def collect_solution_prefixes(self) -> List[Dict[str, Any]]:
        """
        Collect all solution prefixes and their corresponding MC values from the search tree.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing solution prefixes and their MC values.
        """
        collected_data = []
        for node in self.T.nodes:
            solution_prefix = node.solution_prefix
            mc_value = node.MC
            collected_data.append({
                "solution_prefix": solution_prefix,
                "mc_value": mc_value
            })
        return collected_data

    def binary_search_incorrect_step(self, s_ast: State, rollout:Rollout):
        """
        Recursively perform binary search to find all incorrect steps in the rollout and adds middle states as new states.

        Parameters:
        - s_ast (State): The selected parent state.
        - steps (List[str]): The rollout steps as a list.
        - left (int): Left index of the current search interval.
        - right (int): Right index of the current search interval.
        """
        # Separate the rollout into individual steps
        steps, code = rollout.new_thought_steps, rollout.code

        # 保留初始的s_ast的前缀状态
        tmp_s_ast = s_ast
        # 二分法定位第一个错误的thought
        left = 0
        right = len(steps) - 1
        while left < right:
            mid = (left + right) // 2
            new_steps = steps[left:mid + 1]
            new_steps_text_list = [step.thought_text for step in new_steps]
            if new_steps:
                prefix_solution = tmp_s_ast.solution_prefix + '\n' + '\n'.join(new_steps_text_list)
            else:
                prefix_solution = tmp_s_ast.solution_prefix

            # Create new state s_new
            s_new = State(solution_prefix=prefix_solution, parent=tmp_s_ast, all_steps=tmp_s_ast.all_steps + new_steps)
            tmp_s_ast.children.append(s_new)
            self.monte_carlo_estimation(s_new)

            if s_new.MC == 0:
                # Found incorrect step; continue searching in the left half to find earlier incorrect steps
                right = mid
            else:
                # Steps up to mid are correct; continue searching in the right half
                left = mid + 1
                tmp_s_ast = s_new
        self.save_labeled_rollout_to_db(s_ast, steps, code, wrong_id=left)

    def save_labeled_rollout_to_db(self, parent_state: State, steps: List[Step], code:str, wrong_id:int):
        """
        wrong_id:
        -1 : correct trace, save all steps with correct score
        -2: incorrect trace, save all steps with incorrect score
        [0, len(steps)-1]: incorrect trace, save all steps before wrong_id with correct score, and the wrong_id step with incorrect score
        """
        # 保存一个rollout到数据库中
        # 获得select的rollout的代码的执行结果 (总体肯定是不对的，因为是incorrect rollout)
        code_id = self.tokenizer.encode(code)
        full_result_train = self.get_reward(code_id, with_verbal=True, mode='train')
        full_result_test = self.get_reward(code_id, with_verbal=True, mode='test')
        full_result = [(full_result_train[0] + full_result_test[0]) / 2,
                       full_result_train[1] + full_result_test[1]]
        complete_prog_score, verbal_feedbacks = full_result[0], full_result[1]

        if wrong_id not in [-1, -2]:
            current_trace_score = complete_prog_score
            current_trace_train_score = full_result_train[0]
            current_trace_code = code
            current_trace_train_exe = self.gather_saved_feedback(full_result_train[1])
            current_trace_exe = self.gather_saved_feedback(verbal_feedbacks)
        elif wrong_id == -1:  # correct rollout
            current_trace_score = 1.0
            current_trace_train_score = 1.0
            current_trace_code = code
            current_trace_train_exe = [{'error': None, 'output': None}]
            current_trace_exe = [{'error': None, 'output': None}]
        elif wrong_id == -2:
            current_trace_score = complete_prog_score
            current_trace_train_score = full_result_train[0]
            current_trace_code = code
            current_trace_train_exe = self.gather_saved_feedback(full_result_train[1])
            current_trace_exe = self.gather_saved_feedback(verbal_feedbacks)
        else:
            raise ValueError(f"wrong_id should be in [-1, -2, [0, len(steps)-1]], but got {wrong_id}")

        # 保存当前trace
        cur_save_trace_count = len(self.save_mc_process_scores['label'])
        self.save_mc_process_scores['label'][f'trace_{cur_save_trace_count}_steps'] = []

        # 先append prefix部分的thought
        for i, step in enumerate(parent_state.all_steps):
            new_steps = parent_state.all_steps[:(i+1)]
            new_steps_text_list = [step.thought_text for step in new_steps]
            if new_steps:
                current_state = self.cur_prob_instance['prompt'] + '\n' + '\n'.join(new_steps_text_list)
            else:
                current_state = self.cur_prob_instance['prompt']

            if wrong_id == -2:  # incorrect trace and all seen as incorrect steps
                score = 0.0
            else:
                score = 1.0

            self.save_mc_process_scores['label'][f'trace_{cur_save_trace_count}_steps'].append(
                {'state': current_state,
                 'thought': parent_state.all_steps[i].thought_text,
                 'mcs': score,
                 'avg_pass_rate': step.avg_mc_pass_rate,
                 'avg_pass_1': step.avg_pass_1,
                 'total_pass_rate_list': step.total_pass_rate_list[:],
                 'total_pass_1_list': step.total_pass_1_list[:],
                 'saved_flag': step.saved_flag,
                 'current_trace_score': current_trace_score,
                 'current_trace_train_score': current_trace_train_score,
                 'current_trace_code': current_trace_code,
                 'current_trace_train_exe': current_trace_train_exe,
                 'current_trace_exe': current_trace_exe,
                 'wrong_state_flag': wrong_id,
                 'parent_state': parent_state.solution_prefix,
                 'time_stamp': datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                 })
            # step.saved_flag = True

        # 再append本次rollout的thought，这部分都设置saved_flag为False（未被保存）
        for i, step in enumerate(steps):
            if wrong_id not in [-1, -2] and i > wrong_id:
                break

            # set score
            if wrong_id == -2 or i == wrong_id:
                score = 0.0
            else:
                score = 1.0

            new_steps = steps[:(i+1)]
            new_steps_text_list = [step.thought_text for step in new_steps]
            if new_steps:
                current_state = parent_state.solution_prefix + '\n' + '\n'.join(new_steps_text_list)
            else:
                current_state = parent_state.solution_prefix

            self.save_mc_process_scores['label'][f'trace_{cur_save_trace_count}_steps'].append(
                {'state': current_state,
                 'thought': steps[i].thought_text,
                 'mcs': score,
                 'avg_pass_rate': step.avg_mc_pass_rate,
                 'avg_pass_1': step.avg_pass_1,
                 'total_pass_rate_list': step.total_pass_rate_list[:],
                 'total_pass_1_list': step.total_pass_1_list[:],
                 'saved_flag': step.saved_flag, # 总是未保存的
                 'current_trace_score': current_trace_score,
                 'current_trace_train_score': current_trace_train_score,
                 'current_trace_code': current_trace_code,
                 'current_trace_train_exe': current_trace_train_exe,
                 'current_trace_exe': current_trace_exe,
                 'wrong_state_flag': wrong_id,
                 'parent_state': parent_state.solution_prefix,
                 'time_stamp': datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                 })
            # step.saved_flag = True


    def add_correct_rollout_to_tree(self, parent_state: State, rollout: Rollout):
        """
        Add the correct rollout to the tree as a child of parent_state.
        """
        new_steps, code = rollout.new_thought_steps, rollout.code
        if parent_state.solution_prefix:
            new_solution_prefix = (parent_state.solution_prefix + '\n\n' + rollout.text).strip()
        else:
            new_solution_prefix = self.cur_prob_instance['prompt'] + '\n' + rollout.text
        new_state = State(solution_prefix=new_solution_prefix, parent=parent_state, all_steps=parent_state.all_steps+new_steps)
        new_state.MC = 1.0  # Since the rollout is correct
        new_state.total_rollouts = 0
        new_state.correct_rollouts = 0
        self.T.add_state(new_state)
        parent_state.children.append(new_state)  # Add to parent's children

    def compute_selection_score(self, state: State, rollout: Rollout) -> float:
        """
        Compute selection score: Score(s, r) = Q(s, r) + U(s)
        """
        Q_s_r = self.compute_Q(state, rollout)
        U_s = self.compute_U(state)
        score = Q_s_r + U_s
        return score

    def compute_Q(self, state: State, rollout: Rollout) -> float:
        """
        Compute Q(s, r) = alpha^{1 - MC(s)} * beta^{len(r)/L}, where len(r) is based on word count.
        """
        # Count words in the rollout
        word_count = len(rollout.text.split())
        length_penalty = word_count / self.L
        Q_value = (self.alpha ** (1 - state.MC)) * (self.beta ** length_penalty)
        return Q_value

    def compute_U(self, state: State) -> float:
        """
        Compute U(s) = c_puct * sqrt(sum_{s'} N(s')) / (1 + N(s))
        """
        N_total = sum(s.N for s in self.T.nodes if 0.0< s.MC < 1.0 or s.parent is None)
        if N_total == 0:
            N_total = 1  # Prevent division by zero
        U_s = self.c_puct * (math.sqrt(N_total)) / (1 + state.N)
        return U_s
