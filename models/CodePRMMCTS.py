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
from prm import *
from executors import AppsExecutor, HumanevalExecutor
from ChatModels import GPT35Chat, GPT2Chat
import re
import json
from collections import deque


class CodePRMMCTS:
    def __init__(self, args):
        self.args = args
        self.gamma = 0.9
        self.save_mid_json = []
        self.save_mc_process_scores = {}
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # general tokenizer
        if 'gpt2' in args.arch or 'neo' in args.arch:  # gpt2, gpt-neo
            raise ValueError("MCTSAg only supports GPT3.5")
        else:
            self.generator = GPT35Chat(args.generator_local_model_actor, args.arch, self.tokenizer, args, self.save_mid_json, api_port=args.generator_api_port)

        # llm verifier
        if self.args.prm_model_name != args.arch and 'ft' not in self.args.prm_model_name:
            self.llm_verifier = GPT35Chat(args.prm_local_model_actor, args.prm_model_name, self.tokenizer, args, call_mode='api', api_port=args.prm_api_port)
        elif 'ft' in self.args.prm_model_name:
            if 'value' in self.args.prm_model_name:
                self.llm_verifier = ValuePRM(args)
            else:
                self.llm_verifier = TokenPRM(args)
        else:
            self.llm_verifier = self.generator

        # dataset
        if args.dataset in ['apps', 'codeforces']:
            self.executor = AppsExecutor(args)
        elif args.dataset == 'humaneval':
            self.executor = HumanevalExecutor(args)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        # selection algorithm
        if args.uct_alg == 'uct':
            self.node_choose_policy = uct_tree_policy
        elif args.uct_alg == 'p_uct':
            self.node_choose_policy = p_uct_tree_policy
        elif args.uct_alg == 'var_p_uct':
            self.node_choose_policy = var_p_uct_tree_policy
            self.ucb_base = args.ucb_base

        # parameter initial
        self.root = None
        self.cached_reward = {}
        self.cached_value = {}
        self.cached_verbal_feedback = {}
        self.cur_prob_instance = None
        self.sample_times = []
        self.st = time.time()

        self.current_trace_score = 0.0
        self.current_trace_train_score = 0.0
        self.current_trace_code = None
        self.current_trace_exe = None
        self.current_trace_train_exe = None

        self.max_think_times = args.max_think_times
        self.current_think_times = 1

    def generate(self, problem_instance):
        self.st = time.time()
        self.cur_prob_instance = problem_instance
        raw_prompt = problem_instance['prompt']
        initial_state = raw_prompt
        if len(self.tokenizer.encode(initial_state)) >= self.args.horizon:
            return None

        if self.args.json_save_mc and self.args.verify_pos_mode == 'omega_mc':
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
            else:
                raise ValueError(f"Unknown dataset: {self.args.dataset}")
            self.save_mc_process_scores['label'] = {}

        self.mcts_procedure(initial_state)

        if len(self.cached_value) == 0:
            code_id = self.generator.get_rationale_predicted_sequence(self.tokenizer.encode(initial_state))
            complete_prog_score = self.get_reward(code_id)

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

        # save json data
        if self.args.dataset == 'humaneval':
            problem_id = problem_instance['task_id'].split('/')[-1]
        elif self.args.dataset in ['apps', 'codeforces']:
            problem_id = problem_instance['index']

        if self.args.json_save_mc and self.args.verify_pos_mode == 'omega_mc':
            self.save_mc_process_scores['final_program'] = complete_programs[best_idx]
            self.save_mc_process_scores['test_reward'] = test_rewards[best_idx]

            result_loc = f"{get_proj_path()}/results/{self.args.dataset}/Experiment_{self.args.experiment_idx}/{self.args.repeat_id}/{self.args.APPDifficulty}/mc_labels/"
            os.makedirs(result_loc, exist_ok=True)
            result_loc = os.path.join(result_loc, f"{problem_id}.json")
            with open(result_loc, "w") as f:
                json.dump(self.save_mc_process_scores, f)

        output_dict['all_programs'] = complete_programs
        output_dict['all_train_rewards'] = train_rewards
        output_dict['all_test_rewards'] = test_rewards
        output_dict['avg_sample_time'] = np.mean(np.array(self.sample_times))

        self.cached_reward = {}
        self.cached_value = {}
        self.cached_verbal_feedback = {}
        self.generator.clean_cache()
        self.save_mid_json = []
        self.generator.save_mid_json = self.save_mid_json
        self.save_mc_process_scores = {}
        self.args.rollout_count = -1

        return output_dict

    def mcts_procedure(self, initial_state):
        """
        Compute the entire MCTS procedure wrt to the selected tree policy.
        Function tree_policy is a function taking an agent + a list of ChanceNodes as argument
        and returning the one chosen by the tree policy.
        """
        # 开始时，root是None
        decision_node_num = 0
        self.root = MineDecisionNode(None, initial_state, generator=self.generator, id=decision_node_num, tokenizer=self.tokenizer)
        self.root.__expand__()
        decision_node_num += 1
        # 如果rollouts=1，产生的程序存在cached_rewards里面的只有一个完整程序，其实select那一步就已经选了走哪个完整程序了
        print("Performing rollouts.")
        for rollout_count in range(self.args.rollout):  # 这个rollout控制的是选择次数，如果从根节点开始，第一次选第一层，第二次可能选的是第二层，第三次选第三层
            self.args.rollout_count = rollout_count
            rewards = []  # Rewards collected along the tree for the current rollout
            node = self.root  # Current node

            # Selection & Expansion
            node = self.selection_expansion_phase(node, decision_node_num, rewards)

            # Evaluation: Generate code and possibly rethink
            state = node.state
            prev_thought_score = 0.0
            self.current_think_times = 1
            for renewchild_count in range(self.max_think_times):
                # 生成code
                code_id = self.generator.get_rationale_predicted_sequence(self.tokenizer.encode(state))
                code = self.tokenizer.decode(code_id)

                # 得到code执行结果
                if self.args.json_save_mc and self.args.verify_pos_mode == 'omega_mc':
                    # 保存数据时，用所有的test和train的数据来测试
                    full_result_train = self.get_reward(code_id, with_verbal=True, mode='train')
                    full_result_test = self.get_reward(code_id, with_verbal=True, mode='test')
                    full_result = [(full_result_train[0] + full_result_test[0]) / 2,
                                   full_result_train[1] + full_result_test[1]]
                    complete_prog_score, verbal_feedbacks = full_result[0], full_result[1]
                    failed_tests, verbalFeedback, saved_feedbacks = self.gather_feedback(verbal_feedbacks, state, code)
                    self.current_trace_train_exe = self.gather_saved_feedback(full_result_train[1])
                    self.current_trace_train_score = full_result_train[0]
                    self.current_trace_score = complete_prog_score
                    self.current_trace_code = code
                    self.current_trace_exe = saved_feedbacks
                    # 反传得到的feedback
                    self.backpropagate_mc_result(node, code, complete_prog_score, saved_feedbacks)
                else:
                    full_result = self.get_reward(code_id, with_verbal=True)
                    complete_prog_score, verbal_feedbacks = full_result[0], full_result[1]
                    failed_tests, verbalFeedback, saved_feedbacks = self.gather_feedback(verbal_feedbacks, state, code)

                # value estimation
                value = self.value_estimation(state, code, complete_prog_score, failed_tests, node)
                if tuple(code_id) not in self.cached_value.keys():
                    self.cached_value[tuple(code_id)] = value

                # 统计上一轮rethink是否成功
                if renewchild_count >= 1:
                    if complete_prog_score > prev_thought_score:
                        self.args.rethink_effective_nums += 1
                    if complete_prog_score < prev_thought_score:
                        self.args.rethink_failed_nums += 1
                    if complete_prog_score == 1.0:
                        self.args.rethink_success_nums += 1
                elif renewchild_count == 0 and complete_prog_score == 1.0:
                    self.args.no_rethink_success_num += 1

                # rethink
                if failed_tests != '':
                    # Rethink (主动判断整个trace上那个thought有问题，然后更新这个thought)
                    if self.current_think_times >= self.max_think_times:
                        # node是叶子结点，实际rethink的可能不是这个节点，这里的second chance flag是用来标识这条trace只能rethink这么多次
                        node.second_chance_flag = False
                    if not node.second_chance_flag: # 出错了但是没有rethink次数,保留下这条新被探索的trace
                        if self.args.json_save_mc and self.args.verify_pos_mode == 'omega_mc':
                            # form thoughts states list
                            cur_state, thoughts, id_node_dict, state_list = self.form_thought_state_list(node)
                            wrong_id, wrong_node = self.omega_mc_judge(node, id_node_dict, thoughts)
                            self.save_labels_rollout_to_db(id_node_dict, node, wrong_id, state_list, thoughts)
                    else:
                        self.args.rethink_total_nums += 1
                        prev_thought_score = complete_prog_score
                        node = self.renew_trace_wrong_thoughts(node, cur_code=code, pure_feedback=failed_tests.strip())
                        self.current_think_times += 1
                        state = node.state
                else:
                    # 没有出错的情况下同样通过mc判定trace错误，并进行保存
                    if self.args.json_save_mc and self.args.verify_pos_mode == 'omega_mc':
                        # form thoughts states list
                        cur_state, thoughts, id_node_dict, state_list = self.form_thought_state_list(node)
                        self.save_labels_rollout_to_db(id_node_dict, node, -1, state_list, thoughts)
                    break  # 这里需要一个break，因为如果代码正确，不应当继续执行max_think_times的循环

            # Expand new child
            verbalFeedback = verbalFeedback if (failed_tests != '') else ''
            node.__expand__(verbal_feedback=verbalFeedback)

            # Backpropagation of scaled reward
            self.backpropagate_value(node, rewards, value)
            self.sample_times.append(time.time() - self.st)

    def selection_expansion_phase(self, root_node, decision_node_num, rewards):
        select = True
        node = root_node
        while select:
            if (type(node) == MineDecisionNode):  # DecisionNode
                node = self.node_choose_policy(self, node.children)  # 根据P-UCB从node的children中选择一个最大值的node， node is now a ChanceNode
            else:  # ChanceNode，（状态，动作）节点，相当于树中的一条边
                next_state = self.transition(node.parent.state, node.action)
                rewards.append(0.0)  # 做完动作没有terminal的情况下，reward为0，后面backpropagation主要靠estimation

                new_state = True  # 如果树有很多层，这里的while循环会从根节点一层一层往下走，直到找到一个新的state_p
                for i in range(len(node.children)):  # 其实ChanceNode只有一个child, 或者没有child(更常见，因为是还没有探索过的节点)
                    if node.children[i].state == next_state:
                        # Shun: state_p already in the tree, point node to the corresponding Decision Node
                        node = node.children[i]
                        new_state = False
                        break
                if new_state and len(node.children) > 0 and type(node) == MineChanceNode:
                    assert False
                if new_state:  # 一开始如果是三个rollouts，就三个root的children都会经过这里
                    select = False  # Selected a ChanceNode

        # 打印选择的路径
        path_text = self.get_node_path(node)
        print(f"selection_{self.args.rollout_count}: \n{path_text}")
        if type(node) == MineChanceNode:
            # chance node 只有一个子节点，就是加上了那个动作的节点,但每一个decision node在创建的时候都会带有3个可能的动作
            node.children.append(
                MineDecisionNode(node, next_state, generator=self.generator, id=decision_node_num, decision_memory=node.chance_memory, tokenizer=self.tokenizer))
            decision_node_num += 1
            node = node.children[-1]  # 就是新增加的decision node
        assert (type(node) == MineDecisionNode)

        return node

    def value_estimation(self, state, code, complete_prog_score, failed_tests, node):
        if self.args.reward_mode == 'pub_test':
            value = complete_prog_score
        elif self.args.reward_mode == 'pub_llm':
            if complete_prog_score == 1.0:
                evaluation = self.get_evaluation(state, code)
                value = 0.8 * complete_prog_score + 0.2 * evaluation
            else:
                value = complete_prog_score
        elif self.args.reward_mode == 'pub_prm':
            if complete_prog_score < 1.0:
                value = complete_prog_score
            else:
                if 'ft' not in self.args.prm_model_name:
                    # 首先提取出报错信息
                    pure_feedback = failed_tests.strip()
                    if failed_tests == '':
                        pure_feedback = 'All public test cases passed!'
                    # 让LLM给每个state打分
                    # form thoughts states list
                    cur_state, thoughts, id_node_dict, state_list = self.form_thought_state_list(node)

                    # 让LLM一次性给所有thought打分
                    thought_score = self.llm_score_thought(state_list[-1], code, pure_feedback)
                    if len(thought_score) >= len(thoughts):
                        thought_score = thought_score[:len(thoughts)]
                    else:
                        thought_score = [np.random.rand() for _ in range(len(thoughts))]

                    # 让LLM一个一个给thought打分
                    # thought_score = []
                    # for i, each_state in enumerate(state_list):
                    #     score = self.llm_score_thought(each_state, code, pure_feedback)
                    #     thought_score.append(score)

                    prm_score = agg_prm_scores(self.args.prm_agg_mode, thought_score)
                    value = 1.0 * complete_prog_score + 1.0 * prm_score
                else:
                    # 首先提取出报错信息
                    pure_feedback = failed_tests.strip()
                    if failed_tests == '':
                        pure_feedback = 'All public test cases passed!'
                    # form thoughts states list
                    cur_state, thoughts, id_node_dict, state_list = self.form_thought_state_list(node)

                    _, scores = self.llm_verifier.process_judge(self.cur_prob_instance['prompt'],
                                                                [t[1] for t in thoughts],
                                                                code=code,
                                                                feedback=pure_feedback)
                    prm_score = agg_prm_scores(self.args.prm_agg_mode, scores)
                    value = 1.0 * complete_prog_score + 1.0 * prm_score
        return value

    def backpropagate_mc_result(self, node, code, code_exe_score, saved_feedback):
        # 记录在所有节点的mc return中
        tmp_node = node.parent
        assert (type(tmp_node) == MineChanceNode)
        while tmp_node:
            tmp_node.mc_returns.append(code_exe_score)
            tmp_node = tmp_node.parent.parent

        # 所有父节点存储生成的code和结果
        tmp_node = node.parent
        while tmp_node:
            tmp_node.mc_code_exe.append(
                {
                    'code': code,
                    'execution_feedback': saved_feedback,
                    'test_pass_rate': code_exe_score,
                }
            )
            tmp_node = tmp_node.parent.parent

    def backpropagate_value(self, node, rewards, value):
        node.visits += 1
        node = node.parent
        assert (type(node) == MineChanceNode)
        while node:
            if len(rewards) != 0:
                value = rewards.pop() + self.gamma * value
            node.sampled_returns.append(value)
            node.parent.visits += 1
            node = node.parent.parent
        assert len(rewards) == 0

    def renew_trace_wrong_thoughts(self, ori_node, cur_code, pure_feedback):
        # form thoughts states list
        cur_state, thoughts, id_node_dict, state_list = self.form_thought_state_list(ori_node)
        # locate wrong node
        wrong_id, wrong_node = self.locate_wrong_node(cur_state, cur_code, pure_feedback, id_node_dict, state_list, ori_node, thoughts)
        # refine thought
        return self.refine_thoughts(ori_node, wrong_id, wrong_node, cur_state, cur_code, pure_feedback)

    def refine_thoughts(self, ori_node, wrong_id, wrong_node, cur_state, cur_code, pure_feedback):
        if isinstance(wrong_id, list):
            to_refine_ids = wrong_id
            to_refine_nodes = wrong_node
            to_bfs_node = wrong_node[wrong_id.index(min(wrong_id))]
        else:
            to_refine_ids = [wrong_id]
            to_refine_nodes = [wrong_node]
            to_bfs_node = wrong_node

        for i, refine_id in enumerate(to_refine_ids):
            # 将错误的thought进行修正
            system_msg = f"You are an expert programmer."
            input_prompt = f"""
{cur_state}\n{cur_code}\n{pure_feedback}
Above is the combination of problem, thoughts and code. 
Each thought is bounded with an id number at the beginning of the thought.
* Revise and enhance the {refine_id}-Thought above in the thoughts by providing a improved new thought to replace it.
* Remember that you only need to provide the new thought in one or two sentences, not the code.
"""
            print('\n-----------14renew action input prompt')
            print(input_prompt)
            response, _ = self.generator.generate_response_api(input_prompt, top_k=1, max_length=1024,
                                                               system_message=system_msg)
            if 'Thought:' not in response:
                response = '\nThought: ' + response

            print('\n-----------15renew action response')
            print(response)

            to_refine_nodes[i].action = response
            to_refine_nodes[i].parent.possible_actions[to_refine_nodes[i].chance_id] = response

        # 用BFS更新wrong node的所有子树节点的state
        queue = deque()
        queue.append(to_bfs_node.children[-1])
        while queue:
            # 从队列中取出当前的DecisionNode
            current_decision_node = queue.popleft()

            # 为当前DecisionNode更改状态,清空其mc_returns和code_exe
            current_decision_node.renew_state(initial_state=self.cur_prob_instance['prompt'])
            current_decision_node.parent.mc_returns = []
            current_decision_node.parent.mc_code_exe = []
            current_decision_node.parent.saved_flag = False  # rethink之后，其saved_flag设置为False

            # 检查当前DecisionNode是否有children
            if hasattr(current_decision_node, 'children'):
                # 遍历当前DecisionNode的所有ChanceNode
                for chance_node in current_decision_node.children:
                    # 将指向的子节点（DecisionNode）加入队列
                    if len(chance_node.children) > 0:
                        queue.append(chance_node.children[-1])

        # 返回更新之后的node
        return ori_node

    def form_thought_state_list(self, ori_node):
        # 用list方式形成新的 state
        thoughts = []
        id_node_dict = {}
        cur_node = ori_node.parent
        assert (type(cur_node) == MineChanceNode)
        while cur_node:
            if len(self.tokenizer.encode(cur_node.action)) >= 3:  # 有的action是/n就不添加了,这里可能导致无限循环,如果停在一个action为/n的节点上，所有更新cur_node的代码一定要在if外面
                thoughts.append([cur_node, cur_node.action])
            cur_node = cur_node.parent.parent
        thoughts = thoughts[::-1]
        # 给thought添加序号
        for i in range(len(thoughts)):
            if 'Thought:' in thoughts[i][1]:
                thoughts[i][1] = '\n' + thoughts[i][1].split('Thought:')[1]
            thoughts[i][1] = f"\n" + f"{i+1}-Thought: " + thoughts[i][1].strip()
            id_node_dict[i+1] = thoughts[i][0]

        # each state list
        state_list = []
        tmp_state = self.cur_prob_instance['prompt']
        for thought in thoughts:
            tmp_state = tmp_state + thought[1]
            state_list.append(tmp_state)

        # current state
        cur_state = self.cur_prob_instance['prompt']
        if len(thoughts) > 0:
            cur_state = cur_state + f'\nThoughts:\n'
            for thought in thoughts:
                cur_state = cur_state + thought[1]
        return cur_state, thoughts, id_node_dict, state_list


    def locate_wrong_node(self, cur_state, cur_code, pure_feedback, id_node_dict, state_list, ori_node, thoughts):
        # 定位错误thought
        if self.args.verify_pos_mode == 'llm':
            # 让LLM判断几号出错了
            wrong_id = self.llm_locate_wrong_thought(cur_state, cur_code, pure_feedback, thoughts)
            wrong_node = id_node_dict[wrong_id]
        elif self.args.verify_pos_mode == 'random':
            wrong_id = np.random.choice(list(id_node_dict.keys()))
            wrong_node = id_node_dict[wrong_id]
        elif self.args.verify_pos_mode == 'last':
            wrong_id = len(thoughts)
            wrong_node = id_node_dict[wrong_id]
        elif self.args.verify_pos_mode == 'omega_mc':
            wrong_id, wrong_node = self.omega_mc_judge(ori_node, id_node_dict, thoughts)
            # 存储链条上的正确结点和错误结点到json文件中
            if self.args.json_save_mc:
                self.save_labels_rollout_to_db(id_node_dict, ori_node, wrong_id, state_list, thoughts)
        elif self.args.verify_pos_mode in ['llm_score', 'llm_score_thresh']:
            # 获得step scores
            if 'ft' not in self.args.prm_model_name:
                # 让LLM一次性给所有thought打分
                step_scores = self.llm_score_thought(state_list[-1], cur_code, pure_feedback)
                if len(step_scores) >= len(thoughts):
                    step_scores = step_scores[:len(thoughts)]
                else:
                    step_scores = [np.random.rand() for _ in range(len(thoughts))]

                # # 让LLM一个一个打分
                # step_scores = []
                # for i, each_state in enumerate(state_list):
                #     score = self.llm_score_thought(each_state, cur_code, pure_feedback)
                #     step_scores.append(score)
            else:
                _, step_scores = self.llm_verifier.process_judge(self.cur_prob_instance['prompt'],
                                                                [t[1] for t in thoughts],
                                                                code=cur_code,
                                                                feedback=pure_feedback)

            # 设置wrong_id 为得分最低的 / 得分低于某个threshold的
            if self.args.verify_pos_mode == 'llm_score_thresh':
                wrong_id = [i + 1 for i, score in enumerate(step_scores) if score < self.args.prm_verify_thresh]
                if len(wrong_id) == 0:
                    wrong_id = [step_scores.index(min(step_scores)) + 1]
                wrong_node = [id_node_dict[i] for i in wrong_id]
            else:
                wrong_id = step_scores.index(min(step_scores)) + 1
                wrong_node = id_node_dict[wrong_id]
        else:
            raise ValueError("Unknown verify_pos_mode")
        return wrong_id, wrong_node

    def omega_mc_judge(self, ori_node, id_node_dict, thoughts):
        # 由于backward机制，只要子节点的sample times够次数，则所有父节点的sample times也够次数
        while len(ori_node.parent.mc_returns) < self.args.mc_nums:
            state = ori_node.state
            code_id = self.generator.get_rationale_predicted_sequence(self.tokenizer.encode(state), temperature=0.7)
            code = self.tokenizer.decode(code_id)
            if self.args.json_save_mc:
                # 保存数据时，用所有的test和train的数据来测试
                full_result_train = self.get_reward(code_id, with_verbal=True, mode='train')
                full_result_test = self.get_reward(code_id, with_verbal=True, mode='test')
                full_result = [(full_result_train[0] + full_result_test[0]) / 2,
                               full_result_train[1] + full_result_test[1]]
            else:
                full_result = self.get_reward(code_id, with_verbal=True)
            complete_prog_score, verbal_feedbacks = full_result[0], full_result[1]
            saved_feedback = self.gather_saved_feedback(verbal_feedbacks)
            self.backpropagate_mc_result(ori_node, code, complete_prog_score, saved_feedback)

        # 对id_node_dict中的节点按照二分法，如果一个节点sample returns中有大于0的，则该节点为对，说明错误的在后一半
        left = 1
        right = len(id_node_dict)
        while left < right:
            mid = (left + right) // 2
            if np.max(id_node_dict[mid].mc_returns) < 1.0: # 没有能够产生正确代码，注意这里和原文中的mc不同，应<1.0保持意义一样
                right = mid
            else:
                left = mid + 1
        wrong_id = left
        wrong_node = id_node_dict[wrong_id]
        return wrong_id, wrong_node

    def save_labels_rollout_to_db(self, id_node_dict, ori_node, wrong_id, state_list, thoughts):
        cur_save_trace_count = len(self.save_mc_process_scores['label'])
        self.save_mc_process_scores['label'][f'trace_{cur_save_trace_count}_steps'] = []
        for id in id_node_dict.keys():
            # 将mc_returns中等于1.0的mc为1.0，其他为0.0
            tmp_mc_returns = [1.0 if mc == 1.0 else 0.0 for mc in id_node_dict[id].mc_returns]
            score = sum(tmp_mc_returns) / len(tmp_mc_returns)

            # selection text - start decision
            full_trace_path_text = self.get_node_path(ori_node.parent)
            current_node_path_text = self.get_node_path(id_node_dict[id])
            selection_text = f"selection_{self.args.rollout_count}: \n{full_trace_path_text}"
            cur_node_path_text = f"cur_thought_path: \n{current_node_path_text}"

            # save labeled thoughts
            if id <= wrong_id:
                self.save_mc_process_scores['label'][f'trace_{cur_save_trace_count}_steps'].append(
                    {'selection_trace': selection_text,
                     'state': state_list[id - 1],
                     'thought_index_in_trace': id,
                     'thought': thoughts[id - 1][1],
                     'thought_path': cur_node_path_text,
                     'code_exe': id_node_dict[id].mc_code_exe[:],
                     'saved_flag': id_node_dict[id].saved_flag,
                     'mcs': score,
                     'current_trace_score': self.current_trace_score,
                     'current_trace_train_score': self.current_trace_train_score,
                     'current_trace_code': self.current_trace_code,
                     'current_trace_train_exe': self.current_trace_train_exe,
                     'current_trace_exe': self.current_trace_exe
                     })
                # 保存后，将其saved_flag设置为True
                id_node_dict[id].saved_flag = True

    def llm_locate_wrong_thought(self, state, code, pure_feedback, thoughts):
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
            wrong_id = int(response_id[0])
        except Exception as e:
            print(f"Error in parsing evaluation response: {repr(e)}{e}")
            wrong_id = len(thoughts)  # 修改最近的一个
        if wrong_id <= 0 or wrong_id > len(thoughts):
            wrong_id = len(thoughts)
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
* The JSON should be a **dict**, the dict items are split with comma ','.
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

    def get_evaluation(self, cur_state, cur_code=None):
        # 原文件中verbal memory的evaluation方式
        system_msg = f"You are a evaluator that evaluates the code is suitable for solving a given problem."
        input_prompt = f"""
{cur_state}\n\n{cur_code}
Above is a Python code problem with the thoughts and code to solve the problem. The code could pass all the example test cases, however, it may or may not be completely correct.
Please evaluate and return the correctness score in range [-1, 1]
Evaluate the correctness of the code and give only ONE evaluation score.
The code's correctness is whether it can pass all the possible unseen test cases of the problem, not just the given ones.
Example Answers: 
{{\"evaluation\": -0.5,  \"explanation\": \"The code is far from correct for solving the problem.\"}}
{{\"evaluation\": 1.0, \"explanation\": \"The generated code is the correct solution that can pass all the possible test cases and strange corner cases too. \"}} 
{{\"evaluation\": 0.1, \"explanation\": \"The code is not the correct solution but can pass some simple test cases. \"}} 
{{\"evaluation\": 0.85, \"explanation\": \"The code can pass most test cases while may fail on some corner cases. \"}} 
"""

        print('\n--------------5 evaluation input prompt')
        print(input_prompt)

        response, _ = self.generator.generate_response_api(input_prompt, top_k=1, max_length=1024, system_message=system_msg)

        print('\n--------------6 evaluation response')
        print(response)

        try:
            float_pattern = re.compile(r'-?\d+\.?\d*')
            response_scores = [float(match) for match in float_pattern.findall(response)]
            evaluation = response_scores[0]
        except Exception as e:
            print(f"Error in parsing evaluation response: {repr(e)}{e}")
            evaluation = 0.0
        return evaluation

    def gather_feedback(self, verbal_feedbacks, state, code):
        failed_tests = ''
        verbalFeedback = ''
        failed_test_list = []

        # 删除code_exe_data['execution_feedback']中不是dict的那些元素
        saved_feedbacks = self.gather_saved_feedback(verbal_feedbacks=verbal_feedbacks)

        for k, feedback in enumerate(saved_feedbacks):
            failed_tests += f"\n\n## Failed test {k + 1}: \n{feedback['output']}"
            failed_test_list.append(feedback['output'])

        if failed_tests != '':
            verbalFeedback = f"""
{state}\n\n{code}
Above is the combination of problem, thoughts and code. The code is generated following the thoughts to solve the problem.
However, the code generated following the thoughts doesn't pass some test cases. Here are the test cases the code doesn't pass:
{failed_tests}\n
"""
            self.args.verbal_length_check_num += 1
            if len(self.tokenizer.encode(verbalFeedback)) > 12000:
                self.args.verbal_length_exd_num += 1
                verbalFeedback = self.tokenizer.decode(self.tokenizer.encode(verbalFeedback)[:8000])
        return failed_tests, verbalFeedback, saved_feedbacks

    def gather_saved_feedback(self, verbal_feedbacks):
        saved_feedback = [item for item in verbal_feedbacks if isinstance(item, dict)]
        ori_length = len(verbal_feedbacks)
        saved_feedback = [
            {
                'error': shorten_text_no_tokenizer(item['error'], total_length=50000),
                'output': shorten_text_no_tokenizer(item['output'], total_length=50000)
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
                    'output': shorten_text_block(item['output'], self.tokenizer, total_length=512,
                                                 section_length=512)
                }
                for item in saved_feedback
            ]
        saved_feedback = tmp_save
        return saved_feedback

    def get_node_path(self, node):
        selection_print = []
        tmp_node = node
        selection_print.append(f'{tmp_node.chance_id}')
        while tmp_node.parent.parent:
            tmp_node = tmp_node.parent.parent
            selection_print.append(f'{tmp_node.chance_id}')
        tmp_text = '->'.join(selection_print[::-1])
        path_text = f"{'root->' + tmp_text}"
        return path_text

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
        next_state = s + a
        return next_state

    def ucb(self, node):
        """
        Upper Confidence Bound of a chance node
        """
        return chance_node_value(node) + self.args.ucb_constant * sqrt(log(node.parent.visits)) / (1 + len(node.sampled_returns))

    def p_ucb(self, node):
        """
        Upper Confidence Bound of a chance node, weighted by prior probability
        """
        return chance_node_value(node) + self.args.ucb_constant * node.prob * sqrt(log(node.parent.visits)) / (1 + len(node.sampled_returns))

    def var_p_ucb(self, node):
        """
        Upper Confidence Bound of a chance node, the ucb exploration weight is a variable
        """
        ucb_parameter = log((node.parent.visits + self.ucb_base + 1) / self.ucb_base) + self.args.ucb_constant
        return chance_node_value(node) + ucb_parameter * node.prob * sqrt(log(node.parent.visits)) / (1 + len(node.sampled_returns))


class MineDecisionNode:
    """
    Decision node class, labelled by a state

    Args:
        dp: default policy, used to prioritize and filter possible actions
    """

    def __init__(self, parent, state, generator=None, id=None, decision_memory=[], tokenizer=None):
        self.id = id
        self.parent = parent
        self.state = state
        if self.parent is None:  # Root node
            self.depth = 0
        else:  # Non root node
            self.depth = parent.depth + 1
        self.generator = generator
        self.tokenizer = tokenizer
        self.explored_children = 0
        # this decision node should be visited at least once, otherwise p-uct makes no sense for this node
        self.visits = 1
        # used to save any information of the state
        # we use this for saving complete programs generated from it
        self.info = {}
        self.decision_memory = decision_memory
        self.second_chance_flag = True

    def __expand__(self, verbal_feedback=''):
        if verbal_feedback == '':
            expand_prompt_id = self.tokenizer.encode(self.state)
            top_k_line_predict, top_k_scores = self.generator.get_top_k_rationale_predict(expand_prompt_id)
        else:
            expand_prompt_id = self.generator.tokenizer.encode(verbal_feedback)
            top_k_line_predict, top_k_scores = self.generator.get_top_k_rationale_predict(expand_prompt_id, with_verbal=True)

        self.possible_actions = [self.tokenizer.decode(line) for line in top_k_line_predict]
        self.action_scores = top_k_scores

        # populate its children
        self.children = [MineChanceNode(self, (act, score), chance_memory=self.decision_memory, chance_id=chance_id) for chance_id, (act, score) in enumerate(zip(self.possible_actions, self.action_scores))]

    def renew_state(self, initial_state):
        # split way
        thoughts = []
        cur_node = self.parent
        while cur_node:
            thoughts.append(cur_node.action)
            cur_node = cur_node.parent.parent
        cur_state = initial_state
        if len(thoughts) > 0:
            for thought in thoughts[::-1]:
                cur_state = cur_state + thought  # 注意这里一定不要加别的，因为selection会根据这个判断节点是否被访问过
        self.state = cur_state

    def is_fully_expanded(self):
        return all([child.expanded() for child in self.children])


class MineChanceNode:
    """
    Chance node class, labelled by a state-action pair
    The state is accessed via the parent attribute
    """

    def __init__(self, parent, action_and_score, chance_memory=[], chance_id=None):
        self.parent = parent
        self.action = action_and_score[0]
        self.depth = parent.depth
        self.children = []
        self.prob = action_and_score[1]  # the probability that this action should be token, provided by default policy
        self.sampled_returns = []
        self.mc_returns = []  # only for mc process evaluation
        self.mc_code_exe = []
        self.chance_memory = chance_memory
        self.chance_id = chance_id
        self.saved_flag = False

    def expanded(self):
        return len(self.children) > 0


def chance_node_value(node, mode="best"):
    """
    Value of a chance node
    """
    if len(node.sampled_returns) == 0:
        return 0

    if mode == "best":
        # max return (reasonable because the model is deterministic?)
        return max(node.sampled_returns)
    elif mode == "sample":
        # Use average return
        return sum(node.sampled_returns) / len(node.sampled_returns)
    else:
        raise Exception(f"Unknown tree search mode {mode}")