# -*- coding:utf-8 _*-
# Process directory
import sys
import os
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path[:cur_path.find('CodePRM')] + 'CodePRM')  # 这里要改为你自己的项目的主目录
import warnings
warnings.filterwarnings('ignore')

# CUDA devices
os.environ['CUDA_VISIBLE_DEVICES'] = '4'  # 必须放在import各种python的包之前运行
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
# torch.multiprocessing.set_start_method('spawn')

import shutil
from llm_service import *
import copy
from dataSet import *
import pickle
from tqdm import tqdm
import time

# openai keys
import openai
os.environ["TOKENIZERS_PARALLELISM"] = "false"

API_KEY_SOURCE_LIST = {
    'openai': 'xxx'
}
API_KEY_SOURCE_NAME = 'openai'
API_KEY = API_KEY_SOURCE_LIST[API_KEY_SOURCE_NAME]

openai.api_key = API_KEY
# os.environ["http_proxy"] = "http://127.0.0.1:8888"
# os.environ["https_proxy"] = "http://127.0.0.1:8888"
# os.environ["all_proxy"] = "socks5://127.0.0.1:8889"
os.environ["OPENAI_API_KEY"] = API_KEY

# imports
import torch
from Processor import Processor
from argparse import ArgumentParser
from accelerate import Accelerator
from utils import *
import pprint


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

def seed_everything(seed):
    # 1. 设置Python内置的random模块的全局随机种子
    random.seed(seed)
    # 2. 设置Python的哈希种子，使得哈希函数的行为是可预测的。这在某些使用字典和集合的操作中可能会影响顺序，从而影响结果。
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 3. 设置Numpy库的全局随机种子
    np.random.seed(seed)
    # 4. 设置PyTorch库的CPU部分的随机种子
    torch.manual_seed(seed)
    # 5. 设置PyTorch库的GPU部分的随机种子。如果使用多块GPU进行并行计算，确保所有GPU的随机数生成器都使用相同的种子。
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 6. 确保使用确定性算法。这对结果的可复现性非常重要，但可能会影响计算性能。
    torch.backends.cudnn.deterministic = True
    # 7. 关闭CuDNN的自动优化功能。这可以保证每次运行的计算图和内存分配都是一致的，从而提高结果的可复现性。
    torch.backends.cudnn.benchmark = False

def get_available_gpus():
    # 读取 CUDA_VISIBLE_DEVICES 环境变量
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible_devices == "":
        # 如果未设置 CUDA_VISIBLE_DEVICES，则认为没有 GPU 可用
        return 0
    else:
        # 解析 CUDA_VISIBLE_DEVICES 并计算可用 GPU 数量
        return len(cuda_visible_devices.split(","))

def align_args(args):
    """
    APPS:
    intro: 4000-4999
    inter: 0000-2999
    competition: 3000-3999
    """
    if args.model == 'TreeDataCollect':
        args.json_save_mc = True

    if args.dataset == 'apps':
        if args.index is None and not args.json_save_mc:
            if args.APPDifficulty == 'introductory':
                args.start = 4000
                args.end = 4100
            elif args.APPDifficulty == 'interview':
                args.start = 0
                args.end = 100
            elif args.APPDifficulty == 'competition':
                args.start = 3000
                args.end = 3100

    if args.dataset == 'codeforces':
        args.start = 0
        args.end = 10

    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.arch in ['gpt3.5', 'kimi', 'gpt4', 'gpt4o-mini', 'gpt4o', 'o1-preview', 'o1-mini','gpt2', 'gpt-neo',]:
        args.call_mode = 'api'

    if args.experiment_idx == 2:
        args.rerun = True

    args.save = f"{args.save}/{args.dataset}/Experiment_{args.experiment_idx}/{args.repeat_id}/{args.APPDifficulty}/"
    print(f'save dir: {args.save}')

    if args.rerun:
        if os.path.exists(args.save):
            shutil.rmtree(args.save)
            print(f"'{args.save}' has been removed and recreated.")
            # 这是为了每次都重新生成结果，文件夹里不会混杂两个实验的结果
            # 且可以通过rerun=False来继续跑之前的实验

    os.makedirs(args.save, exist_ok=True)
    print(pprint.pformat(vars(args)))
    return args


model_choices = ['CodePRMMCTS',
                 'bs',
                 'BestOfNStep',
                 'TreeDataCollect',
                 ]
arch_choices = ['gpt4o-mini',
                'deepseek-reasoner',
                'DeepSeek-Coder-V2-Lite-Instruct',
                'Qwen2.5-Coder-7B-Instruct',
                'Qwen2.5-Coder-7B-Instruct-value-ft-priv-True',
                'Qwen2.5-Coder-7B-Instruct-value-ft-priv-False',
                ] # DeepSeek-Coder-V2-Lite-Instruct is 16B


def main():
    parser = ArgumentParser("CodePRM")
    # -------------------- 跑实验前设置一下 , 文件上方还有一个cuda device设置一下--------------------------
    parser.add_argument('-d', '--dataset', type=str, default='codeforces', choices=['apps', 'humaneval', 'codeforces'])
    parser.add_argument('-m', '--model', type=str, default='CodePRMMCTS', choices=model_choices)
    parser.add_argument('-eid', '--experiment-idx', type=int, default=2)
    parser.add_argument('-r', '--repeat_id', type=int, default=1)
    parser.add_argument("--arch", default="gpt4o-mini", choices=arch_choices)
    parser.add_argument("--rollout", default=4, type=int, help="The maximum number of rollouts for PG-TD.")
    parser.add_argument("--save", type=str, default="./results", help="Directory to save generated code.")
    parser.add_argument('--rerun', action='store_true', default=False,
                        help="If True, rerun if the output file already exists.")
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument("--small_part_run", type=str2bool, default=True,
                        help="If True, the questions that all methods pass by gpt3.5 could be avoid. Only for advanced methods.")
    parser.add_argument('--json_save_all', type=str2bool, default=False, help='If True, save all json files.')
    parser.add_argument('--api_key_source_name', type=str, default=API_KEY_SOURCE_NAME)
    parser.add_argument('--generator_api_port', type=int, default=12344)
    parser.add_argument('--prm_api_port', type=int, default=12345)
    parser.add_argument('--no_verb', type=str2bool, default=False)
    # -------------------------- dataset --------------------------
    parser.add_argument("-i", "--index", default=None, type=int)
    # 129, 81 is hard, could be used for single test; humaneval_10 for reset child; 24 for verbal feedback
    # single trace 65 humaneval can have wrong trace
    parser.add_argument("--start", default=4000, type=int)
    parser.add_argument("--end", default=4100, type=int)
    # ------------- APPS -------------
    parser.add_argument("--APPDifficulty", default="1200", choices=['introductory', 'interview', 'competition', '1200', '1500', '1700', '1300', '1400', '1600'],
                        help="The difficulty of the problems to solve.")
    # ------------- CodePRMMCTS --------------------
    parser.add_argument("--json_save_mc", type=str2bool, default=False, help="If True, save the json files of MC labels.")
    parser.add_argument("--verify_pos_mode", default='llm', choices=['llm', 'random', 'last', 'omega_mc', 'llm_score', 'llm_score_thresh'])
    parser.add_argument("--prm_verify_thresh", default=0.5, type=float)
    parser.add_argument("--mc_nums", default=3, type=int)
    # ------------- PRM --------------------
    parser.add_argument("--prm_model_name", default='Qwen2.5-Coder-7B-Instruct-value-ft-priv-True', choices=arch_choices)
    parser.add_argument('--reward_mode', type=str, default='pub_test', choices=['pub_test', 'all_test', 'pub_llm', 'pub_prm', 'random', 'prm'])
    parser.add_argument('--prm_agg_mode', type=str, default='max', choices=['max', 'min', 'mean', 'last'])
    parser.add_argument('--with_execution_enhance', type=str2bool, default=True)
    # ------------- TreeDataCollect --------------------
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--c_puct", type=float, default=0.125)
    parser.add_argument("--L", type=int, default=2048)
   # -------------------------- mcts --------------------
    parser.add_argument("--width", default=3, type=int, help="The maximum number of children for any node.")
    parser.add_argument("--horizon", default=16000, type=int, help="The maximum number of tokens to generate.")
    parser.add_argument("--num-beams", default=1, type=int, help="The number of beams for beam search or PG-TD.")
    parser.add_argument("--num-samples", default=1, type=int, help="The number of samples for Sampling + Filtering.")
    parser.add_argument("--max-sample-times", default=768, type=int, help="The maximum number of Transformer generation function calls."
                                                                          "Program stops when this number is reached (default to be 512 * 1.5 = 768).")
    parser.add_argument("--ucb-constant", default=4., type=float)
    parser.add_argument("--ucb-base", default=10., type=float)
    parser.add_argument("--uct-alg", default="var_p_uct", choices=["uct", "p_uct", "var_p_uct"],
                        help="The UCT algorithm to use."
                             "`uct` is the original UCT algorithm,"
                             "`p_uct` is the UCT algorithm with PUCT,"
                             "and `var_p_uct` is the UCT algorithm with variable PUCT.")
    parser.add_argument('--top-k-cache-steps', type=int, default=1024, help="Number of forward steps to cache top k caches, default 1024 means the whole horizon.")
    parser.add_argument("--public-cases-type", type=str, default='half', help="Number of public test cases to use for evaluation.")
    parser.add_argument("--ts-mode", default="best", choices=["best", "sample"], help="Tree search mode within the evaluation step. `best` uses beam search, `sample` uses sampling.")
    parser.add_argument("--max_think_times", default=2, type=int, help="The max num of think times")
    # ------------------------- 全局变量 -----------------------
    parser.add_argument('--total_input_token_num', type=int, default=0, help='The maximum number of tokens to input.')
    parser.add_argument('--total_output_token_num', type=int, default=0, help='The maximum number of tokens to output.')
    parser.add_argument('--failed_json_num', type=int, default=0, help='The number of failed json format output.')
    parser.add_argument('--all_json_num', type=int, default=0, help='The number of all json format output.')
    parser.add_argument('--verbal_length_exd_num', type=int, default=0, help='The number of length of verbal length too long')
    parser.add_argument('--verbal_length_check_num', type=int, default=0, help='The number of length check of verbal feedback')
    parser.add_argument('--rollout_count', type=int, default=-1, help='The rollout count')
    parser.add_argument('--generate_tests_total', type=int, default=0, help='The generate tests count')
    parser.add_argument('--failed_generate_tests_count', type=int, default=0, help='The failed generated tests count total')
    parser.add_argument('--rethink_total_nums', type=int, default=0, help='The total number of rethink')
    parser.add_argument('--rethink_effective_nums', type=int, default=0, help='The effective number of rethink')
    parser.add_argument('--rethink_failed_nums', type=int, default=0, help='The failed number of rethink')
    parser.add_argument('--rethink_success_nums', type=int, default=0, help='The success number of rethink')
    parser.add_argument('--no_rethink_success_num', type=int, default=0, help='The success number of no rethink')
    # ---------------------- 全局常量 ----------------------------------
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--cudaDevice', type=str, default='cuda')

    args = parser.parse_args()
    args = align_args(args)

    # seed everything
    if args.repeat_id == 2:
        seed_everything(5)
    elif args.repeat_id == 3:
        seed_everything(10)

    # dataset
    if args.index is not None:
        problem_indices = [args.index]
    elif args.end is not None:
        problem_indices = range(args.start, args.end)
    else:
        raise Exception("Don't know what problems to solve.")

    if args.dataset == 'apps':
        data_handler = APPSHandler(problem_indices, args)
    elif args.dataset == 'codeforces':
        data_handler = CodeforcesHandler(problem_indices, args)
    elif args.dataset == 'humaneval':
        if args.index is None:
            problem_indices = [i for i in range(164)]
            # self.problem_indices = [99, 81, 145, 118, 91, 163, 141, 119, 120, 147, 83, 32, 126, 75, 76, 10, 132, 155, 125, 115, 108, 129]
        data_handler = HumanevalHandler(problem_indices, args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # 对codeforces需要重新定向problem_indices
    if args.dataset == 'codeforces':
        problem_indices = [problem_instance['index'] for problem_instance in data_handler.problems]

    # running & parallel running
    # 加载prm local model和generator local model
    generator_local_model_actor = None
    prm_local_model_actor = None
    if 'ft' in args.prm_model_name:
        prm_local_model_actor = LocalModelActor(args.prm_model_name)
    # if args.arch in ['DeepSeek-Coder-V2-Lite-Instruct', 'Qwen2.5-Coder-7B-Instruct']:
    #     generator_local_model_actor = LocalModelActor(args.arch, vllm_mode=True)

    setattr(args, "generator_local_model_actor", generator_local_model_actor)
    setattr(args, "prm_local_model_actor", prm_local_model_actor)
    runner = Processor(args)
    runner.run(problem_indices, data_handler.problems)


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
