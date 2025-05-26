import numpy as np
import torch
from utils import *
import os
import json


def reset_test_cases(problem_instance):
    train_in_outs, test_in_outs = {}, {}

    total_in_outs = {
        'inputs': [],
        'outputs': []
    }
    for i, input_data in enumerate(problem_instance["train_in_outs"]['inputs']):
        total_in_outs['inputs'].append(input_data)
        total_in_outs['outputs'].append(problem_instance["train_in_outs"]['outputs'][i])
    for i, input_data in enumerate(problem_instance["test_in_outs"]['inputs']):
        total_in_outs['inputs'].append(input_data)
        total_in_outs['outputs'].append(problem_instance["test_in_outs"]['outputs'][i])

    public_test_cases_num = min(15, len(total_in_outs['inputs']) // 2)
    private_test_cases = len(total_in_outs['inputs']) - public_test_cases_num
    if public_test_cases_num < 1 or private_test_cases < 1:
        raise Exception(f"Not enough test cases: {public_test_cases_num}, {private_test_cases}.")
    train_in_outs['inputs'] = total_in_outs['inputs'][:public_test_cases_num]
    train_in_outs['outputs'] = total_in_outs['outputs'][:public_test_cases_num]
    test_in_outs['inputs'] = total_in_outs['inputs'][public_test_cases_num:]
    test_in_outs['outputs'] = total_in_outs['outputs'][public_test_cases_num:]
    return train_in_outs, test_in_outs



class CodeforcesHandler:
    def __init__(self, problem_indices, args):
        self.problems = []
        problem_count = 0
        all_problems = []
        base_dir = os.path.join(get_proj_path(), "dataProcess", "codeforces")

        # 遍历子文件夹和JSON文件
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".json"):  # 检查文件是否是JSON文件
                    json_path = os.path.join(root, file)
                    try:
                        # 打开并读取JSON文件
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            data['file_path']= json_path
                            all_problems.append(data)  # 将读取的数据加入列表
                    except Exception as e:
                        print(f"Failed to read {json_path}: {e}")

        for problem_instance in all_problems:
            problem_instance['code_type'] = 'standard_input'
            problem_instance['method_name'] = None

            # difficulty filtering
            if args.APPDifficulty is not None:
                if int(problem_instance['rating']) != int(args.APPDifficulty):
                    # r_file_path = problem_instance['file_path']
                    # os.remove(r_file_path)
                    continue

            if args.model == 'TreeDataCollect':
                if problem_count <= max(problem_indices):
                    problem_count += 1
                    continue
            else:
                if problem_count > max(problem_indices):
                    # r_file_path = problem_instance['file_path']
                    # os.remove(r_file_path)
                    continue

            problem_instance['index'] = problem_count

            # problem instance formation
            input_prompt = "\nQUESTION:\n"
            input_prompt += problem_instance['full_description']
            input_prompt += "\nUse Standard Input format"  # \n"
            input_prompt += "\nANSWER:\n"
            problem_instance["prompt"] = input_prompt

            # test cases for train and test
            train_in_outs, test_in_outs = reset_test_cases(problem_instance)
            problem_instance["train_in_outs"] = train_in_outs
            problem_instance["test_in_outs"] = test_in_outs
            self.problems.append(problem_instance)
            problem_count += 1
