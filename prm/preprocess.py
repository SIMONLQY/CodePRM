from utils import *
import shutil
import os
from tqdm import tqdm
import numpy as np
import random

# 随机删除一半元素的函数
def remove_half(data_list):
    # 随机抽取一半的元素
    num_to_keep = len(data_list) // 2
    sampled_data = random.sample(data_list, num_to_keep)
    return sampled_data


def get_problem_traces(mc_jsons, mc_jsons_dir):
    # 收集所有 traces，按 problem_id 分组
    problem_traces = {}
    for _, mc_json in tqdm(enumerate(mc_jsons), total=len(mc_jsons), desc="First Pass - Collecting Data"):
        with open(os.path.join(mc_jsons_dir, mc_json), 'r') as f:
            json_data = json.load(f)
        problem_id = int(json_data['ques_id'])
        question = json_data['question']

        if problem_id not in problem_traces:
            problem_traces[problem_id] = {
                'question': question,
                'traces': []
            }

        for trace_id, trace_datas in json_data['label'].items():
            if len(trace_datas) == 0:
                print(f"empty trace: {problem_id}-{trace_id}")
                continue
            # 存储 trace_datas 以便后续处理
            problem_traces[problem_id]['traces'].append(trace_datas)
    return problem_traces

def get_problem_step_max_lengths(problem_traces):
    problem_max_lengths = {}
    for problem_id, data in tqdm(problem_traces.items(), desc="Calculating Max Lengths"):
        problem_max_lengths[problem_id] = {}
        for trace_datas in data['traces']:
            for _, step_data in enumerate(trace_datas):
                # 使用步骤的state_step作为唯一标识符
                step_anchor = f"{step_data['state']}\n\n{step_data['thought']}"
                if step_anchor not in problem_max_lengths[problem_id]:
                    problem_max_lengths[problem_id][step_anchor] = len(step_data.get('total_pass_rate_list', []))
                else:
                    if len(step_data.get('total_pass_rate_list', [])) > problem_max_lengths[problem_id][step_anchor]:

                        problem_max_lengths[problem_id][step_anchor] = len(step_data.get('total_pass_rate_list', []))

    # 统计信息
    max_lengths = []
    for problem_id, data in problem_max_lengths.items():
        for step_anchor, length in data.items():
            max_lengths.append(length)
    mean = np.mean(max_lengths)
    median = np.median(max_lengths)
    max_value = np.max(max_lengths)
    min_value = np.min(max_lengths)

    print(
        f"-------max lengths distribution-------\n"
        f"平均数: {mean}\n"
        f"中位数: {median}\n"
        f"最大值: {max_value}\n"
        f"最小值: {min_value}\n"
    )


    return problem_max_lengths


def read_raw_data(dataset, exp_id, difficulty, step_tag):
    train_json_datas = []
    test_json_datas = []
    statistics = {
        'train_trace_count': 0,
        'train_step_count': 0,
        'train_pos_step_count': 0,
        'train_neg_step_count': 0,
        'test_trace_count': 0,
        'test_step_count': 0,
        'test_pos_step_count': 0,
        'test_neg_step_count': 0,
    }
    mc_jsons_dir = f"{get_proj_path()}/results/{dataset}/Experiment_{exp_id}/1/{difficulty}/mc_labels/"
    mc_jsons = os.listdir(mc_jsons_dir)

    # 对于humaneval，只能用id为100-163的64道题目
    if dataset == 'humaneval':
        mc_jsons = [mc_json for mc_json in mc_jsons if 100 <= int(mc_json.split('.')[0]) <= 163]

    # question id range的前90%加入train，后10%加入test
    question_ids = [int(mc_json.split('.')[0]) for mc_json in mc_jsons]
    min_question_id = min(question_ids)
    train_split_thre = min_question_id + 0.9 * (max(question_ids) - min_question_id)

    # # 收集所有 traces，按 problem_id 分组
    # problem_traces = get_problem_traces(mc_jsons, mc_jsons_dir)
    # # 计算每个 problem_id 下每个 step 的最大 total_pass_rate_list 长度
    # problem_max_lengths = get_problem_step_max_lengths(problem_traces)

    trace_step_nums = []
    for _, mc_json in tqdm(enumerate(mc_jsons)):
        with open(f"{mc_jsons_dir}/{mc_json}", 'r') as f:
            json_data = json.load(f)
        problem_id = int(json_data['ques_id'])

        for trace_id, trace_datas in json_data['label'].items():
            if len(trace_datas) == 0:
                print(f"empty trace: {problem_id}-{trace_id}")
                continue

            process = f""
            process_before_code = f""
            trace_label_list = []
            trace_mcs_list = []
            trace_saved_flag_list = []
            for step_data in trace_datas:
                # # 获取当前步骤的 total_pass_rate_list 长度
                # current_length = len(step_data.get('total_pass_rate_list', []))
                # # 获取该步骤在该 problem_id 下的最大长度
                # step_anchor = f"{step_data['state']}\n\n{step_data['thought']}"
                # max_length = problem_max_lengths[problem_id][step_anchor]
                # # 设置 saved_flag
                # if current_length >= max_length:
                #     trace_saved_flag_list.append(False)
                # else:
                #     trace_saved_flag_list.append(True)

                # 如果mcs大于0则append 1， 否则append0
                trace_mcs_list.append(step_data['mcs'])
                trace_label_list.append(step_data['avg_pass_rate'])
                # trace_label_list.append(1 if step_data['mcs'] > 0 else 0)
                trace_saved_flag_list.append(step_data['saved_flag'])
                process += f"{step_data['thought'].strip()} {step_tag}"
                process_before_code += f"{step_data['thought'].strip()}\n"

            # thought pos的太多会导致不balance，跳过
            # if all(trace_label_list):
            #     rand_num = np.random.rand()
            #     if rand_num < 0.7:
            #         continue

            # thought太多会超出输入长度，放弃
            if len(trace_datas) > 5:
                continue

            # 如果trace中所有step都被保存了，则跳过
            if all(trace_saved_flag_list):
                continue

            # # 随机drop一些intro和inter难度下，代码pub pass 且 真的pass的trace
            # if dataset == 'apps':
            #     if (trace_datas[-1]['current_trace_train_score'] == 1.0 and (problem_id < 3000 or problem_id >3999) and
            #             trace_datas[-1]['current_trace_score'] == 1.0):
            #         rand_num = np.random.rand()
            #         if rand_num < 0.5:
            #             continue
            #     # 随机drop一些intro和inter难度下的trace，对错误和正确的分别drop
            #     if max(trace_mcs_list) == 0.0 and (problem_id < 3000 or problem_id > 3999):
            #         rand_num = np.random.rand()
            #         if rand_num < 0.7:
            #             continue
            #     # 随机drop一些comp难度错误的trace
            #     if max(trace_mcs_list) == 0.0 and (3000 <= problem_id <= 3999):
            #         rand_num = np.random.rand()
            #         if rand_num < 0.4:
            #             continue
            #
            # if dataset == 'apps':
            #     rand_num = np.random.rand()
            #     if rand_num < 0.5:
            #         continue

            # if dataset == 'codeforces':
            #     if trace_datas[-1]['current_trace_train_score'] == 1.0 and trace_datas[-1]['current_trace_score'] == 1.0:
            #         rand_num = np.random.rand()
            #         if rand_num < 0.5:
            #             continue

            # 统计信息
            trace_step_nums.append(0)
            for i, step_data in enumerate(trace_datas):
                trace_step_nums[-1] += 1  # 统计的是trace的平均长度，而不是计算loss的平均长度，因为计算loss大部分都是最后一步
                if not trace_saved_flag_list[i]:
                    if problem_id < train_split_thre:
                        statistics['train_step_count'] += 1
                        if step_data['mcs'] > 0:
                            statistics['train_pos_step_count'] += 1
                        else:
                            statistics['train_neg_step_count'] += 1
                    else:
                        statistics['test_step_count'] += 1
                        if step_data['mcs'] > 0:
                            statistics['test_pos_step_count'] += 1
                        else:
                            statistics['test_neg_step_count'] += 1

            # 加上这个trace的code和exe feedback
            # 应当是trace中所有thought都包含生成的code
            trace_score = trace_datas[-1]['current_trace_score']
            trace_train_score = trace_datas[-1]['current_trace_train_score']
            current_trace_code = trace_datas[-1]['current_trace_code']
            current_trace_train_exe = trace_datas[-1]['current_trace_train_exe']

            if trace_train_score == 1.0:
                feedback = 'All public test cases passed!'
            else:
                feedback_str = f""
                test_case_count = 0
                for single_test_feedback in current_trace_train_exe:
                    if isinstance(single_test_feedback, dict) and test_case_count < 5:
                        feedback_str += f"## Failed test {test_case_count + 1}: \n{single_test_feedback['output']}\n"
                        test_case_count += 1
                feedback = feedback_str.strip()

            input_fine_tune_data = {
                'question_id': problem_id,
                'question': json_data['question'],
                'process_before_code': process_before_code,
                'code': current_trace_code,
                'feedback': feedback,
                'process': process,
                'mcs': trace_mcs_list,
                'label': trace_label_list,
                'saved_flag': trace_saved_flag_list
            }

            # 根据problem id划分train/test
            if problem_id < train_split_thre:
                train_json_datas.append(input_fine_tune_data)
                statistics['train_trace_count'] += 1
            else:
                test_json_datas.append(input_fine_tune_data)
                statistics['test_trace_count'] += 1
    return train_json_datas, test_json_datas, statistics, trace_step_nums

def form_data_json(dataset_exp_id_dict, step_tag):
    total_train_json_datas = []
    total_test_json_datas = []

    # 读取已有的train和test数据
    # train_jsons_dir = f"{get_proj_path()}/dataProcess/ft_data/all/mc_prm_data/train/"
    # test_jsons_dir = f"{get_proj_path()}/dataProcess/ft_data/all/mc_prm_data/test/"
    # with open(f"{train_jsons_dir}/data_list.json", 'r') as f:
    #     total_train_json_datas = json.load(f)
    # with open(f"{test_jsons_dir}/data_list.json", 'r') as f:
    #     total_test_json_datas = json.load(f)

    # # 对已有的 train 和 test 数据随机删除一半
    # total_train_json_datas = remove_half(total_train_json_datas)
    # total_test_json_datas = remove_half(total_test_json_datas)

    total_trace_step_nums = []
    total_statistics = {
        'train_trace_count': 0,
        'train_step_count': 0,
        'train_pos_step_count': 0,
        'train_neg_step_count': 0,
        'test_trace_count': 0,
        'test_step_count': 0,
        'test_pos_step_count': 0,
        'test_neg_step_count': 0
    }
    all_pos_mc_scores = []
    for dataset in dataset_exp_id_dict.keys():
        difficulty_choices = ['introductory', 'interview', 'competition']
        if 'codeforces' in dataset:
            difficulty_choices = ['1200', '1500', '1700', '1300', '1400', '1600']

        exp_id_list = dataset_exp_id_dict[dataset]
        for exp_id in exp_id_list:
            for difficulty in difficulty_choices:
                print('------------------')
                print(f"dataset: {dataset}, exp_id: {exp_id}")
                if not os.path.isdir(f"{get_proj_path()}/results/{dataset}/Experiment_{exp_id}/1/{difficulty}/mc_labels/"):
                    continue
                one_exp_train_json_datas, one_exp_test_json_datas, one_exp_statistics, one_exp_trace_step_nums = (
                    read_raw_data(dataset, exp_id, difficulty, step_tag))
                ##
                for one_finetune_data in one_exp_train_json_datas:
                    for label in one_finetune_data['label']:
                        if label > 0.0:
                            all_pos_mc_scores.append(label)
                ##
                total_train_json_datas.extend(one_exp_train_json_datas)
                total_test_json_datas.extend(one_exp_test_json_datas)
                total_trace_step_nums.extend(one_exp_trace_step_nums)
                total_statistics['train_trace_count'] += one_exp_statistics['train_trace_count']
                total_statistics['train_step_count'] += one_exp_statistics['train_step_count']
                total_statistics['train_pos_step_count'] += one_exp_statistics['train_pos_step_count']
                total_statistics['train_neg_step_count'] += one_exp_statistics['train_neg_step_count']
                total_statistics['test_trace_count'] += one_exp_statistics['test_trace_count']
                total_statistics['test_step_count'] += one_exp_statistics['test_step_count']
                total_statistics['test_pos_step_count'] += one_exp_statistics['test_pos_step_count']
                total_statistics['test_neg_step_count'] += one_exp_statistics['test_neg_step_count']

    print(f"avg pos mc score: {np.mean(all_pos_mc_scores)}")
    print(f"median pos mc score:  {np.median(all_pos_mc_scores)}")
    print(f"max pos mc score: {np.max(all_pos_mc_scores)}")
    print(f"min pos mc score: {np.min(all_pos_mc_scores)}")
    print(f"total_statistics: \n{total_statistics}")

    # 统计trace step nums分布
    mean = np.mean(total_trace_step_nums)
    median = np.median(total_trace_step_nums)
    max_value = np.max(total_trace_step_nums)
    min_value = np.min(total_trace_step_nums)
    std_dev = np.std(total_trace_step_nums)
    variance = np.var(total_trace_step_nums)
    print(f"-------trace step nums distribution-------")
    print(f"平均数: {mean}")
    print(f"中位数: {median}")
    print(f"最大值: {max_value}")
    print(f"最小值: {min_value}")
    print(f"标准差: {std_dev}")
    print(f"方差: {variance}")


    train_jsons_dir = f"{get_proj_path()}/dataProcess/ft_data/all/mc_prm_data/train/"
    test_jsons_dir = f"{get_proj_path()}/dataProcess/ft_data/all/mc_prm_data/test/"
    os.makedirs(train_jsons_dir, exist_ok=True)
    os.makedirs(test_jsons_dir, exist_ok=True)

    # 将json datas保存到data.json
    # with open(f"{train_jsons_dir}/data_list.json", 'w') as f:
    #     json.dump(total_train_json_datas, f)
    # with open(f"{test_jsons_dir}/data_list.json", 'w') as f:
    #     json.dump(total_test_json_datas, f)


def data_list_analysis(jsons_dir):
    with open(f"{jsons_dir}/data_list.json", 'r') as f:
        json_data = json.load(f)
    total_step_count = 0
    loss_step_count = 0
    trace_count = 0
    pos_step_count = 0
    neg_step_count = 0
    # partial_data = []
    for sample in json_data:
        trace_count += 1
        total_step_count += len(sample['label'])
        loss_step_count += sum([1 for sf in sample['saved_flag'] if not sf])
        # sample['mcs']中大于0的代表一个pos step
        pos_step_count += sum([1 for mcs in sample['mcs'] if mcs > 0])
        neg_step_count += sum([1 for mcs in sample['mcs'] if mcs <= 0])
        # if sample['question_id'] == 1:
        #     partial_data.append(sample)

    print(f'------------------{jsons_dir}------------------')
    print(f"trace_count: {trace_count}")
    print(f"total stepcount: {total_step_count}")
    print(f"loss step count: {loss_step_count}")
    print(f"pos_step_count: {pos_step_count}")
    print(f"neg_step_count: {neg_step_count}")
    # # 将json datas保存到data.json
    # with open(f"{jsons_dir}/data_list_little.json", 'w') as f:
    #     json.dump(partial_data, f)


if __name__=='__main__':
    # 从exp生成data
    step_tag = '\n\n\n\n\n'  # ки
    dataset_exp_id_dict = {
        'apps': [54],
        'humaneval': [43],
        'codeforces': [11]
    }
    form_data_json(dataset_exp_id_dict, step_tag)

    # 统计数据
    # train_jsons_dir = f"{get_proj_path()}/dataProcess/ft_data/all/mc_prm_data/train/"
    # data_list_analysis(train_jsons_dir)
    # test_jsons_dir = f"{get_proj_path()}/dataProcess/ft_data/all/mc_prm_data/test/"
    # data_list_analysis(test_jsons_dir)