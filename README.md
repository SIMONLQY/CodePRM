# CodePRM
Repository for [CodePRM: Execution Feedback-enhanced Process Reward Model for Code Generation].

## Data Preparation
Download the raw data of APPS and Codeforces.

Then change the path in function "get_raw_data_path()" in "utils/util.py" to your raw data path.

### Data Collection
Then, to collect data for PRM training, run run.py

	python run.py --dataset apps -m TreeDataCollect -eid 0 --arch gpt4o-mini --rollout 16 --mc_nums 9 --start 4100 --end 4300


## Train PRM:

If not want to train one PRM yourself, we provide the lora fine-tuned CodePRM adapter in "dataProcess/ft_results/models/". You may need to change the "base_model_name_or_path" in adapter_config.json to your own path of the vanilla LLM.


If you want to train one PRM yourself, follow the following steps.

### Preprocess data
First, preprocess the collected data by running prm/preprocess.py

    python prm/preprocess.py

Then the training data would be ready in dataProcess/ft_data/all/.

Prepare the vanilla LLM also, and put it under "{get_raw_data_path()}/LLMWeights/".

### Fine-tuning CodePRM

Then, to fine-tune CodePRM, run prm/finetune_value_prm.py

	python finetune_value_prm.py --privileged True


## Run Generate-Verify-Refine (GVR) Inference
We provide two search algorithms with the GVR framework: Best-of-N and MCTsh. To run the GVR inference, run run.py

    python run.py --dataset apps -m CodePRMMCTS -eid 2 --rollout 16 --APPDifficulty 1200

The parameter settings are also in run.py
Set the `experiment_idx` before each run. And the experiment result will be saved in "results/{dataset}/Experiment_{experiment_idx}/".

If you want to see the result, run "ExpPro.py", change the dataset and playlist contain the "experiment_idx" you want to see.
