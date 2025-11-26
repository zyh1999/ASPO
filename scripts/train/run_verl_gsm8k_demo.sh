#!/usr/bin/env bash
set -xeuo pipefail

############################################
# 1. 如果本地没有 GSM8K 数据，就自动下载并写到 data/train & data/test
############################################

data_dir=./data
train_dir=$data_dir/train
test_dir=$data_dir/test

train_base=$train_dir/gsm8k_train
val_base=$test_dir/gsm8k_test

if [ ! -f "${train_base}.json" ] || [ ! -f "${val_base}.json" ] || \
   [ ! -f "${train_base}.parquet" ] || [ ! -f "${val_base}.parquet" ]; then
  python - << 'EOF'
import os, json

import pandas as pd
from datasets import load_dataset

data_dir = "data"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

train_base = os.path.join(train_dir, "gsm8k_train")
val_base = os.path.join(test_dir, "gsm8k_test")

def dump(split, base):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    json_path = base + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        for ex in ds:
            row = {
                # verl 的 RLHFDataset 期望有 prompt 字段：单轮对话
                "prompt": [
                    {"role": "user", "content": ex["question"]}
                ],
                # 奖励函数需要 ground_truth，用 reward_model 字段存
                "reward_model": {"ground_truth": ex["answer"]},
                # default_compute_score 支持 "openai/gsm8k"
                "data_source": "openai/gsm8k",
                "extra_info": {},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote", json_path)

    # 再从 jsonlines 读入，存成真正的 parquet 文件，避免 pandas/pyarrow 报错
    df = pd.read_json(json_path, lines=True)
    # PyArrow 不支持“没有子字段的 struct 列”，而 extra_info 目前都是空 dict，先删掉
    if "extra_info" in df.columns:
        df = df.drop(columns=["extra_info"])
    parquet_path = base + ".parquet"
    df.to_parquet(parquet_path)
    print("wrote", parquet_path)

dump("train", train_base)
dump("test", val_base)
EOF
fi

# 传给 verl 一个 .parquet 路径（现在我们已经真的生成了 parquet）
TRAIN_FILE=${train_base}.parquet
TEST_FILE=${val_base}.parquet

############################################
# 2. 训练配置（简化版，风格类似 code.sh）
############################################

project_name='verl-gsm8k-demo'
exp_name='verl-gsm8k-demo'

MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
CKPTS_DIR=./output/${project_name}/${exp_name}
mkdir -p "$CKPTS_DIR"

############################################
# 3. 启动 DAPO + GRPO 训练（仿照 code.sh）
############################################

# 序列长度与 batch 配置
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 32))

# GRPO 配置，仿照 run_archer2.0_qwen2.5_1.5b_code.sh / math.sh
train_prompt_bsz=64          # data.train_batch_size
train_prompt_mini_bsz=16     # actor_rollout_ref.actor.ppo_mini_batch_size
n_resp_per_prompt=8          # actor_rollout_ref.rollout.n（每个 prompt 生成多少条）
gen_prompt_bsz=$((train_prompt_bsz * 1))  # data.gen_batch_size

python -m dapo.main_dapo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TEST_FILE}" \
  data.prompt_key=prompt \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.max_prompt_length=${max_prompt_length} \
  data.max_response_length=${max_response_length} \
  data.gen_batch_size=${gen_prompt_bsz} \
  data.train_batch_size=${train_prompt_bsz} \
  actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  reward_model.enable=False \
  reward_model.reward_manager=dapo \
  data.reward_fn_key=data_source \
  trainer.project_name="${project_name}" \
  trainer.experiment_name="${exp_name}" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.default_local_dir="${CKPTS_DIR}" \
  "$@" 2>&1 | tee "${CKPTS_DIR}/${project_name}_${exp_name}.log"


