import json
from pathlib import Path

DATA = Path(__file__).parent / "data" / "sample.jsonl"

# 已发布 adapter（Qwen3.6-35B-A3B QLoRA v2）那次训练用的超参，照默认跑 = 复现同一次训练。
# 之前从 models/ 下的清单文件读，那份清单已随模型目录一起删掉，值直接写在这里。
BASE = "Qwen/Qwen3.6-35B-A3B"
ADAPTER = {
    "format": "PEFT LoRA",
    "rank": 32,
    "alpha": 64,
    "dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    # 按 Qwen3.6-35B-A3B 的模块命名写死；换基座必须改（不确定就用 "all-linear"）。
    "target_modules": (
        r"^(model\.language_model(?=\.).*\.(shared_expert_gate|down_proj|out_proj|"
        r"in_proj_a|in_proj_b|q_proj|in_proj_z|gate_proj|up_proj|in_proj_qkv|"
        r"k_proj|v_proj|o_proj))$"
    ),
}
TRAIN = {
    "epochs": 2,
    "seed": 42,
    "learning_rate": 2e-4,
    "lr_scheduler": "cosine",
    "warmup_ratio": 0.03,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "precision": "bf16",
    "gradient_checkpointing": True,
    "max_sequence_length": 2048,
    "optimizer": "adamw_torch_fused",
    "weight_decay": 0.1,
    "adam_beta": [0.9, 0.95],
}

MEMORY_CATEGORIES = ("knowledge", "emotion", "persona")

SYSTEM = {
    ("memory", "zh"): "你是VoiceMem，一个有记忆的个人AI伴侣。用下面检索到的记忆和用户画像自然地"
                      "回应，不要把记忆原样念给用户。",
    ("memory", "en"): "You are VoiceMem, a personal AI companion with memory. Reply naturally "
                      "using the retrieved memories and user profile below; do not recite them "
                      "verbatim to the user.",
    ("casual", "zh"): "你是VoiceMem，一个聪明有温度的AI伴侣。基于当下对话自然地回应。",
    ("casual", "en"): "You are VoiceMem, a warm and thoughtful AI companion. Reply naturally "
                      "based on the current conversation.",
}


def system_prompt(category, lang):
    kind = "memory" if category in MEMORY_CATEGORIES else "casual"
    return SYSTEM[(kind, lang)]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def warmup_steps(n_rows, epochs):
    # transformers 5.x 删了 warmup_ratio，只剩 warmup_steps，这里自己换算。
    # 按单进程算，多卡要再除以卡数。
    per_step = TRAIN["per_device_train_batch_size"] * TRAIN["gradient_accumulation_steps"]
    return max(1, round(n_rows * epochs / per_step * TRAIN["warmup_ratio"]))
