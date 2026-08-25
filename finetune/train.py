import argparse

from swift import SftArguments, sft_main

from dataset import load
from utils import ADAPTER, BASE, DATA, TRAIN, warmup_steps

p = argparse.ArgumentParser()
p.add_argument("--data", default=str(DATA))
p.add_argument("--out", default="out/voicemem-qlora")
p.add_argument("--base", default=BASE)
p.add_argument("--rank", type=int, default=ADAPTER["rank"])
p.add_argument("--alpha", type=int, default=ADAPTER["alpha"])
p.add_argument("--epochs", type=float, default=TRAIN["epochs"])
p.add_argument("--lr", type=float, default=TRAIN["learning_rate"])
p.add_argument("--max-len", type=int, default=TRAIN["max_sequence_length"])
p.add_argument("--no-4bit", action="store_true")
args = p.parse_args()

rows = load(args.data)

sft_main(SftArguments(
    model=args.base,
    dataset=[args.data],
    output_dir=args.out,
    tuner_type="lora",
    lora_rank=args.rank,
    lora_alpha=args.alpha,
    lora_dropout=ADAPTER["dropout"],
    lora_bias=ADAPTER["bias"],
    # 这条正则按 Qwen3.6-35B-A3B 的模块命名写死，换基座必须换
    target_regex=ADAPTER["target_modules"],
    quant_bits=None if args.no_4bit else 4,
    quant_method=None if args.no_4bit else "bnb",
    torch_dtype="bfloat16",
    max_length=args.max_len,
    # 只有最后一轮 assistant 算 loss，历史轮不算
    loss_scale="last_round",
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    lr_scheduler_type=TRAIN["lr_scheduler"],
    warmup_steps=warmup_steps(len(rows), args.epochs),
    per_device_train_batch_size=TRAIN["per_device_train_batch_size"],
    gradient_accumulation_steps=TRAIN["gradient_accumulation_steps"],
    gradient_checkpointing=TRAIN["gradient_checkpointing"],
    weight_decay=TRAIN["weight_decay"],
    optim=TRAIN["optimizer"],
    seed=TRAIN["seed"],
    split_dataset_ratio=0.1,
))
