#!/usr/bin/env python3
"""对一次已经跑完的评测重新判分——不重跑检索和作答。

    python evaluation/score.py --file results/locomo.json
    python evaluation/score.py --file results/locomo.json --judge gpt-4o --out results/locomo-gpt4o.json

检索和作答是贵的那一半（每题一次 search + 一次生成），判分是便宜的一半。分开之后：
换裁判模型、修了判分口径的 bug、想看换个裁判分数稳不稳——都只重跑便宜的那半。
run.py --no-score 则是只做贵的那半。

原始问题从数据集重新读（按 question_id 对上），不是从结果文件里凑——rubric、meta
这些判分要用的字段结果文件里没存。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation import datasets                              # noqa: E402
from evaluation.run import make_llm, provenance, summarize   # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="对已有评测结果重新判分")
    p.add_argument("--file", required=True, help="run.py 产出的结果 json")
    p.add_argument("--out", default="", help="写到哪，默认覆盖 --file")
    p.add_argument("--judge", default="", help="裁判模型，默认沿用原来那次的")
    p.add_argument("--dataset", default="", choices=[""] + datasets.names(),
                   help="默认沿用结果文件里记的")
    p.add_argument("--data", default="", help="数据集文件路径，默认沿用结果文件里记的")
    args = p.parse_args()

    src = Path(args.file)
    blob = json.loads(src.read_text(encoding="utf-8"))
    cfg = blob.get("config", {})

    dataset = args.dataset or cfg.get("dataset", "")
    data = args.data or cfg.get("data", "")
    judge_model = args.judge or cfg.get("judge", "gpt-4o-mini")
    if not dataset or not data:
        raise SystemExit("结果文件里没记 dataset/data，用 --dataset 和 --data 指定")
    if not Path(data).exists():
        raise SystemExit(f"数据集不在了：{data}\n用 --data 指到它现在的位置")

    ds = datasets.get(dataset)
    # 按 (对话 id, 题 id) 索引：question_id 只在单段对话内唯一，跨段会撞
    # （LoCoMo 每段都是 q0/q1/…），只用 q.id 会拿到别段的标准答案。
    questions = {(c.id, q.id): q for c in ds.load(data) for q in c.questions}
    judge = make_llm(judge_model)

    n = sum(len(r["items"]) for r in blob["results"])
    print(f"重新判分：{n} 题，裁判 {judge_model}（原来是 {cfg.get('judge', '?')}）", flush=True)

    changed, missing = 0, 0
    for r in blob["results"]:
        got = total = 0.0
        for it in r["items"]:
            q = questions.get((r["conversation_id"], it["question_id"]))
            if q is None:               # 数据集变了，这题对不上——保留原分并记一笔
                missing += 1
                got += it["correct"]
                total += it["total"]
                continue
            s = ds.score(q, it["predicted"], judge)
            if s.correct != it["correct"]:
                changed += 1
            it["correct"], it["total"], it["note"] = s.correct, s.total, s.note
            got += s.correct
            total += s.total
        r["score"], r["total"] = got, total

    blob["summary"] = summarize(blob["results"], dataset)
    blob["config"] = {**cfg, "judge": judge_model, "rescored": True}
    blob["provenance"] = provenance()

    out = Path(args.out or src)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")

    s = blob["summary"]
    if missing:
        print(f"警告：{missing} 题在数据集里找不到，保留了原分数")
    print(f"改判 {changed}/{n} 题")
    print(f"得分 {s['score']:.0f}/{s['total']:.0f}  =  {s['accuracy']:.1%}")
    print(f"结果已存 {out}")


if __name__ == "__main__":
    main()
