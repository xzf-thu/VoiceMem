#!/usr/bin/env python3
"""VoiceMem 评测入口：一条命令跑完一个 benchmark。

    python evaluation/run.py --dataset locomo --data data/locomo.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# mem0 的遥测会在 ~/.mem0 开一个全局 qdrant 并独占文件锁，多进程/多线程评测会
# 撞 "already accessed by another instance"。评测用不上它。必须在 import 之前设。
os.environ.setdefault("MEM0_TELEMETRY", "False")

from evaluation import datasets  # noqa: E402


# ── ④ 答案模型 / 判分裁判：都走 OpenAI 兼容接口，一处配置 ─────────────────────

def make_llm(model: str):
    """返回 ``fn(system, user) -> str``。换成自建端点就设 OPENAI_BASE_URL。"""
    from openai import OpenAI
    client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)

    def call(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return (resp.choices[0].message.content or "").strip()
    return call


def provenance() -> dict:
    """跑这次评测时的环境。写进结果文件——半年后看到一个数字，能查出它是哪份代码跑的。"""
    import platform
    import subprocess
    from datetime import datetime, timezone

    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except Exception:
            return ""

    def version(pkg):
        from importlib.metadata import PackageNotFoundError, version as v
        try:
            return v(pkg)
        except PackageNotFoundError:
            return ""

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),   # 改动没提交就跑，数字对不回代码
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: version(p) for p in ("voicemem", "mem0ai", "openai", "qdrant-client")},
    }


ANSWER_SYSTEM = """你要根据「记忆」回答问题。

记忆里的"用户"和问题里提到的人是同一个人——记忆是从这个人自己的话里抽出来的，
主语写成"用户"只是抽取时的措辞。不要因为称呼不同就判定记忆里没有。

只能用下面这些记忆作答，不要编造记忆里没有的信息。记忆里确实没有的，才回答"不知道"。
回答要短——直接给答案，不要复述问题、不要解释推理过程。

记忆：
{memory}"""


# ── ②③ 一段对话的完整评测 ────────────────────────────────────────────────────

def run_conversation(conv, args, answer_llm, judge_llm) -> dict:
    """建库 → 灌对话 → 逐题检索作答 → 判分。返回这段对话的结果。"""
    from voicemem import VoiceMem
    from voicemem.memory_api import build_memory_context

    # 每段对话一个独立记忆库：混在一起等于把别的对话的答案也喂了进去
    root = Path(args.memory_root) / conv.id
    vm = VoiceMem(memory_root=str(root), user_id=conv.id, mode=args.mode)

    t0 = time.time()
    for turn in conv.turns:
        try:
            vm.ingest(turn.text, speaker=turn.speaker or "user",
                      observed_at=turn.observed_at or None, async_facts=False)
        except Exception as e:
            print(f"  [{conv.id}] ingest failed (skipping this turn): {e}", flush=True)
    ingest_s = time.time() - t0

    items, got, total = [], 0.0, 0.0
    for q in conv.questions:
        t1 = time.time()
        result = vm.search(q.text, top_k=args.top_k)
        search_ms = (time.time() - t1) * 1000
        memory = build_memory_context(result)

        answer = answer_llm(ANSWER_SYSTEM.format(memory=memory or "（没有相关记忆）"), q.text)
        s = (datasets.Score(0.0, 1.0, "未判分") if args.no_score
             else ds_score(args, q, answer, judge_llm))

        got += s.correct
        total += s.total
        items.append({
            "question_id": q.id, "question": q.text, "gold": q.answer,
            "predicted": answer, "correct": s.correct, "total": s.total,
            "note": s.note, "category": q.category,
            "search_ms": round(search_ms, 1),
            "memory_tokens": len(memory) // 2,      # 中文按 2 字符≈1 token 粗算
            "memory": memory if args.save_memory else "",
        })

    return {"conversation_id": conv.id, "ingest_seconds": round(ingest_s, 1),
            "score": got, "total": total, "items": items}


def ds_score(args, q, answer, judge_llm):
    return datasets.get(args.dataset).score(q, answer, judge_llm)


# ── 汇总 ─────────────────────────────────────────────────────────────────────

def summarize(results: list[dict], dataset: str) -> dict:
    got = sum(r["score"] for r in results)
    total = sum(r["total"] for r in results)
    by_cat: dict[str, list[float]] = {}
    lat, toks = [], []
    for r in results:
        for it in r["items"]:
            by_cat.setdefault(it["category"] or "all", []).append(
                it["correct"] / it["total"] if it["total"] else 0.0)
            lat.append(it["search_ms"])
            toks.append(it["memory_tokens"])
    return {
        "dataset": dataset,
        "conversations": len(results),
        "questions": sum(len(r["items"]) for r in results),
        "score": round(got, 2), "total": round(total, 2),
        "accuracy": round(got / total, 4) if total else 0.0,
        "by_category": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_cat.items())},
        "median_search_ms": round(sorted(lat)[len(lat) // 2], 1) if lat else 0.0,
        "median_memory_tokens": sorted(toks)[len(toks) // 2] if toks else 0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="VoiceMem 评测：一条命令跑完一个 benchmark")
    p.add_argument("--dataset", required=True, choices=datasets.names(),
                   help="跑哪个 benchmark（见 evaluation/datasets/）")
    p.add_argument("--data", required=True, help="数据集文件路径")
    p.add_argument("--out", default="", help="结果 json，默认 results/<dataset>.json")
    p.add_argument("--answer-model", default=os.environ.get("EVAL_ANSWER_MODEL", "gpt-4o-mini"),
                   help="拿记忆作答的模型")
    p.add_argument("--judge", default=os.environ.get("EVAL_JUDGE_MODEL", "gpt-4o-mini"),
                   help="判分的裁判模型")
    p.add_argument("--mode", default="left_brain_single",
                   help="VoiceMem mode；纯文本评测用 left_brain_single 就够，"
                        "要连右脑一起测用 text_mode")
    p.add_argument("--top-k", type=int, default=5, help="每题检索几条记忆")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 段对话（调试用）")
    p.add_argument("--workers", type=int, default=4, help="并发跑几段对话")
    p.add_argument("--memory-root", default="", help="记忆库落盘目录，默认 results/<dataset>_memory")
    p.add_argument("--resume", action="store_true", help="接着上次跑，跳过已完成的对话")
    p.add_argument("--save-memory", action="store_true", help="把每题检索到的记忆也存进结果（体积大，便于复核）")
    p.add_argument("--inspect", action="store_true", help="只解析数据集并打印前几条，不跑评测")
    p.add_argument("--no-score", action="store_true",
                   help="只生成答案不判分，之后用 evaluation/score.py 判（换裁判不用重跑）")
    args = p.parse_args()

    out = Path(args.out or f"results/{args.dataset}.json")
    if not args.memory_root:
        args.memory_root = str(out.parent / f"{args.dataset}_memory")

    # ① 读数据集
    convs = datasets.get(args.dataset).load(args.data)
    if args.limit:
        convs = convs[:args.limit]

    if args.inspect:                        # 先确认解析对了再花钱跑
        print(f"Parsed {len(convs)} conversations, "
              f"{sum(len(c.questions) for c in convs)} questions\n")
        for c in convs[:2]:
            print(f"[{c.id}] {len(c.turns)} turns / {len(c.questions)} questions")
            for t in c.turns[:3]:
                print(f"   {t.observed_at or '(no date)'} {t.speaker}: {t.text[:60]}")
            for q in c.questions[:2]:
                print(f"   Q: {q.text[:60]}")
                print(f"   A: {q.answer[:60]}" if q.answer else f"   rubric: {q.rubric[:2]}")
            print()
        return

    done: dict[str, dict] = {}
    if args.resume and out.exists():
        done = {r["conversation_id"]: r
                for r in json.loads(out.read_text(encoding="utf-8")).get("results", [])}
        print(f"Resuming: {len(done)} conversations already done, skipping", flush=True)

    prov = provenance()
    if prov["git_dirty"]:
        print("Warning: working tree is dirty; these numbers map to no commit", flush=True)
    answer_llm, judge_llm = make_llm(args.answer_model), make_llm(args.judge)
    todo = [c for c in convs if c.id not in done]
    results = list(done.values())

    print(f"{datasets.display_name(args.dataset)}: {len(todo)} conversations to run "
          f"(answer={args.answer_model}, judge={args.judge}, top_k={args.top_k})", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_conversation, c, args, answer_llm, judge_llm): c
                   for c in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            conv = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                print(f"  [{conv.id}] failed: {e}", flush=True)
                continue
            results.append(r)
            acc = r["score"] / r["total"] if r["total"] else 0
            print(f"  [{i}/{len(todo)}] {r['conversation_id']}  "
                  f"{r['score']:.0f}/{r['total']:.0f} ({acc:.0%})", flush=True)
            # 每段都落盘：跑几小时的评测中途挂了不用从头再来
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"summary": summarize(results, args.dataset),
                                       "config": vars(args), "provenance": prov,
                                       "results": results},
                                      ensure_ascii=False, indent=2), encoding="utf-8")

    s = summarize(results, args.dataset)
    name = datasets.display_name(args.dataset)
    print(f"\n{name}: {s['conversations']} conversations \u00b7 {s['questions']} questions\n")
    print(f"Score: {s['score']:.0f}/{s['total']:.0f} = {s['accuracy']:.1%}\n")
    if len(s["by_category"]) > 1:
        width = max(len(k) for k in s["by_category"])
        for k, v in s["by_category"].items():
            print(f"  {k:<{width + 2}}{v:.1%}")
        print()
    print(f"Median retrieval latency: {s['median_search_ms']:.0f} ms")
    print(f"Median retrieved memory: {s['median_memory_tokens']} tokens")
    print(f"\nSaved to {out}")
    if args.no_score:
        print(f"Not scored. Score with: python evaluation/score.py --file {out}")


if __name__ == "__main__":
    main()
