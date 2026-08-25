import argparse
import collections
import json

from swift import InferRequest, RequestConfig, TransformersEngine

from dataset import answer, load, prompt, question
from utils import BASE, DATA

p = argparse.ArgumentParser()
p.add_argument("--adapter", required=True)
p.add_argument("--data", default=str(DATA))
p.add_argument("--base", default=BASE)
p.add_argument("--max-tokens", type=int, default=256)
p.add_argument("--out", default="")
p.add_argument("--quiet", action="store_true")
args = p.parse_args()

rows = load(args.data)
engine = TransformersEngine(args.base, adapters=[args.adapter])
config = RequestConfig(max_tokens=args.max_tokens, temperature=0.0)
resps = engine.infer([InferRequest(messages=prompt(r)) for r in rows], config)

out = []
for row, resp in zip(rows, resps):
    pred = resp.choices[0].message.content
    ref = answer(row)
    out.append({
        "question": question(row),
        "ref": ref,
        "pred": pred,
        "meta": row["meta"],
        "pred_chars": len(pred or ""),
        "ref_chars": len(ref or ""),
    })
    if not args.quiet:
        print(json.dumps(out[-1], ensure_ascii=False))

if args.out:
    from utils import write_jsonl
    write_jsonl(out, args.out)


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0


def table(key):
    groups = collections.defaultdict(list)
    for o in out:
        groups[o["meta"][key]].append(o)
    for name, rows_ in sorted(groups.items()):
        empty = sum(1 for o in rows_ if not (o["pred"] or "").strip())
        print(f"  {name:<12} n={len(rows_):<4} "
              f"回答长度中位数 {median([o['pred_chars'] for o in rows_]):<5} "
              f"(参考 {median([o['ref_chars'] for o in rows_])})"
              f"{f'  空回答 {empty}' if empty else ''}")


print(f"\n{'=' * 56}")
print(f"{len(out)} 条 · base {args.base} · adapter {args.adapter}")
print("按 category:")
table("category")
print("按 lang:")
table("lang")
print("\n注：这里只报覆盖和长度，不报正确率——回复质量得人看或另接裁判模型。")
