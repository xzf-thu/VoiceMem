# evaluation — 跑一条命令，出一个数字

```bash
export OPENAI_API_KEY=sk-...

python evaluation/run.py --dataset locomo --data data/locomo.json
```
跑完直接打印，结果同时写进 `results/locomo.json`：

```text
locomo  10 段对话 · 152 题
得分 139/152  =  91.4%
   multi_hop                88.2%
   temporal                 85.7%
   single_hop               95.1%
检索中位数 12ms · 记忆中位数 298 tokens
结果已存 results/locomo.json
```

## 常用参数

| 参数 | 干什么 | 默认 |
|---|---|---|
| `--dataset` / `--data` | 用哪个适配器 / 数据文件 | 必填 |
| `--answer-model` | 拿记忆作答的模型 | `gpt-4o-mini` |
| `--judge` | 判分的裁判模型 | `gpt-4o-mini` |
| `--top-k` | 每题检索几条记忆 | `5` |
| `--mode` | `left_brain_single`=只测事实记忆；`text_mode`=连右脑一起 | `left_brain_single` |
| `--workers` | 并发跑几段对话 | `4` |
| `--limit` | 只跑前 N 段（调试用） | 全部 |
| `--resume` | 接着上次跑，跳过已完成的对话 | 关 |
| `--save-memory` | 把每题检索到的记忆也存进结果，便于人工复核 | 关 |
| `--inspect` | 只解析数据集并打印，不跑评测 | 关 |
| `--no-score` | 只生成答案不判分，之后用 `score.py` 判 | 关 |

结果每跑完一段就落盘，所以跑几小时的评测中途挂了，加 `--resume` 接着跑即可。

## 重新判分

检索 + 作答是贵的那一半（每题一次 search 加一次生成），判分是便宜的一半。想换裁判
模型、或者修了判分口径的 bug，不用重跑贵的那半：

```bash
python evaluation/score.py --file results/locomo.json --judge gpt-4o
```

原始问题是从数据集重新读的（按对话 id + 题 id 对上），不是从结果文件里凑——rubric、
meta 这些判分要用的字段结果文件里没存。会打印改判了几题。

想彻底分两段跑，生成时加 `--no-score`。

## 结果文件里有什么

```json
{
  "summary":    { "accuracy": ..., "by_category": {...}, "median_search_ms": ... },
  "config":     { 这次用的全部参数 },
  "provenance": { "git_commit": ..., "git_dirty": ..., "python": ..., "packages": {...} },
  "results":    [ 每段对话每道题的 gold / predicted / 判分理由 ]
}
```

`provenance` 是为了半年后看到一个数字，还能查出它是哪份代码、什么环境跑出来的。
`git_dirty` 为 true 表示跑的时候工作区有未提交改动，这个数字对不回任何一个 commit
——跑正式结果前先提交。

## 评测新 benchmark

一个文件、两个函数，主流程一行不用动。

**1. 复制 `datasets/locomo.py` 改成 `datasets/你的数据集.py`**，实现两个函数：

```python
def load(path: str) -> list[Conversation]:
    """读你的数据文件，转成统一结构。
    Conversation(id, turns=[Turn(speaker, text, observed_at)], questions=[Question(...)])
    """

def score(q: Question, answer: str, judge) -> Score:
    """判这道题对不对。judge(system, user) -> str 是注入进来的裁判模型。
    Score(correct=1.0, total=1.0, note="判分理由")
    rubric 类的评分：correct=满足的要点数, total=总要点数
    """
```

**2. 登记到 `datasets/__init__.py` 的 `get()`**：

```python
table = {"locomo": locomo, "你的数据集": 你的模块}
```

**3. 跑**：

```bash
python evaluation/run.py --dataset 你的数据集 --data data/xxx.json --inspect   # 先验证解析
python evaluation/run.py --dataset 你的数据集 --data data/xxx.json
```