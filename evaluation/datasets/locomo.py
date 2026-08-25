"""LoCoMo：超长多轮对话上的问答。

数据是一个 json 数组，每条 = 一段横跨多个 session 的对话 + 针对它的问答。
公开版大致长这样（字段名各版本有出入，下面的解析对常见变体都做了兼容）::

    [{"sample_id": "conv-26",
      "conversation": {
        "speaker_a": "Alice", "speaker_b": "Bob",
        "session_1_date_time": "2023-05-08 10:00",
        "session_1": [{"speaker": "Alice", "text": "...", "dia_id": "D1:1"}, ...],
        "session_2_date_time": ..., "session_2": [...]},
      "qa": [{"question": "...", "answer": "...", "category": 1}, ...]}]

**跑之前先用 --inspect 确认解析对了**（几段对话、几轮、时间戳有没有读到），
字段对不上就改 load() 里那几行——比在评测跑了两小时之后才发现全错强。
"""
from __future__ import annotations

import json
import re

#: 打印用的名字（CLI 上的键是小写的 "locomo"）
NAME = "LoCoMo"

from evaluation.datasets import Conversation, Question, Score, Turn

#: LoCoMo 的题型编号 → 人话，出分类得分用
CATEGORIES = {
    1: "multi_hop", 2: "temporal", 3: "open_domain",
    4: "single_hop", 5: "adversarial",
}


def load(path: str) -> list[Conversation]:
    raw = json.loads(open(path, encoding="utf-8").read())
    if isinstance(raw, dict):                     # 有的版本是 {"conv-26": {...}}
        raw = [{**v, "sample_id": k} for k, v in raw.items()]

    out: list[Conversation] = []
    for i, sample in enumerate(raw):
        cid = str(sample.get("sample_id") or sample.get("id") or f"conv_{i}")
        out.append(Conversation(id=cid,
                                turns=_turns(sample.get("conversation") or sample),
                                questions=_questions(sample.get("qa") or sample.get("questions") or [])))
    return out


def _turns(conv: dict) -> list[Turn]:
    """把 session_1 / session_2 … 按编号顺序摊平成一串对话。

    每个 session 有自己的日期（session_N_date_time），要带上——LoCoMo 有一整类
    时序题（"这件事发生在哪次之前"），日期丢了这类题直接归零。
    """
    sessions: list[tuple[int, list, str]] = []
    for key, val in conv.items():
        m = re.fullmatch(r"session_(\d+)", str(key))
        if m and isinstance(val, list):
            when = conv.get(f"session_{m.group(1)}_date_time") or conv.get(f"{key}_date_time") or ""
            sessions.append((int(m.group(1)), val, str(when)))
    sessions.sort()

    turns: list[Turn] = []
    for _, msgs, when in sessions:
        for m in msgs:
            text = (m.get("text") or m.get("clean_text") or m.get("utterance") or "").strip()
            if not text:
                continue
            # 有的样本把图片描述单独放一个字段，一并喂进去，否则"照片里那只猫"这类题没依据
            if m.get("blip_caption"):
                text = f"{text}（图：{m['blip_caption']}）"
            turns.append(Turn(speaker=str(m.get("speaker") or "user"),
                              text=text, observed_at=_date(when)))
    return turns


def _date(when: str) -> str:
    """"2023-05-08 10:00" / "8 May, 2023" → ISO 日期；认不出就留空。"""
    if not when:
        return ""
    if m := re.search(r"(\d{4})-(\d{2})-(\d{2})", when):
        return m.group(0)
    if m := re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", when):
        months = {m_: f"{i:02d}" for i, m_ in enumerate(
            ["jan", "feb", "mar", "apr", "may", "jun",
             "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
        mm = months.get(m.group(2)[:3].lower())
        if mm:
            return f"{m.group(3)}-{mm}-{int(m.group(1)):02d}"
    return ""


def _questions(qa: list) -> list[Question]:
    out = []
    for j, q in enumerate(qa):
        text = (q.get("question") or "").strip()
        if not text:
            continue
        ans = q.get("answer", q.get("adversarial_answer", ""))
        out.append(Question(
            id=str(q.get("question_id") or f"q{j}"), text=text,
            answer="" if ans is None else str(ans),
            category=CATEGORIES.get(q.get("category"), str(q.get("category", ""))),
            meta={"evidence": q.get("evidence", [])},
        ))
    return out


JUDGE = """判断「模型答案」和「标准答案」说的是不是同一件事。

标准答案：{gold}
模型答案：{pred}

宽松一点：意思对就算对，措辞、详略、格式不同都不扣分；标准答案是日期/数字时，
值对就算对。模型答案明显是别的信息、或者答不知道，才算错。

只回一个词：对 / 错"""


def score(q: Question, answer: str, judge) -> Score:
    """比对标准答案。LoCoMo 的答案多是短语，字面比对漏判太多，交给裁判模型。"""
    gold = (q.answer or "").strip()
    if not gold:                       # 没有标准答案的题不计入总分，免得拉低/抬高
        return Score(correct=0.0, total=0.0, note="无标准答案，跳过")

    pred = (answer or "").strip()
    if pred and gold.lower() in pred.lower():      # 明显包含就不必花裁判的钱了
        return Score(correct=1.0, note="字面命中")

    verdict = judge("你是评测裁判，只回「对」或「错」。",
                    JUDGE.format(gold=gold, pred=pred or "（空）"))
    ok = verdict.strip().startswith(("对", "Y", "y", "T", "t"))
    return Score(correct=1.0 if ok else 0.0, note=verdict.strip()[:40])
