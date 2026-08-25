"""把问句里的相对时间词展开成绝对日期，再拿去检索。

为什么需要：抽取会把"下周三下午三点体检"归一成"Jiaqi 将在 2026年8月26日（周三）
下午三点进行体检"——库里存的是绝对日期。而用户问的是"我下周有什么安排"，这句话里
**一个绝对日期都没有**，向量对不上，实测三条下周日程一条都检索不到::

    「我下周有什么安排」   日程命中 0/3
    「8月26号我要干嘛」    日程命中 3/3

差别只在问法。所以在检索前把"下周"就地展开成那七天的日期，拼在问句后面——
向量里有了 8月26日 这样的字面，才够得着库里那条。

只改**拿去检索的那份文本**，不改用户说的话，也不写进记忆。

    expand_relative_dates("我下周有什么安排")
    → "我下周有什么安排（2026年8月24日 2026年8月25日 … 2026年8月30日）"

识别不到相对时间词就原样返回，一分钱不花（纯正则，无模型）。
"""
from __future__ import annotations

import re
from datetime import date, timedelta

#: 相对时间词 → (起始偏移, 天数)。偏移是相对"今天"的天数。
#: 周相关的偏移在 _resolve 里按当天星期几现算，这里用 None 占位。
_SPANS: dict[str, tuple[int | None, int]] = {
    "前天":     (-2, 1),
    "昨天":     (-1, 1),
    "今天":     (0, 1),
    "今日":     (0, 1),
    "明天":     (1, 1),
    "明日":     (1, 1),
    "后天":     (2, 1),
    "大后天":   (3, 1),
    "这几天":   (0, 3),
    "最近几天": (-3, 4),
    "接下来几天": (0, 4),
    "未来几天": (0, 4),
    # 周：偏移按当天星期几算
    "上周":     (None, 7),
    "上个星期": (None, 7),
    "这周":     (None, 7),
    "本周":     (None, 7),
    "这个星期": (None, 7),
    "下周":     (None, 7),
    "下个星期": (None, 7),
    "下星期":   (None, 7),
}

#: 周相关的词 → 相对"本周一"的周偏移
_WEEK_OFFSET = {
    "上周": -1, "上个星期": -1,
    "这周": 0, "本周": 0, "这个星期": 0,
    "下周": 1, "下个星期": 1, "下星期": 1,
}

#: 长的词要先匹配，否则"下个星期"会先被"下周"之外的短词切碎。
_WORDS = sorted(_SPANS, key=len, reverse=True)
_RE = re.compile("|".join(re.escape(w) for w in _WORDS))

#: 一次最多展开几个日期。问"最近三个月"这种展开出来上百个日期，
#: 会把问句本身的语义冲淡，反而检索更差。
_MAX_DAYS = 8


def _resolve(word: str, today: date) -> list[date]:
    """一个相对时间词覆盖哪几天。"""
    if word in _WEEK_OFFSET:
        monday = today - timedelta(days=today.weekday())      # 本周一
        start = monday + timedelta(weeks=_WEEK_OFFSET[word])
        return [start + timedelta(days=i) for i in range(7)]
    offset, days = _SPANS[word]
    start = today + timedelta(days=offset or 0)
    return [start + timedelta(days=i) for i in range(days)]


def expand_relative_dates(query: str, today: date | None = None) -> str:
    """在问句后面补上它涉及的绝对日期。没有相对时间词就原样返回。"""
    if not query:
        return query
    words = _RE.findall(query)
    if not words:
        return query

    today = today or date.today()
    days: list[date] = []
    for word in words:
        for d in _resolve(word, today):
            if d not in days:
                days.append(d)
    if not days or len(days) > _MAX_DAYS:
        return query

    days.sort()
    # 写成"2026年8月26日"这种格式，跟抽取归一后的写法对齐——格式不一样就白展开了。
    stamps = " ".join(f"{d.year}年{d.month}月{d.day}日" for d in days)
    return f"{query}（{stamps}）"
