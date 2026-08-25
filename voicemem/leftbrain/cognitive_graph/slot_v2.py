"""Slot V2 taxonomy — life-domain slots for memory retrieval.

Seven slots that describe *which domain of life* a memory belongs to.
Used for LLM-based query classification and embedding-based memory tagging.
"""

from __future__ import annotations

from enum import Enum


class SlotV2(str, Enum):
    """Seven life-domain slots for memory retrieval (V2).

    Each value is the canonical string stored in the ``memory_tags`` table.
    Using ``str, Enum`` lets the values be compared and serialised directly
    as plain strings without calling ``.value``.
    """

    WORK          = "work"
    FINANCE       = "finance"
    RELATIONSHIPS = "relationships"
    HEALTH        = "health"
    GOALS         = "goals"
    DAILY_LIFE    = "daily_life"
    KNOWLEDGE     = "knowledge"


# ---------------------------------------------------------------------------
# Semantic descriptions used to build slot embeddings (write-side tagging).
# Rich bilingual descriptions so the embedder maps memories correctly.
# ---------------------------------------------------------------------------

SLOT_V2_DESCRIPTIONS: dict[str, str] = {
    SlotV2.WORK: (
        "Work, career, job, company, colleagues, projects, meetings, work performance, "
        "professional development, workplace, business tasks, 工作, 职业, 公司, 项目, 同事, "
        "职场, 会议, 老板, 汇报, 升职, 跳槽, 辞职"
    ),
    SlotV2.FINANCE: (
        "Money, salary, income, expenses, investments, savings, debt, financial goals, "
        "spending, budget, 财务, 收入, 薪资, 投资, 支出, 存款, 贷款, 消费, 理财, 股票, 房贷"
    ),
    SlotV2.RELATIONSHIPS: (
        "Friends, family, romantic relationships, social connections, interpersonal dynamics, "
        "people I care about, social events, 朋友, 家人, 感情, 人际关系, 父母, 恋人, 社交, "
        "同学, 邻居, 亲戚, 吵架, 和好"
    ),
    SlotV2.HEALTH: (
        "Physical health, exercise, diet, sleep, mental health, fitness, wellness, "
        "illness, doctor, 健康, 运动, 饮食, 睡眠, 医疗, 身体, 锻炼, 生病, 看病, 心情, "
        "焦虑, 压力, 体重"
    ),
    SlotV2.GOALS: (
        "Goals, ambitions, future plans, dreams, aspirations, long-term vision, "
        "self-improvement, personal development, 目标, 计划, 梦想, 未来, 理想, 志向, "
        "成长, 提升, 年度计划, 五年计划"
    ),
    SlotV2.DAILY_LIFE: (
        "Daily life, hobbies, leisure, entertainment, routines, habits, personal preferences, "
        "lifestyle, daily activities, 日常, 爱好, 习惯, 娱乐, 生活方式, 休闲, 购物, "
        "旅游, 读书, 电影, 游戏, 做饭"
    ),
    SlotV2.KNOWLEDGE: (
        "Knowledge, learning, concepts, skills, facts, technology, ideas, "
        "education, research, 知识, 学习, 技能, 概念, 信息, 原理, 方法, 工具, 技术, "
        "课程, 书籍, 研究"
    ),
}

# Convenience list of all canonical slot value strings.
ALL_SLOT_V2_VALUES: list[str] = [s.value for s in SlotV2]

# Related slots — when retrieving for a primary slot, also fetch summaries of these.
SLOT_RELATIONS: dict[str, list[str]] = {
    "work":          ["finance", "relationships", "goals"],
    "finance":       ["work", "goals"],
    "health":        ["daily_life", "goals"],
    "relationships": ["work", "daily_life"],
    "goals":         ["work", "finance", "health"],
    "daily_life":    ["health", "relationships"],
    "knowledge":     ["work", "goals"],
}

__all__ = [
    "SlotV2",
    "SLOT_V2_DESCRIPTIONS",
    "ALL_SLOT_V2_VALUES",
    "SLOT_RELATIONS",
]
