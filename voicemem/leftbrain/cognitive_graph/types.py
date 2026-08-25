"""认知图核心数据类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .slot_v2 import SlotV2


class EntityType(str, Enum):
    USER = "user"
    PERSON = "person"
    ORGANIZATION = "organization"
    PROJECT = "project"
    TASK = "task"
    KNOWLEDGE = "knowledge"
    PREFERENCE = "preference"
    PLACE = "place"
    ROUTINE = "routine"
    ASSET = "asset"
    EVENT = "event"


# 槽位统一用 SlotV2（work/finance/relationships/health/goals/daily_life/
# knowledge，见 slot_v2.py）——这是 Classify()/检索路径实际在用的唯一 taxonomy。
# 之前这里还有一套独立的 7 类 SlotType（people/projects/knowledge/tasks/
# places/routines/assets），只被 AnchorRouter 读取，且是 EntityType 的纯派生
# 值（ENTITY_TYPE_TO_SLOT 是个确定性映射），跟检索用的 SlotV2 完全不通、纯属
# 冗余的第二套 taxonomy。AnchorRouter 需要的"这个实体是什么类型"信息本来就是
# EntityType 自己就有的，已改成直接读 entity_type，不再需要这层转译。


class EntityMemoryRole(str, Enum):
    """实体在某条记忆里的角色。"""
    SUBJECT = "subject"    # 主语：是动作的发起者（"Lang 说了…"）
    OBJECT = "object"      # 宾语：是动作的对象（"关于 Lang 的讨论"）
    CONTEXT = "context"    # 背景：作为上下文被提及（"在 project X 里"）
    OWNER = "owner"        # 所有者：记忆直接属于这个实体（"Caroline 的偏好"）


@dataclass
class Entity:
    id: str
    user_id: str
    entity_type: EntityType
    name: str               # 展示名（原始大小写）
    name_norm: str          # 归一化名（用于去重合并）
    slot: SlotV2
    description: str = ""   # d_v：实体描述，首次创建时用原句填充，可追加更新
    confidence: float = 1.0
    importance: float = 0.5                        # 0–1，影响展示排序和节点大小
    aliases: list[str] = field(default_factory=list)   # 别名列表（合并实体用）
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class EntityEdge:
    id: str
    user_id: str
    from_entity_id: str
    to_entity_id: str
    relation_type: str
    role_label: str | None = None
    confidence: float = 1.0
    weight: float = 1.0                             # 随共现次数累加，读取时按 updated_at 做时间衰减
    edge_type: str = "weak"                         # strong | weak，由 weight 阈值派生
    status: str = "active"                         # active | deprecated | merged
    evidence_memory_ids: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class EntityMemoryLink:
    id: str
    memory_id: str
    entity_id: str
    user_id: str
    role: EntityMemoryRole = EntityMemoryRole.CONTEXT  # 实体在该记忆中的角色
    relation_hint: str | None = None               # 关系描述（"collaborates_on"）
    created_at: str = ""


@dataclass
class MemoryRecord:
    """认知图自己持有的记忆元数据（与向量库里的 memory_id 一一对应）。"""
    id: str              # 与 voicemem_leftbrain.sqlite 里的 memory_id 相同
    user_id: str
    slot: SlotV2
    memory_type: str     # fact | event | preference | routine | …
    content: str         # 记忆文本
    confidence: float = 1.0
    sensitivity: float = 0.0   # 敏感度 0–1（越高越私密，影响展示和分享）
    ttl: int | None = None     # 过期秒数，None = 永不过期
    created_at: str = ""
    updated_at: str = ""


@dataclass
class SlotProfile:
    user_id: str
    slot: SlotV2
    summary: str = ""                              # LLM 生成的槽位摘要文本
    entity_ids: list[str] = field(default_factory=list)  # 该槽位下的实体 id 列表
    entity_count: int = 0
    memory_count: int = 0
    last_updated: str = ""


@dataclass
class AffectiveEdge:
    """右脑情感边（预留接口）。"""
    id: str
    user_id: str
    from_entity_id: str
    to_entity_id: str | None = None
    trigger_frame: str | None = None
    emotion: str | None = None
    appraisal: str | None = None
    response_policy: str | None = None
    confidence: float = 1.0                        # 情感判断置信度
    evidence_memory_ids: list[str] = field(default_factory=list)  # 触发来源记忆
    created_at: str = ""


@dataclass
class EntityAnnotation:
    """LLM 从单条 fact 中识别出的单个实体。"""
    name: str
    entity_type: EntityType
    role: str | None = None    # LLM 给出的角色描述，会被规范化为 EntityMemoryRole


@dataclass
class RelationAnnotation:
    """LLM 从单条 fact 中识别出的实体间关系。"""
    from_name: str
    to_name: str
    relation_type: str
    role_label: str | None = None
    confidence: float = 1.0


@dataclass
class AnnotatedFact:
    """LLM 对单条 fact 的完整标注结果。"""
    fact_text: str
    slot: SlotV2
    entities: list[EntityAnnotation] = field(default_factory=list)
    relations: list[RelationAnnotation] = field(default_factory=list)
    confidence: float = 1.0
