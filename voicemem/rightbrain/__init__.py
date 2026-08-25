"""右脑 Experience Layer。

两层：

  · **记忆层** heartnote + response_experience，按锚点检索（store/experience_repository）
  · **判断层** rb_traits + rb_evidence，一个节点 = 一条关于这个人的判断，
    claim 带向量，按 query 语义检索，作为 source="profile" 的 rb_hit 返回
    （见 traits_store.py 和 brain._rb_trait_hits）

判断层取代了原来的 slot→entity→heartnote 图。那套结构里 entity 一层身兼三职
（判断 / 话题 / 情绪词），实测滚成「悲伤 ×61」「佳琪 ×52」这样的大杂烩，
描述还要靠事后巩固批处理补。旧表（rb_slots/rb_entities）只读保留一个版本，
不再写入。
"""
from .anchor_router import AnchorRouter
from .attribution_manager import AttributionManager
from .brain import RightBrain, RightBrainHit
from .experience_repository import ExperienceRepository
from .graph_store import RBEntity, RBSlot, RightBrainGraphStore
from .store import RightBrainStore
from .types import (
    CurrentSignals,
    MemoryAnchor,
    MemoryQueryPlan,
    RightBrainContext,
    RightBrainMemory,
)

__all__ = [
    "AnchorRouter",
    "AttributionManager",
    "RightBrain",
    "RightBrainHit",
    "ExperienceRepository",
    "RightBrainGraphStore",
    "RBSlot",
    "RBEntity",
    "RightBrainStore",
    "CurrentSignals",
    "MemoryAnchor",
    "MemoryQueryPlan",
    "RightBrainContext",
    "RightBrainMemory",
]
