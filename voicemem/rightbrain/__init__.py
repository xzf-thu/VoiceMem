"""右脑 Experience Layer。

heartnote + response_experience 的动态检索系统。
用户画像（slot 描述）也在右脑，按 query 作为 source="profile" 的 rb_hit 检索
（见 engine._rb_graph_hits），和其它 slot 一样——不再有独立的前刺层。
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
