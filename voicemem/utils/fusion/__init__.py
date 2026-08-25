"""胼胝体 fusion：双通道检索、回复记忆与 turn 编排。

注意：不在包初始化时导入 ``orchestrator``，避免包级循环依赖。
请使用 ``from voicemem.utils.fusion.orchestrator import process_user_turn`` 等显式导入。
"""

from voicemem.utils.fusion.config import FusionConfig
from voicemem.utils.fusion.left_channel import (
    LeftBrainGraphSearchClient,
    LeftBrainSearchClient,
    search_left_channel,
    search_left_for_reply,
)
from voicemem.utils.fusion.prompt_builder import (
    build_left_only_reply_context_prompt,
    build_reply_context_prompt,
)
from voicemem.utils.fusion.reply_memory import (
    PersonaProvider,
    StubPersonaProvider,
    build_normal_left_reply_memory,
    build_omni_attribution_context,
    build_reply_memory,
    retrieve_for_reply,
    retrieve_left_for_normal_turn,
    run_anomaly_turn,
    run_rightbrain_memory_extract,
)
from voicemem.utils.fusion.right_channel import (
    FixtureRightChannelRelevanceFilter,
    OpenAIRightChannelRelevanceFilter,
    RightChannelRelevanceFilter,
    search_right_channel,
)
from voicemem.utils.fusion.types import (
    AnomalyTurnResult,
    FusionRetrievalResult,
    LeftMemoryHit,
    ReplyContextBundle,
    ReplyContextPrompt,
    ReplyRetrievalBundle,
    RightMemoryHit,
)

__all__ = [
    "AnomalyTurnResult",
    "FixtureRightChannelRelevanceFilter",
    "FusionConfig",
    "FusionRetrievalResult",
    "LeftBrainGraphSearchClient",
    "LeftBrainSearchClient",
    "LeftMemoryHit",
    "OpenAIRightChannelRelevanceFilter",
    "PersonaProvider",
    "ReplyContextBundle",
    "ReplyContextPrompt",
    "ReplyRetrievalBundle",
    "RightChannelRelevanceFilter",
    "RightMemoryHit",
    "StubPersonaProvider",
    "build_left_only_reply_context_prompt",
    "build_normal_left_reply_memory",
    "build_omni_attribution_context",
    "build_reply_context_prompt",
    "build_reply_memory",
    "retrieve_for_reply",
    "retrieve_left_for_normal_turn",
    "run_anomaly_turn",
    "run_rightbrain_memory_extract",
    "search_left_channel",
    "search_left_for_reply",
    "search_right_channel",
]
