"""Qwen2.5-Omni 多模态异常轮情绪归因。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from voicemem.utils.audio.emotion.attribution_prompt import (
    TURN_ATTRIBUTION_SYSTEM_PROMPT,
    build_turn_attribution_user_prompt,
    parse_turn_attribution_response,
)
from voicemem.utils.audio.emotion.types import TurnAttributionLLMResult, TurnEmotionRecord
from voicemem.utils.audio.emotion.omni_generate import (
    build_omni_text_gen_kwargs,
    suppress_omni_text_inference_noise,
)
from voicemem.utils.audio.emotion.vad_qwen_prompt import _get_sequences


@runtime_checkable
class OmniTurnAttributor(Protocol):
    def analyze_turn_with_audio(
        self,
        *,
        audio_path: str,
        asr_text: str | None,
        left_memory_block: str,
        emotion_graph_context: str | None,
        turn: TurnEmotionRecord,
    ) -> TurnAttributionLLMResult:
        ...


@dataclass
class FixtureOmniTurnAttributor:
    """单测固定输出，不加载 Omni 权重。"""

    result: TurnAttributionLLMResult | None = None

    def analyze_turn_with_audio(
        self,
        *,
        audio_path: str,
        asr_text: str | None,
        left_memory_block: str,
        emotion_graph_context: str | None,
        turn: TurnEmotionRecord,
    ) -> TurnAttributionLLMResult:
        _ = (audio_path, asr_text, left_memory_block, emotion_graph_context, turn)
        if self.result is not None:
            return self.result
        from voicemem.utils.audio.emotion.types import EmotionGraphDelta, EmotionGraphNodeInput, EmotionSignal

        return TurnAttributionLLMResult(
            analysis_text="Fixture attribution: subdued tone; likely frustration about a blocked experiment.",
            emotion=EmotionSignal(
                label="frustration",
                valence=turn.vad.valence,
                arousal=turn.vad.arousal,
                intensity="high",
                confidence=0.75,
            ),
            acoustic_evidence=["lower vocal energy", "noticeable pauses"],
            semantic_evidence=["mentions experiment failure"],
            related_nodes=[
                EmotionGraphNodeInput(local_id="topic_1", name="main experiment", node_type="Topic")
            ],
            graph_delta=EmotionGraphDelta(
                nodes=[
                    EmotionGraphNodeInput(local_id="topic_1", name="main experiment", node_type="Topic")
                ],
                edges=[],
            ),
            retrieval_snippet=["main experiment", "frustration"],
        )


class QwenOmniEmotionAttributor:
    """注入已加载的 Qwen2.5-Omni processor/model/tokenizer。"""

    def __init__(
        self,
        processor: Any,
        model: Any,
        tokenizer: Any,
        *,
        system_prompt: str = TURN_ATTRIBUTION_SYSTEM_PROMPT,
        max_new_tokens: int = 768,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> None:
        self.processor = processor
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def analyze_turn_with_audio(
        self,
        *,
        audio_path: str,
        asr_text: str | None,
        left_memory_block: str,
        emotion_graph_context: str | None,
        turn: TurnEmotionRecord,
    ) -> TurnAttributionLLMResult:
        import torch

        user_text = build_turn_attribution_user_prompt(
            turn=turn,
            asr_text=asr_text,
            left_context_summary=left_memory_block,
            emotion_graph_context=emotion_graph_context,
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "audio", "path": audio_path},
                ],
            },
        ]
        gen_kwargs = build_omni_text_gen_kwargs(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            tokenizer=self.tokenizer,
        )
        self.model.eval()
        with torch.no_grad(), suppress_omni_text_inference_noise():
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)
            out = self.model.generate(**inputs, **gen_kwargs)
            sequences = _get_sequences(out)
            input_len = inputs["input_ids"].shape[-1]
            to_decode = sequences[:, input_len:] if sequences.shape[-1] > input_len else sequences
            text = (
                self.processor.batch_decode(
                    to_decode,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                if to_decode.shape[-1] > 0
                else ""
            )
        return parse_turn_attribution_response(text)
