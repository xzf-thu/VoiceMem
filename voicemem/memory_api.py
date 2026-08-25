"""One-line memory for an EXISTING chat system.

The whole point of this module: someone who already has a working dialogue
loop should be able to add long-term memory by changing ONE line --

    from voicemem import inject

    reply = llm.chat(inject(messages))       # <- the one line

``inject(messages)`` takes an OpenAI-style messages list (the de-facto
industry format -- OpenAI/Claude/Qwen/DeepSeek/ollama/vLLM/LangChain all
speak it), finds the latest user message, retrieves relevant left-brain
facts + right-brain persona from this user's memory, inserts them as a
system message right before that user message, schedules the new utterance
for background ingestion, and returns the augmented list. The caller's LLM
call is untouched -- this module never talks to the reply model.

For systems that don't use messages-format prompts, the same machinery is
exposed as two primitives:

    m = Memory(user_id="alice")
    context = m.recall("what did I say about my job?")   # -> str to splice
    m.remember("I got promoted today!")                   # store, background

Honest costs, not hidden: ``inject``/``recall`` run voicemem's real
retrieval pipeline (Classify -> Search: one LLM call + one embedding call,
~1-2s wall time against OpenAI defaults), so the host conversation gains
that much latency per turn; ``remember`` is fire-and-forget in a daemon
thread and adds none. voicemem's own internal calls need an
OpenAI-compatible endpoint (``OPENAI_API_KEY``, optionally
``OPENAI_BASE_URL`` for ollama/vLLM) -- the same requirement mem0-style
memory libraries have.
"""
from __future__ import annotations

import threading
from typing import Any


def build_memory_context(result: Any, max_rb: int = 3) -> str:
    """Render a SearchResult into the memory-context text block.

    Single source of truth for how retrieved memory becomes prompt text --
    the openai_voice_demo bridge delegates here too, so "what the demo does"
    and "what we tell integrators to do" can never drift apart. Returns ""
    when there is nothing worth injecting.

    ``max_rb``: right-brain hits are priority-sorted by Search(); capping at
    3 was a real measured latency fix in the demo (all 5 measurably delayed
    the reply model's first token), kept as the default here.
    """
    lines: list[str] = []
    hits = getattr(result, "hits", None) or []
    if hits:
        lines.append("MEMORY CONTEXT (things you remember about the user):")
        for hit in hits:
            # 带上事件日期（跟右脑 heartnote 的 [YYYY-MM-DD] 前缀同一个格式）：
            # 事实正文里多是"上周""一个多月了"这种相对说法，没有绝对日期，
            # "这事在那事之前吗"就无从判断。
            when = getattr(hit, "observed_at", "") or ""
            lines.append(f"- [{when}] {hit.text}" if when else f"- {hit.text}")
    rb_hits = getattr(result, "rb_hits", None) or []
    if rb_hits:
        # 右脑内容必须跟事实分开标注。它是画像/情绪归因/回复经验——写给模型看的
        # 内部笔记（"⚠ 避免重复：…（下次：…）""应对方式：…"），不是能对用户讲的话。
        # 不标注就只是跟在事实后面的几行文本，模型会照着念，回复立刻变成
        # "你的应对方式是通过健身缓解压力"这种听起来像读档案的句子。
        lines.append("")
        lines.append("HOW TO SPEAK TO THIS USER (internal — never quote or paraphrase aloud):")
        lines.extend(f"- {h.content}" for h in rb_hits[:max_rb])
        lines.append("Let these shape your tone, what you bring up, and what you leave alone. "
                     "Never state them back to the user.")
    return "\n".join(lines)


class Memory:
    """Per-user memory handle wrapping the full VoiceMem pipeline behind
    three calls: ``inject`` / ``recall`` / ``remember``.

    Audio-native perception (voiceprint/scene/emotion-from-audio) is OFF by
    default -- an existing text chat system shouldn't inherit those heavy
    optional dependencies. Pass ``audio_native=True`` (with the
    corresponding install extras) to enable them.
    """

    def __init__(self, user_id: str = "default", memory_root: str | None = None,
                 audio_native: bool = False, **voicemem_kwargs: Any) -> None:
        from voicemem.core import VoiceMem
        self.user_id = user_id
        self._vm = VoiceMem(
            memory_root=memory_root,
            user_id=user_id,
            enable_scene=audio_native,
            enable_music=audio_native,
            enable_abnormal_sound=audio_native,
            enable_voiceprint=audio_native,
            enable_emotion=audio_native,
            **voicemem_kwargs,
        )

    # ── primitives ───────────────────────────────────────────────────────────

    def recall(self, text: str, top_k: int = 5) -> str:
        """Retrieve memory relevant to *text*, rendered as a prompt-ready
        block (may be ""). This is the officially-supported
        Classify() -> Search() pattern, packaged."""
        c = self._vm.Classify(text)
        result = self._vm.Search(text, slots=c.slots, entities=c.entities, top_k=top_k)
        return build_memory_context(result)

    def remember(self, text: str, speaker: str = "user", **ingest_kwargs: Any) -> None:
        """Store one user utterance. Runs in a daemon thread so the host
        conversation is never blocked; errors are printed, not raised
        (a memory-write failure must not break the host's turn)."""
        def _run() -> None:
            try:
                self._vm.Ingest(text, speaker=speaker, **ingest_kwargs)
            except Exception as e:  # noqa: BLE001 -- background, nothing to propagate to
                print(f"[voicemem] background remember() failed: {e}", flush=True)
        threading.Thread(target=_run, daemon=True).start()

    def flush(self) -> None:
        """Optional: call when a conversation formally ends -- runs the
        session-boundary batch work (see VoiceMem.Flush)."""
        self._vm.Flush()

    # ── the one line ─────────────────────────────────────────────────────────

    def inject(self, messages: list[dict], top_k: int = 5,
               remember: bool = True) -> list[dict]:
        """The one-line integration. Returns a NEW messages list (the
        caller's list is never mutated) with a system message of retrieved
        memory inserted immediately before the latest user message; that
        user message is also queued for background ingestion (disable with
        ``remember=False`` if the host stores turns itself elsewhere).

        No user message / no relevant memory -> the list comes back
        unchanged (aside from being a copy), so it is always safe to call
        unconditionally.
        """
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user" and isinstance(messages[i].get("content"), str):
                last_user_idx = i
                break
        if last_user_idx is None:
            return list(messages)

        user_text = messages[last_user_idx]["content"]
        context = self.recall(user_text, top_k=top_k)
        if remember:
            self.remember(user_text)
        if not context:
            return list(messages)

        out = list(messages)
        out.insert(last_user_idx, {"role": "system", "content": context})
        return out


# ── module-level lazy default, so the integration is literally one line ─────
# (`from voicemem import inject` + `inject(messages)` -- no visible setup;
# the default Memory is built on first use, one per user_id.)

_DEFAULT_INSTANCES: dict[str, Memory] = {}
_DEFAULT_LOCK = threading.Lock()


def _default(user_id: str | None) -> Memory:
    import os
    uid = user_id or os.environ.get("VOICEMEM_USER_ID", "default")
    with _DEFAULT_LOCK:
        if uid not in _DEFAULT_INSTANCES:
            _DEFAULT_INSTANCES[uid] = Memory(user_id=uid)
        return _DEFAULT_INSTANCES[uid]


def inject(messages: list[dict], user_id: str | None = None, **kwargs: Any) -> list[dict]:
    return _default(user_id).inject(messages, **kwargs)


def recall(text: str, user_id: str | None = None, **kwargs: Any) -> str:
    return _default(user_id).recall(text, **kwargs)


def remember(text: str, user_id: str | None = None, **kwargs: Any) -> None:
    _default(user_id).remember(text, **kwargs)
