"""Mem0 OSS / Platform V3 对齐：additive（ADD-only）记忆抽取。

- System：``data/additive_extraction_prompt.txt``（与 upstream ``ADDITIVE_EXTRACTION_PROMPT`` 一致）
- User：``mem0_additive_prompt_build.generate_additive_extraction_prompt``
- OpenAI Chat JSON：顶层 ``memory`` 数组，元素含 ``id`` / ``text`` / ``attributed_to`` / 可选 ``linked_memory_ids``

默认 Chat 模型：``gpt-4o-mini``（可用 ``OPENAI_MODEL`` 或 ``OpenAIAdditiveExtractorConfig(model=...)`` 覆盖）
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from voicemem.leftbrain.mem0_additive_prompt_build import (
    generate_additive_extraction_prompt,
    load_additive_system_prompt,
)

from voicemem.utils.common.cost_log import log_usage as _log_usage


def remove_code_blocks(content: str) -> str:
    text = content.strip()
    m = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.I)
    return m.group(1).strip() if m else text


def extract_json(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx : end_idx + 1]
    return text


@dataclass(frozen=True)
class ExtractedAdditiveMemory:
    """单条 additive 抽取结果（与 Mem0 输出 schema 对齐）。"""

    local_id: str
    text: str
    attributed_to: str
    linked_memory_ids: tuple[str, ...] = ()
    # "谁的什么属性"，不含具体值，如 "User's favorite restaurant" / "User's job"。
    # 只用于冲突判定阶段的第二次候选检索（见 voice_input.py），空串 = 事件/经历类，
    # 没有可覆盖的属性。由抽取 prompt 顺带输出（_ATTRIBUTE_ADDENDUM），零额外调用。
    attribute: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExtractedAdditiveMemory:
        lids = raw.get("linked_memory_ids") or []
        if not isinstance(lids, list):
            lids = []
        attr = raw.get("attribute")
        return cls(
            local_id=str(raw.get("id", "")),
            text=str(raw.get("text", "")).strip(),
            attributed_to=str(raw.get("attributed_to", "user")),
            linked_memory_ids=tuple(str(x) for x in lids),
            attribute=str(attr).strip() if isinstance(attr, str) else "",
        )


#: 抽取模型反复无视提示词硬塞进来的两类垃圾，按词面拦掉。
#:
#: 为什么非要在代码里拦：上游提示词开篇就是"每一条可记的信息都必须捕获，漏抽就是
#: 丢上下文"，我们在 addendum 里写的"别抽请求/别抽助手自己的话"压不过它——实测
#: 改了两版提示词都照抽不误。
#:
#: 为什么这两类特别有害：它们的措辞跟当前这轮对话几乎一样，所以下一个问题一来就
#: 得高分，直接把真正相关的记忆挤出 top-k。实测"给我推荐几个菜"存进去之后，下次
#: 再问吃什么，过敏那条就掉出前五，模型于是推荐了含肉的菜。
_JUNK_PATTERNS = (
    # 助手自己说的话被当成关于用户的事实
    "助手推荐", "助手建议", "助手提供", "助手回答", "助手表示", "助手告诉",
    "assistant recommended", "assistant suggested", "assistant provided",
    "assistant replied", "assistant explained", "assistant told",
    # 这一轮的临时请求，不是长期事实
    "询问推荐", "请求推荐", "要求推荐", "询问建议", "请求建议",
    # 问助手「你还记得我吗 / 你对我什么印象」也是一次性请求。这类特别毒：
    # 措辞跟下次同类提问几乎一样，一问就排最前，把真正的画像挤出 top-k。
    "询问对他的认识", "询问对方是否记得", "询问是否记得", "对他的认识",
    "询问对自己的印象", "询问印象", "是否记得他", "是否记得自己",
    "asked for recommendations", "requested recommendations",
    "asked for suggestions", "wants suggestions", "is asking for",
)


def _is_junk(text: str) -> bool:
    low = text.lower()
    return any(pat in text or pat in low for pat in _JUNK_PATTERNS)


def _strip_request_clause(text: str) -> str:
    """一条 memory 里既有长期事实、又有一次性请求时，把请求那半句切掉。

    「我下周要考GRE，你有什么书推荐吗」会被抽成一条
    「用户下周要参加GRE考试，并询问推荐的GRE书籍。」——整条命中 _JUNK_PATTERNS
    的"询问推荐"，于是**连同"下周要考GRE"这个真事实一起丢掉**。实测就是这么丢的：
    右脑存下了原话，左脑一条都没有，用户后面问起来完全想不起有这回事。

    做法保守：只在最后一个逗号处切，切完必须仍是一句完整的事实（≥8 字、不再命中
    junk），否则维持原样照旧丢弃——宁可漏记，也不要往库里塞半截话。
    """
    for sep in ("，并", "，还", "，同时", "，然后", "，", ", and ", ", "):
        if sep not in text:
            continue
        head = text.rsplit(sep, 1)[0].strip().rstrip("，,")
        if len(head) >= 8 and not _is_junk(head):
            return head + ("。" if not head.endswith(("。", ".")) else "")
    return ""


def parse_additive_memory_response(raw_json: str) -> list[ExtractedAdditiveMemory]:
    data = json.loads(raw_json)
    if not isinstance(data, dict):
        raise ValueError("additive response must be a JSON object")
    mem = data.get("memory")
    if mem is None:
        raise ValueError('additive JSON missing "memory" array')
    if not isinstance(mem, list):
        raise ValueError('"memory" must be a list')
    out: list[ExtractedAdditiveMemory] = []
    for item in mem:
        if not isinstance(item, dict):
            continue
        t = (item.get("text") or "").strip()
        if not t:
            continue
        if _is_junk(t):
            kept = _strip_request_clause(t)
            if kept:
                print(f"[extract] 切掉请求那半句：{t[:36]} → {kept[:36]}", flush=True)
                t = kept
            else:
                print(f"[extract] 丢弃（助手自己的话/一次性请求）：{t[:40]}", flush=True)
                continue
        # 合并调用多带回来的标注：暂存给下游的 annotator 用，省掉那次 LLM 调用。
        # 字段缺了就什么都不存，annotator 查不到会照旧自己调一次。
        if any(k in item for k in ("slot", "entities", "relations")):
            from voicemem.leftbrain import merged_extraction
            merged_extraction.put_annotation(t, {
                "slot": item.get("slot", ""),
                "entities": item.get("entities") or [],
                "relations": item.get("relations") or [],
                "confidence": item.get("confidence", 0.9),
            })
        out.append(ExtractedAdditiveMemory.from_dict(item))

    # 右脑标签是**整段**的，不挂在某条 fact 上
    traits = data.get("traits")
    if isinstance(traits, list):
        from voicemem.leftbrain import merged_extraction
        pairs = [(str(x.get("slot", "")).strip(), str(x.get("label", "")).strip())
                 for x in traits if isinstance(x, dict)]
        merged_extraction.put_traits(_MERGED_UTTERANCE.get("text", ""),
                                     [p for p in pairs if p[0] and p[1]])
    emo = data.get("emotion")
    if isinstance(emo, str):
        from voicemem.leftbrain import merged_extraction
        merged_extraction.put_emotion(_MERGED_UTTERANCE.get("text", ""), emo.strip())
    return out


# parse 函数拿不到原话（它只收 JSON 字符串），而 traits 要按原话做 key 才能被
# _extract_rb_traits 取到。extract() 在调用前把原话放这儿。
_MERGED_UTTERANCE: dict[str, str] = {}


@dataclass
class OpenAIAdditiveExtractorConfig:
    """OpenAI Chat 客户端配置。"""

    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    use_input_language: bool = True

    def resolved_model(self) -> str:
        return (self.model or os.environ.get("OPENAI_MODEL", "").strip() or "gpt-4o-mini").strip()


# 追加在 upstream 抽取 prompt 之后（不改 additive_extraction_prompt.txt 本身）。
# 目的：让每条 fact 顺带带上"谁的什么属性"这个不含值的短语。冲突判定阶段用它
# 再搜一次候选——"最喜欢的餐厅是 A" 和 "最喜欢的餐厅是 B" 两句话向量并不像
# （值不同），但都和 "User's favorite restaurant" 很像，这样旧值必进比较窗口
# （benchmark 上 UPDATE 漏判的主因就是旧值挤不进 top-k）。抽取本来就要调一次
# LLM，多一个字段零成本。
_LANGUAGE_RULE = """

# LANGUAGE

Write each memory in the SAME language the user spoke it in. A Chinese utterance
becomes a Chinese memory, an English one an English memory. Never translate.

Retrieval is embedding-based: a Chinese question and an English memory about the
same thing are far apart in vector space, so a translated memory quietly becomes
unfindable. Keep proper nouns and product names as the user said them.
"""


_VOICE_ADDENDUM = """

# VOICE CONTEXT: what is NOT worth remembering

These transcripts come from live speech, so much of what is said is about the
conversation itself rather than about the user. Do NOT extract:
- Audibility and connection checks from either side ("can you hear me?", "能听到吗",
  "在吗"), and the replies confirming audibility.
- Identity probes that carry no new information ("do you know who I am?",
  "你记得我叫什么吗"). If the user actually states their name, extract the name —
  but never the question itself.
- Remarks about the exchange rather than the user's life ("why aren't you replying",
  "别打断我", "你回复得好快", "在跑吗").
- Fragments the ASR clearly garbled: isolated syllables, filler, or text with no
  recoverable meaning ("lolo", "嗯那个", "呃", "清华分对").
- Anything whose only content is that something was said or asked. Never write a
  memory of the form "User asked X" or "Assistant said X" unless X itself is a fact
  about the user's world.
- The user's request for this turn. "User wants dinner suggestions", "User asked for
  restaurant recommendations" — wanting something right now is not a lasting fact.
  A standing preference is ("User only drinks pour-over coffee"); a one-off ask is not.
- Never write "Speaker 0" (or any "Speaker N") into a memory as if it were a name —
  it means the voice has not been matched to a known person yet. In that case say
  "用户" / "the user" instead, and do not assume they are anyone named elsewhere
  in the conversation.
  When the message DOES carry a real name ("Jiaqi: 我搬去杭州了"), keep that name as
  the subject. Rewriting a named speaker to "用户" makes the memory unfindable by
  questions that use the name ("Jiaqi 搬到哪个城市了？").
- The assistant's own answer: suggestions it made, options it listed, information it
  looked up. Those are the assistant's output, not facts about the user. Extract from
  an assistant turn only when the user CONFIRMED something about themselves in it.

These two are the most damaging kind of junk: they are worded like the current
conversation, so they score high on the very next query and push the real memories
out of top-k — the allergy stops being retrieved right when it matters.

The test: would this still be useful to know a week from now, in a different
conversation? If not, do not extract it. Extracting nothing from a turn is a valid
and common outcome — an empty result is better than a worthless memory.

"""

_ATTRIBUTE_ADDENDUM = """

# ADDITIONAL FIELD: attribute

For each memory object, ALSO output an "attribute" field (string):
- If the memory states a fact of the form "<subject>'s <attribute> is <value>" (a preference, status, or personal detail that could later change), set "attribute" to "<subject>'s <attribute>" WITHOUT the value. Examples:
  - "User's favorite restaurant is Sushi Zen" -> "attribute": "User's favorite restaurant"
  - "User works as a data analyst at Grab" -> "attribute": "User's job"
  - "User lives in Clementi" -> "attribute": "User's residence"
  - "User's sister Jiahui is in her final year of high school" -> "attribute": "Jiahui's school year"
- If the memory is an event, experience, or one-off narrative ("User went hiking last Sunday", "User had a fight with a coworker"), set "attribute" to "".
- Keep the subject name explicit ("User" for the account owner, the person's name otherwise). Keep it short (2-6 words). Write it in English regardless of the memory language.
"""


class OpenAIMem0V3AdditiveExtractor:
    """OpenAI Chat + Mem0 additive system/user prompt 抽取。"""

    def __init__(self, config: OpenAIAdditiveExtractorConfig | None = None) -> None:
        self._cfg = config or OpenAIAdditiveExtractorConfig()
        self._system = load_additive_system_prompt() + _LANGUAGE_RULE + _ATTRIBUTE_ADDENDUM + _VOICE_ADDENDUM

    def extract(
        self,
        *,
        new_messages: list[dict[str, str]],
        summary: str | dict[str, str] | None = None,
        recently_extracted_memories: list[Any] | None = None,
        existing_memories: list[dict[str, Any]] | None = None,
        last_k_messages: list[dict[str, str]] | None = None,
        observation_date: str | None = None,
        current_date: str | None = None,
        custom_instructions: str | None = None,
    ) -> list[ExtractedAdditiveMemory]:
        """对 ``new_messages`` 跑一次 additive ADD 抽取。"""
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("请安装: pip install openai>=1.0") from e

        api_key = self._cfg.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("缺少 OPENAI_API_KEY（或在 OpenAIAdditiveExtractorConfig(api_key=...) 传入）")

        # timeout 必须显式设：openai 客户端默认超时长达 10 分钟，上游偶尔一个
        # 请求悬住整个 ingest 就停摆；60s 内没回来就让重试机制接手
        kw_client: dict[str, Any] = {"api_key": api_key, "timeout": 60.0, "max_retries": 2}
        if self._cfg.base_url:
            kw_client["base_url"] = self._cfg.base_url
        client = OpenAI(**kw_client)

        user_content = generate_additive_extraction_prompt(
            summary=summary,
            recently_extracted_memories=recently_extracted_memories,
            existing_memories=existing_memories,
            new_messages=new_messages,
            last_k_messages=last_k_messages,
            current_date=current_date,
            timestamp=observation_date,
            custom_instructions=custom_instructions,
            use_input_language=self._cfg.use_input_language,
        )

        # 合并模式：让这一次调用顺便把 slot/实体/关系/右脑标签也吐出来，
        # 省掉下游 annotator 和 _extract_rb_traits 各自那次 LLM 往返。
        #
        # 追加到**用户消息**末尾。接在 system 后面时顶层的 emotion/traits 会被模型
        # 丢掉（用户消息里的输出格式说明写死了顶层只有 "memory"，它照做），
        # 右脑于是每轮都拿到空的，一个节点都长不出来。见 merged_extraction。
        from voicemem.leftbrain import merged_extraction
        system = self._system
        if merged_extraction.enabled():
            user_content = user_content + merged_extraction.prompt_addendum()
            _MERGED_UTTERANCE["text"] = " ".join(
                (m.get("content") or "") for m in new_messages
                if (m.get("role") or "user") != "assistant").strip()
        else:
            _MERGED_UTTERANCE.pop("text", None)

        resp = client.chat.completions.create(
            model=self._cfg.resolved_model(),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        _log_usage("extract", self._cfg.resolved_model(), getattr(resp, "usage", None))
        raw_text = (resp.choices[0].message.content or "").strip()
        json_str = extract_json(remove_code_blocks(raw_text))
        return parse_additive_memory_response(json_str)

    def extract_from_asr(
        self,
        user_text: str,
        *,
        assistant_reply: str | None = None,
        summary: str | dict[str, str] | None = None,
        recently_extracted_memories: list[Any] | None = None,
        existing_memories: list[dict[str, Any]] | None = None,
        last_k_messages: list[dict[str, str]] | None = None,
        observation_date: str | None = None,
        current_date: str | None = None,
        custom_instructions: str | None = None,
    ) -> list[ExtractedAdditiveMemory]:
        """便捷：本轮用户 ASR + 可选助手回复；``existing_memories`` 供去重/link（UUID ``id`` + ``text``）。"""
        msgs: list[dict[str, str]] = [{"role": "user", "content": user_text.strip()}]
        if assistant_reply is not None and assistant_reply.strip():
            msgs.append({"role": "assistant", "content": assistant_reply.strip()})
        return self.extract(
            new_messages=msgs,
            summary=summary,
            recently_extracted_memories=recently_extracted_memories,
            existing_memories=existing_memories,
            last_k_messages=last_k_messages,
            observation_date=observation_date,
            current_date=current_date,
            custom_instructions=custom_instructions,
        )


# ── Mem0 V1 风格 Conflict Resolver ────────────────────────────────────────────

_CONFLICT_SYSTEM_PROMPT = """You are a smart memory manager which controls the memory of a system.
You can perform four operations: (1) add into the memory, (2) update the memory, (3) delete from the memory, and (4) no change.

Compare newly retrieved facts with the existing memory. For each new fact, decide whether to:
- ADD: Add it to the memory as a new element
- UPDATE: Update an existing memory element (use the existing ID, keep the most informative version)
- DELETE: Delete an existing memory element that directly contradicts the new fact
- NONE: Make no change (fact already present or not worth storing)

Guidelines:
1. ADD if the fact is genuinely new information not covered by existing memories.
2. UPDATE if the fact updates/refines an existing memory about the same topic (same ID, new text).
3. DELETE if the fact directly contradicts an existing memory (e.g. user switched habits, changed preference).
4. NONE if the fact duplicates an existing memory with no meaningful new context.

Return JSON only:
{
  "memory": [
    {"id": "<existing-id or new sequential id>", "text": "<memory text>", "event": "ADD|UPDATE|DELETE|NONE", "old_memory": "<old text, only for UPDATE>"},
    ...
  ]
}

Rules:
- For ADD: use a new sequential string id ("new_0", "new_1", ...).
- For UPDATE/DELETE/NONE: use the exact existing UUID id from the provided list.
- For UPDATE: include "old_memory" with the original text.
- Only emit entries where something actually changes (ADD or UPDATE or DELETE). You may omit NONE entries.
"""

# 上面那版是精简改写（1.5k 字），把 UPDATE 的触发条件写成了"同一个话题就合并"。
# 实测这条太松：7 月"带孩子去博物馆"被判成和 5 月"喜欢画画放松"同话题而并进去，
# 具体细节被揉平（sunrise、"认识朋友 4 年"整条消失），日期还沿用被并那条的旧日期。
# 这里换成 mem0 官方原版（github.com/mem0ai/mem0 mem0/configs/prompts.py 的
# DEFAULT_UPDATE_MEMORY_PROMPT，5310 字，逐字节照抄）：它把 UPDATE 限定在"讲的是
# 同一件事"，并给了 4 组正反例，新事实默认走 ADD。抽取 prompt 已经是照抄的官方版，
# 这一步补齐后整条写入链路与 mem0 对齐。
_CONFLICT_SYSTEM_PROMPT = (
    Path(__file__).resolve().parent / "data" / "mem0_update_memory_prompt.txt"
).read_text(encoding="utf-8")

# mem0 官方这份 prompt 是单用户记忆场景设计的，"同话题 = 更新" 这条规则放到
# 多人家庭场景里很危险：好几个人都可能被问到同一类问题（"你最喜欢的慈善机构
# 是哪个"），如果判断时不认人只认话题，后说话的人会把前面完全不相关的另一个
# 人的记忆覆盖掉（实测：Nancy 先说了自己的慈善偏好，Kenneth/Eric/Jennifer 后面
# 各自说自己的偏好时，其中一条真的把 Nancy 那条覆盖没了）。这里在官方原文之外
# 追加一条强制规则，不改动 mem0_update_memory_prompt.txt 本身。
_CONFLICT_SYSTEM_PROMPT += """

IMPORTANT — multi-person household rule (in addition to the rules above):
Memories in this system can be about DIFFERENT named people living in the same household (e.g. "Nancy's preferred charity is X" vs "Jennifer's preferred charity is Y" are about two different people, not the same fact).
Before choosing UPDATE or DELETE for any existing memory, check whether the named subject/person in the new fact is the SAME as the named subject/person in that existing memory:
- If the subjects are different named people, you MUST NOT use UPDATE or DELETE on that memory, even if the topic/category is identical — treat the new fact as ADD instead.
- Only use UPDATE/DELETE when the existing memory is unambiguously about the SAME person as the new fact (same name, or both clearly refer to the generic account owner "User").
- If a fact's subject is ambiguous or not named, prefer ADD over UPDATE — a duplicate is far cheaper to have than silently overwriting a different person's information.
"""

_SINGLE_VALUED_RULE = """
IMPORTANT — single-valued attribute rule (in addition to the rules above):
Some facts describe a SINGLE-VALUED attribute of a person: something that has exactly one current value and gets replaced when it changes — favorite X (favorite food/restaurant/color/team/song...), current job/employer/role, residence/address, age, phone number, relationship status, current school/grade, the ONE pet's name, the date of a specific planned event.
- If a new fact and an existing memory are about the SAME person and the SAME single-valued attribute but state DIFFERENT values, the new fact SUPERSEDES the old one: choose UPDATE with the new value. This applies even when the new fact contains NO negation or explicit reference to the old value ("User's favorite restaurant is B" simply replaces "User's favorite restaurant is A"). The most recent statement wins.
- MULTI-VALUED attributes (hobbies, friends, foods the person likes in general, places visited, skills, allergies) accumulate: a new value is ADD, not UPDATE, unless it directly contradicts an existing one.
- Events, experiences, and one-off narratives ("went hiking on Sunday", "had dinner with Mom") are NEVER merged into unrelated preference memories and never replace each other: always ADD. Do not fold specific details of one event into another memory.
"""


def _build_update_memory_message(
    retrieved_old_memory: list, new_facts: list, speaker_name: str | None = None,
    system_prompt: str | None = None,
) -> str:
    """拼出送给模型的完整消息，与 mem0 官方 ``get_update_memory_messages`` 同款。

    官方是把「判定 prompt + 旧记忆 + 新事实 + 输出结构说明」拼成一条 user 消息发出去，
    不拆 system/user。照它拼有两个实际好处：输出结构说明和我们的解析器逐字段对得上；
    以及那段说明里带 "JSON" 字样——官方判定 prompt 正文一个 json 字都没有，拆开发
    会被 OpenAI 的 ``response_format=json_object`` 直接拒掉（400）。

    speaker_name:
        本轮新事实的说话人真名（已知时传入）。facts 文本里通常已经带了人名，
        但那是"软信号"——额外把当前说话人明说一遍，给模型一个更硬的锚点去判断
        "已有记忆和新事实是不是同一个人"，而不是只能从两段文本里自己猜。
    """
    if retrieved_old_memory:
        current_memory_part = f"""
    Below is the current content of my memory which I have collected till now. You have to update it in the following format only:

    ```
    {retrieved_old_memory}
    ```

    """
    else:
        current_memory_part = """
    Current memory is empty.

    """

    return f"""{system_prompt if system_prompt is not None else _CONFLICT_SYSTEM_PROMPT}

    {current_memory_part}

    The new retrieved facts are mentioned in the triple backticks. You have to analyze the new retrieved facts and determine whether these facts should be added, updated, or deleted in the memory.
    {f"These new facts were all spoken by: {speaker_name}. Use this as the authoritative subject when checking whether an existing memory is about the same person (per the multi-person household rule above)." if speaker_name else ""}

    ```
    {new_facts}
    ```

    You must return your response in the following JSON structure only:

    {{
        "memory" : [
            {{
                "id" : "<ID of the memory>",                # Use existing ID for updates/deletes, or new ID for additions
                "text" : "<Content of the memory>",         # Content of the memory
                "event" : "<Operation to be performed>",    # Must be "ADD", "UPDATE", "DELETE", or "NONE"
                "old_memory" : "<Old memory content>"       # Required only if the event is "UPDATE"
            }},
            ...
        ]
    }}

    Follow the instruction mentioned below:
    - Do not return anything from the custom few shot prompts provided above.
    - If the current memory is empty, then you have to add the new retrieved facts to the memory.
    - You should return the updated memory in only JSON format as shown below. The memory key should be the same if no changes are made.
    - If there is an addition, generate a new key and add the new memory corresponding to it.
    - If there is a deletion, the memory key-value pair should be removed from the memory.
    - If there is an update, the ID key should remain the same and only the value needs to be updated.

    Do not return anything except the JSON format.
    """


@dataclass
class ConflictResolution:
    event: str          # ADD | UPDATE | DELETE | NONE
    memory_id: str      # existing UUID for UPDATE/DELETE, "new_N" for ADD
    text: str           # new text (ADD/UPDATE) or empty (DELETE)
    old_memory: str = ""


class ConflictResolver:
    """Mem0 V1 风格：对新提取的事实和已有记忆做 ADD/UPDATE/DELETE/NONE 决策。"""

    def __init__(self, config: OpenAIAdditiveExtractorConfig | None = None,
                 single_valued_rule: bool = True) -> None:
        self._cfg = config or OpenAIAdditiveExtractorConfig()
        # 单值属性"新值覆盖"规则（VOICEMEM_CONFLICT_WIDE 消融开关经 voice_input 传入）
        self._system = _CONFLICT_SYSTEM_PROMPT + (_SINGLE_VALUED_RULE if single_valued_rule else "")

    def resolve(
        self,
        new_facts: list[str],
        existing_memories: list[dict[str, str]],  # [{"id": uuid, "text": ...}]
        speaker_name: str | None = None,
    ) -> list[ConflictResolution]:
        """返回每条 fact 的操作决策。existing_memories 为空时全部 ADD。

        speaker_name: 本轮新事实的说话人真名，已知时传入，帮助模型判断
        UPDATE/DELETE 候选是否真的和新事实是同一个人（见模块顶部的
        multi-person household rule）。
        """
        if not new_facts:
            return []

        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("请安装: pip install openai>=1.0") from e

        api_key = self._cfg.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("缺少 OPENAI_API_KEY")

        kw: dict[str, Any] = {"api_key": api_key}
        if self._cfg.base_url:
            kw["base_url"] = self._cfg.base_url
        client = OpenAI(**kw)

        # UUID → 序号映射（防幻觉）：官方 prompt 的示例 id 全是 0/1/2 这样的小整数，
        # 直接把 36 位 UUID 丢进去，模型很容易写错一两位，UPDATE/DELETE 就会指向一条
        # 不存在的记忆而被静默丢掉。mem0 官方同样做这一步（main.py "Map UUIDs to
        # integers (anti-hallucination)"），拿回结果再映射回真实 UUID。
        uuid_mapping: dict[str, str] = {}
        indexed: list[dict[str, str]] = []
        for idx, mem in enumerate(existing_memories or []):
            uuid_mapping[str(idx)] = str(mem.get("id", ""))
            indexed.append({"id": str(idx), "text": str(mem.get("text", ""))})

        user_content = _build_update_memory_message(indexed, new_facts, speaker_name=speaker_name,
                                                    system_prompt=self._system)
        # 上游那个 prompt 要求**对每条候选都输出一个决定**，包括 "event":"NONE"。
        # 候选是按相似度取的（每条新事实 top-15 + 属性 top-10），实测一轮 25 条，
        # 于是模型要吐 25 个 JSON 对象、其中二十几个是 NONE——这一次调用就要 10.2s，
        # 占整轮 ingest 的一半，而且库越大越慢。
        # 改成只吐真正要动的那几条：没被提到的一律当 NONE。消费端本来就是按事件
        # 遍历、不假设一一对应（voice_input.py "若 resolve 成功，按决策执行"），
        # 所以少几条 NONE 不影响任何行为。
        if os.environ.get("VOICEMEM_RESOLVE_OMIT_NONE", "1") != "0":
          user_content += (
            "\n\nIMPORTANT — output size:\n"
            "Return ONLY the memories whose event is ADD, UPDATE or DELETE.\n"
            "OMIT every memory you would mark NONE — anything absent from your\n"
            "output is treated as NONE. Most turns change nothing, so "
            '{"memory": []} is a normal and correct answer.'
          )

        resp = client.chat.completions.create(
            model=self._cfg.resolved_model(),
            messages=[{"role": "user", "content": user_content}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip()

        try:
            data = json.loads(remove_code_blocks(raw))
        except json.JSONDecodeError:
            data = json.loads(extract_json(raw))

        results: list[ConflictResolution] = []
        for item in data.get("memory", []):
            event = str(item.get("event", "NONE")).upper()
            if event not in ("ADD", "UPDATE", "DELETE", "NONE"):
                event = "NONE"
            raw_id = str(item.get("id", ""))
            # UPDATE/DELETE 必须落到真实 UUID 上；序号映射不上就说明模型编了一个
            # 不存在的 id，此时降级成 ADD 而不是拿假 id 去改库（改不动＝事实丢失）
            if event in ("UPDATE", "DELETE"):
                if raw_id in uuid_mapping:
                    raw_id = uuid_mapping[raw_id]
                elif raw_id not in set(uuid_mapping.values()):
                    event = "ADD" if event == "UPDATE" else "NONE"
            results.append(ConflictResolution(
                event=event,
                memory_id=raw_id,
                text=str(item.get("text", "")).strip(),
                old_memory=str(item.get("old_memory", "")),
            ))
        return results
