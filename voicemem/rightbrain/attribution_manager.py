"""右脑归因批处理：短期归因（entity.description，每3轮）+ 长期归因（slot.description，session边界）。

短期：某个entity这几轮新关联了哪些memory，综合这些memory内容，更新这个entity的
description，同时顺手把memory item本身也精炼一下（去掉冗余，保留核心）。

长期：session结束时，看这个slot下所有entity（含它们的description），
综合出一个更上位的slot描述——类似"这方面呈现出的人格画像"。
"""

from __future__ import annotations

from typing import Callable

from .graph_store import RightBrainGraphStore
from .store import RightBrainStore


def _is_cjk(text: str) -> bool:
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    return cjk / max(sum(1 for c in text if c.isalpha()) + cjk, 1) >= 0.3


class AttributionManager:
    def __init__(
        self,
        graph_store: RightBrainGraphStore,
        rb_store: RightBrainStore,
        llm_fn: Callable[[str], str],
    ) -> None:
        self._graph = graph_store
        self._rb_store = rb_store
        self._llm = llm_fn

    # ── 短期归因：entity.description ────────────────────────────────────────

    def run_short_term(self, user_id: str, entity_ids: list[str]) -> None:
        for eid in entity_ids:
            ent = self._graph.get_entity(eid)
            if ent is None:
                continue
            mem_ids = self._graph.get_memories_for_entity(eid)
            contents = []          # (mid, content, already_refined)
            for mid in mem_ids:
                mem = self._rb_store.get_memory(mid)
                if mem is not None and mem.content:
                    contents.append((mid, mem.content, bool((mem.metadata or {}).get("refined"))))
            if not contents:
                continue

            src = [c for _, c, _ in contents]
            new_desc = self._summarize_entity(ent.name, src)
            # 同上：语言跟证据对不上就不写，留着上一版描述——画像每轮都会拼进
            # system prompt，中英混杂比少一条更糟。
            if new_desc and _is_cjk(new_desc) != _is_cjk(" ".join(src)):
                new_desc = ""
            if new_desc:
                self._graph.set_entity_description(eid, new_desc)

            # 精炼每条 memory item（去冗余）——但每条只精炼一次（metadata.refined
            # 打标）。以前每轮短期归因都会把同一批记忆反复重写：每次都是有损
            # 改写，多轮累积后数字/名字/时间这类具体细节被逐渐磨掉，而且每条
            # 都是一次同步 LLM 调用，随轮次线性烧钱。
            for mid, content, already_refined in contents:
                if already_refined:
                    continue
                refined = self._refine_memory_item(ent.name, content)
                # 换语言了就丢弃：prompt 里写了"必须保持原句语言"但模型不保证遵守，
                # 一旦把中文原话精炼成英文，存的就不再是用户说过的话了（记忆是证据，
                # 翻译过的证据没法用），后续归纳出的画像也跟着中英混杂。
                if refined and _is_cjk(refined) != _is_cjk(content):
                    print(f"[Attribution] 精炼后语言变了，保留原句：{content[:30]}")
                    refined = ""
                if refined and refined != content:
                    self._rb_store.update_content(mid, refined)
                self._rb_store.merge_metadata(mid, {"refined": True})

    def _summarize_entity(self, entity_name: str, contents: list[str]) -> str:
        snippets = "\n".join(f"- {c}" for c in contents[:20])
        prompt = (
            f"以下是跟「{entity_name}」这个特质相关的一批记忆片段：\n{snippets}\n\n"
            "请用一句话（30字以内）概括这个特质在用户身上的具体表现。\n"
            "硬性要求：写具体行为和事实，不写抒情形容；平实白话；"
            "不要重复列举原句；片段里没有依据的不要写；"
            "输出语言与片段一致（片段是英文就用英文写）。"
        )
        return self._llm_text(prompt)

    def _refine_memory_item(self, entity_name: str, content: str) -> str:
        prompt = (
            f"这条记忆跟用户的「{entity_name}」这个特质有关：\n「{content}」\n\n"
            "如果这句话有冗余/口语化水分/多余的形容词，帮忙精炼成更简洁平实的一句话，"
            "保留原意和关键细节，不要丢失信息；如果已经很简洁就原样返回。\n"
            "必须保持原句的语言（原句是英文就输出英文，不要翻译）。\n"
            "只输出精炼后的这一句话，不要加任何解释。"
        )
        return self._llm_text(prompt)

    # ── 长期归因：slot.description ───────────────────────────────────────────

    def run_long_term(self, user_id: str, slot_ids: list[str]) -> None:
        for sid in slot_ids:
            slot = self._graph.get_slot(sid)
            if slot is None:
                continue
            # 只汇总真正有 memory 证据的 entity——种子占位的空 entity（比如
            # 从没触发过的情绪标签）不该参与人格总结，否则会凭空编出没观测到的特质。
            entities = [
                e for e in self._graph.get_entities_for_slot(user_id, sid)
                if self._graph.get_memories_for_entity(e.id)
            ]
            if not entities:
                continue
            new_desc = self._summarize_slot(slot.name, entities)
            src = " ".join(f"{e.name}{e.description or ''}" for e in entities)
            if new_desc and _is_cjk(new_desc) != _is_cjk(src):
                new_desc = ""
            if new_desc:
                self._graph.set_slot_description(sid, new_desc)

    def _summarize_slot(self, slot_name: str, entities) -> str:
        lines = []
        for e in entities:
            if e.description:
                lines.append(f"- {e.name}：{e.description}")
            else:
                lines.append(f"- {e.name}")
        listing = "\n".join(lines[:30])
        # 曾经只要求"2-3句话概括整体画像"，没限字数没限文风——GPT 会写成
        # 100多字的排比抒情段（"呈现出一种复杂的状态，既渴望…又…"），这种
        # 描述每轮都被拼进 system prompt（prestimulus 人格块 + profile hit），
        # 是实测里整个 prompt 最大的 token 开销，而且空话多、信息密度低。
        prompt = (
            f"以下是用户在「{slot_name}」这个维度上目前呈现出的具体特质：\n{listing}\n\n"
            f"请把用户在「{slot_name}」维度的画像概括成一句话，50字以内。\n"
            "硬性要求：只写上面条目里有依据的具体观察；平实白话，像同事间的"
            "备注，不像心理报告；禁止排比、比喻和抒情；信息不够就少写，"
            "不要凭空拔高或补全；输出语言与条目一致（条目是英文就用英文）。"
        )
        return self._llm_text(prompt)

    # ── LLM 辅助 ──────────────────────────────────────────────────────────────

    def _llm_text(self, prompt: str) -> str:
        """跟 core.py::_llm_json 不同，这里要的是纯文本输出，不是JSON。"""
        raw = self._llm(prompt)
        return raw.strip() if raw else ""
