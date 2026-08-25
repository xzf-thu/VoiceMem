"""左脑子图判定：检索触发（run_for_retrieved_pool），判断一次检索候选池里
的 entity 共现子图要不要正式升级成一个新 slot。不再有 ingest 时按 slot_ref
扫描的 session 边界触发路径——候选池只来自实际检索的结果。

流程：
  1. 候选池是调用方传入的一批 memory_id（比如一次检索的 top-k 结果），不按
     slot_ref 分开扫描——选取步骤不该替共现结构预先猜"哪些更相关"，那是
     连通分量+聚合系数该做的事
  2. 建 entity 共现图：node=entity，两个 entity 出现在同一条 memory 里就连边
  3. 找连通子图，过滤 entity 数 >= 3 的
  4. 排除全员 can_split=False 的子图（除非有 can_split=True 的新 entity 混进来激活）
  5. 每个连通分量内部贪心剥离（反复去掉当前子集里度数最低的节点），每剥一层都
     算一次关联系数 ρ（论文公式：候选子集跟用户历史每次查询激活的实体集合的
     平均 Jaccard 相似度，见 GraphEntityStore.compute_rho），保留这条分量剥离
     序列里 ρ 最高的那一层——分量本身大不代表真的该独立成簇，可能剥掉一两个
     弱关联节点后 ρ 反而更高。
  6. 跨分量比较各自剥出来的最优子集，ρ 全局最高的那个作为本轮候选
     （一次调用只出一个候选，其余分量留到下次检索命中时再看）
  7. ρ 没过 MIN_SYNERGY_THRESHOLD 这条线的候选，直接跳过（不调用LLM，
     省成本），但不永久拉黑——证据不够跟判定过不重要是两回事，can_split
     保持 True，以后共现证据攒够了还能重新被考虑
  8. LLM 判定：是不是同一个事件/主题，且足够重要 → 是就新建 slot 并把这批 entity
     迁过去（不产生新 memory）；否就把这批 entity 的 can_split 永久置 False
"""

from __future__ import annotations

import json
from typing import Callable

from .graph_entity_store import GraphEntity, GraphEntityStore

MIN_SUBGRAPH_SIZE = 3

# ρ 下限——ρ 是 Jaccard 相似度的均值，取值范围 [0,1]，跟之前"密度×计数"
# （可以到几十）完全不是一个量级，不能照搬旧阈值。这里先给一个宽松值：
# 没过线的候选本来就不调LLM（省成本），真正的质量把关交给下一步LLM判定，
# 阈值本身先不做硬卡死。等积累了真实查询数据、看到 ρ 的实际分布后再收紧。
MIN_SYNERGY_THRESHOLD = 0.05

# 一个子图最多能覆盖候选池里多大比例的记忆——这条跟打分公式无关，独立保留：
# 剥离过程还是要防止"糊成一大团"这种候选进入 ρ 打分（分量本身跨了半个库时，
# 先把它按覆盖比例砍掉，逼贪心继续往下剥到足够紧的那一层，ρ 才有意义去评估）。
MAX_TOUCHED_FRACTION = 0.25


def _connected_components(adjacency: dict[str, set[str]]) -> list[set[str]]:
    seen: set[str] = set()
    components: list[set[str]] = []
    for start in adjacency:
        if start in seen:
            continue
        comp = {start}
        queue = [start]
        seen.add(start)
        while queue:
            node = queue.pop()
            for nb in adjacency.get(node, ()):
                if nb not in seen:
                    seen.add(nb)
                    comp.add(nb)
                    queue.append(nb)
        components.append(comp)
    return components


def _synergy(
    entity_ids: set[str],
    mem_to_entities: dict[str, set[str]],
    rho_fn: Callable[[set[str]], float],
) -> tuple[float, set[str]]:
    """给定一批 entity id，返回 (ρ 关联系数, 涉及的 memory 集合)。
    ρ 由调用方传入的 rho_fn 计算（GraphEntityStore.compute_rho，论文公式：
    候选子集跟用户历史每次查询激活集合的平均 Jaccard 相似度）。"""
    touched = {mid for mid, eids in mem_to_entities.items() if eids & entity_ids}
    return rho_fn(entity_ids), touched


def _densest_subset(
    component: set[str],
    adjacency: dict[str, set[str]],
    mem_to_entities: dict[str, set[str]],
    min_size: int,
    rho_fn: Callable[[set[str]], float],
    max_touched: int | None = None,
) -> tuple[set[str], float, set[str]]:
    """贪心剥离（Charikar peeling）：反复去掉当前子集里度数最低的节点，
    每层都算一次 ρ，返回整条剥离路径上 ρ 最高的那一层（包括不剥、即整个
    连通分量本身，作为剥离路径的起点）。

    max_touched 非空时，只有覆盖记忆数不超过它的那些层才有资格当最优解——
    见 MAX_TOUCHED_FRACTION 的说明。整条剥离路径每一层都超标时（说明这批
    entity 无论怎么剥都铺得太广，本质上不是一条活动线），返回 ρ=0.0，
    让它落到 MIN_SYNERGY_THRESHOLD 之下被跳过。注意跳过不等于拉黑：
    can_split 保持 True，以后共现结构变了还能重新被考虑。
    """
    def _ok(touched: set[str]) -> bool:
        return max_touched is None or len(touched) <= max_touched

    current = set(component)
    best_set: set[str] = set(current)
    best_synergy = 0.0
    best_touched: set[str] = set()
    found_ok = False

    synergy, touched = _synergy(current, mem_to_entities, rho_fn)
    if _ok(touched):
        best_set, best_synergy, best_touched, found_ok = set(current), synergy, touched, True

    while len(current) > min_size:
        degrees = {eid: len(adjacency[eid] & current) for eid in current}
        weakest = min(degrees, key=degrees.get)
        current = current - {weakest}
        synergy, touched = _synergy(current, mem_to_entities, rho_fn)
        if _ok(touched) and (not found_ok or synergy > best_synergy):
            best_set, best_synergy, best_touched, found_ok = set(current), synergy, touched, True

    if not found_ok:
        return set(component), 0.0, set()
    return best_set, best_synergy, best_touched


class SubgraphManager:
    """tag_fn: 新 slot 判定通过时，把促成这个 slot 诞生的那批 memory 反过来在
    cog_store.memory_tags 里也打上新 slot 的标签——不这么做的话，GraphEntityStore
    里的 slot_ref 迁移只是内部记账（供下次子图分析用），检索路径
    （SearchCogGraph/memory_ids_for_slots_v2）完全不读 GraphEntityStore，新 slot
    从检索角度永远是空的，Classify() 就算分类出这个新 slot 也查不到任何东西。
    tag_fn 签名 (user_id, memory_id, slot_name) -> None，是追加式打标签
    （upsert_memory_tags 本身是按 (memory_id, slot) 唯一键 upsert，不会清掉这条
    memory 已有的旧标签），所以老的宽泛slot("relationships")和新的窄slot都能查到
    这条memory——宽查询保住召回，窄查询获得精度。"""

    def __init__(
        self,
        graph_store: GraphEntityStore,
        dynamic_slot_store,
        llm_fn: Callable[[str], str],
        tag_fn: Callable[[str, str, str], None] | None = None,
    ) -> None:
        self._graph = graph_store
        self._dyn = dynamic_slot_store
        self._llm = llm_fn
        self._tag = tag_fn

    def run_for_retrieved_pool(
        self,
        user_id: str,
        memory_ids: set[str],
        memory_content_lookup: Callable[[str], str | None],
        session_id: str | None = None,
    ) -> dict:
        """候选池直接是一批 memory_id（比如一次真实检索的 top-k 结果），不按
        slot_ref 分开扫描——检索本身不经过 GraphEntityStore 的 slot_ref
        （SearchCogGraph 走的是 cog_store 的实体索引和向量检索），所以同一个
        entity 即使因为不同 ingest 批次被 _llm_tag_memories 打上不同 slot
        标签、分别注册在不同的 slot_ref 下，只要它们出现在同一批 memory_id
        里就能被重新放到一起建共现图。真正判定共现/重要性的逻辑
        （_promote_or_reject）不做任何区分对待，跟候选池怎么组装出来的无关。
        """
        if not memory_ids:
            return {"status": "no_memories"}

        ent_by_id: dict[str, GraphEntity] = {}
        mem_to_entities: dict[str, set[str]] = {}
        for mid in memory_ids:
            ents = self._graph.get_entities_for_memory(user_id, mid)
            if not ents:
                continue
            for e in ents:
                ent_by_id[e.id] = e
            mem_to_entities[mid] = {e.id for e in ents}

        if not ent_by_id:
            return {"status": "no_memories"}

        return self._promote_or_reject(user_id, ent_by_id, mem_to_entities, memory_content_lookup, session_id)

    def _promote_or_reject(
        self,
        user_id: str,
        ent_by_id: dict[str, GraphEntity],
        mem_to_entities: dict[str, set[str]],
        memory_content_lookup: Callable[[str], str | None],
        session_id: str | None = None,
    ) -> dict:
        """共享核心：从 (ent_by_id, mem_to_entities) 出发，建共现图→连通分量→
        贪心剥离找最优子集→LLM判定→新建slot(晋升) 或 can_split=False(拒绝)。

        新建 slot 时的 parent_slots 直接从参与晋升的 entity 各自当前的
        slot_ref 推导（此时还没调用 update_slot_ref，读到的就是它们原本
        所在的 slot_ref）——候选池里的 entity 可能横跨好几个 slot_ref（同一个
        entity 因为不同 ingest 批次被打上不同标签），推导出来就是真正的多父
        slot 集合，不需要调用方指定或猜；如果候选池恰好都在同一个 slot_ref
        下，自然收敛成单元素集合。"""
        adjacency: dict[str, set[str]] = {eid: set() for eid in ent_by_id}
        for eids in mem_to_entities.values():
            eid_list = list(eids)
            for i in range(len(eid_list)):
                for j in range(i + 1, len(eid_list)):
                    adjacency[eid_list[i]].add(eid_list[j])
                    adjacency[eid_list[j]].add(eid_list[i])

        components = [c for c in _connected_components(adjacency) if len(c) >= MIN_SUBGRAPH_SIZE]
        components = [c for c in components if any(ent_by_id[eid].can_split for eid in c)]
        if not components:
            return {"status": "no_eligible_subgraph"}

        # 每个连通分量各自剥出 ρ 最高的子集，跨分量取全局最高的那个。
        # session_id 传了就把 Q 限定在当前会话（论文字面：Q 是当前会话的查询
        # 集合，不是终身历史）——没传（没有 session 概念的调用方）时
        # compute_rho 自己退回旧的全历史行为。
        rho_fn = lambda h: self._graph.compute_rho(user_id, h, session_id=session_id)  # noqa: E731
        max_touched = max(MIN_SUBGRAPH_SIZE,
                          int(len(mem_to_entities) * MAX_TOUCHED_FRACTION))
        candidates = [
            _densest_subset(c, adjacency, mem_to_entities, MIN_SUBGRAPH_SIZE, rho_fn, max_touched)
            for c in components
        ]
        best, synergy, touched_memories = max(candidates, key=lambda t: t[1])
        synergy_score = round(synergy, 4)

        # ρ 没过线的候选，直接跳过LLM判定——"证据还不够"跟"LLM判定过
        # 不重要"是两回事，can_split 保持 True，以后共现证据攒够了还能重新被
        # 考虑，不像 LLM 拒绝那样永久拉黑。
        if synergy_score < MIN_SYNERGY_THRESHOLD:
            names = [ent_by_id[eid].name for eid in best]
            return {"status": "below_threshold", "entities": names, "synergy": synergy_score}

        names = [ent_by_id[eid].name for eid in best]
        sample_texts = [memory_content_lookup(mid) for mid in list(touched_memories)[:5]]
        sample_texts = [t for t in sample_texts if t]

        # 已有动态 slot 一并交给 LLM 判定——主题跟已有 slot 高度重合时并入
        # 已有的那个，不允许再起一个雷同的新名字（"家庭与成长"/"家庭与爱"/
        # "关爱与接纳"各建一个，同一片记忆被切碎，查哪个都只看到一角）
        existing = [(s.name, s.description) for s in self._dyn.get_dynamic_slots(user_id)]
        verdict = self._judge_subgraph(names, sample_texts, existing)

        if verdict:
            kind, slot_name, slot_desc = verdict
            if kind == "new":
                parent_slots = sorted({ent_by_id[eid].slot_ref for eid in best if ent_by_id[eid].slot_ref})
                self._dyn.create_dynamic_slot(user_id, slot_name, slot_desc, parent_slots=parent_slots)
            for eid in best:
                self._graph.update_slot_ref(eid, slot_name)
            if self._tag is not None:
                for mid in touched_memories:
                    try:
                        self._tag(user_id, mid, slot_name)
                    except Exception:
                        pass
            return {
                "status": "slot_created" if kind == "new" else "merged_into_slot",
                "name": slot_name, "entities": names, "synergy": synergy_score,
            }

        for eid in best:
            if ent_by_id[eid].can_split:
                self._graph.mark_cannot_split(eid)
        return {"status": "rejected", "entities": names, "synergy": synergy_score}

    def _judge_subgraph(
        self,
        entity_names: list[str],
        sample_texts: list[str],
        existing_slots: list[tuple[str, str]] | None = None,
    ) -> tuple[str, str, str] | None:
        """三次独立 LLM 校验（相关性/重要性/信息完整性），三个都过才通过；
        任一没过直接拒绝，不再往下问剩下的（省成本，拒绝理由已经确定）。
        返回 ("existing", 已有slot名, "") / ("new", 新名字, 描述) / None(拒绝)。

        三项标准分开问而不是糅合成一次问答，是因为三者关注点不同、容易
        互相稀释：相关性问的是"这是不是同一件事"，重要性问的是"这件事
        在这个用户的记忆里够不够分量单独占一个格子"，信息完整性问的是
        "现在攒的这些内容够不够撑起一个有用的分类，而不是一个空壳标签"。
        糅合成一次问答时，模型容易只答出其中一条最突出的理由就下结论，
        另外两条被隐式带过，拒绝/通过的判断依据说不清楚。
        """
        relevance = self._check_relevance(entity_names, sample_texts, existing_slots)
        if relevance is None:
            return None
        kind, slot_name, slot_desc = relevance
        # 并入已有主题：那个主题第一次被创建时已经独立验证过重要性/信息
        # 完整性，这次只是"这批新证据也属于同一个已验证过的主题"，不需要
        # 重新过一遍——不然一个已经成立的活动线，只因为这次凑巧带来的样本
        # 少，就会被后两项校验误拒，导致新证据点没法并入。
        if kind == "existing":
            return relevance
        if not self._check_importance(slot_name, sample_texts):
            return None
        if not self._check_completeness(slot_name, slot_desc, sample_texts):
            return None
        return relevance

    def _check_relevance(
        self,
        entity_names: list[str],
        sample_texts: list[str],
        existing_slots: list[tuple[str, str]] | None,
    ) -> tuple[str, str, str] | None:
        """标准一（相关性）：这几个概念是不是共同指向同一条具体活动线？
        如果是，识别出是哪一条（已有的就并入，没有就起个名字+描述）。
        """
        snippets = "\n".join(f"- {t}" for t in sample_texts) or "(无)"
        existing_part = ""
        if existing_slots:
            listing = "\n".join(f"- {n}: {d}" for n, d in existing_slots)
            existing_part = (
                f"\n用户已有这些细分主题：\n{listing}\n\n"
                "如果这几个概念指向的主题跟上面已有的某一个本质上是同一回事"
                "（哪怕措辞不同），必须并入那个已有主题，不要另起一个意思雷同的新名字。\n"
            )
        # 判定标准是「活动线」而不是「主题」。这两者差别很大，实测直接决定成败：
        # 按主题判会劈出「骄傲与自我接纳」这类情感/价值观标签——它横切所有话题，
        # 越劈越大（覆盖了半个库），而且跟提问方式对不上。用户问的是"她哪天做的
        # 那个盘子""她七月哪天去露营的"，检索单元是**具体活动**，不是情绪。
        # 按活动线判才会劈出「陶艺」「露营」「领养进程」这种——同一件事反复做、
        # 每次是可区分的一场，正好对上这类问题。
        prompt = (
            f"以下几个概念在用户的记忆里经常一起出现：{', '.join(entity_names)}\n\n"
            f"相关记忆片段：\n{snippets}\n"
            f"{existing_part}\n"
            "请只判断一件事：这几个概念是不是共同指向**同一条具体的活动线**——"
            "即用户反复投入的某件具体的事（一项爱好、一类外出活动、一个正在推进的"
            "过程），它在时间上发生过多次、每一次是可以区分开的一场。\n\n"
            "算的例子：陶艺（报班→做碗→做盘子→受伤停了）、露营（好几次不同的"
            "露营，各有日期和地点）、领养（查机构→参加说明会→递申请→通过面试）、"
            "跑步、养宠物。\n"
            "不算的例子：\n"
            "- 情绪、价值观、心境：自我接纳、成长、幸福、感恩、勇气、归属感\n"
            "- 泛泛的人际或家庭概念：家人的爱、朋友的支持、亲密关系\n"
            "这类东西横切所有话题，归成一类之后什么都能匹配，反而查不准。\n"
            "名字必须是这件活动本身（名词），不能是它带来的感受。\n\n"
            "这一步先不用管这件事够不够重要、内容够不够丰富——那是接下来两步"
            "单独判断的事，这里只回答\"这是不是同一条活动线\"。\n"
            "- 属于已有主题 → {\"split\": true, \"existing\": \"已有主题名\"}\n"
            "- 是全新的活动线 → {\"split\": true, \"name\": \"2-6字中文名\", \"description\": \"一句描述\"}\n"
            "- 不是同一条活动线，是几个碰巧同时出现彼此无关的概念 → {\"split\": false}\n"
            "只输出 JSON。"
        )
        raw = self._llm(prompt)
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except Exception:
            return None
        if not data.get("split"):
            return None
        existing_names = {n for n, _ in (existing_slots or [])}
        chosen = str(data.get("existing", "") or "").strip()
        if chosen and chosen in existing_names:
            return ("existing", chosen, "")
        name = str(data.get("name", "")).strip()
        desc = str(data.get("description", "")).strip()
        if not name:
            return None
        # 模型没走 existing 字段但起的名字跟已有的撞了 → 同样按并入处理
        if name in existing_names:
            return ("existing", name, "")
        return ("new", name, desc)

    def _check_importance(self, slot_name: str, sample_texts: list[str]) -> bool:
        """标准二（重要性）：这条活动线在用户记忆里的分量够不够，值得单独
        建一个可检索的分类，而不是零星提过一两次的琐事。"""
        snippets = "\n".join(f"- {t}" for t in sample_texts) or "(无)"
        prompt = (
            f"用户的记忆里有一条可能的活动线：「{slot_name}」\n\n"
            f"相关记忆片段：\n{snippets}\n\n"
            "请只判断一件事：这件事在用户的生活/记忆里够不够重要、够不够"
            "被反复提及，值得单独给它建一个可检索的分类（而不是随口一提的"
            "琐事，或者只出现过一次就不再提起）？\n"
            "只被顺口提过一两次、后续没有再展开的，不算重要。\n\n"
            "{\"important\": true} 或 {\"important\": false}，只输出 JSON。"
        )
        raw = self._llm(prompt)
        if not raw:
            return False
        try:
            data = json.loads(raw)
        except Exception:
            return False
        return bool(data.get("important"))

    def _check_completeness(self, slot_name: str, slot_desc: str, sample_texts: list[str]) -> bool:
        """标准三（信息完整性）：现在攒的这些记忆片段是不是有足够具体的
        信息量，能撑起一个真正有用的分类，而不是一个内容空洞的空壳标签。"""
        snippets = "\n".join(f"- {t}" for t in sample_texts) or "(无)"
        prompt = (
            f"用户的记忆里有一条活动线：「{slot_name}」（{slot_desc or '无描述'}）\n\n"
            f"相关记忆片段：\n{snippets}\n\n"
            "请只判断一件事：这些记忆片段里的信息量是不是已经足够具体"
            "（比如有具体的时间、地点、进展、细节这类实质内容），能支撑起一个"
            "以后检索时真正有用的分类？如果这些片段全都很空洞、笼统"
            "（比如只反复说\"提到了这件事\"而没有任何具体细节），信息量不够，"
            "现在建这个分类只是个没有内容的空壳。\n\n"
            "{\"complete\": true} 或 {\"complete\": false}，只输出 JSON。"
        )
        raw = self._llm(prompt)
        if not raw:
            return False
        try:
            data = json.loads(raw)
        except Exception:
            return False
        return bool(data.get("complete"))
