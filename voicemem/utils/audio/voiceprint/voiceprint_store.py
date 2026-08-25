"""声纹管家。

三路判断：
  score >= match_thr  → match：直接精化画像
  cand_thr <= score < match_thr → candidate：先临时建一个新画像，同时留一条
                                   candidate 记录；后续同一个人的声音继续
                                   出现、累积出的子质心跟原画像重新交叉打分
                                   过 match_thr 时，find_resolvable_candidates()
                                   / merge_persons() 自动把两个画像合并回去
                                   （见 core.py::_reconcile_speaker_candidates）。
                                   若原画像（top）分叉后一直样本极少（<=2，
                                   说明它此后每次都输给分叉画像、没机会再被
                                   真实匹配更新）而分叉画像（fork）已经靠自己
                                   攒到>=3条真实样本，改用更宽松的 cand_thr，
                                   否则这种候选永远卡在原画像那条噪声样本上、
                                   无法被回收。
  score < cand_thr    → new：当新人，建新画像
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voicemem.utils.audio.voiceprint import Profile, SubCentroid, l2norm


@dataclass
class IdentifyResult:
    person_id: str   # match/new → 真实 person_id；candidate → candidate_id
    score: float
    action: str      # "match" | "candidate" | "new"


class VoiceprintStore:
    """维护 person_id → Profile 的声纹库。线程安全。"""

    def __init__(
        self,
        store_dir: Path,
        match_threshold: float = 0.50,
        candidate_threshold: float = 0.40,
        merge_threshold: float = 0.65,
    ):
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._match_thr = match_threshold
        self._cand_thr = candidate_threshold
        # 候选自动回收合并用，故意比 match_thr 更高——见 find_resolvable_candidates
        # 里的说明：合并是永久、几乎不可逆的操作，不能用单句匹配那道门槛。
        self._merge_thr = merge_threshold
        self._lock = threading.Lock()
        self._profiles: dict[str, Profile] = {}
        # person_id → {obs_count, confidence, created_at}
        # 真实姓名绑定不在这里：那是 VoiceprintRegistry（voice_input.py）的职责，
        # 存在 voiceprint_registry.json，靠"Hi, I am X" 自报姓名触发 bind()。
        self._meta: dict[str, dict] = {}
        # candidate_id → {vec, top_match_person_id, score, context, ts}
        self._candidates: dict[str, dict] = {}
        self._load()

    # ── 持久化 ──────────────────────────────────────────────────────────────

    def _meta_path(self) -> Path:
        return self._dir / "voiceprint_meta.json"

    def _profile_path(self, person_id: str) -> Path:
        return self._dir / f"profile_{person_id}.npz"

    def _load(self) -> None:
        if not self._meta_path().exists():
            return
        data = json.loads(self._meta_path().read_text(encoding="utf-8"))
        self._meta = data.get("persons", {})
        self._candidates = data.get("candidates", {})
        for pid in self._meta:
            path = self._profile_path(pid)
            if not path.exists():
                continue
            npz = np.load(path, allow_pickle=True)
            prof = Profile(
                w_max=float(npz.get("w_max", [40.0])[0]),
                t_split=float(npz.get("t_split", [0.55])[0]),
            )
            vecs = npz["vecs"]  # [k, D]
            ws   = npz["ws"]    # [k]
            prof.subs = [SubCentroid(vecs[i], float(ws[i])) for i in range(len(ws))]
            self._profiles[pid] = prof

    def _save(self) -> None:
        data = {"persons": self._meta, "candidates": self._candidates}
        self._meta_path().write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for pid, prof in self._profiles.items():
            if not prof.subs:
                continue
            vecs = np.stack([s.vec for s in prof.subs])
            ws   = np.array([s.w   for s in prof.subs])
            np.savez(
                self._profile_path(pid),
                vecs=vecs, ws=ws,
                w_max=np.array([prof.w_max]),
                t_split=np.array([prof.t_split]),
            )

    # ── 主接口 ──────────────────────────────────────────────────────────────

    def identify(
        self, vec: np.ndarray, context: str = "", pinned_person_id: str | None = None,
    ) -> IdentifyResult:
        """给定声纹向量，返回 IdentifyResult。

        ``pinned_person_id``：调用方（core.py）在同一个 session 里，一旦靠自我
        介绍明确过"这句话是谁说的"，就会把这个 person_id 传进来。原因：
        自我介绍触发的拆分（``create_person``）只用当次这一句话的向量建了个
        全新画像，样本量是 1；而这段 session 剩下的每一句仍然会走下面纯声纹
        打分选全局最高分——分数常年打不过对方那个攒了几十条样本的老画像，
        于是刚拆出来的新画像永远没机会再被真实匹配更新，等于白拆（实测
        Nancy/Jennifer 案例：拆分只在自报姓名那一句触发过一次，之后 Jennifer
        每一句都被判回 Nancy 的画像，分数还比 Nancy 自己说话时更高）。
        这里只要 pinned 对象自己的分数还过得去（>=cand_thr，没到"完全不像"
        的地步），就优先信 session 已经确认过的这个人，而不是纯声纹最高分——
        既打破"新画像永远太薄"的死循环，也避免把不属于这个人的样本继续
        更新进另一个人的画像里。
        """
        vec = l2norm(np.asarray(vec, dtype=np.float64))
        with self._lock:
            if not self._profiles:
                pid = self._new_person()
                self._profiles[pid].update(vec, quality=1.0)
                self._meta[pid]["obs_count"] = 1
                self._save()
                return IdentifyResult(pid, 1.0, "new")

            scores = {pid: prof.score(vec) for pid, prof in self._profiles.items()}
            best_pid = max(scores, key=scores.__getitem__)
            best_score = scores[best_pid]

            if (
                pinned_person_id is not None
                and pinned_person_id in scores
                and pinned_person_id != best_pid
                and scores[pinned_person_id] >= self._cand_thr
            ):
                best_pid, best_score = pinned_person_id, scores[pinned_person_id]

            if best_score >= self._match_thr:
                self._profiles[best_pid].update(vec, quality=float(best_score))
                self._meta[best_pid]["obs_count"] += 1
                conf = self._meta[best_pid]["confidence"]
                self._meta[best_pid]["confidence"] = round(min(1.0, conf + 0.02), 4)
                self._save()
                return IdentifyResult(best_pid, best_score, "match")

            elif best_score >= self._cand_thr:
                # 候选不能继续作为 person_id 返回：调用方会把它当作不稳定
                # ID，导致多人记忆无法挂载。先建独立声纹保证隔离；候选记录
                # 记住"这个新声纹当初差一点就判给了 top_match_person_id"，
                # 后续靠 find_resolvable_candidates() 自动核对——短句的单次
                # 声纹分数噪声很大，但等新声纹自己也攒够几条真实匹配之后，
                # 拿两边攒起来的画像互相打分会比单条向量准得多；真的收敛到
                # 同一个人时才自动并回去，而不是永远靠 verify_candidate 桩
                # 等一个没人调用的人工/LLM 确认。
                pid = self._new_person()
                self._add_candidate(vec, best_pid, best_score, context, forked_person_id=pid)
                self._profiles[pid].update(vec, quality=1.0)
                self._meta[pid]["obs_count"] = 1
                self._save()
                return IdentifyResult(pid, best_score, "new")

            else:
                pid = self._new_person()
                self._profiles[pid].update(vec, quality=1.0)
                self._meta[pid]["obs_count"] = 1
                self._save()
                return IdentifyResult(pid, best_score, "new")

    def create_person(self, vec: np.ndarray, context: str = "") -> IdentifyResult:
        """强制把当前声纹拆成独立人物，供名字冲突时使用。"""
        vec = l2norm(np.asarray(vec, dtype=np.float64))
        with self._lock:
            pid = self._new_person()
            self._profiles[pid].update(vec, quality=1.0)
            self._meta[pid]["obs_count"] = 1
            self._meta[pid]["context"] = context[:120]
            self._save()
            return IdentifyResult(pid, 0.0, "new")

    def _new_person(self) -> str:
        pid = f"person_{uuid.uuid4().hex[:8]}"
        self._profiles[pid] = Profile()
        self._meta[pid] = {
            "obs_count": 0,
            "confidence": 0.5,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        return pid

    def _add_candidate(
        self, vec: np.ndarray, top_match: str, score: float, context: str,
        forked_person_id: str,
    ) -> str:
        cid = f"cand_{uuid.uuid4().hex[:8]}"
        self._candidates[cid] = {
            "vec": vec.tolist(),
            "top_match_person_id": top_match,
            "forked_person_id": forked_person_id,
            "score": round(float(score), 4),
            "context": context,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._save()
        return cid

    # ── 候选自动回收 ────────────────────────────────────────────────────────

    def find_resolvable_candidates(self, updated_pid: str) -> list[tuple[str, str, str, float]]:
        """``updated_pid`` 刚拿到一次新的真实 match 之后调用：检查是否有候选记录

        因为这次更新而能被判定为"其实是同一个人"了。``updated_pid`` 可能是候选
        记录里的 ``top_match_person_id``（原声纹又被真实匹配上了），也可能是
        ``forked_person_id``（当初被拆出去的那个声纹自己又攒了新的真实匹配）——
        两种情况都用当前两边最新的画像互相打分，而不是候选记录里存的那条孤立
        的旧向量，样本多了噪声自然被压下去。

        返回 ``[(candidate_id, absorb_pid, into_pid, score), ...]``，只读、不
        修改任何状态；是否真的合并、要不要处理姓名冲突交给调用方（``core.py``
        持有 VoiceprintRegistry，这里的 store 不认姓名）。
        """
        with self._lock:
            out: list[tuple[str, str, str, float]] = []
            for cid, c in self._candidates.items():
                top_pid = c.get("top_match_person_id")
                fork_pid = c.get("forked_person_id")
                if updated_pid not in (top_pid, fork_pid):
                    continue
                top_prof = self._profiles.get(top_pid)
                fork_prof = self._profiles.get(fork_pid)
                if top_prof is None or fork_prof is None or not top_prof.subs or not fork_prof.subs:
                    continue  # 有一边已经被合并掉/删掉了
                cross = max(
                    float(np.dot(l2norm(a.vec), l2norm(b.vec)))
                    for a in top_prof.subs for b in fork_prof.subs
                )
                # 正常情况下要求 merge_thr 级别的信心——比单句 match_thr 更高，
                # 因为合并是永久、几乎不可逆的操作（两个画像焊成一个，名字互相
                # 覆盖），不能只按"这一句像不像"的门槛。实测过 cross=0.542（刚
                # 过 match_thr 一点点）就足够把两个真实的不同人（Nancy/Jennifer，
                # 本身声音就分不太开）永久焊死、还带偏了名字，事后没法拆。
                #
                # 但如果 top（分叉发生时命中的原画像）样本数一直很薄
                # （obs_count<=2）、而 fork（分叉出去的候选画像）已经靠自己攒到
                # 至少 3 条真实样本——这说明 identify() 后续每次都判给了 fork
                # （分数更稳、样本更多），top 从分叉那一刻起就再也没机会被真实
                # 匹配更新过，它那 1-2 条样本的噪声永远没法被摊薄，cross 分数会
                # 被死死摁在 merge_thr 以下（Nancy 案例：max_cross 卡在 0.47
                # 附近）。只在这种"top 结构性地不可能自证、fork 已有真实积累"的
                # 组合下才改用更宽松的 cand_thr——本来就是"值得建候选、可能是
                # 同一个人"那条线；不对称地只放宽这一种情况，避免放宽到"候选
                # 刚建、fork 只有 1 条样本"这种本该继续等待积累的正常场景，也
                # 避免放宽到"两边都已经各自独立攒了好几条样本"这种更可能是两个
                # 不同人的场景。
                top_obs = self._meta.get(top_pid, {}).get("obs_count", 0)
                fork_obs = self._meta.get(fork_pid, {}).get("obs_count", 0)
                thin_and_stuck = top_obs <= 2 and fork_obs >= 3
                threshold = self._cand_thr if thin_and_stuck else self._merge_thr
                if cross >= threshold:
                    out.append((cid, fork_pid, top_pid, cross))
            return out

    def merge_persons(self, absorb_pid: str, into_pid: str) -> None:
        """把 ``absorb_pid`` 的画像并入 ``into_pid``，删除 ``absorb_pid``，
        并清掉所有指向这两个 id 的候选记录（已经有结论了，不用再等确认）。"""
        with self._lock:
            absorb_prof = self._profiles.pop(absorb_pid, None)
            into_prof = self._profiles.get(into_pid)
            if absorb_prof is None or into_prof is None:
                return
            into_prof.subs.extend(absorb_prof.subs)
            into_prof.consolidate()
            while len(into_prof.subs) > into_prof.k_max:
                into_prof._merge_closest()

            absorb_meta = self._meta.pop(absorb_pid, {})
            into_meta = self._meta[into_pid]
            into_meta["obs_count"] = into_meta.get("obs_count", 0) + absorb_meta.get("obs_count", 0)
            into_meta["confidence"] = round(
                min(1.0, max(into_meta.get("confidence", 0.5), absorb_meta.get("confidence", 0.5)) + 0.02), 4
            )

            self._candidates = {
                cid: c for cid, c in self._candidates.items()
                if absorb_pid not in (c.get("top_match_person_id"), c.get("forked_person_id"))
                and into_pid not in (c.get("top_match_person_id"), c.get("forked_person_id"))
            }

            path = self._profile_path(absorb_pid)
            if path.exists():
                path.unlink()
            self._save()

    # ── 辅助 ────────────────────────────────────────────────────────────────

    def get_meta(self, person_id: str) -> dict:
        return dict(self._meta.get(person_id, {}))

    def list_persons(self) -> list[dict]:
        with self._lock:
            return [
                {"person_id": pid, **self._meta.get(pid, {})}
                for pid in self._profiles
            ]
