"""MusicMemoryStore — 背景音乐/哼唱识别记忆（audiomem 2.5）。

复用 voicemem/voiceprint/adaptive_centroid.py 的自适应多中心机制，但这次聚类的
不是声纹而是 AST 环境音 embedding：反复出现的同一段背景音乐/哼唱会被识别
成"熟悉的调子"，而不是每次都当新东西记一遍。

跟声纹三级判断（match/candidate/new，见 voiceprint_store.py）不同，这里没有
"认错人、需要人工确认"那种后果——认错一首歌不会破坏画像或伤害用户，所以只做
两级：match（并入已知调子）/ new（建一个新调子档案）。

match_threshold=0.80 是没有真实数据校准过的经验默认值（不像 voiceprint 那边
用 MagicData 做过 EER 分析）——AST embedding 是分类模型的中间表示，同一类
声音（比如"都是钢琴曲"）本身就会有较高的相似度，阈值需要偏高一些才能把
"同一首歌"和"同一类音乐"区分开；后续如果拿到真实重复哼唱的数据，应该重新
校准。
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
class TuneIdentifyResult:
    tune_id: str
    score: float
    action: str        # "match" | "new"
    heard_count: int   # 这个调子被识别到的累计次数（含本次）


class MusicMemoryStore:
    """维护 tune_id → Profile 的"熟悉调子"库。线程安全。"""

    def __init__(self, store_dir: Path, match_threshold: float = 0.80):
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._match_thr = match_threshold
        self._lock = threading.Lock()
        self._profiles: dict[str, Profile] = {}
        # tune_id → {labels, heard_count, created_at}
        self._meta: dict[str, dict] = {}
        self._load()

    # ── 持久化 ──────────────────────────────────────────────────────────────

    def _meta_path(self) -> Path:
        return self._dir / "music_meta.json"

    def _profile_path(self, tune_id: str) -> Path:
        return self._dir / f"profile_{tune_id}.npz"

    def _load(self) -> None:
        if not self._meta_path().exists():
            return
        data = json.loads(self._meta_path().read_text(encoding="utf-8"))
        self._meta = data.get("tunes", {})
        for tid in self._meta:
            path = self._profile_path(tid)
            if not path.exists():
                continue
            npz = np.load(path, allow_pickle=True)
            prof = Profile(
                w_max=float(npz.get("w_max", [40.0])[0]),
                t_split=float(npz.get("t_split", [0.55])[0]),
            )
            vecs = npz["vecs"]
            ws = npz["ws"]
            prof.subs = [SubCentroid(vecs[i], float(ws[i])) for i in range(len(ws))]
            self._profiles[tid] = prof

    def _save(self) -> None:
        data = {"tunes": self._meta}
        self._meta_path().write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for tid, prof in self._profiles.items():
            if not prof.subs:
                continue
            vecs = np.stack([s.vec for s in prof.subs])
            ws = np.array([s.w for s in prof.subs])
            np.savez(
                self._profile_path(tid),
                vecs=vecs, ws=ws,
                w_max=np.array([prof.w_max]),
                t_split=np.array([prof.t_split]),
            )

    # ── 主接口 ──────────────────────────────────────────────────────────────

    def identify(self, vec: np.ndarray, labels: list[str] | None = None) -> TuneIdentifyResult:
        """给定这段音乐/哼唱的 AST embedding，返回识别结果（match/new）。"""
        vec = l2norm(np.asarray(vec, dtype=np.float64))
        with self._lock:
            if not self._profiles:
                tid = self._new_tune(labels)
                self._profiles[tid].update(vec, quality=1.0)
                self._meta[tid]["heard_count"] = 1
                self._save()
                return TuneIdentifyResult(tid, 1.0, "new", 1)

            scores = {tid: prof.score(vec) for tid, prof in self._profiles.items()}
            best_tid = max(scores, key=scores.__getitem__)
            best_score = scores[best_tid]

            if best_score >= self._match_thr:
                self._profiles[best_tid].update(vec, quality=float(best_score))
                self._meta[best_tid]["heard_count"] += 1
                self._save()
                return TuneIdentifyResult(
                    best_tid, best_score, "match", self._meta[best_tid]["heard_count"]
                )

            tid = self._new_tune(labels)
            self._profiles[tid].update(vec, quality=1.0)
            self._meta[tid]["heard_count"] = 1
            self._save()
            return TuneIdentifyResult(tid, best_score, "new", 1)

    def _new_tune(self, labels: list[str] | None) -> str:
        tid = f"tune_{uuid.uuid4().hex[:8]}"
        self._profiles[tid] = Profile()
        self._meta[tid] = {
            "labels": labels or [],
            "heard_count": 0,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        return tid

    # ── 辅助 ────────────────────────────────────────────────────────────────

    def get_meta(self, tune_id: str) -> dict:
        return dict(self._meta.get(tune_id, {}))

    def list_tunes(self) -> list[dict]:
        with self._lock:
            return [{"tune_id": tid, **self._meta.get(tid, {})} for tid in self._profiles]
