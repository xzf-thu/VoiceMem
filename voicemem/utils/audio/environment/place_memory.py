"""PlaceMemoryStore — 熟悉地点自动聚类（audiomem 2.11）。

跟 MusicMemoryStore（music_memory.py）几乎一样的结构，复用 adaptive_centroid
做自适应多中心聚类，但这次聚的是"具体地点"而不是"具体调子"：同一个粗粒度
scene_tag（比如都是 café）下，不同的具体咖啡厅背景声学特征（混响、底噪、
装修材质带来的频响差异）其实不一样，AST 环境 embedding 能捕捉到这种差异
——反复到访同一个具体地点会被识别成"熟悉的地方"，而不是每次都当新地方记
一遍。这是 audiomem 2.12（熟悉环境主动提示"上次在这里"）的前置依赖。

跟声纹三级判断不同，这里也只做两级：match（识别为已知地点）/ new（新地点）
——认错地点不会有声纹认错人那种社交后果，不需要人工确认候选。

match_threshold=0.80 是跟 MusicMemoryStore 一样没有真实数据校准过的经验默认
值（同样的道理：AST embedding 对"同一类地方"本身就会给出较高相似度，阈值
要偏高才能把"同一个具体地方"和"同一类地方"区分开），后续有真实重复到访
录音数据时应该重新校准。
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from voicemem.utils.audio.voiceprint import Profile, SubCentroid, l2norm


@dataclass
class PlaceIdentifyResult:
    place_id: str
    score: float
    action: str                    # "match" | "new"
    visit_count: int               # 累计到访次数（含本次）
    previous_visit_at: str | None  # 上一次被识别到访的时间；None=第一次到访
    scene: str | None              # 这个地点关联的粗粒度场景标签（如果有）


class PlaceMemoryStore:
    """维护 place_id → Profile 的"熟悉地点"库。线程安全。"""

    def __init__(self, store_dir: Path, match_threshold: float = 0.80):
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._match_thr = match_threshold
        self._lock = threading.Lock()
        self._profiles: dict[str, Profile] = {}
        # place_id → {scene, visit_count, first_seen, last_seen, created_at}
        self._meta: dict[str, dict] = {}
        self._load()

    # ── 持久化 ──────────────────────────────────────────────────────────────

    def _meta_path(self) -> Path:
        return self._dir / "place_meta.json"

    def _profile_path(self, place_id: str) -> Path:
        return self._dir / f"profile_{place_id}.npz"

    def _load(self) -> None:
        if not self._meta_path().exists():
            return
        data = json.loads(self._meta_path().read_text(encoding="utf-8"))
        self._meta = data.get("places", {})
        for pid in self._meta:
            path = self._profile_path(pid)
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
            self._profiles[pid] = prof

    def _save(self) -> None:
        data = {"places": self._meta}
        self._meta_path().write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for pid, prof in self._profiles.items():
            if not prof.subs:
                continue
            vecs = np.stack([s.vec for s in prof.subs])
            ws = np.array([s.w for s in prof.subs])
            np.savez(
                self._profile_path(pid),
                vecs=vecs, ws=ws,
                w_max=np.array([prof.w_max]),
                t_split=np.array([prof.t_split]),
            )

    # ── 主接口 ──────────────────────────────────────────────────────────────

    def identify(
        self, vec: np.ndarray, scene: str | None = None, when: datetime | None = None,
    ) -> PlaceIdentifyResult:
        """给定这段录音的 AST embedding，返回地点识别结果（match/new）。"""
        vec = l2norm(np.asarray(vec, dtype=np.float64))
        now_str = (when or datetime.now()).isoformat()
        with self._lock:
            if not self._profiles:
                pid = self._new_place(scene, now_str)
                self._profiles[pid].update(vec, quality=1.0)
                self._meta[pid]["visit_count"] = 1
                self._save()
                return PlaceIdentifyResult(pid, 1.0, "new", 1, None, scene)

            scores = {pid: prof.score(vec) for pid, prof in self._profiles.items()}
            best_pid = max(scores, key=scores.__getitem__)
            best_score = scores[best_pid]

            if best_score >= self._match_thr:
                prev_last_seen = self._meta[best_pid].get("last_seen")
                self._profiles[best_pid].update(vec, quality=float(best_score))
                self._meta[best_pid]["visit_count"] += 1
                self._meta[best_pid]["last_seen"] = now_str
                self._save()
                return PlaceIdentifyResult(
                    best_pid, best_score, "match",
                    self._meta[best_pid]["visit_count"], prev_last_seen,
                    self._meta[best_pid].get("scene"),
                )

            pid = self._new_place(scene, now_str)
            self._profiles[pid].update(vec, quality=1.0)
            self._meta[pid]["visit_count"] = 1
            self._save()
            return PlaceIdentifyResult(pid, best_score, "new", 1, None, scene)

    def _new_place(self, scene: str | None, now_str: str) -> str:
        pid = f"place_{uuid.uuid4().hex[:8]}"
        self._profiles[pid] = Profile()
        self._meta[pid] = {
            "scene": scene,
            "visit_count": 0,
            "first_seen": now_str,
            "last_seen": now_str,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        return pid

    # ── 辅助 ────────────────────────────────────────────────────────────────

    def get_meta(self, place_id: str) -> dict:
        return dict(self._meta.get(place_id, {}))

    def list_places(self) -> list[dict]:
        with self._lock:
            return [{"place_id": pid, **self._meta.get(pid, {})} for pid in self._profiles]
