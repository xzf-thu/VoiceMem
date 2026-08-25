"""RoutineStore — 生活声音规律：自动建立 routine 记忆（audiomem 2.7）。

按"场景 + 大致时间段"记录场景到达事件（场景变化时记一次，不是每句话都记，
见 core.py Ingest 里只在 scene_changed 时调用 observe()）。同一天同一场景
同一时间段只算一次（PRIMARY KEY 天然去重）。当某个 (场景, 时间段) 组合
在 routine_threshold 个不同的日子里都出现过，就判定为一条生活规律——只在
刚跨过阈值的这一次触发建立 routine 记忆，之后同一 (scene, bucket) 不会
重复生成（见 routines 表的存在性检查）。

时间段用 3 小时一档（一天 8 档），给"差不多这个点"留误差空间，不要求
分钟级精确重合。
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

_BUCKET_LABELS = [
    "deep night (00:00-03:00)", "early morning (03:00-06:00)",
    "morning (06:00-09:00)", "late morning (09:00-12:00)",
    "early afternoon (12:00-15:00)", "afternoon (15:00-18:00)",
    "evening (18:00-21:00)", "night (21:00-24:00)",
]


def bucket_label(bucket: int) -> str:
    return _BUCKET_LABELS[bucket % 8]


class RoutineStore:
    """维护"场景在什么时间段规律性出现"的观测与已建立的 routine。"""

    def __init__(self, db_path: Path, routine_threshold: int = 3) -> None:
        self._path = db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._threshold = routine_threshold
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def _ensure_schema(self) -> None:
        with self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS scene_observations (
                user_id  TEXT NOT NULL,
                scene    TEXT NOT NULL,
                bucket   INTEGER NOT NULL,
                obs_date TEXT NOT NULL,
                ts       TEXT NOT NULL,
                PRIMARY KEY (user_id, scene, bucket, obs_date)
            );
            CREATE TABLE IF NOT EXISTS routines (
                user_id        TEXT NOT NULL,
                scene          TEXT NOT NULL,
                bucket         INTEGER NOT NULL,
                established_at TEXT NOT NULL,
                distinct_days  INTEGER NOT NULL,
                PRIMARY KEY (user_id, scene, bucket)
            );
            """)

    def observe(self, user_id: str, scene: str, dt: datetime) -> dict:
        """记录一次场景到达观测。

        Returns
        -------
        dict
            ``{is_new_routine, distinct_days, bucket}``。``is_new_routine=True``
            表示这次观测让 (scene, bucket) 刚好跨过 routine_threshold 个不同
            日子，值得建立一条新的 routine 记忆——同一 (user, scene, bucket)
            只会触发一次。
        """
        bucket = dt.hour // 3
        obs_date = dt.date().isoformat()
        with self._conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO scene_observations VALUES (?,?,?,?,?)",
                (user_id, scene, bucket, obs_date, dt.isoformat()),
            )
            row = c.execute(
                "SELECT COUNT(DISTINCT obs_date) FROM scene_observations "
                "WHERE user_id=? AND scene=? AND bucket=?",
                (user_id, scene, bucket),
            ).fetchone()
            distinct_days = row[0] if row else 0

            already = c.execute(
                "SELECT 1 FROM routines WHERE user_id=? AND scene=? AND bucket=?",
                (user_id, scene, bucket),
            ).fetchone()

            is_new_routine = False
            if distinct_days >= self._threshold and not already:
                c.execute(
                    "INSERT INTO routines VALUES (?,?,?,?,?)",
                    (user_id, scene, bucket, dt.isoformat(), distinct_days),
                )
                is_new_routine = True

        return {
            "is_new_routine": is_new_routine,
            "distinct_days": distinct_days,
            "bucket": bucket,
        }

    def list_routines(self, user_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT scene, bucket, established_at, distinct_days "
                "FROM routines WHERE user_id=?",
                (user_id,),
            ).fetchall()
        return [
            {"scene": r[0], "bucket": r[1], "bucket_label": bucket_label(r[1]),
             "established_at": r[2], "distinct_days": r[3]}
            for r in rows
        ]
