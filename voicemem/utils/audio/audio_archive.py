"""AudioArchive — 记录每条记忆对应的原始 WAV 文件路径。

存储结构（SQLite）::

    audio_archive(
        memory_id   TEXT NOT NULL,
        user_id     TEXT NOT NULL,
        audio_path  TEXT NOT NULL,
        archived_at TEXT NOT NULL,
        PRIMARY KEY (memory_id, user_id)
    )
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AudioArchive:
    """把 memory_id → audio_path 的对应关系持久化到 SQLite。"""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def _ensure_schema(self) -> None:
        with self._conn() as c:
            c.execute("""
            CREATE TABLE IF NOT EXISTS audio_archive (
                memory_id   TEXT NOT NULL,
                user_id     TEXT NOT NULL,
                audio_path  TEXT NOT NULL,
                archived_at TEXT NOT NULL,
                PRIMARY KEY (memory_id, user_id)
            )
            """)

    def record(self, memory_ids: list[str], user_id: str, audio_path: str) -> None:
        """将一批 memory_id 与 audio_path 关联存储。"""
        now = _now()
        with self._conn() as c:
            c.executemany(
                "INSERT OR IGNORE INTO audio_archive VALUES (?,?,?,?)",
                [(mid, user_id, audio_path, now) for mid in memory_ids],
            )

    def get_audio_path(self, memory_id: str, user_id: str) -> str | None:
        """查询某条记忆的原始 WAV 路径，不存在时返回 None。"""
        with self._conn() as c:
            row = c.execute(
                "SELECT audio_path FROM audio_archive WHERE memory_id=? AND user_id=?",
                (memory_id, user_id),
            ).fetchone()
        return row[0] if row else None

    def get_by_audio(self, audio_path: str, user_id: str) -> list[str]:
        """查询某个 WAV 文件对应的所有 memory_id。"""
        with self._conn() as c:
            rows = c.execute(
                "SELECT memory_id FROM audio_archive WHERE audio_path=? AND user_id=?",
                (audio_path, user_id),
            ).fetchall()
        return [r[0] for r in rows]

    def cleanup_expired(self, retention_days: int = 30) -> list[str]:
        """删除超过保留期的原始 WAV 文件本体，数据库里的映射行不动。

        memory_id → audio_path 的映射永久保留，只是文件本体过期后被物理删除——
        `get_audio_path()` / `GetOriginalAudio()` 之后靠 ``Path.exists()`` 判断
        "已过期"，不需要额外的 flag。同一个 audio_path 可能对应多条 memory_id
        （一次录音抽出多条记忆），这里按 audio_path 去重，只删一次文件。

        Returns
        -------
        list[str]
            本次实际删除的文件路径（去重后）。
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
        with self._conn() as c:
            rows = c.execute(
                "SELECT DISTINCT audio_path FROM audio_archive WHERE archived_at < ?",
                (cutoff,),
            ).fetchall()

        deleted: list[str] = []
        for (audio_path,) in rows:
            p = Path(audio_path)
            if p.exists():
                try:
                    p.unlink()
                    deleted.append(audio_path)
                except OSError:
                    pass
        return deleted
