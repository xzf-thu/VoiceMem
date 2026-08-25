"""SceneTrigger — 声学触发提醒系统。

用户说"到公交上提醒我打电话给妈妈"，系统存储一条 scene-triggered reminder，
当检测到场景变为 transit 时自动触发。

存储结构（SQLite）::

    scene_triggers(
        id          TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL,
        trigger_scene TEXT NOT NULL,   -- 触发场景 tag，如 "transit"
        message     TEXT NOT NULL,     -- 提醒内容
        created_at  TEXT NOT NULL,
        status      TEXT NOT NULL,     -- "pending" | "fired" | "cancelled"
        required_label TEXT            -- 具体声音标签（如"bus"），NULL=粗粒度场景对上即可
    )

    scene_state(
        user_id     TEXT PRIMARY KEY,
        last_scene  TEXT,              -- 上次检测到的场景
        updated_at  TEXT
    )
"""
from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from voicemem.utils.audio.environment.scene_classifier import SceneTag


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 用户描述场景的自然语言 → (scene_tag, 需要在 raw_matches 里命中的具体 AudioSet 标签)
#
# required_label 为 None 表示粗粒度场景对上就行；不为 None 时，check_and_fire()
# 还得在这次检测到的具体标签里找到它，才会真正触发——避免"公交"和"地铁"
# 都被粗分成 transit，导致设置"上公交提醒我"却在坐地铁时被误触发。
_SCENE_ALIASES: dict[str, tuple[SceneTag, str | None]] = {
    # transit —— 细分到具体交通工具
    "公交": (SceneTag.TRANSIT, "bus"), "公交车": (SceneTag.TRANSIT, "bus"),
    "地铁": (SceneTag.TRANSIT, "subway"), "地铁上": (SceneTag.TRANSIT, "subway"),
    "火车": (SceneTag.TRANSIT, "train"), "火车上": (SceneTag.TRANSIT, "train"),
    # 这几个是泛指，用户没说具体交通工具，粗粒度场景对上就行
    "通勤": (SceneTag.TRANSIT, None), "上班路上": (SceneTag.TRANSIT, None),
    "下班路上": (SceneTag.TRANSIT, None), "路上": (SceneTag.TRANSIT, None),
    "出行": (SceneTag.TRANSIT, None), "坐车": (SceneTag.TRANSIT, None),
    # office
    "办公室": (SceneTag.OFFICE, None), "公司": (SceneTag.OFFICE, None),
    "上班": (SceneTag.OFFICE, None), "工位": (SceneTag.OFFICE, None),
    # home
    "家": (SceneTag.HOME, None), "到家": (SceneTag.HOME, None), "回家": (SceneTag.HOME, None),
    # café
    "咖啡厅": (SceneTag.CAFE, None), "咖啡馆": (SceneTag.CAFE, None),
    "星巴克": (SceneTag.CAFE, None), "cafe": (SceneTag.CAFE, None),
    # outdoor
    "外面": (SceneTag.OUTDOOR, None), "户外": (SceneTag.OUTDOOR, None),
    "出门": (SceneTag.OUTDOOR, None), "散步": (SceneTag.OUTDOOR, None),
    # meeting
    "会议": (SceneTag.MEETING, None), "开会": (SceneTag.MEETING, None),
    "会议室": (SceneTag.MEETING, None),
    # quiet
    "安静": (SceneTag.QUIET, None), "独处": (SceneTag.QUIET, None),
}

# 触发词：识别用户是否在创建一个 scene trigger
_TRIGGER_PATTERNS = [
    "提醒我", "remind me", "记得提醒",
    "到时候提醒", "别忘了提醒",
]

_SCENE_ARRIVAL_PATTERNS = [
    "到", "上了", "进了", "等到", "等我到",
    "等我上", "到了", "到达",
]


@dataclass
class SceneTrigger:
    id: str
    user_id: str
    trigger_scene: str   # SceneTag.value
    message: str
    created_at: str
    status: str          # "pending" | "fired" | "cancelled"
    required_label: str | None = None   # 具体声音标签，如"bus"；None=不细分


class SceneTriggerStore:
    """SQLite 存取 scene-triggered reminders。"""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self._path)
        c.row_factory = sqlite3.Row
        return c

    def _ensure_schema(self) -> None:
        with self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS scene_triggers (
                id            TEXT PRIMARY KEY,
                user_id       TEXT NOT NULL,
                trigger_scene TEXT NOT NULL,
                message       TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                status        TEXT NOT NULL DEFAULT 'pending',
                required_label TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_st_user ON scene_triggers(user_id, status);

            CREATE TABLE IF NOT EXISTS scene_state (
                user_id    TEXT PRIMARY KEY,
                last_scene TEXT,
                updated_at TEXT
            );
            """)

    def create(
        self, user_id: str, trigger_scene: str, message: str,
        required_label: str | None = None,
    ) -> SceneTrigger:
        t = SceneTrigger(
            id=str(uuid.uuid4()),
            user_id=user_id,
            trigger_scene=trigger_scene,
            message=message,
            created_at=_now(),
            status="pending",
            required_label=required_label,
        )
        with self._conn() as c:
            c.execute(
                "INSERT INTO scene_triggers VALUES (?,?,?,?,?,?,?)",
                (t.id, t.user_id, t.trigger_scene, t.message, t.created_at, t.status, t.required_label),
            )
        return t

    def get_pending(self, user_id: str, scene: str) -> list[SceneTrigger]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM scene_triggers WHERE user_id=? AND trigger_scene=? AND status='pending'",
                (user_id, scene),
            ).fetchall()
        return [_row_to_trigger(r) for r in rows]

    def fire(self, trigger_id: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE scene_triggers SET status='fired' WHERE id=?",
                (trigger_id,),
            )

    def cancel(self, trigger_id: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE scene_triggers SET status='cancelled' WHERE id=?",
                (trigger_id,),
            )

    def get_last_scene(self, user_id: str) -> str | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT last_scene FROM scene_state WHERE user_id=?", (user_id,)
            ).fetchone()
        return row["last_scene"] if row else None

    def update_scene(self, user_id: str, scene: str) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO scene_state (user_id, last_scene, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET
                       last_scene=excluded.last_scene,
                       updated_at=excluded.updated_at""",
                (user_id, scene, _now()),
            )


def _row_to_trigger(row: sqlite3.Row) -> SceneTrigger:
    return SceneTrigger(
        id=row["id"], user_id=row["user_id"],
        trigger_scene=row["trigger_scene"], message=row["message"],
        created_at=row["created_at"], status=row["status"],
        required_label=row["required_label"],
    )


# ── 触发解析（从用户语句中提取 scene trigger 意图） ───────────────────────────

def parse_trigger_intent(text: str) -> tuple[SceneTag | None, str, str | None]:
    """从用户语句中解析场景触发意图。

    Returns:
        (trigger_scene, reminder_message, required_label) 如果识别到触发意图，
        否则 (None, "", None)。required_label 非空时，触发前还得在检测到的具体
        声音标签里核实一遍（见 check_and_fire），避免"公交"和"地铁"混在一起。

    Examples::
        "到公交上提醒我打电话给妈妈" → (TRANSIT, "打电话给妈妈", "bus")
        "等我到办公室了提醒我发邮件" → (OFFICE, "发邮件", None)
        "我今天跑步了" → (None, "", None)
    """
    text_lower = text.lower()

    # 必须同时含有到达词 + 提醒词
    has_arrival = any(p in text for p in _SCENE_ARRIVAL_PATTERNS)
    has_remind  = any(p in text_lower for p in _TRIGGER_PATTERNS)
    if not (has_arrival and has_remind):
        return None, "", None

    # 识别场景
    matched_scene: SceneTag | None = None
    required_label: str | None = None
    for alias, (scene, label) in _SCENE_ALIASES.items():
        if alias in text:
            matched_scene = scene
            required_label = label
            break

    if matched_scene is None:
        return None, "", None

    # 提取提醒内容：取"提醒我"之后的部分
    for kw in _TRIGGER_PATTERNS:
        idx = text.find(kw)
        if idx != -1:
            message = text[idx + len(kw):].strip()
            if message:
                return matched_scene, message, required_label

    return matched_scene, text, required_label  # 整句作为提醒内容


# ── 场景变化检查器 ────────────────────────────────────────────────────────────

@dataclass
class TriggerFireResult:
    scene: str
    fired: list[SceneTrigger]   # 这次触发的提醒列表
    scene_changed: bool


def check_and_fire(
    store: SceneTriggerStore,
    user_id: str,
    new_scene: str,
    raw_labels: list[str] | None = None,
) -> TriggerFireResult:
    """检查新场景是否触发待命提醒，触发后标记为 fired。

    在每次 Ingest 结束后调用，传入当前检测到的 scene_tag，以及这次判定该场景
    时命中的具体 AudioSet 标签（raw_labels，如 ["Bus", "Vehicle"]）。

    trigger.required_label 非空时（比如"公交"对应的"bus"），必须在 raw_labels
    里找到子串匹配才会真正触发——避免"上公交提醒我"在坐地铁时被误触发（地铁
    和公交都粗分在 transit 里，但具体标签不一样）。required_label 为空的
    trigger（比如泛指"通勤"）不做这层校验，粗粒度场景对上就触发，跟以前一样。
    """
    last_scene = store.get_last_scene(user_id)
    scene_changed = (last_scene != new_scene)

    raw_labels_lower = [l.lower() for l in (raw_labels or [])]

    fired: list[SceneTrigger] = []
    if scene_changed and new_scene not in (SceneTag.UNKNOWN.value, SceneTag.QUIET.value):
        pending = store.get_pending(user_id, new_scene)
        for trigger in pending:
            if trigger.required_label and not any(
                trigger.required_label in rl for rl in raw_labels_lower
            ):
                continue  # 具体场景没对上（比如要"bus"却检测到的是"train"），先不触发
            store.fire(trigger.id)
            fired.append(trigger)
            print(f"  [scene_trigger] 🔔 {trigger.message} (scene={new_scene})", flush=True)

    store.update_scene(user_id, new_scene)
    return TriggerFireResult(scene=new_scene, fired=fired, scene_changed=scene_changed)
