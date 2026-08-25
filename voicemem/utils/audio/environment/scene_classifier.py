"""SceneClassifier — maps AudioSet labels (from the AST detector) to high-level scene tags.

scene_tag 取值:
    office   — 键盘、打印机、空调、办公室噪音
    outdoor  — 风声、雨声、鸟鸣、交通噪音
    transit  — 公共交通：公交/地铁/火车/飞机
    café     — 餐厅/咖啡厅：餐具、背景人声、咖啡机
    meeting  — 多人会议：回声、混响、多人声
    home     — 家居：电视/洗衣机/炊事
    quiet    — 安静：几乎无背景音
    unknown  — 无法判断

用法::
    from voicemem.utils.audio.environment.scene_classifier import classify_scene, SceneTag
    tag, conf = classify_scene([("Computer keyboard", 0.72), ("Typing", 0.55)])
    # → SceneTag.OFFICE, 0.635
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SceneTag(str, Enum):
    OFFICE  = "office"
    OUTDOOR = "outdoor"
    TRANSIT = "transit"
    CAFE    = "café"
    MEETING = "meeting"
    HOME    = "home"
    QUIET   = "quiet"
    UNKNOWN = "unknown"


# AudioSet 标签关键词 → scene_tag 映射（关键词均为小写）
_SCENE_KEYWORDS: dict[SceneTag, list[str]] = {
    SceneTag.OFFICE: [
        "keyboard", "typing", "typewriter", "printer", "photocopier",
        "air conditioning", "ventilation fan", "computer", "office",
        "mechanical fan", "white noise",
    ],
    SceneTag.OUTDOOR: [
        "wind", "rain", "thunder", "raindrop", "drizzle",
        "bird", "chirp", "tweet", "crow", "owl", "insect", "cricket",
        "stream", "river", "waterfall", "ocean", "wave",
        "traffic", "road", "highway", "nature",
    ],
    SceneTag.TRANSIT: [
        "bus", "train", "subway", "metro", "rail", "railroad",
        "aircraft", "airplane", "helicopter", "engine", "motor",
        "vehicle", "car", "truck", "automobile",
        "horn", "siren", "emergency vehicle",
    ],
    SceneTag.CAFE: [
        "restaurant", "cafe", "coffee", "espresso", "cutlery",
        "silverware", "dishes", "clatter", "glass", "clink",
        "background music", "crowd", "chatter", "babble",
    ],
    SceneTag.MEETING: [
        "echo", "reverberation", "reverb", "conference",
        "auditorium", "classroom", "lecture",
    ],
    SceneTag.HOME: [
        "television", "tv set", "washing machine", "microwave",
        "kettle", "boiling", "frying", "cooking", "vacuum cleaner",
        "door", "sink", "water tap", "refrigerator", "dishwasher",
        "toilet", "shower",
    ],
}

# 预编译 label → scene 查找表（把上面的列表拍平）
_LABEL_TO_SCENE: dict[str, SceneTag] = {}
for _scene, _keywords in _SCENE_KEYWORDS.items():
    for _kw in _keywords:
        _LABEL_TO_SCENE[_kw] = _scene


@dataclass
class SceneResult:
    tag: SceneTag
    confidence: float
    raw_matches: list[tuple[str, float]]  # [(label, score), ...]


def classify_scene(
    audioset_results: list[tuple[str, float]],
    quiet_threshold: float = 0.10,
    min_evidence: float = 0.35,
    min_margin: float = 0.10,
) -> SceneResult:
    """AudioSet (label, score) 列表 → SceneResult.

    Args:
        audioset_results:  ASTEnvironmentDetector 返回的 (label, score) 对列表。
                        如果传入空列表，返回 quiet（检测到安静环境）。
        quiet_threshold: 所有 score 都低于此值则判定为 quiet。
    """
    if not audioset_results:
        return SceneResult(tag=SceneTag.QUIET, confidence=1.0, raw_matches=[])

    max_score = max(s for _, s in audioset_results)
    if max_score < quiet_threshold:
        return SceneResult(tag=SceneTag.QUIET, confidence=1.0 - max_score, raw_matches=[])

    # 对每个 scene 累加匹配标签的得分
    scene_scores: dict[SceneTag, float] = {}
    matches_by_scene: dict[SceneTag, list[tuple[str, float]]] = {}

    for label, score in audioset_results:
        label_lower = label.lower()
        matched_scene: SceneTag | None = None

        # 关键词匹配（包含匹配，不要求完全相等）
        for kw, scene in _LABEL_TO_SCENE.items():
            if kw in label_lower or label_lower in kw:
                matched_scene = scene
                break

        if matched_scene is not None:
            scene_scores[matched_scene] = scene_scores.get(matched_scene, 0.0) + score
            matches_by_scene.setdefault(matched_scene, []).append((label, score))

    if not scene_scores:
        return SceneResult(tag=SceneTag.UNKNOWN, confidence=0.0, raw_matches=audioset_results)

    ranked = sorted(scene_scores, key=scene_scores.__getitem__, reverse=True)
    best_scene = ranked[0]
    best_evidence = scene_scores[best_scene]
    runner_up = scene_scores[ranked[1]] if len(ranked) > 1 else 0.0

    # 原实现只以“已映射标签之间的占比”报告 confidence。因此一个随机的
    # 0.15 标签若恰好是唯一匹配，就会变成 1.00 置信度。语音主导、背景很弱
    # 的片段正好会大量触发这个问题。这里把绝对证据和类别间隔都纳入判断：
    # 证据不足/相互冲突时明确返回 unknown，而不是编造一个环境场景。
    if best_evidence < min_evidence or best_evidence - runner_up < min_margin:
        return SceneResult(tag=SceneTag.UNKNOWN, confidence=round(best_evidence, 3), raw_matches=audioset_results)

    total = sum(scene_scores.values())
    dominance = best_evidence / total if total > 0 else 0.0
    confidence = best_evidence * dominance

    return SceneResult(
        tag=best_scene,
        confidence=round(confidence, 3),
        raw_matches=matches_by_scene.get(best_scene, []),
    )


# ── 响应风格建议 ──────────────────────────────────────────────────────────────

SCENE_RESPONSE_STYLE: dict[SceneTag, str] = {
    SceneTag.OFFICE:  "用户当前在办公室，请给出简洁专业的回复，不超过3句。",
    SceneTag.OUTDOOR: "用户当前在户外，请给出简短易记的回复，1-2句即可。",
    SceneTag.TRANSIT: "用户当前在通勤途中，请极简回复，一句话为佳。",
    SceneTag.CAFE:    "用户当前在咖啡厅，环境较嘈杂，请简短回复。",
    SceneTag.MEETING: "用户当前在会议中，请极简回复或询问是否稍后再说。",
    SceneTag.HOME:    "用户当前在家中，环境安静，可以适当展开回复。",
    SceneTag.QUIET:   "环境安静，可以正常展开回复。",
    SceneTag.UNKNOWN: "",
}


def scene_to_response_directive(scene: SceneTag) -> str:
    """返回注入 rb_directive 的场景响应风格提示。"""
    return SCENE_RESPONSE_STYLE.get(scene, "")


# ── 情景绑定记忆：从查询文本反推场景（audiomem 2.1） ────────────────────────────

# 用户提问里提到的场景关键词 → SceneTag。Ingest 时每条记忆会被打上
# scene:<tag>（粗粒度，见 core.py Ingest 里 upsert_memory_tags 的调用），
# 这里只需要反推出同样粗粒度的 SceneTag 去匹配，不需要 scene_trigger.py
# 那种细分到具体交通工具的 required_label。
_QUERY_SCENE_KEYWORDS: dict[SceneTag, list[str]] = {
    SceneTag.TRANSIT: [
        "公交", "地铁", "火车", "飞机", "通勤", "路上", "车上", "坐车",
        "出行", "上班路上", "下班路上",
    ],
    SceneTag.OFFICE: ["办公室", "公司", "上班", "工位", "办公"],
    SceneTag.HOME:   ["家里", "在家", "回家", "家中"],
    SceneTag.CAFE:   ["咖啡厅", "咖啡馆", "星巴克", "café", "cafe"],
    SceneTag.MEETING: ["会议", "开会", "会议室"],
    SceneTag.OUTDOOR: ["户外", "外面", "散步", "出门", "外出"],
    SceneTag.QUIET:  ["安静", "独处"],
}


def infer_scene_from_text(text: str) -> SceneTag | None:
    """从用户查询语句里反推场景意图，用于按场景缩窄检索。

    例如 "我在公交上跟你说的那件事" → SceneTag.TRANSIT，Search() 拿它当
    scene_filter 用，去匹配 Ingest 时打上的 scene:<tag> 标签。只做关键词
    匹配，没命中返回 None（不缩窄，走原来的全量检索）。
    """
    for scene, keywords in _QUERY_SCENE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return scene
    return None
