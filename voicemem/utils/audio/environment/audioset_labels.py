"""Shared AudioSet label constants used by the environmental-sound detectors.

Both the AST detector (``environment_detector_ast.py``) and the CLAP detector
key off the same AudioSet ontology, so the speech-class index set and the
music/abnormal-sound keyword lists live here instead of being duplicated or
imported from one detector module into another.
"""
from __future__ import annotations

# AudioSet label indices for speech/voice classes — excluded from environment output
_SPEECH_LABEL_INDICES = {
    0,    # Speech
    1,    # Male speech, man speaking
    2,    # Female speech, woman speaking
    3,    # Child speech, kid speaking
    4,    # Conversation
    5,    # Narration, monologue
    6,    # Babbling
    7,    # Speech synthesizer
    8,    # Shout
    9,    # Bellow
    10,   # Whoop
    11,   # Yell
    12,   # Children shouting
    13,   # Screaming
    14,   # Whispering
    15,   # Laughter
    16,   # Baby laughter
    17,   # Giggling
    18,   # Snicker
    19,   # Belly laugh
    20,   # Chuckle, chortle
    21,   # Crying, sobbing
    22,   # Baby cry, infant cry
    23,   # Whimper
    24,   # Wail, moan
    25,   # Sigh
    26,   # Singing
    27,   # Choir
    28,   # Yodeling
    29,   # Chant
    30,   # Mantra
    31,   # Male singing
    32,   # Female singing
    33,   # Child singing
    34,   # Synthetic singing
    35,   # Rapping
    36,   # Humming
    37,   # Groan
    38,   # Grunt
    39,   # Whistling
    40,   # Breathing
    41,   # Wheeze
    42,   # Snoring
    43,   # Gasp
    44,   # Pant
    45,   # Snort
    46,   # Cough
    47,   # Throat clearing
    48,   # Sneeze
    49,   # Sniff
}

# 音乐/哼唱相关标签关键词（小写）——上面 _SPEECH_LABEL_INDICES 会把 Singing/
# Humming/Whistling 当"类语音"过滤掉（避免用户自己说话被误当成环境音），
# 但背景音乐/哼唱识别记忆（audiomem 2.5）恰恰需要这些标签，所以单独走一遍
# 关键词匹配，不受场景标签那层过滤影响。
_MUSIC_KEYWORDS = [
    "music", "singing", "humming", "whistling", "song", "singer",
    "musical instrument", "chant", "yodeling",
]

# 异常环境音关键词（小写）——破碎声/警报/尖叫（audiomem 2.6）。同样绕开
# _SPEECH_LABEL_INDICES 那层过滤（Screaming/Yell 本来会被当"类语音"丢弃）。
_ABNORMAL_KEYWORDS = [
    "glass", "shatter", "smash", "crash", "explosion", "gunshot", "gunfire",
    "alarm", "siren", "scream", "yell",
]
