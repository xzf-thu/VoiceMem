"""voicemem.utils —— 支撑左右脑的能力/工具层。

- ``audio/``    音频原生感知：ASR、说话人声纹、情绪 VAD、声学环境/场景。
- ``common/``   跨切面工具：会话追踪、turn_id、语音接入适配、图公共小工具、成本日志、配置。
- ``fusion/``   左右脑输出的融合/回复编排（独立于主引擎的一套上层编排）。

这些是「怎么感知、怎么协调」的工具，不是记忆本身——记忆的左/右脑在
``voicemem.leftbrain`` / ``voicemem.rightbrain``。子模块按需惰性 import（见顶层
``voicemem/__init__.py`` 的 PEP 562 映射），``import voicemem`` 不会拉起 torch/sherpa。
"""
