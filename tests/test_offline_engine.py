#!/usr/bin/env python3
"""README「作为离线记忆引擎运行」那一块，原样跑一遍。

    export OPENAI_API_KEY=sk-...
    python tests/test_offline_engine.py

跑通 = pip 装出来的 voicemem 是好的。
换音频：VOICEMEM_TEST_AUDIO=/你的/音频.wav
"""
import os
import sys
import tempfile
from pathlib import Path

# 相对脚本位置找，不是相对 cwd——不然换个目录跑就找不到音频了
REPO = Path(__file__).resolve().parent.parent
AUDIO = os.environ.get("VOICEMEM_TEST_AUDIO") or str(REPO / "assets/speech.wav")

# 用临时库，别污染已有的记忆
ROOT = tempfile.mkdtemp(prefix="voicemem_test_")

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("没设 OPENAI_API_KEY")
if not os.path.exists(AUDIO):
    sys.exit(f"找不到音频 {AUDIO}")


# ── 以下是 README 那一块 ────────────────────────────────────────────────────
from voicemem import VoiceMem

vm = VoiceMem(
    mode="normal",
    openai_key=os.environ["OPENAI_API_KEY"],
    top_k=5,
    memory_root=ROOT,
)

# 本地模型懒加载，先热起来：不预热的话第一次 ingest 要多等二十几秒
vm.warmup(verbose=True)

# 存：音频文件
# 内部跑 ASR / 声纹 / 场景 / 情绪感知 / Embedding 抽取
print("入库开始")
vm.ingest(audio=AUDIO)          # 我喜欢吃马卡龙
print("入库结束")

# 写入慢是因为要抽事实、打标签、建图；查询走纯向量检索，跟写入无关
print("检索开始")
result = vm.search("我喜欢吃什么？")
print("检索结束")

print("音频：", result.result_leftbrain, result.result_rightbrain)
audio_ok = bool(result.result_leftbrain)


# 存：左脑信息文本（无情感）
vm = VoiceMem(
    mode="leftbrain_only",
    openai_key=os.environ["OPENAI_API_KEY"],
    top_k=5,
    memory_root=ROOT,
)

vm.ingest("我是素食主义者，对坚果过敏。")

result = vm.search("我的饮食禁忌是什么？")

print("文本：", result.result_leftbrain)
text_ok = bool(result.result_leftbrain)
# ── README 那一块到此为止 ───────────────────────────────────────────────────


import shutil
shutil.rmtree(ROOT, ignore_errors=True)

if audio_ok and text_ok:
    print("\n通过")
else:
    print(f"\n失败：音频={audio_ok} 文本={text_ok}")
    sys.exit(1)
