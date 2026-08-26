"""存和查：音频进 → 双脑；文本进 → 只有左脑。

    export OPENAI_API_KEY=sk-...
    python examples/01_memory.py

embedding 和 slots 都走本地 E5，跟 web demo 一个配置：检索这条路 0 LLM、0 网络，
本体 ~10ms。用默认的 OpenAI embedding 也能跑，但每次检索要发一次 HTTP，
说话时那 0–500ms 的投机预取就来不及了（README 里 134ms 说的就是本地这套）。
"""
import os
from pathlib import Path

from voicemem import VoiceMem

# 相对脚本位置，不是相对 cwd——从哪个目录跑都找得到
AUDIO = str(Path(__file__).resolve().parent.parent / "assets/input.wav")

LOCAL = {
    "embedding": {"provider": "local"},
    "slots": {"provider": "local"},
    "api_key": os.environ["OPENAI_API_KEY"],   # 只用于写入侧抽事实
    # 单独一个库。**向量维度不同的库不能混用**——本地 E5 是 384 维、OpenAI 是
    # 1536 维，指同一个目录会直接报 shapes (n,384) and (1536,) not aligned。
    # 不写这行就会跟默认库（多半是 OpenAI 维度建的）撞上。
    "memory_root": str(Path(__file__).resolve().parent / "example_memory"),
}
# 注意 top_k 不能写进 from_config —— 它不在认的键里，会被静默丢掉。
# 取几条在 search() 上传。

vm = VoiceMem.from_config({**LOCAL, "mode": "normal"})

# 本地模型是懒加载的：不预热的话第一次 ingest 要多等二十几秒（E5 / FunASR /
# 感知那套全在那时候加载）。web demo 一直这么做，这里也一样。
vm.warmup(verbose=True)

# 存：音频文件
# 内部跑 ASR / 声纹 / 场景 / 情绪感知 / Embedding 抽取
vm.ingest(audio=AUDIO)  # 我是素食主义者，对坚果过敏。

result = vm.search("我的饮食禁忌是什么？", top_k=5)

print(result.result_leftbrain, result.result_rightbrain)


# 存：左脑信息文本（无情感）
vm = VoiceMem.from_config({**LOCAL, "mode": "leftbrain_only"})

vm.ingest("我是素食主义者，对坚果过敏。")

result = vm.search("我的饮食禁忌是什么？", top_k=5)

print(result.result_leftbrain, result.result_rightbrain)
