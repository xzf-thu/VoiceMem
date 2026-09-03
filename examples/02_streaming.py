"""流式接口：喂音频块，说完一轮就拿到这一轮的全部感知结果。

    python examples/02_streaming.py speech.wav
"""
import asyncio
import os
import sys
from pathlib import Path
from pprint import pprint

import numpy as np
import soundfile as sf

from voicemem import VoiceMem

# 本地 E5：检索 0 网络，投机预取才来得及（跟 web demo 同一套配置）
vm = VoiceMem.from_config({
    "mode": "normal",
    "embedding": {"provider": "local"},
    "slots": {"provider": "local"},
    "api_key": os.environ["OPENAI_API_KEY"],   # 只用于写入侧抽事实
    # 单独一个库：本地 E5 是 384 维，跟默认库（OpenAI 1536 维）混用会直接报
    # shapes (n,384) and (1536,) not aligned
    "memory_root": str(Path(__file__).resolve().parent / "example_memory"),
})
WAV = sys.argv[1] if len(sys.argv) > 1 else str(
    Path(__file__).resolve().parent.parent / "assets/speech.wav")

# 本地模型懒加载：不预热的话第一块音频要等二十几秒的模型加载
vm.warmup(verbose=True)


async def main():
    audio, sr = sf.read(WAV, dtype="float32")
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

    stream = vm.stream(
        src_rate=sr,
        vad_threshold=0.5,
        on_partial=lambda t: print(f"\r[partial] {t}", end="", flush=True),
    )

    step = int(sr * .032)

    for i in range(0, len(pcm), step):
        st = await stream.feed(pcm[i:i + step].tobytes())
        print(f"\n[state] {st.state}")

        FIELDS = [
            "result_leftbrain",
            "result_rightbrain",
            "speaker_id",
            "speaker_voiceprint",
            "emotion",
            "transcript",
            "entity",
            "slots",
            "text_embedding",
        ]

        if st.state == "turn_over":
            pprint({key: getattr(st, key) for key in FIELDS})


asyncio.run(main())
