"""只听不答：麦克风 → 转写 → 记忆检索。没有 LLM 回复，没有 TTS。

    export OPENAI_API_KEY=sk-...
    python examples/06_mic_memory.py

说一句话，你会看到三件事按顺序发生：

    [听]  实时转写，你还在说的时候就在出字
    [查]  你说完的那一刻，相关记忆**已经在手上了**——检索是在你说话期间
          后台跑完的（0–500ms 投机预取），不占你说完之后的时间
    [存]  这一轮写进记忆库，下次就能被查到

想看它真的记住了：说「我对花生过敏」，然后隔几句再问「我不能吃什么」。

Ctrl-C 退出。
"""
import asyncio
import os
import queue
import sys
from pathlib import Path

import sounddevice as sd

from voicemem import VoiceMem

SR = 16000          # 麦克风采样率
BLOCK = 512         # 每块样本数：32ms @16k，跟 VAD 的帧长对齐

vm = VoiceMem.from_config({
    "mode": "normal",
    "embedding": {"provider": "local"},   # 记忆向量：本地，0 网络
    "slots": {"provider": "local"},       # 槽位分类：本地，0 LLM
    "api_key": os.environ["OPENAI_API_KEY"],   # 只在写入侧抽事实时用
    # 单独一个库：本地 E5 是 384 维，跟默认库（OpenAI 1536 维）混用会直接报
    # shapes (n,384) and (1536,) not aligned
    "memory_root": str(Path(__file__).resolve().parent / "example_memory"),
})


def show_partial(text):
    print(f"\r[听] {text}", end="", flush=True)


async def main():
    vm.warmup()

    # sounddevice 的回调跑在自己的线程里，不能直接 await。用队列过一道，
    # 让事件循环这边去取——回调里只做搬运，一点都别阻塞，否则会丢音频。
    blocks: queue.Queue = queue.Queue()

    def on_audio(indata, frames, time_info, status):
        blocks.put(bytes(indata))

    stream = vm.stream(src_rate=SR, on_partial=show_partial)

    with sd.RawInputStream(samplerate=SR, blocksize=BLOCK, dtype="int16",
                           channels=1, callback=on_audio):
        print("说话吧（Ctrl-C 退出）\n", flush=True)
        while True:
            pcm = await asyncio.to_thread(blocks.get)
            st = await stream.feed(pcm)
            if st.state != "turn_over":
                continue

            print(f"\r[听] {st.transcript}")

            left = st.result_leftbrain or []
            right = st.result_rightbrain or []
            if left or right:
                print("[查] 说完这一刻已经检索好的记忆：")
                for m in left:
                    print(f"       左脑  {m}")
                for m in right:
                    print(f"       右脑  {m}")
            else:
                print("[查] 还没有相关记忆（库是空的，多说几句就有了）")

            # 写入是秒级的，丢线程别挡住麦克风
            asyncio.create_task(asyncio.to_thread(vm.ingest, st.transcript))
            print("[存] 已写入，下次可被检索\n", flush=True)


try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("\n再见。")
    sys.exit(0)
