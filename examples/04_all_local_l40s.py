"""全开源组件，一张 L40S 就跑得动：ASR / VAD / 记忆向量都在本地，LLM 走 vLLM，TTS 本地。

    # 另开一个终端起 vLLM（48G 卡，留一半显存给 TTS 和感知模型）
    vllm serve Qwen/Qwen3-8B --port 8000 --gpu-memory-utilization 0.5

    pip install sounddevice voxcpm pywebrtc-audio
    bash scripts/download_models.sh          # FunASR / silero / E5
    python examples/04_all_local_l40s.py

全程不出机器：转写是 FunASR，判说完是 silero，记忆向量和 slot 分类是本地 E5，
抽事实和回复都打给本机那个 vLLM，出声是 VoxCPM2。回声消除和打断在 _audio.py。
"""
import asyncio

from _audio import AudioIO, SR

from voicemem import VoiceMem
from voicemem.tts import speak_stream, tts_stream

# 本机 vLLM 的 OpenAI 兼容口。api_key 随便填一个，vLLM 不校验。
VLLM = {"model": "Qwen/Qwen3-8B", "api_key": "EMPTY",
        "base_url": "http://127.0.0.1:8000/v1"}

# TTS 也走本地。VoxCPM2 是 2B、音质好，但首帧要 0.5s 起，而且跟 vLLM 抢同一张卡；
# 要低延迟就换 piper（provider 写 "local"，几十 M，首帧几十 ms，
# 用 VOICEMEM_TTS_MODEL 指向 voice 的 .onnx）。
TTS = {"tts": {"provider": "voxcpm"}}

# embedding 的 provider 有三类来源，一份配置管所有通道（记忆向量、slot 锚点、
# 实体去重、右脑判断表）：
#   local        本地 E5（multilingual-e5-small）。0 网络、约 10ms——实时语音只能
#                用这一类，投机预取那点时间预算里发不起 HTTP。
#   openai       OpenAI 或任何兼容端点。认 base_url，所以 TEI / vLLM 也走这个：
#                  {"provider": "openai", "config": {"model": "BAAI/bge-m3",
#                   "base_url": "http://127.0.0.1:8080/v1", "api_key": "EMPTY"}}
#   其余名字     转给 mem0 的 EmbedderFactory：ollama / huggingface / gemini /
#                aws_bedrock / azure_openai / vertexai / together / lmstudio /
#                fastembed / langchain。config 原样透传给 mem0：
#                  {"provider": "ollama", "config": {"model": "nomic-embed-text",
#                   "ollama_base_url": "http://127.0.0.1:11434"}}
#
# 换 embedder 要注意两件事：
#   · 选跟你语言匹配的模型。all-MiniLM-L6-v2 那种纯英文模型做中文检索**不报错、
#     只是全错**（实测"我在哪读书"第一条返回"对坚果过敏"），最难查的一类问题。
#   · 库里的老向量维度会对不上，被跳过时有警告，右脑检索和实体去重在重新 embed
#     之前是空的：python3 tools/reembed.py <space> --apply
vm = VoiceMem.from_config({
    "mode": "normal",
    "embedding": {"provider": "local"},                # 记忆向量：本地 E5
    "slots":     {"provider": "local"},                # slot 分类：本地 E5，0 LLM
    "llm":   {"provider": "openai", "config": VLLM},   # 写入侧抽事实 → vLLM
    "reply": {"provider": "openai", "config": VLLM},   # 回复 → vLLM
})


async def warmup(stream):
    """模型全是懒加载的：不预热的话第一轮要现加载 E5 / FunASR / silero / 感知那套
    / VoxCPM，用户对着 [ready] 说完第一句要等二十几秒。"""
    await asyncio.to_thread(vm.warmup)                   # E5 + FunASR + silero + 感知
    await asyncio.to_thread(vm.search, "预热")           # 向量库
    await stream.feed(b"\x00" * 320)
    async for _ in tts_stream("你好。", TTS):             # VoxCPM 权重
        break


async def main():
    loop = asyncio.get_running_loop()
    stream = vm.stream(src_rate=SR)

    stop = asyncio.Event()                               # 用户插话了 → 这一轮到此为止
    audio = AudioIO(loop, on_barge_in=stop.set)

    print("[warmup] 正在加载模型…", flush=True)
    await warmup(stream)

    audio.start()
    print("[ready] 说话吧", flush=True)

    try:
        while True:
            st = await stream.feed(await audio.mic.get())
            if st.state != "turn_over":
                continue

            print(f"\n你：{st.transcript}\n助手：", end="", flush=True)

            # 记忆在你说话的时候就预取好了，这里直接回复；边生成边合成，不等全文。
            stop.clear()
            audio.assistant_started()
            async for pcm in speak_stream(vm.reply_stream(st), TTS,
                                          on_delta=lambda d: print(d, end="", flush=True)):
                if stop.is_set():
                    print("\n[打断]", flush=True)
                    break
                audio.play(pcm)
            audio.assistant_done()

            # 存记忆是秒级的，丢线程里——摆在这条循环上会挡住下一轮读麦克风。
            # 助手说了什么 reply_stream 已经登记过（被打断时记的是用户真听到的那半句）。
            asyncio.create_task(asyncio.to_thread(vm.ingest, st.transcript, async_facts=True))
    finally:
        audio.close()


asyncio.run(main())
