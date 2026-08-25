"""接 Realtime 语音模型：voicemem 只管记忆，音频平行喂给 gpt-realtime / qwen-omni-realtime。

    pip install sounddevice pywebrtc-audio "websockets>=14"

    OPENAI_API_KEY=sk-...    python examples/05_realtime_gpt_qwen.py gpt
    DASHSCOPE_API_KEY=sk-... python examples/05_realtime_gpt_qwen.py qwen

麦克风的每一块音频（已消过回声）走两条路：一条进 realtime 出原生语音，一条进
voicemem 做投机预取。本地 VAD 判「说完了」的那一刻记忆已经是现成的，直接跟着
response.create 发过去——所以记忆不占回复前面那段时间。

两条路各有各的消费者（uplink / 主循环），不排在一起：feed() 的 ASR 是在事件循环上
跑的，串在一起的话它抖一下，发给 realtime 的上行音频也跟着抖。

两家的事件名一样（session.update / input_audio_buffer.append / response.create /
response.*audio.delta），只有 session 那一段的结构不同，见下面两个 dict。
"""
import argparse
import asyncio
import base64
import json
import os

import numpy as np
import websockets
from _audio import AudioIO, SR

from voicemem import VoiceMem
from voicemem.utils.audio.stream_io import resample

PERSONA = ("你是语音助手。用你记得的事，但别念出来，也别用「我记得你说过」开头。"
           "句子短，一次说一两句就停。")

GPT = {
    "url": "wss://api.openai.com/v1/realtime?model=gpt-realtime",
    "key_env": "OPENAI_API_KEY",
    "in_rate": 24000,
    # turn_detection 在 session.audio.input 下，**不是顶层**——写成顶层会被静默拒绝。
    # create_response=False：什么时候回复由我们决定（等本地判完一轮、记忆预取好）；
    # interrupt_response=True：用户一开口服务端直接掐掉正在播的回复。
    "session": {"type": "realtime", "audio": {
        "input": {"turn_detection": {"type": "server_vad", "create_response": False,
                                     "interrupt_response": True}},
        "output": {"voice": "marin"}}},
    # 记忆写入侧还要个普通 chat 模型抽事实，这条路用 OpenAI 的。
    "llm": {"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY"),
            "base_url": None},
}

QWEN = {
    "url": "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3-omni-flash-realtime",
    "key_env": "DASHSCOPE_API_KEY",
    "in_rate": 16000,
    "session": {"modalities": ["text", "audio"], "voice": "Chelsie",
                "input_audio_format": "pcm16", "output_audio_format": "pcm16",
                "turn_detection": {"type": "server_vad", "create_response": False,
                                   "interrupt_response": True}},
    # 抽事实也留在阿里这边：DashScope 的 OpenAI 兼容模式。
    "llm": {"model": "qwen-plus", "api_key": os.environ.get("DASHSCOPE_API_KEY"),
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
}

# 服务端 VAD 判完一句会自己 commit 音频缓冲，我们随后那次 commit 就撞上空缓冲。
# 说得太短时它不会自动 commit，所以手动那次得留着——这条属于预期内。
# response_cancel_not_active：打断有两条路（本地 AEC 和服务端 VAD），互为备份，
# 谁先到算谁的，慢的那个扑空是正常的。
_EXPECTED_ERRORS = ("input_audio_buffer_commit_empty", "response_cancel_not_active")


def to_wire(pcm: bytes, rate: int) -> str:
    """麦克风块（16k，AEC 的原生档）→ 这一家要的采样率 → base64。"""
    if rate != SR:
        f = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
        out = resample(f, src=SR, dst=rate)
        pcm = (np.clip(out, -1, 1) * 32767).astype(np.int16).tobytes()
    return base64.b64encode(pcm).decode()


async def main(p, name):
    vm = VoiceMem.from_config({
        "mode": "normal",
        "embedding": {"provider": "local"},     # 检索 0 网络，投机预取才来得及
        "slots":     {"provider": "local"},
        "llm": {"provider": "openai", "config": p["llm"]},
    })

    loop = asyncio.get_running_loop()
    to_ws: asyncio.Queue = asyncio.Queue()      # 上行给 realtime
    to_vm: asyncio.Queue = asyncio.Queue()      # 同一份音频，给记忆预取
    stream = vm.stream(src_rate=SR)
    turn = {"text": "", "reply": "", "live": False}

    print("[warmup] 正在加载模型…", flush=True)
    await asyncio.to_thread(vm.warmup)           # E5 + FunASR + silero + 感知
    await asyncio.to_thread(vm.search, "预热")   # 向量库
    await stream.feed(b"\x00" * 320)

    key = os.environ[p["key_env"]]

    async with websockets.connect(
            p["url"], additional_headers={"Authorization": f"Bearer {key}"}) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": p["session"]}))

        def on_barge_in():
            """本地 AEC 听到人声（播放缓冲已经清了）→ 让服务端也别再生成。"""
            if not turn["live"]:
                return
            turn["live"] = False
            print("\n[打断]", flush=True)
            asyncio.create_task(ws.send(json.dumps({"type": "response.cancel"})))

        audio = AudioIO(loop, on_barge_in=on_barge_in)

        async def split():
            """一份音频，两条路。"""
            while True:
                pcm = await audio.mic.get()
                to_ws.put_nowait(pcm)
                to_vm.put_nowait(pcm)

        async def uplink():
            """音频上行独占一条协程，不跟本地 ASR 排队。"""
            while True:
                pcm = await to_ws.get()
                await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                          "audio": to_wire(pcm, p["in_rate"])}))

        async def pump():
            """事件流只能有一个消费者，收音频/文本/收尾都在这里——所以这里面
            一律不许有阻塞调用。"""
            async for raw in ws:
                ev = json.loads(raw)
                t = ev.get("type", "")

                if t.endswith("audio.delta"):                  # gpt: response.output_audio.delta
                    if turn["live"]:
                        audio.play(base64.b64decode(ev["delta"]))
                elif t.endswith("audio_transcript.delta"):
                    if turn["live"]:
                        turn["reply"] += ev["delta"]
                        print(ev["delta"], end="", flush=True)
                elif t.endswith("input_audio_buffer.speech_started"):
                    # 服务端 VAD 听到人声：它那侧已经掐了回复，本地缓冲也得清，
                    # 不然那几秒照样播完，听感是「打断没用」。
                    if turn["live"]:
                        turn["live"] = False
                        audio.stop_playing()
                elif t.endswith("response.done") or t.endswith("response.cancelled"):
                    turn["live"] = False
                    audio.assistant_done()
                    # 存记忆是秒级的，丢线程里；卡在这儿等于整条会话不响应。
                    asyncio.create_task(asyncio.to_thread(
                        vm.ingest, turn["text"], agent_reply=turn["reply"], async_facts=True))
                    turn["reply"] = ""
                elif t == "error":
                    if (ev.get("error") or {}).get("code") not in _EXPECTED_ERRORS:
                        print(f"\n[{name}] {ev.get('error')}", flush=True)

        tasks = [asyncio.create_task(f()) for f in (pump, uplink, split)]

        audio.start()
        print(f"[ready] {name}，说话吧", flush=True)

        try:
            while True:
                st = await stream.feed(await to_vm.get())
                if st.state != "turn_over":
                    continue

                turn.update(text=st.transcript, reply="", live=True)
                print(f"\n你：{st.transcript}\n助手：", end="", flush=True)

                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                # 记忆走 per-response 的 instructions，不是 session.update——后者是
                # 会话级设置，这一轮模型根本读不到（库里明明检索到了，它还说没听清）。
                await ws.send(json.dumps({"type": "response.create", "response": {
                    "instructions": f"{PERSONA}\n\n{st.memory_context}"}}))
                audio.assistant_started()
        finally:
            for t in tasks:
                t.cancel()
            audio.close()


ap = argparse.ArgumentParser()
ap.add_argument("provider", choices=["gpt", "qwen"], nargs="?", default="gpt")
args = ap.parse_args()

asyncio.run(main({"gpt": GPT, "qwen": QWEN}[args.provider], args.provider))
