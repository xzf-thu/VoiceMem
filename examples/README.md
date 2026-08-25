# examples

六个能直接跑的例子，从「当记忆库用」到「接 Realtime 语音模型」。

```bash
pip install voicemem
export OPENAI_API_KEY=sk-...
```

| | 干什么 | 额外要什么 |
|---|---|---|
| [`01_memory.py`](01_memory.py) | 存和查 —— 最小用法 | 音频那半段要 `bash scripts/download_models.sh` |
| [`02_streaming.py`](02_streaming.py) | 流式接口：喂音频块，看每一轮算出了什么 | 同上 |
| [`03_simple_agent_with_voicemem_memory.py`](03_simple_agent_with_voicemem_memory.py) | 完整语音 agent：边听边取记忆、说话时能被打断 | 一个麦克风 |
| [`04_all_local_l40s.py`](04_all_local_l40s.py) | 全开源组件，一张 L40S 流式跑起来，全程不出机器 | 一个本机 vLLM + `pip install sounddevice voxcpm pywebrtc-audio` |
| [`05_realtime_gpt_qwen.py`](05_realtime_gpt_qwen.py) | 接 gpt-realtime / qwen-omni-realtime，记忆随 response 注入 | `pip install sounddevice pywebrtc-audio "websockets>=14"` |
| [`06_mic_memory.py`](06_mic_memory.py) | **只听不答**：麦克风 → 转写 → 检索记忆。没有 LLM、没有 TTS | 一个麦克风 |

## 01 · 存和查

```bash
python examples/01_memory.py
```

两种输入，区别只在 `mode`：

```python
VoiceMem(mode="normal")           # 音频 → 双脑（ASR / 声纹 / 场景 / 情绪都跑）
VoiceMem(mode="leftbrain_only")   # 文本 → 只有左脑（事实），不做情绪归因
```

查出来的东西分两半：`result.result_leftbrain` 是事实，`result.result_rightbrain`
是画像和情绪。

## 02 · 流式接口

```bash
python examples/02_streaming.py speech.wav
```

按块喂音频，每块回一个状态；VAD 判定说完时 `state` 变成 `turn_over`，这时候
`memory_context` 早就算好了 —— 检索是在你还在说的时候后台跑完的，不占回复前面
那段时间。

`turn_over` 那一刻能拿到的全部字段：

```
result_leftbrain / result_rightbrain    这一轮检索到的记忆
speaker_id / speaker_voiceprint         谁在说
emotion / transcript                    情绪 / 转写
entity / schema / text_embedding        抽出的实体、槽位、向量
```

## 03 · 完整语音 agent

```bash
python examples/03_simple_agent_with_voicemem_memory.py
```

麦克风 → voicemem 边听边预取记忆 → OpenAI 带着记忆回答 → OpenAI TTS 出声，
**你一开口它就闭嘴**。想看 web 版（带脑图和记忆面板）用 `python web/run.py`。

## 04 · 全开源，一张 L40S

```bash
vllm serve Qwen/Qwen3-8B --port 8000 --gpu-memory-utilization 0.5   # 另开一个终端
python examples/04_all_local_l40s.py
```

一条链路上没有任何 api：转写 FunASR、判说完 silero、记忆向量和 slot 分类本地 E5、
抽事实和回复打给本机 vLLM、出声 VoxCPM2。换模型只改 `VLLM` 那个 dict —— vLLM 是
OpenAI 兼容口，所以 `llm` 和 `reply` 两段填上 `base_url` 就完事。

48G 的卡大概这么分：vLLM 给 0.5（Qwen3-8B bf16 ~16G + KV），VoxCPM2 ~5G，
E5 / FunASR / silero 加起来 ~2G。显存紧就把 TTS 换成 piper（`{"tts": {"provider": "local"}}`，
几十 M）。

首帧延迟看 TTS：VoxCPM 是音质档（0.5s 起，还跟 vLLM 抢卡），piper 是低延迟档
（几十 ms）。模型全是懒加载的，例子里在 `[ready]` 之前先把 E5 / FunASR / silero /
VoxCPM 各跑一次预热 —— 不然第一句要现加载，等好几秒。

回声消除和打断在 [`_audio.py`](_audio.py)（04 / 05 共用），见下面那节。

## 05 · Realtime：gpt / qwen

```bash
OPENAI_API_KEY=sk-...    python examples/05_realtime_gpt_qwen.py gpt
DASHSCOPE_API_KEY=sk-... python examples/05_realtime_gpt_qwen.py qwen
```

麦克风每一块音频走两条路：一条 `input_audio_buffer.append` 进 realtime 出原生语音，
一条 `stream.feed()` 进 voicemem 预取记忆。本地 VAD 判完一轮时记忆已经现成，跟着
`response.create` 一起发过去。

**记忆必须走 per-response 的 `instructions`，不能用 `session.update`** —— 后者是会话级
设置，实测这一轮模型读不到：问「我的猫叫什么」，库里明明检索到「叫墨墨」，它还答
「你刚提过但我没听清」。

服务端 VAD 只借来做打断（`create_response=False` + `interrupt_response=True`），
什么时候回复仍由本地决定 —— 否则它抢着生成的那个 response 里没有我们注入的记忆。

两家的事件名是一样的，只有 `session` 那一段结构不同（gpt 的 `turn_detection` 在
`session.audio.input` 下，写成顶层会被静默拒绝；qwen 是扁的）。输入采样率也不同，
gpt 24k、qwen 16k，例子里统一录 24k 再降。

三处是延迟上必须这么写的，不是随手：

* **播放要过一层缓冲**。realtime 推音频比实时播放快得多，在事件泵里直接
  `out.write()` 会阻塞——而事件泵是 websocket 的唯一消费者，一卡就不读了，TCP
  反压上去，`speech_started` / `response.done` 全被推迟到缓冲里的音频播完为止。
* **打断要清本地缓冲**。`interrupt_response=True` 只让服务端停止生成，本地已经
  收到的那几秒照样播完，听感是「打断没用」。
* **上行音频和本地 ASR 各走一条协程**。`stream.feed()` 里的 ASR 是在事件循环上跑
  的，串在一起的话它抖一下，发给 realtime 的上行音频也跟着抖。

存记忆（`vm.ingest`）是秒级的同步调用，两个例子都丢进线程 —— 摆在事件泵或主循环
上会把整条会话卡住。

打断有两条路，互为备份：服务端 VAD 的 `interrupt_response`，和本地 AEC 的
`speech_probability`。谁先到算谁的，慢的那个会撞上 `response_cancel_not_active`，
属于预期内，例子里忽略掉了。

## 06 · 只听不答

```bash
python examples/06_mic_memory.py
```

02 是从文件喂音频，这个是从麦克风。整条链路上没有生成模型——说话、转写、检索、
写入，四步都看得见：

```
[听] 我对花生过敏
[查] 说完这一刻已经检索好的记忆：
       左脑  用户是素食主义者
[存] 已写入，下次可被检索
```

`[查]` 那一步的意义是它**不占说完之后的时间**——检索是在你还在说的时候后台跑完的
（0–500ms 投机预取）。想验证它真的记住了：说「我对花生过敏」，隔几句再问
「我不能吃什么」。

## `_audio.py` · 回声消除

外放的时候麦克风录到的是「你的声音 + 喇叭里助手的声音」。**AEC**（回声消除）拿
喇叭正在放的那份信号作参考，从麦克风信号里减掉它，剩下的才是真正的人声。不做
这一步会坏两件事：转写把助手的话当成你说的存进记忆；VAD 一直以为有人在说话，
助手一开口就把自己掐了。

所以播放必须和录音走**同一个** `sd.Stream` —— 只有在同一个回调里才拿得到跟这一帧
麦克风严格对齐的参考信号。整条设备跑 16k：WebRTC 的 APM 只吃 8/16/32/48k，
而 TTS 和 realtime 吐的是 24k，`play()` 里降一次。

`AudioIO` 顺带把打断也做了：助手说话期间，`speech_probability` 连续 0.5s 过阈值就
清播放缓冲并回调。刚开口那 0.6s 是宽限期 —— 那时候麦克风里几乎只有助手自己的
声音，AEC 还没收敛，不设宽限期它一出声就把自己掐了。

浏览器不用管这些，`getUserMedia` 自带 AEC（`web/run.py` 就靠它）。03 里内联了一份
同样的东西，那个例子刻意保持单文件自足。

## 为什么都用 `from_config` 而不是 `VoiceMem(openai_key=...)`

这些例子的 embedding 和 slots 都走本地 E5：

```python
vm = VoiceMem.from_config({
    "mode": "normal",
    "embedding": {"provider": "local"},   # 记忆向量：本地，0 网络
    "slots":     {"provider": "local"},   # 槽位分类：本地，0 LLM
    "api_key":   os.environ["OPENAI_API_KEY"],   # 只在写入侧抽事实时用
})
```

这不是可选的优化 —— **投机预取那 0–500ms 预算里不能走网络**。你说话的时候检索
就在后台跑，说完时记忆得是现成的；embedding 要是每次发一个 HTTP 去 OpenAI，
光往返就吃掉整个预算。README 里的 134ms 说的就是这套配置，实测 search 本体 ~10ms。

`VoiceMem(openai_key=...)` 这种默认构造用的是 OpenAI embedding，也能跑，只是
每轮检索都要联网。

## 顺带一个坑：库不能混用

不传 `memory_root` 时所有例子和 web demo 共用同一个库（包目录下的
`results/voice_memory`）。**向量维度不同的库不能混用** —— 本地 E5 是 384 维、
OpenAI 是 1536 维，指同一个目录会直接报：

```
ValueError: shapes (25,384) and (1536,) not aligned
```

例子现在跟 demo 用同一套本地配置，所以不会撞上。要各跑各的就显式指定
`memory_root`。
