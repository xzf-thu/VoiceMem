# VoiceMem

<p align="center">
  <img src="assets/Voicemem_logo.webp" alt="VoiceMem Logo" width="100%">
</p>

<p align="center">
  <strong>中文</strong> | <a href="README_en.md">English</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.19833">技术报告 📖</a> /
  <a href="https://huggingface.co/zhifeixie/VoiceMem_Default_Models_Env">VoiceMem Utils 🤗</a> /
  <a href="https://huggingface.co/zhifeixie/VoiceMem_MF_Qwen3_6_35B_A3B_Qlora">VoiceMem Model Families 🤗</a> /
  <a href="https://huggingface.co/datasets/zhifeixie/VoiceMem-ChatMem400k">ChatMem-400K 🤗</a>
</p>

<p align="center">
  <a href="https://github.com/xzf-thu/Mega-ASR/raw/main/assets/wechat.jpg">
    <img src="https://img.shields.io/badge/WeChat-Join%20Group-07C160?logo=wechat&logoColor=white" alt="WeChat">
  </a>
  <a href="https://xzf-thu.github.io/Mega-ASR/">
    <img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page">
  </a>
  <a href="https://x.com/XieZhifei14110">
    <img src="https://img.shields.io/badge/X-@XieZhifei14110-black?logo=x&logoColor=white" alt="X">
  </a>
</p>

---

我们带来 **VoiceMem**，为语音模型增加最后一个组件：灵魂，让它真正越来越懂你。VoiceMem 建立在<strong>「流式双脑」</strong>架构之上，提供**精准、有情感、懂人格、低延迟且最便宜的记忆服务**。本仓库将<strong>「永久保持全部开源」</strong>。

快速理解 VoiceMem：

* **左脑：** 直接管理信息，在 Top-3 限制下维持 Mem0 的满载性能。
* **右脑：** 用长短期情绪归因管理「情商」，含交叉节点、与左脑信息联合维护。
* **低延迟：** 通过压缩信息、分层存储、流式查询（0–300 ms 投机预取），几乎不增加延迟。
* **简单实用：** 单轮查询约 300 token；架构全部解耦，全部组件（含底层记忆引擎）都可更换。

<p align="center">
  <img src="assets/teaser.webp" alt="VoiceMem Logo" width="100%">
</p>

## 🎬 Demo

> **注意：** 播放前需要先取消静音。

<div align="center">
  <video
    src="https://private-user-images.githubusercontent.com/201621992/637588589-34d46638-20db-4943-a88b-b3826c16f156.mp4"
    width="1000"
    controls>
  </video>
</div>

## 📚 目录

* [🚀 快速开始](#-快速开始)
* [🧠 VoiceMem 双脑流式架构](#-voicemem基于流式双脑架构的记忆系统)
* [🤖 VoiceMem 官方记忆模型](#-voicemem-模型系列)
* [🔌 使用 VoiceMem 定制你的语音智能体](#-使用-voicemem-定制你的语音智能体)
* [🛠️ 模型微调](#️-模型微调)
* [📊 评测代码](#-评测)
* [致谢](#致谢)
* [许可证](#许可证)

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/lang-jiaqi/Voicemem_open.git
cd Voicemem_open

# 安装记忆系统（含 ASR / 声纹 / 场景 / 情绪 / 本地 embedding 全套内置组件）
pip install voicemem

# 可选：用我们微调的 Qwen 回复模型
pip install "voicemem[slm]"
```

### 下载所需模型

```bash
pip install -U huggingface_hub

hf download zhifeixie/VoiceMem_Default_Models_Env --local-dir ./models
```

### 基础用法 <a id="interfaces"></a>

#### 作为离线记忆引擎运行

```python
from voicemem import VoiceMem

vm = VoiceMem(
    mode="normal",
    openai_key="api_xxx",
    top_k=5,
)

# 本地模型是懒加载的，先热起来，别让第一次调用去等加载
vm.warmup()

# 存：音频文件
# 内部跑 ASR / 声纹 / 场景 / 情绪感知 / Embedding 抽取
print("入库开始")
vm.ingest(audio="assets/input.wav")  # 我是素食主义者，对坚果过敏。
print("入库结束")

# 查：写入慢是因为要抽事实、打标签、建图；查询走的是纯向量检索，跟写入无关
print("检索开始")
result = vm.search("我的饮食禁忌是什么？")
print("检索结束")

print(result.result_leftbrain, result.result_rightbrain)


# 存：左脑信息文本（无情感）
vm = VoiceMem(
    mode="leftbrain_only",
    openai_key="api_xxx",
    top_k=5,
)

vm.ingest("我是素食主义者，对坚果过敏。")

result = vm.search("我的饮食禁忌是什么？")
```

#### 以流式方式运行 VoiceMem

可以把 VoiceMem 的流式接口看作一个持续处理音频的 VAD 接口。

下面这段：先显式存一条事实，再喂一段**问句**音频，看记忆是怎么在人还没说完时就查好的；最后照例走一次入库判断。

```python
import asyncio
import os
from pprint import pprint

import numpy as np
import soundfile as sf

from voicemem import VoiceMem

# 沿用上面那个 vm；单独跑这段就自己建一个
vm = VoiceMem(mode="normal", openai_key=os.environ["OPENAI_API_KEY"], top_k=5)

# 本地模型是懒加载的，先热起来，别让第一块音频去等模型加载
vm.warmup()

# 先存一条事实，等下那个问句才有东西可查
vm.ingest("我是素食主义者，对坚果过敏。")

SPEC_MIN_CHARS = 6          
searching = False


def on_partial(text):
    """边说边出字。够长了就说明后台这一刻已经开查了。"""
    global searching
    print(f"\r[partial] {text}", end="", flush=True)
    if not searching and len(text) >= SPEC_MIN_CHARS:
        searching = True
        print("\n[检索开始] 人还没说完，后台已经在查了", flush=True)


async def main():
    # 这段音频里是一个问句：「我的饮食禁忌是什么？」
    audio, sr = sf.read("assets/question.wav", dtype="float32")
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

    stream = vm.stream(src_rate=sr, vad_threshold=0.5, on_partial=on_partial)
    step = int(sr * .032)

    for i in range(0, len(pcm), step):
        st = await stream.feed(pcm[i:i + step].tobytes())
        if st.state != "turn_over":                
            continue

        # VAD 确认这一轮说完了。记忆早在说话过程中就查好了，这里直接取，不再等
        print("[检索结束]")
        print("转写  ", st.transcript)
        print("左脑  ", st.result_leftbrain)         
        print("右脑  ", st.result_rightbrain)
        pprint({k: getattr(st, k) for k in
                ["speaker_id", "speaker_voiceprint", "emotion",
                 "entity", "schema", "text_embedding"]})

        # 每一轮都要走一次入库判断
        print("[入库] LLM 正在判断这句话值不值得入库…", flush=True)
        res = vm.ingest(st.transcript)
        print(f"[入库] 抽出 {res['facts_count']} 条事实 -> {res['memory_ids']}")
       


asyncio.run(main())
```

### VoiceMem 交互式演示

演示代码在仓库里（pip 装的包只有库本身），先确认已经克隆并进入仓库目录。

```bash
python web/run.py
```

然后访问：

```text
http://localhost:8787
```

## 🧠 VoiceMem：基于流式双脑架构的记忆系统

**VoiceMem** 是一个面向实时语音智能体的记忆系统。

VoiceMem 不把所有记忆放进同一个检索数据库，而是将记忆拆分成两个互相配合的部分：

<p align="center">
  <img src="./docs/images/fig-architecture.webp" alt="VoiceMem 系统架构" width="80%">
</p>

* **左脑**通过 Schema 和 Entity 组织事实记忆，用于更加准确地检索信息。
* **右脑**通过独立节点和跨实体节点管理人格、情绪和关系信息。

<p align="center">
  <img src="./docs/images/stages.webp" alt="VoiceMem Logo" width="90%">
</p>

整个流程都是**流式**的。

在用户仍然说话时，VoiceMem 会持续完成音频分段、语音转写、记忆提取，并把结构化信息写入记忆图中。

查询时，VoiceMem 会**先路由，再排序，最后只把 Top-K 条记忆注入模型上下文**，从而在保留相关信息的同时控制上下文长度。

### 主要特性

* 🎯 **精准** — 在 **LoCoMo 上达到 91.2%**，Mem0 为 **61.68%**，并且只需要 **Top-5** 条记忆。
* ❤️ **有情感、懂人格** — 不只记住**用户说过什么**，还会记住**用户是谁、用户有什么感受**。在 **PersonaMem 上达到 69.44%**。
* 🎧 **多模态** — 可以从真实世界音频中记住**语音、说话人、声音事件、多人对话和音乐**。
* ⚡ **低延迟** — 响应时间为 **134 ms**，Mem0 为 **1,440 ms**，并支持在语音轮次内部进行流式检索。
* 💰 **低 Token 消耗** — 每次只使用 **430 个记忆 token**，Mem0 为 **6,956**，EverMemOS 为 **1,899**。

---

## 🤖 VoiceMem 模型系列

我们通过三阶段 OPD 训练流程构建 **ChatMem-400K**：

1. **Memory-world construction**
2. **SLM-validated online on-policy distillation（OPD）**
3. **Human refinement**

同一套流程在人工编辑后形成 **ChatMem-Bench**，评测语音模型是否能够在长期沉淀中形成对用户的理解。

VoiceMem 家族开源模型包括 **Qwen2.5-Omni、Qwen3-Omni 和 Step-Audio2-Mini**。这些模型可以在对话时接受并理解 VoiceMem 提供的记忆信息。

<p align="center">
  <img src="./docs/images/fig-opd.webp" alt="VoiceMem OPD 流程" width="90%">
</p>

## 🔌 使用 VoiceMem 定制你的语音智能体

你可以将 VoiceMem 接入自己的语音模型，用于构建带有长期记忆能力的实时语音智能体。

整体流程如下：

**麦克风 → VoiceMem 监听语音并提前检索相关记忆 → 你的模型读取这些记忆并生成回答**

```bash
export OPENAI_API_KEY=sk-...
# 仅在写入记忆时用于事实信息提取。
# 记忆检索完全在本地运行。

python examples/03_simple_agent_with_voicemem_memory.py
```

换成你自己的模型：把生成那一步换掉就行，记忆那半边一行都不用动。

```python
def my_reply(text, memory_context):        # 同步函数也可以，会自动丢线程
    return my_model.generate(system=memory_context, user=text)

vm = VoiceMem(reply=my_reply)
```

## 🛠️ 模型微调

VoiceMem 提供完整的微调代码，可用于训练自己的 VoiceMem Model Family Adapter。

默认训练配置与发布的 `checkpoint-3318` 使用的配置一致。

使用默认参数运行下面的命令，可以复现相同的 Adapter：

```bash
pip install ms-swift==4.5.2 bitsandbytes

python finetune/train.py --data data/train.jsonl
```

训练数据格式、GPU 显存要求，以及如何更换基础模型，请参阅 **[finetune/README.md](finetune/README.md)**。

## 📊 评测

评测流程完全开源，并且可以复现。

## Benchmarking: Fully open-source and alls reproducible.
<img src="./assets/evaluation.webp" alt="VoiceMem Logo" width="100%">

### 运行评测

只需要一条命令即可运行 Benchmark：

```bash
export OPENAI_API_KEY=sk-...

# 建议先运行仓库中自带的小型示例，
# 确认环境和配置没有问题。
# 2 个对话，5 个问题。
python evaluation/run.py \
    --dataset locomo \
    --data evaluation/examples/locomo_sample.json

# 然后运行完整数据集。
python evaluation/run.py \
    --dataset locomo \
    --data data/locomo.json
```

示例结果：

```text
LoCoMo: 10 conversations · 152 questions

Score: 139/152 = 91.4%

  multi_hop     88.2%
  temporal      85.7%
  single_hop    95.1%

Median retrieval latency: 12 ms
Median retrieved memory: 298 tokens
```

在运行完整评测之前，可以加入 `--inspect`，检查数据集是否被正确解析。

这个模式不会调用模型，因此也不会产生 API 费用：

```bash
python evaluation/run.py \
    --dataset locomo \
    --data data/locomo.json \
    --inspect
```

评测过程中，回答模型**只会收到检索得到的记忆**，不会收到原始对话历史。

如果直接把完整对话交给模型，Benchmark 测试的就会变成模型的阅读理解能力，而不是记忆系统本身的能力。

完整评测流程，以及添加新 Benchmark 的方法，请参阅 **[evaluation/README.md](evaluation/README.md)**。添加一个新的 Benchmark 只需要增加一个文件并实现两个函数。

## 致谢

我们感谢以下优秀的开源项目：

* [mem0](https://github.com/mem0ai/mem0) — 向量记忆引擎
* [FunASR](https://github.com/modelscope/FunASR) — 基于 `paraformer-zh-streaming` 的流式 ASR
* [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — Silero VAD、3D-Speaker 说话人验证，以及备用流式 ASR
* [intfloat/multilingual-e5](https://huggingface.co/intfloat/multilingual-e5-small) — 本地 Embedding 和 Slot 分类

VoiceMem 同时使用 OpenAI API 提供 Chat、TTS 和 Realtime 功能。

## 许可证

VoiceMem 基于 **Apache License 2.0** 开源。

详细信息请参阅 [LICENSE](LICENSE)。
