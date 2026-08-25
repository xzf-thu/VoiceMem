# VoiceMem

<p align="center">
  <img src="assets/Voicemem_logo.png" alt="VoiceMem Logo" width="100%">
</p>

<p align="center">
  <a href="README.md">中文</a> | <strong>English</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.19833">Technical Report 📖</a> /
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

We introduce **VoiceMem**, adding the final component to voice models: a memory that helps them understand you better over time. VoiceMem is built on a <strong>streaming dual-brain</strong> architecture and provides **accurate, emotional, personality-aware, low-latency, and low-cost memory services**. This repository will <strong>remain fully open source</strong>.

A quick overview of VoiceMem:

* **Left Brain:** Directly manages factual information and maintains strong retrieval performance under a Top-3 memory limit.
* **Right Brain:** Manages emotional intelligence through short-term and long-term emotional attribution, including cross-entity nodes and joint maintenance with Left Brain information.
* **Low Latency:** Uses information compression, hierarchical storage, and streaming retrieval with 0–500 ms speculative prefetching, adding very little extra latency.
* **Simple and Practical:** Each query uses about 300 memory tokens. The architecture is fully decoupled, and every component, including the underlying memory engine, can be replaced independently.

<p align="center">
  <img src="assets/teaser.png" alt="VoiceMem Overview" width="100%">
</p>

## 🎬 Demo

> **Note:** Please unmute the video before playback.

<div align="center">
  <video
    src="https://private-user-images.githubusercontent.com/201621992/637588589-34d46638-20db-4943-a88b-b3826c16f156.mp4"
    width="1000"
    controls>
  </video>
</div>

## 📚 Overview

* [🚀 Quick Start](#-quick-start)
* [🧠 VoiceMem Dual-Brain Streaming Architecture](#-voicemem-memory-with-a-streaming-dual-brain-architecture)
* [🤖 VoiceMem Model Families](#-voicemem-model-families)
* [🔌 Customize Your Voice Agent with VoiceMem](#-customize-your-voice-agent-with-voicemem)
* [🛠️ Finetuning](#️-finetuning)
* [📊 Evaluation](#-evaluation)
* [Acknowledgements](#acknowledgements)
* [License](#license)

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/lang-jiaqi/Voicemem_open.git
cd Voicemem_open

# Install the memory system (bundles ASR / speaker ID / scene / emotion / local embedding)
pip install voicemem

# Optional: run our fine-tuned Qwen reply model
pip install "voicemem[slm]"
```

### Required Model Download

```bash
pip install -U huggingface_hub

hf download zhifeixie/VoiceMem_Default_Models_Env --local-dir ./models
```

### Basic Usage <a id="interfaces"></a>

#### Run as an Offline Memory Engine

```python
from voicemem import VoiceMem

vm = VoiceMem(
    mode="normal",
    openai_key="api_xxx",
    top_k=5,
)

# Local models load lazily -- warm them up so the first call doesn't pay for it.
vm.warmup()

# Store an audio file.
# VoiceMem internally runs ASR, speaker recognition,
# scene and emotion analysis, and embedding extraction.
print("ingest start")
vm.ingest(audio="assets/input.wav")  # I am vegetarian and allergic to nuts.
print("ingest done")

# Writing is slow because it extracts facts, tags them and builds the graph.
# Reading is a pure vector lookup -- independent of write cost.
print("search start")
result = vm.search("What are my dietary restrictions?")
print("search done")

print(result.result_leftbrain, result.result_rightbrain)


# Store factual text directly without emotional information.
vm = VoiceMem(
    mode="leftbrain_only",
    openai_key="api_xxx",
    top_k=5,
)

vm.ingest("I am vegetarian and allergic to nuts.")

result = vm.search("What are my dietary restrictions?")
```

#### Run VoiceMem in Streaming Mode

The streaming interface continuously processes incoming audio and can be used in a VAD-style audio pipeline.

The example below stores one fact explicitly, then feeds a **question** as audio to show how the memory is already retrieved before the speaker finishes. It ends with the usual ingest step — a question carries no new fact, so the LLM decides not to store anything, but the call itself must still happen.

```python
import asyncio
import os
from pprint import pprint

import numpy as np
import soundfile as sf

from voicemem import VoiceMem

# Reuses the vm above; building one here so the block runs standalone
vm = VoiceMem(mode="normal", openai_key=os.environ["OPENAI_API_KEY"], top_k=5)

# Local models load lazily -- warm them up so the first audio chunk doesn't wait
vm.warmup()

# Store one fact first, so the question below has something to find
vm.ingest("I am vegetarian and allergic to nuts.")

SPEC_MIN_CHARS = 6          # partial length that triggers speculative prefetch (vm.stream default)
searching = False


def on_partial(text):
    """Partial transcripts as they arrive. Long enough = the search already started."""
    global searching
    print(f"\r[partial] {text}", end="", flush=True)
    if not searching and len(text) >= SPEC_MIN_CHARS:
        searching = True
        print("\n[search start] speaker isn't done yet, retrieval already running", flush=True)


async def main():
    # This audio is a question: "What are my dietary restrictions?"
    audio, sr = sf.read("assets/question.wav", dtype="float32")
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

    stream = vm.stream(src_rate=sr, vad_threshold=0.5, on_partial=on_partial)
    step = int(sr * .032)

    for i in range(0, len(pcm), step):
        st = await stream.feed(pcm[i:i + step].tobytes())
        if st.state != "turn_over":                 # "<speak>" / "<silence>"
            continue

        # VAD confirmed end of turn. Memory was fetched while the user spoke -- just read it
        print("[search end]")
        print("transcript  ", st.transcript)
        print("left brain  ", st.result_leftbrain)   # -> the vegetarian / nut-allergy fact
        print("right brain ", st.result_rightbrain)
        pprint({k: getattr(st, k) for k in
                ["speaker_id", "speaker_voiceprint", "emotion",
                 "entity", "schema", "text_embedding"]})

        # Every turn runs the ingest decision -- don't skip it. Whether an utterance
        # is worth storing is the LLM's call, not the caller's guess
        print("[ingest] LLM deciding whether this is worth storing...", flush=True)
        res = vm.ingest(st.transcript)
        print(f"[ingest] extracted {res['facts_count']} facts -> {res['memory_ids']}")
        # A question holds no new information about the user -> 0 facts, empty ids. Expected


asyncio.run(main())
```

### Interactive Demo with VoiceMem

The demo lives in the repo (the pip package ships the library only) — clone it and run from the repo root.

```bash
python web/run.py
```

Then open:

```text
http://localhost:8787
```

## 🧠 VoiceMem: Memory with a Streaming Dual-Brain Architecture

**VoiceMem** is a memory system built for real-time voice agents.

Instead of storing every type of memory in a single retrieval database, VoiceMem separates memory into two complementary parts:

<p align="center">
  <img src="./docs/images/fig-architecture.webp" alt="VoiceMem Architecture" width="80%">
</p>

* **Left Brain** organizes factual memory using schemas and entities for more accurate retrieval.
* **Right Brain** manages personality, emotion, and relationships using independent and cross-entity memory nodes.

<p align="center">
  <img src="./docs/images/stages.png" alt="VoiceMem Processing Pipeline" width="90%">
</p>

The entire pipeline is **streaming**.

While the user is still speaking, VoiceMem continuously segments audio, transcribes speech, extracts useful memories, and writes structured information into the memory graph.

At query time, VoiceMem **routes first, ranks second, and injects only the Top-K memories into the model context**. This keeps the context small while preserving the most relevant information.

### Key Features

* 🎯 **Accurate** — Reaches **91.2% on LoCoMo**, compared with **61.68% for Mem0**, using only **Top-5** memories.
* ❤️ **Emotional & Personal** — Remembers not only **what the user said**, but also **who the user is and how they feel**. Reaches **69.44% on PersonaMem**.
* 🎧 **Multimodal** — Remembers **speech, speakers, sound events, multi-speaker conversations, and music** from real-world audio.
* ⚡ **Fast** — Responds in **134 ms**, compared with **1,440 ms for Mem0**, with streaming retrieval inside the voice turn.
* 💰 **Low Token Usage** — Uses only **430 memory tokens**, compared with **6,956 for Mem0** and **1,899 for EverMemOS**.

---

## 🤖 VoiceMem Model Families

We build **ChatMem-400K** through a three-stage OPD training pipeline:

1. **Memory-world construction**
2. **SLM-validated online on-policy distillation (OPD)**
3. **Human refinement**

After human editing, the same pipeline produces **ChatMem-Bench**, which evaluates whether a voice model can build a long-term understanding of the user over time.

The open-source VoiceMem model family includes **Qwen2.5-Omni, Qwen3-Omni, and Step-Audio2-Mini**. These models can receive and understand memory information provided by VoiceMem during conversations.

<p align="center">
  <img src="./docs/images/fig-opd.webp" alt="VoiceMem OPD Pipeline" width="90%">
</p>

## 🔌 Customize Your Voice Agent with VoiceMem

You can integrate VoiceMem with your own voice model to build a real-time voice agent with long-term memory.

The basic flow is:

**microphone → VoiceMem listens and prefetches relevant memories → your model reads those memories and generates a response**

```bash
export OPENAI_API_KEY=sk-...
# Only used for fact extraction when writing memories.
# Memory retrieval runs entirely locally.

python examples/03_simple_agent_with_voicemem_memory.py
```

To use your own model, replace the generation step — the memory half stays as is:

```python
def my_reply(text, memory_context):        # a sync function is fine, it runs off-thread
    return my_model.generate(system=memory_context, user=text)

vm = VoiceMem(reply=my_reply)
```

## 🛠️ Finetuning

VoiceMem provides the complete finetuning pipeline for training your own VoiceMem Model Family adapter.

The default training configuration matches the one used for the released `checkpoint-3318`.

Running the following command with the default settings reproduces the same adapter:

```bash
pip install ms-swift==4.5.2 bitsandbytes

python finetune/train.py --data data/train.jsonl
```

See **[finetune/README.md](finetune/README.md)** for the training data format, GPU memory requirements, and instructions for using a different base model.

## 📊 Evaluation

The evaluation pipeline is fully open source and reproducible.

<p align="center">
  <img src="./assets/evaluation.png" alt="VoiceMem Evaluation Results" width="100%">
</p>

### Run Evaluation

A benchmark can be started with a single command:

```bash
export OPENAI_API_KEY=sk-...

# Start with the small example included in the repository
# to make sure everything is set up correctly.
# 2 conversations, 5 questions.
python evaluation/run.py \
    --dataset locomo \
    --data evaluation/examples/locomo_sample.json

# Then run the full dataset.
python evaluation/run.py \
    --dataset locomo \
    --data data/locomo.json
```

Example result:

```text
LoCoMo: 10 conversations · 152 questions

Score: 139/152 = 91.4%

  multi_hop     88.2%
  temporal      85.7%
  single_hop    95.1%

Median retrieval latency: 12 ms
Median retrieved memory: 298 tokens
```

Before running a full evaluation, add `--inspect` to check how the dataset is parsed.

This mode does not call the model and does not incur API costs:

```bash
python evaluation/run.py \
    --dataset locomo \
    --data data/locomo.json \
    --inspect
```

During evaluation, the answering model receives **only the retrieved memories**, not the original conversation history.

If the model receives the full conversation, the benchmark becomes a reading-comprehension test rather than an evaluation of the memory system itself.

See **[evaluation/README.md](evaluation/README.md)** for the complete evaluation protocol and instructions for adding a new benchmark. Adding a benchmark only requires one file and two functions.

## Acknowledgements

We thank the following excellent open-source projects:

* [mem0](https://github.com/mem0ai/mem0) — vector memory engine
* [FunASR](https://github.com/modelscope/FunASR) — streaming ASR with `paraformer-zh-streaming`
* [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — Silero VAD, 3D-Speaker speaker verification, and fallback streaming ASR
* [intfloat/multilingual-e5](https://huggingface.co/intfloat/multilingual-e5-small) — local embeddings and slot classification

VoiceMem also uses OpenAI APIs for Chat, TTS, and Realtime functionality.

## License

VoiceMem is open source under the **Apache License 2.0**.

See [LICENSE](LICENSE) for details.