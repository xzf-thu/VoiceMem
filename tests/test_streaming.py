#!/usr/bin/env python3
"""README「以流式方式运行 VoiceMem」那一块，原样跑一遍。

    export OPENAI_API_KEY=sk-...
    python tests/test_streaming.py

跑的就是 README 里那段：先显式存一条事实 → 喂一段**问句**音频 → 看记忆是不是在
人还没说完时就查好了 → 最后照例走一次入库判断（问句没有新事实，应当判成不入库）。

跑通 = 流式接口是好的，而且 README 那段的输出跟它写的一致。
换音频：VOICEMEM_TEST_AUDIO=/你的/音频.wav（换了之后下面那几条断言的预期也得换）
"""
import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path
from pprint import pprint

import numpy as np
import soundfile as sf

from voicemem import VoiceMem

REPO = Path(__file__).resolve().parent.parent
AUDIO = os.environ.get("VOICEMEM_TEST_AUDIO") or str(REPO / "assets/question.wav")
ROOT = tempfile.mkdtemp(prefix="voicemem_test_")

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("没设 OPENAI_API_KEY")
if not os.path.exists(AUDIO):
    sys.exit(f"找不到音频 {AUDIO}")

vm = VoiceMem(mode="normal", openai_key=os.environ["OPENAI_API_KEY"], top_k=5,
              memory_root=ROOT)

# 本地模型懒加载，先热起来
vm.warmup(verbose=True)

# ① 先存一条事实，等下那个问句才有东西可查
print("\n[入库] 存一条事实：我是素食主义者，对坚果过敏。", flush=True)
seed = vm.ingest("我是素食主义者，对坚果过敏。")
print(f"[入库] 抽出 {seed['facts_count']} 条事实 -> {seed['memory_ids']}", flush=True)

SPEC_MIN_CHARS = 6          # partial 到几个字起投机预取（vm.stream 的默认值）
searching = False
turns = 0
result = {}                 # 这一轮的检索/入库结果，跑完在下面做断言


def on_partial(text):
    global searching
    print(f"\r[partial] {text}", end="", flush=True)
    if not searching and len(text) >= SPEC_MIN_CHARS:
        searching = True
        print("\n[检索开始] 人还没说完，后台已经在查了（投机预取）", flush=True)


async def main():
    global turns

    # 这段音频里是一个问句：「我的饮食禁忌是什么？」
    audio, sr = sf.read(AUDIO, dtype="float32")

    # 末尾补 1s 静音。VAD 要连续 0.5s 静音才判「说完了」，而文件在说完那一刻就
    # 结束了，凑不满就永远等不到 turn_over。真麦克风没这问题——人不说话的时候
    # 麦克风照样在出静音帧。
    audio = np.concatenate([audio, np.zeros(sr, dtype="float32")])

    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

    stream = vm.stream(src_rate=sr, vad_threshold=0.5, on_partial=on_partial)
    step = int(sr * .032)

    for i in range(0, len(pcm), step):
        st = await stream.feed(pcm[i:i + step].tobytes())
        if st.state != "turn_over":                 # "<speak>" / "<silence>"
            continue
        turns += 1

        # ② VAD 确认这一轮说完了。记忆早在说话过程中就查好了，这里直接取，不再等
        print("[检索结束]")
        print("转写  ", st.transcript)
        print("左脑  ", st.result_leftbrain)
        print("右脑  ", st.result_rightbrain)
        pprint({k: getattr(st, k) for k in
                ["speaker_id", "speaker_voiceprint", "emotion",
                 "entity", "schema", "text_embedding"]})

        # ③ 每一轮都要走一次入库判断，这一步不能省：值不值得存是 LLM 说了算，
        #    不是调用方自己猜的
        print("[入库] LLM 正在判断这句话值不值得入库…", flush=True)
        res = vm.ingest(st.transcript)
        print(f"[入库] 抽出 {res['facts_count']} 条事实 -> {res['memory_ids']}")

        result.update(transcript=st.transcript,
                      leftbrain=list(st.result_leftbrain),
                      facts_count=res["facts_count"],
                      memory_ids=list(res["memory_ids"] or []))


asyncio.run(main())

shutil.rmtree(ROOT, ignore_errors=True)

# ── 断言：README 那段声称的三件事 ────────────────────────────────────────────
fails = []

if not turns:
    fails.append("一轮 turn_over 都没有——VAD 没判出「说完了」")
if not searching:
    fails.append("检索开始从来没触发——partial 一直没攒够 "
                 f"{SPEC_MIN_CHARS} 个字，投机预取没跑起来")
if turns and not result.get("leftbrain"):
    fails.append("左脑一条都没查到——那条素食/坚果过敏的事实没被检索回来")
if turns and result.get("facts_count"):
    fails.append(f"问句被判成值得入库了（facts_count={result['facts_count']}，"
                 f"memory_ids={result['memory_ids']}）——README 写的是这一步应当抽不出新事实")

print()
if fails:
    for f in fails:
        print("失败：" + f)
    sys.exit(1)

print(f"通过：转写 {result['transcript']!r}；左脑 {len(result['leftbrain'])} 条；"
      f"问句入库判断 {result['facts_count']} 条事实（预期 0）")
