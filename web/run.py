"""voicemem web demo —— 对话核心（EOU 0–300ms 投机预取）。管道在 utils.py，渲染在 index.html。

看点：本地 ASR+VAD 边听边算，partial 一到就后台起投机 Search（本地 E5 向量 + 本地 slot
分类，注入的 LocalQueryClassifier，0 LLM 0 网络），VAD 在 300ms 确认说完时记忆早已算好，
交给两条 ~10 行控制流的只剩「发 LLM / 发 Realtime」。说到一半停顿又续上（barge-in）→ 取消投机。

跑（参数见 ``--help``；每个都能用同名环境变量给默认值）::

    export OPENAI_API_KEY=sk-...
    python web/run.py                     # 默认 realtime
    python web/run.py \\
      --mode llm_tts \\                    # 没有 Realtime 权限时走这条
      --port 8787 \\
      --spec_min_chars 6 \\
      --gamble_ms 200 \\
      --confirm_ms 300

默认走 ``realtime``（OpenAI 原生语音）：一次往返直接出声，不像 llm_tts 那样要
"LLM 出文本(~1.0s) → 攒够一句 → TTS 合成(~1.2s)" 两段串行，体验差一截。
key 没有 Realtime 权限就用 ``--mode llm_tts``，那条路只要普通 chat + TTS，
TTS 还能换成本地离线模型（``TTS_BACKEND=local``）。

注意：记忆向量用本地 384 维 E5（投机预算内不能走网络）。换过旧 demo（OpenAI 1536 维）留了
记忆库的，维度不兼容——先清掉记忆目录再跑。
"""
import argparse
import asyncio
import base64
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import uvicorn

HERE = Path(__file__).resolve().parent
_ROOT = HERE.parent
sys.path.insert(0, str(HERE))                       # 让 `import utils` 找到同目录管道层
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("VOICEMEM_MODELS_DIR", str(_ROOT / "models"))
# 记忆空间锚在**仓库根**，不跟当前目录走。否则 `cd web && python run.py` 会在
# web/ 底下另建一个空的 voicemem_memoryspace/demo，用户对着空库说半天话，
# 还以为记忆没生效（实测就这么踩过）。
os.environ.setdefault("VOICEMEM_MEMORYSPACE_ROOT", str(_ROOT / "voicemem_memoryspace"))


# ══════════════════ 命令行参数（同名环境变量给默认值，两种都行）══════════════════
# 放在下面那两个重 import 之前：utils / voicemem 会拉起 torch + sentence-transformers，
# 排在它们后面的话 `--help` 得先等模型库加载完。被 import 时不吃 sys.argv（传 []）。

def _parse(argv):
    p = argparse.ArgumentParser(description="voicemem web demo（脑图 + 0–300ms 投机预取）")
    p.add_argument("--mode", choices=["llm_tts", "realtime"],
                   default=os.environ.get("DEMO_MODE", "realtime"),
                   help="回复控制流：realtime=OpenAI 原生语音（默认，体验最好）；"
                        "llm_tts=LLM 流→TTS 流（不需要 Realtime 权限，可换本地 TTS）")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("VOICEMEM_PORT", 8787)))
    p.add_argument("--spec_min_chars", type=int, default=6,
                   help="partial 转写到几个字起投机预取")
    p.add_argument("--gamble_ms", type=int, default=200,
                   help="静音多久就赌你说完了，补投机一次")
    p.add_argument("--confirm_ms", type=int, default=300,
                   help="静音多久由 VAD 确认一轮结束，交出 Turn")
    p.add_argument("--config", default=os.environ.get("VOICEMEM_CONFIG"),
                   help="一个 .json，整体覆盖下面的 CONFIG")
    p.add_argument("--space", default=os.environ.get("VOICEMEM_SPACE", "demo"),
                   help="用哪个 memory space（voicemem_memoryspace/<space>/）")
    p.add_argument("--memory_root", default=os.environ.get("VOICEMEM_MEMORY_ROOT", ""),
                   help="直接指定记忆库目录，给了就盖过 --space")
    return p.parse_args(argv)


ARGS = _parse(None if __name__ == "__main__" else [])

import utils                                         # noqa: E402  同目录管道层
from voicemem import VoiceMem                        # noqa: E402

BARGE_DEBUG = os.environ.get("BARGE_DEBUG", "1") != "0"
BARGE_THRESHOLD = float(os.environ.get("BARGE_THRESHOLD", "0.45"))  # 越小越容易被打断
#: 转写要比上次多出这么多个字，才算"他真的插话了"。
#: 之前这里是「连续人声 ≥280ms」，纯 VAD 判定太松——咳嗽、关门、AEC 没压干净的
#: 助手回声都算人声，日志里一串"连续人声 → 请求打断 / 助手没在说，忽略"在空转。
#: 换成等 ASR 真的吐出字，代价是多等一次出字（~200-300ms），换来不会自己掐自己。
#:
#: 这个数字直接决定"插话多久才被听见"：说满 N 个字要时间，实测 3 个字要 ~2 秒。
#: 曾经因为 2 个字会被回声骗到（漏出过 "an" —— 助手说的 Annie 回到麦克风里）
#: 才提到 3。现在回声改由 _is_echo() 按**助手正在说的原文**挡，不再靠字数硬扛，
#: 所以降回 2：插话少说一个字，大约快 300-500ms。
BARGE_MIN_CHARS = int(os.environ.get("BARGE_MIN_CHARS", "2"))
#: 助手刚开口那一小段不允许被打断——那时候麦克风里几乎只有它自己的声音。
BARGE_GRACE_MS = int(os.environ.get("BARGE_GRACE_MS", "500"))
#: OpenAI 那侧用哪种回合/打断判定。semantic_vad 由模型判"这是不是真的在打断"，
#: 对 backchannel（"嗯""对""哦"）不敏感；server_vad 只看有没有声音，所以助手自己
#: 的回声、环境噪声都能把它掐了。模型或 SDK 不支持时会以 error 事件回来（不抛），
#: 日志里看到就改回 TURN_DETECTION=server_vad。
TURN_DETECTION = os.environ.get("TURN_DETECTION", "semantic_vad")
#: semantic_vad 的抢答倾向：low 更愿意等你说完，high 更爱抢。
VAD_EAGERNESS = os.environ.get("VAD_EAGERNESS", "low")
MIC_RATE = 24000                       # 前端上行的采样率（index.html 的 SAMPLE_RATE）
#: 按声纹拦陌生人。默认开——启动时预热过、又在后台线程算，实测对延迟零影响
#: （memory_hits 仍在 EOU 前 0.63s 到达，跟关掉时一样）。
SPEAKER_GATE = os.environ.get("VOICEMEM_SPEAKER_GATE", "1") != "0"   # 打断为什么没触发：看这几行日志
#: 连续几轮认成别人才判"陌生人"。1 = 一轮就翻脸（demo 里演"换个人说话"要的就是
#: 这个）；声纹库脏、老是把主人认成新人时，调到 2 能挡掉大部分误判。
STRANGER_MIN_TURNS = int(os.environ.get("STRANGER_MIN_TURNS", "1"))
#: 每轮都打一行说话人判定（默认只在判成陌生人时打）。
SPEAKER_DEBUG = os.environ.get("SPEAKER_DEBUG", "0") != "0"
MODE = ARGS.mode                                     # llm_tts | realtime
SPEC_MIN_CHARS = ARGS.spec_min_chars                 # partial 起投机
GAMBLE_S  = ARGS.gamble_ms / 1000                    # 赌说完
CONFIRM_S = ARGS.confirm_ms / 1000                   # VAD 确认结束

_RT_PERSONA = (
    # 开宗明义地把"你凭什么存在"讲清楚。模型默认的助理人格非常强势，不明确
    # 给它一个不同的立身之本，它就会退回"您好，有什么可以帮您"。
    "你是这个用户长期在用的语音助手，你们认识很久了。你的价值在于**你记得他**——"
    "你说的每句话，都应该是一个没有记忆的助手说不出来的。\n"
    "\n"
    "【两种记忆，用法完全不同】\n"
    "factual memory 是事实，可以直接提，就像你本来就记得"
    "（「Annie 那事你还好吗」，不是「根据记录，Annie 要转学」）。\n"
    "emotion & characteristics 是他这个人的性格和情绪归因，**只**影响你的语气、"
    "先说什么、什么别碰——一个字都不许说出来。\n"
    "\n"
    # 检索按相关度排，但排在前面不等于跟这句话有关。不说清楚模型会硬凑，
    # 听起来就是答非所问或者莫名其妙翻旧账。
    "【检索到 ≠ 相关】\n"
    "这些记忆是检索出来的，不一定都跟他这句话有关。挑真正有关的用，其余的知道就好。"
    "一条都不相关时，就顺着他这句话往下说，不用勉强提起任何记忆。\n"
    "\n"
    # 最贵的一条。没有它模型会编：记忆里只有「下周要考 GRE」，它张口就是
    # 「数学一直是你的强项吧」——听着像真记得，其实是幻觉，比不记得更糟。
    "【只说记忆里真有的事】\n"
    "没写的细节——分数、科目、他做过什么、谁说过什么、哪一天——一个字都不许补；"
    "记忆里没带日期就别提时间。宁可说得少，也不要编。不知道就直说不知道。\n"
    "\n"
    # 产品感的核心：主动性。这一段是"作为产品"和"作为 demo"最大的分野。
    "【主动，别把活儿推给他】\n"
    "× 「有什么想聊的吗」「有什么可以帮你的吗」「今天过得怎么样」——"
    "这些话没有记忆也说得出来，等于当面告诉他你什么都不记得。\n"
    "√ 直接落到具体的事：「明天那个会，准备得怎么样了」。\n"
    "他说得含糊时（「最近压力好大」「今天好累」），别泛泛安慰、也别只是问「怎么了」。"
    "从记忆里挑出最可能是原因的那件具体的事，说出来问他是不是。猜错他会纠正你。\n"
    "一轮最多问一个问题，而且要具体。没什么可问的就别问，说完就停——"
    "每句都拿问号结尾是在审问，不是聊天。\n"
    "\n"
    # 没有这一段，"ok ok" 会被当成一轮全新对话，模型重新打招呼。
    "【顺着对话走】\n"
    "「ok」「好的」「嗯嗯」「行」这类是收尾或者认可，**不是新话题**。"
    "简短接一句就行，绝对不要重新打招呼、不要重启话题、不要重新自我介绍。\n"
    "刚才聊到哪儿了，看下面「刚才的对话」那一段。\n"
    "\n"
    "【说他是什么样的人】\n"
    "每个判断后面紧跟那件让你这么想的事，别堆形容词——"
    "「你特别有追求」这种话空模型也说得出来。\n"
    "\n"
    "【怎么说话】\n"
    "你是在**说话**，不是在写字。短句，一次说一两句就停。"
    "别复述他刚说的话，别用「我记得你说过」开头，别念清单，"
    "也别用「作为你的助手」这类自我介绍——你们早就认识了。"
)



# 从 1147 字精简到现在这个长度。删掉的和为什么——想加回来先看这里，原文在 git 里：
#
# · 「说三四句，两三个点」等长度/结构规定
#     → demo 口味，不是正确性问题。而且句子越多、TTS 分段越多、接缝越明显。
#       真要控制长度，第一句"一次说一两句就停"已经够了。
# · 「说话方式：语调有起伏、重要的词咬重、问句尾音扬起来、别播音腔」整段
#     → 这是**表演指示**，写在文本 prompt 里是让文字模型去理解、再指望 TTS 猜出来，
#       中间隔了两层。TTS 后端有 instruction 参数（Breeze 有，gpt-4o-mini-tts 也有）
#       直接收这个，效果实在得多。搬过去了就别在这儿重复。
# · 「别每句都用同一个口头禅开头」
#     → 它补的是另一条已经删掉的规则（原来写"用嗯/哎/诶开头"，模型当成每句必须
#       执行）。病根没了，补丁也就不用留。
# · 「问'你对我什么印象'时答的重点是他这个人，不是最近发生的事」
#     → 为某个 demo 问题定制的。上面"每个判断紧跟依据"那条已经覆盖了大半。


_STRANGER = ("说话的不是你认识的那个人——声纹对不上。你对他没有任何记忆。"
             "别把别人的事讲给他听，也别猜他是谁。就当第一次见面，"
             "友好但如实地说你还不认识他。")

#: 这一轮一条记忆都没检索到时追加的一句。
#:
#: 人设里那套"从含糊的一句话里猜出具体那件事"的指示，在有记忆时是这套系统最值钱
#: 的地方，可**一条都没检索到**时它就成了编造的许可证：新建一个空的 Memory Space、
#: 第一句问「我不能吃什么」，它张口就是「你的饮食禁忌里，辣椒和海鲜要注意，之前
#: 你提到过对这些过敏」——两样都是凭空来的。空库的第一句话就撞得上，而那正是别人
#: 第一次用这个 demo 的时刻。
#:
#: 跟 _STRANGER 的区别：那句是"你认识的是另一个人"，这句是"这件事你不知道"。
#: 界面选的语言。助手跟着它走——回复用这个语言，抽出来的记忆也用这个语言写。
#:
#: 记忆的语言必须跟着一起换，不然库里会中英混着长：同一件事今天记成中文、明天
#: 记成英文，检索时两边都只能命中一半。
UI_LANG = "zh"

#: 助手说什么语言。原来是跟着**界面语言开关**走（前端把选择存在 localStorage 里，
#: 每次开页面自动 POST 给后端），结果是：后端默认值永远不生效，上次录英文 demo 切过
#: 一次，之后全程中文提问也照样英文回答，换库、重启都没用。
#: 现在改成跟着**用户这句话的语言**走——问什么语言答什么语言，跟界面无关。
_LANG_MIRROR = ("如果用户用中文问你，就用中文回复。如果用英文问你，就用英文回复。"
                "其他语言也是一样。")


def set_lang(lang: str) -> None:
    global UI_LANG
    UI_LANG = "en" if str(lang).lower().startswith("en") else "zh"
    print(f"[lang] 助手改说 {UI_LANG}", flush=True)


def _lang_note() -> str:
    return _LANG_MIRROR


_NO_MEMORY_NOTE = (
    "这一轮你没有检索到任何相关记忆。所以：**不要提任何具体的事**——"
    "食物、地点、人名、日期、他做过什么、他喜欢什么，一个都不许说，"
    "更不能说「你之前提到过」「我记得你说过」。"
    "如实说这件事你还不知道，然后问他，或者就着他这句话本身聊。"
    "宁可显得记性不好，也不要编——编出来的东西他一眼就看得穿，"
    "而且会让他不再相信你真记得的那些。"
)


#: 问的是"一段声音"时才回放。故意做得很笨——这是个触发词表，不是意图分类器：
#: 多放一次听感上只是"它把当时那段放给你听"，判漏了也只是回到手动点 ▶。
_SOUND_WORDS = ("歌", "曲", "调子", "旋律", "音乐", "哼", "那段声音", "放来听",
                "放给我听", "播一下", "什么声音")


def _musical_memory_ids() -> set[str]:
    """音频里**真的有音乐**的那些记忆。

    ingest 时音乐识别命中过的一轮，会被打上 ``tune:<tune_id>`` 标签（跟
    ``scene:café`` / ``speaker:person_x`` 一样存在 memory_tags 里）。有了它就不用
    靠用户正好说过"我听到一首歌"——在咖啡馆随便聊，那一轮的背景音乐照样被记下来。

    读失败就返回空集：这只是个偏好排序，取不到就退回原来的"第一条有音频的"。
    """
    try:
        tunes = vm._o._audio._music_store().list_tunes()
        ids = [f"tune:{t['tune_id']}" for t in tunes]
        if not ids:
            return set()
        store = vm._o._get_repo()._cognitive_store
        return set(store.memory_ids_for_slots_v2(vm._o._user_id, ids))
    except Exception as e:
        print(f"[web] 读音乐标签失败（不影响回放）：{type(e).__name__}: {e}", flush=True)
        return set()


#: 回放正在进行到什么时候（单调时钟）。这段时间里的空文本轮次一律丢掉。
_REPLAY_UNTIL = 0.0


def _note_replay(memory_id: str) -> None:
    """记下这次回放大概会响多久。

    放出来的录音会从扬声器绕回麦克风，被 VAD 判成新的一轮——转写是空的、左右脑
    都没命中，模型手上什么都没有，于是回一句"抱歉，我没法直接重播"，日志里那段
    音乐还被认成"第 2 次听到"。页面那边已经在回放期间停了上行（见 voicemem.html
    的 holdMic），这里是第二道：万一前端没生效（别的客户端、老页面缓存），
    后端自己也认得出这一轮是自己放出去的。
    """
    global _REPLAY_UNTIL
    path = audio_of(memory_id)
    if not path:
        return
    try:
        import soundfile as sf
        dur = float(sf.info(path).duration)
    except Exception:
        dur = 15.0          # 读不出时长就按上限压一会儿，宁可多挡一轮
    _REPLAY_UNTIL = time.monotonic() + dur + 1.0


def _replaying_now() -> bool:
    return time.monotonic() < _REPLAY_UNTIL


#: 刚听过、还没来得及入库的那段音乐。``{"path": wav, "at": 单调时钟}``
#:
#: 为什么要单独记一份：入库走后台线程，一轮 15-20 秒才写完，而"刚听完就问"恰恰
#: 是最自然的用法——实测放完音乐立刻问「重播一下刚才那首歌」，那条记忆的
#: created_at 跟提问时间只差几秒，查库的时候还没有它。而 tune 识别是同步的
#: （Ingest 立刻带着 recognized_tune 返回），这份缓存在上一轮结束时就写好了，
#: 正好补上那十几秒空窗。
_LAST_TUNE: dict = {"path": "", "at": 0.0}

#: 用这个假 id 请求上面那段录音。真 memory_id 是 uuid，不会撞。
LAST_TUNE_ID = "last:tune"

#: 缓存多久算"刚才"。超过就不认了——半小时前听的歌不该被「刚才那首」捞出来。
LAST_TUNE_TTL_S = float(os.environ.get("VOICEMEM_LAST_TUNE_TTL", "1800"))


def _remember_tune(audio_path: str) -> None:
    _LAST_TUNE.update(path=str(audio_path or ""), at=time.monotonic())
    print(f"  [replay] 记住这段音乐：{audio_path}", flush=True)


def _last_tune_path() -> str:
    p = _LAST_TUNE.get("path") or ""
    if not p or time.monotonic() - float(_LAST_TUNE.get("at") or 0) > LAST_TUNE_TTL_S:
        return ""
    return p if Path(p).exists() else ""


#: 问句里的时间限定。记忆本来就绑了 created_at，「上周三下午在咖啡馆听的那首」
#: 这种要求靠语义检索是碰运气——"上周三""下午"对向量几乎没有影响，撞上哪条算哪条。
#: 所以时间和地点单独解析出来，当**硬条件**筛。
_DAY_WORDS = (
    (("大前天",), -3), (("前天",), -2), (("昨天", "昨晚", "昨儿"), -1),
    (("今天", "今早", "今晚", "今日"), 0),
)
_HOUR_WORDS = (
    (("凌晨", "半夜", "深夜"), (0, 6)),
    (("早上", "早晨", "今早", "一早", "清晨"), (6, 10)),
    (("上午",), (8, 12)),
    (("中午", "晌午"), (11, 14)),
    (("下午",), (12, 18)),
    (("傍晚", "黄昏"), (17, 20)),
    (("晚上", "晚间", "昨晚", "今晚", "夜里"), (18, 24)),
)
#: 星期几。「周三」单说指本周，配合「上周」就是上一周。
_WEEKDAYS = (("周一", "星期一", "礼拜一"), ("周二", "星期二", "礼拜二"),
             ("周三", "星期三", "礼拜三"), ("周四", "星期四", "礼拜四"),
             ("周五", "星期五", "礼拜五"), ("周六", "星期六", "礼拜六"),
             ("周日", "周天", "星期日", "星期天", "礼拜天"))

#: 中文地点说法 → 声学场景标签（scene_classifier.SceneTag）。
#: 场景是每轮录音自动分类出来的，存在 memory_tags 里（scene:café）。
_PLACE_WORDS = (
    (("咖啡馆", "咖啡店", "咖啡厅", "星巴克"), "café"),
    (("办公室", "公司", "工位", "单位"), "office"),
    (("家里", "家中", "在家", "屋里", "房间"), "home"),
    (("外面", "户外", "外边", "路上", "街上", "公园"), "outdoor"),
    (("车上", "地铁", "公交", "路上", "通勤", "火车"), "transit"),
    (("会议", "开会", "会上", "会议室"), "meeting"),
)


#: 「第几首」。候选按时间正序排，序号就是下标——同一段时间里听了好几首时，
#: 「第一首」「上一首」是最自然的说法，而这信息记忆里本来就有（created_at）。
_ORDINALS = (
    (("第一首", "第一段", "第1首", "头一首", "最早那首", "最先那首"), 1),
    (("第二首", "第二段", "第2首"), 2),
    (("第三首", "第三段", "第3首"), 3),
    (("第四首", "第四段", "第4首"), 4),
    (("第五首", "第五段", "第5首"), 5),
)
#: 从后往前数的说法。-1 = 最后一首，-2 = 倒数第二（也就是"上一首"）。
_ORDINALS_BACK = (
    (("最后一首", "最后那首", "最后一段", "最新那首", "最近那首"), -1),
    (("上一首", "前一首", "上一段", "前一段", "前面那首", "上一个"), -2),
)


def _ordinal_of(text: str):
    """问句里的「第几首」→ 1-based 序号（负数表示从后往前数）；没说返回 None。"""
    t = text or ""
    for words, n in _ORDINALS + _ORDINALS_BACK:
        if any(w in t for w in words):
            return n
    return None


def _place_of(text: str) -> str:
    """问句里提到的地点 → 场景标签；没提就返回 ""。"""
    t = text or ""
    return next((tag for words, tag in _PLACE_WORDS if any(w in t for w in words)), "")


def _time_window(text: str):
    """问句 → (起, 止) 两个 datetime；没提时间就返回 None。

    认得出：今天/昨天/前天/大前天、这周/上周/上上周、周一~周日、这个月/上个月、
    N月N号、N天前，以及凌晨/早上/上午/中午/下午/傍晚/晚上这些时段，可以组合
    （「上周三下午」）。认不出来就返回 None 交给"取最近的一条"——别自作聪明猜，
    猜错了放出来的是别的录音，比放不出来更糟。

    「刚才」「刚刚」故意不算时间限定：它们说的是"最近那次"，走严格分支反而会因为
    筛不到而什么都不放。
    """
    import re
    from datetime import datetime, timedelta
    t = text or ""
    now = datetime.now()
    day0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
    span = None                      # (起日, 止日)，止日不含

    m = re.search(r"(\d+)\s*天前", t)
    if m:
        d = day0 - timedelta(days=int(m.group(1)))
        span = (d, d + timedelta(days=1))

    if span is None:
        wd = next((i for i, words in enumerate(_WEEKDAYS) if any(w in t for w in words)), None)
        if wd is not None:
            # 本周一为基准；「上周」再往前推一周
            monday = day0 - timedelta(days=day0.weekday())
            if "上上周" in t or "上上星期" in t:
                monday -= timedelta(days=14)
            elif "上周" in t or "上星期" in t or "上礼拜" in t:
                monday -= timedelta(days=7)
            d = monday + timedelta(days=wd)
            span = (d, d + timedelta(days=1))

    if span is None:
        m = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[号日]", t)
        if m:
            mth, dom = int(m.group(1)), int(m.group(2))
            year = now.year - 1 if mth > now.month else now.year
            try:
                d = datetime(year, mth, dom)
                span = (d, d + timedelta(days=1))
            except ValueError:
                span = None
    if span is None:
        m = re.search(r"(?<![0-9])(\d{1,2})\s*[号日](?!\s*[前后])", t)
        if m:
            dom = int(m.group(1))
            try:
                d = day0.replace(day=dom)
                if d > day0:                       # 这个月还没到，那说的是上个月
                    d = (day0.replace(day=1) - timedelta(days=1)).replace(day=dom)
                span = (d, d + timedelta(days=1))
            except ValueError:
                span = None

    if span is None:
        day = next((d for words, d in _DAY_WORDS if any(w in t for w in words)), None)
        if day is not None:
            d = day0 + timedelta(days=day)
            span = (d, d + timedelta(days=1))

    if span is None:
        if "上上周" in t or "上上星期" in t:
            monday = day0 - timedelta(days=day0.weekday() + 14)
            span = (monday, monday + timedelta(days=7))
        elif "上周" in t or "上星期" in t or "上礼拜" in t:
            monday = day0 - timedelta(days=day0.weekday() + 7)
            span = (monday, monday + timedelta(days=7))
        elif "这周" in t or "本周" in t or "这星期" in t:
            monday = day0 - timedelta(days=day0.weekday())
            span = (monday, monday + timedelta(days=7))
        elif "上个月" in t or "上月" in t:
            first = day0.replace(day=1)
            span = ((first - timedelta(days=1)).replace(day=1), first)
        elif "这个月" in t or "本月" in t:
            first = day0.replace(day=1)
            span = (first, (first + timedelta(days=32)).replace(day=1))

    hours = next((h for words, h in _HOUR_WORDS if any(w in t for w in words)), None)
    if span is None and hours is None:
        return None
    if span is None:                                  # 只说了时段，默认今天
        span = (day0, day0 + timedelta(days=1))
    if hours is None:
        return span
    # 时段只在跨度是一天时才细化——「上周下午」没有意义
    if (span[1] - span[0]).days > 1:
        return span
    return span[0] + timedelta(hours=hours[0]), span[0] + timedelta(hours=hours[1])


def _tune_memories() -> list[dict]:
    """所有能放出来的候选：带 tune: 标签、或者是纯声音轮，并且原声还在。

    为什么 sound_only 也算：直接对着麦克风放音乐、声学又没认出来时（外放、环境
    吵、片段短都会漏），那一轮既没有 tune 标签也没有话，只知道"有段录音、没人
    说话"。它恰恰最可能就是用户要找的那段。普通说话轮不会混进来。

    一条 = {"id", "at": datetime, "scenes": {...}, "text"}，新的在前。
    回放的候选池就是它——按时间、按地点筛都在这个池子里做，不依赖这一轮检索
    碰巧命中了什么。
    """
    from datetime import datetime
    try:
        import sqlite3
        from voicemem.utils.common import space as _space
        c = sqlite3.connect(_space.db(vm._o._memory_root))
        c.row_factory = sqlite3.Row
        rows = c.execute(
            """SELECT m.id, m.content, m.created_at,
                      (SELECT group_concat(s.slot) FROM memory_tags s
                        WHERE s.memory_id = m.id AND s.slot LIKE 'scene:%') scenes,
                      (SELECT u.slot FROM memory_tags u
                        WHERE u.memory_id = m.id AND u.slot LIKE 'tune:%' LIMIT 1) tune,
                      EXISTS (SELECT 1 FROM memory_tags o
                               WHERE o.memory_id = m.id AND o.slot = 'sound_only') sound_only
                 FROM memories m
                 WHERE EXISTS (SELECT 1 FROM memory_tags t
                                WHERE t.memory_id = m.id
                                  AND (t.slot LIKE 'tune:%' OR t.slot = 'sound_only'))
                 ORDER BY m.created_at DESC LIMIT 300""").fetchall()
        c.close()
    except Exception as e:
        print(f"[replay] 列音乐记忆失败：{type(e).__name__}: {e}", flush=True)
        return []

    out = []
    for r in rows:
        if not audio_of(r["id"]):
            continue
        try:
            at = datetime.fromisoformat(r["created_at"]).astimezone().replace(tzinfo=None)
        except Exception:
            continue
        out.append({"id": r["id"], "at": at, "text": r["content"] or "",
                    "sound_only": bool(r["sound_only"]),
                    "tune": (r["tune"] or "").split(":", 1)[-1],
                    "scenes": {x.split(":", 1)[1] for x in (r["scenes"] or "").split(",") if ":" in x}})
    return out


def _archived_memory_ids() -> list[str]:
    """所有存了原声的记忆 id，新的在前。

    按时间找录音时的兜底：音乐识别没命中的那些轮次没有 tune: 标签，但音频照样
    归档了。只认标签的话，用户明明刚放过一首歌，它却回一句"没存到"。
    """
    try:
        import sqlite3
        from voicemem.utils.common import space as _space
        c = sqlite3.connect(_space.db(vm._o._memory_root))
        rows = c.execute("SELECT id FROM memories ORDER BY created_at DESC LIMIT 200").fetchall()
        c.close()
        return [r[0] for r in rows]
    except Exception as e:
        print(f"[web] 列存档记忆失败（不影响回放）：{type(e).__name__}: {e}", flush=True)
        return []


def _created_at(mid: str):
    """这条记忆是什么时候写下的。查不到返回 None。

    向量库那边只存了 date（天粒度），要按"下午"筛得用 sqlite 里的 created_at。

    路径要用 vm 解析好的 _memory_root，**不能拿 ARGS.space**——``space.db()`` 收的
    是目录路径，给它一个裸名字（"musictest"）会当成相对路径，找不到就新建一个空
    库，于是每条记忆都查不到时间，按时间回放静默失效。
    """
    from datetime import datetime
    try:
        import sqlite3
        from voicemem.utils.common import space as _space
        c = sqlite3.connect(_space.db(vm._o._memory_root))
        row = c.execute("SELECT created_at FROM memories WHERE id=?", (mid,)).fetchone()
        c.close()
        if not row or not row[0]:
            return None
        return datetime.fromisoformat(row[0]).astimezone().replace(tzinfo=None)
    except Exception:
        return None


#: 同一首歌的两个片段最多隔多久还算"连着的一首"。VAD 在乐句停顿处断开、
#: 下一段重新起录，中间的空档就是这么来的。
TUNE_GAP_S = float(os.environ.get("VOICEMEM_TUNE_GAP_S", "90"))


#: 一组片段用这个前缀的假 id 请求，后面接用逗号分隔的 memory_id。
#: 走的还是既有的 /api/audio/{memory_id}，前端一个字都不用改。
GROUP_ID_PREFIX = "group:"

#: 拼好的整首放这儿。同一组只拼一次，之后直接命中。
_STITCH_CACHE: dict = {}


def _stitch(memory_ids: list) -> str:
    """把几段录音按顺序拼成一个 wav，返回路径；拼不了就返回第一段。

    采样率不一致时以第一段为准重采样——归档的都是 16k 单声道，真遇到不一致
    也不该让回放整个失败。
    """
    paths = [q for q in (audio_of(m) for m in memory_ids) if q]
    if not paths:
        return ""
    if len(paths) == 1:
        return paths[0]
    key = "|".join(paths)
    if key in _STITCH_CACHE and Path(_STITCH_CACHE[key]).exists():
        return _STITCH_CACHE[key]
    try:
        import numpy as np
        import soundfile as sf
        from voicemem.utils.audio.stream_io import resample as _resample
        chunks, sr0 = [], None
        for q in paths:
            x, sr = sf.read(q, dtype="float32", always_2d=False)
            if getattr(x, "ndim", 1) > 1:
                x = x.mean(axis=1)
            if sr0 is None:
                sr0 = sr
            elif sr != sr0:
                x = _resample(x, sr, sr0)
            chunks.append(x)
        out = TURN_AUDIO_DIR / f"stitch_{uuid.uuid4().hex[:12]}.wav"
        sf.write(out, np.concatenate(chunks), sr0)
        _STITCH_CACHE[key] = str(out)
        total = sum(len(c) for c in chunks) / float(sr0 or 16000)
        print(f"  [replay] 拼好 {len(paths)} 段 → {total:.1f}s", flush=True)
        return str(out)
    except Exception as e:
        print(f"  [replay] 拼接失败，只放第一段：{type(e).__name__}: {e}", flush=True)
        return paths[0]


def _same_song_group(pool: list, pick: dict) -> list:
    """挑中的这一段，连同它前后**同一首、时间相连**的片段，按时间正序。

    一轮录音在 VAD 判到静音时就结束了，而音乐里的乐句停顿、弱拍随便就够长——
    实测一段 16 秒的曲子被切成 7.8s + 5.0s 两轮，回放只放中的一段，听感就是
    "只放了开头"。这些碎片被音乐识别归成同一个 tune_id，把它们按时间拼回去
    就是原来那首。

    tune_id 认不出来（tune:unidentified）时只按时间相邻算——那时没有别的凭据，
    宁可少拼几段，也别把两首不同的歌接在一起。
    """
    tune = pick.get("tune") or ""
    same = [x for x in pool
            if (x.get("tune") or "") == tune and (tune != "unidentified" or x.get("sound_only"))]
    same.sort(key=lambda x: x["at"])
    if pick not in same:
        return [pick]
    i = same.index(pick)
    lo = i
    while lo > 0 and (same[lo]["at"] - same[lo - 1]["at"]).total_seconds() <= TUNE_GAP_S:
        lo -= 1
    hi = i
    while hi + 1 < len(same) and (same[hi + 1]["at"] - same[hi]["at"]).total_seconds() <= TUNE_GAP_S:
        hi += 1
    return same[lo:hi + 1]


def _group_id(group: list) -> str:
    """一组片段 → 一个 id。只有一段时就用它本身的 id，别绕。"""
    if not group:
        return ""
    if len(group) == 1:
        return group[0]["id"]
    return GROUP_ID_PREFIX + ",".join(x["id"] for x in group)


def _prefer_sound_only(cand: list) -> list:
    """候选里有"只有声音、没有说话"的那种，就只用它们。

    「给你听一首歌啊」和后面那段音乐是**两轮**——一轮录音从检测到人声开始、到 VAD
    判定说完为止，所以前一轮存的是他自己那句话（说的时候背景里已经有音乐，于是
    那轮也带 tune 标签），后一轮才是音乐本身。两条都在池子里，挑错了放出来是用户
    自己的声音，听感就是"音乐被截断了"。
    一条纯音乐轮都没有时原样返回——总比什么都不放强。
    """
    only = [x for x in cand if x.get("sound_only")]
    return only or cand


def _replay_id(text: str, result) -> str:
    """这一轮该不该把当时那段原声放回来，返回要放的 memory_id（不放就空串）。

    问的是声音（``_SOUND_WORDS``）才会放。候选池是**所有**带音乐标签、原声还在的
    记忆（``_tune_memories``），不是这一轮碰巧检索到的东西——「上周三在咖啡馆听的
    那首」靠语义检索是碰运气，而时间和地点记忆里本来就有。

      ① 说了时间或地点 → 当硬条件筛，取符合条件里最近的一条。
         筛空就是**真没有**，回空串让模型如实说，不退回去随便挑：退回去的话，
         问「昨天晚上那首歌」会把今天下午的录音放出来，用户以为它记错了时间，
         其实它压根没在按时间找。
      ② 什么条件都没说（"那首歌""刚才那段旋律"）→ 池子里最近的一条。
      ③ 池子是空的 → 退回这一轮检索命中的、和刚听过还没入库的那段。
    """
    if not _wants_sound(text):
        return ""

    pool = _tune_memories()
    window, place = _time_window(text), _place_of(text)
    nth = _ordinal_of(text)

    # 「第一首」没说是哪天的第一首时，默认今天——问"第一首"几乎都是指今天听的
    # 那批里的第一首，拿全库去数会数到几个月前那条。
    if nth is not None and window is None:
        from datetime import datetime, timedelta
        day0 = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        window = (day0, day0 + timedelta(days=1))

    if pool and (window or place):
        cand = pool
        if window:
            cand = [x for x in cand if window[0] <= x["at"] <= window[1]]
        if place:
            cand = [x for x in cand if place in x["scenes"]]
        cand = _prefer_sound_only(cand)
        if cand:
            cand.sort(key=lambda x: x["at"])          # 正序：第一首在最前
            if nth is None:
                pick = cand[-1]                        # 没说第几首就取最近的
            elif -len(cand) <= (nth - 1 if nth > 0 else nth) < len(cand):
                pick = cand[nth - 1 if nth > 0 else nth]
            else:
                print(f"  [replay] 那段时间只有 {len(cand)} 首，没有第 {nth} 首", flush=True)
                return ""
            print(f"  [replay] 按条件挑中 {pick['id'][:12]}（{pick['at']:%m-%d %H:%M}"
                  f"{' / ' + place if place else ''}"
                  f"{' / 第 %d 首' % nth if nth else ''}，共 {len(cand)} 首）", flush=True)
            return _group_id(_same_song_group(pool, pick))
        print(f"  [replay] 池里 {len(pool)} 段音乐，没有符合条件的"
              f"（{'时间 %s~%s ' % (window[0], window[1]) if window else ''}"
              f"{'地点 ' + place if place else ''}）", flush=True)
        return ""

    if pool:
        pick = max(_prefer_sound_only(pool), key=lambda x: x["at"])
        print(f"  [replay] 没说条件，放最近的 {pick['id'][:12]}"
              f"（{pick['at']:%m-%d %H:%M}）", flush=True)
        return _group_id(_same_song_group(pool, pick))

    # 池子是空的：音乐识别没命中，或者刚听完还没入库。
    playable = [h.memory_id for h in (getattr(result, "hits", None) or [])
                if audio_of(h.memory_id)]
    if playable:
        print(f"  [replay] 没有音乐标签，放检索到的 {playable[0][:12]}", flush=True)
        return playable[0]
    if _last_tune_path():
        print("  [replay] 还没入库，放刚听过的那段", flush=True)
        return LAST_TUNE_ID
    print(f"  [replay] 放不了：一段音乐记忆都没有，检索 "
          f"{len(getattr(result, 'hits', None) or [])} 条也都没音频，"
          f"缓存 path={_LAST_TUNE.get('path') or '(空)'}", flush=True)
    return ""


def _turn_detection() -> dict:
    """OpenAI 那侧的回合/打断判定。两种 provider 的参数**不通用**——semantic_vad
    不吃 threshold / prefix_padding_ms / silence_duration_ms，传了整段会被静默拒绝
    （只以 error 事件回来）。所以两套各写各的，别合并。

    create_response=False   什么时候回复由我们决定（本地判完一轮、记忆预取好）
    interrupt_response=True 用户一开口，服务端直接掐掉正在播的回复
    """
    # interrupt_response=False：**不让 OpenAI 那侧打断**。
    # 它的 VAD 是纯声学的，只要能量像人声就触发，对 AEC 残留毫无抵抗力——实测
    # 助手说到 1.4-1.6s 时被自己的回声打断，表现是"说半句突然卡住"
    # （日志：OpenAI VAD 听到人声 live=True 还在播=True 1570ms）。
    # 本地那条是 ASR 确认制，要真的转出几个字才算插话，回声转不出连贯的字。
    # 打断改成只走本地：on_speech 里我们自己发 response.cancel，效果一样。
    base = {"type": TURN_DETECTION, "create_response": False, "interrupt_response": False}
    if TURN_DETECTION == "semantic_vad":
        return {**base, "eagerness": VAD_EAGERNESS}
    # server_vad：默认 0.5 对插话太钝——人隔着扬声器说话，回声消除处理过之后
    # 信号本来就弱，够不到阈值就等于打不断。
    return {**base, "threshold": BARGE_THRESHOLD,
            "prefix_padding_ms": 200, "silence_duration_ms": 320}


#: 要回放时追加的一句。不加的话模型会去"描述"那段音频（"你说那是一首很轻快的
#: 钢琴曲…"）——它根本没听过那段音频，描述全是编的；而且用户马上就要亲耳听到。
_REPLAY_NOTE = ("你手上有他当时那段录音，说完这句就会放给他听。"
                "所以别去描述那段声音是什么样的——你没听过，别编。"
                "就短短一句把它引出来，像「我把当时那段找出来了，你听听是不是这个」，"
                "然后停住，等他听。")

#: 他在找一段声音、但那个时间段确实没有存档时追加的一句。
#: 不加的话模型会顺口答"当然，马上播放"——然后什么都不放。说要播却没播，
#: 比直接说没找到糟得多。
_NO_REPLAY_NOTE = ("他在找一段录音，但你手上**没有**他说的那个时间的录音，"
                   "这一轮不会播任何东西。所以别说「马上播放」「这就放给你听」。"
                   "直说那个时候没有存到，再问一句是不是别的时候，"
                   "或者说说你记得的相关的事。")


def _wants_sound(text: str) -> bool:
    return any(w in (text or "") for w in _SOUND_WORDS)


#: 感知层判出来的情绪 → 一句**可执行的**表演指示。
#:
#: 光在人设里写"要有起伏"没用——那是形容词，模型没有对象可对。给它一个具体的
#: 目标（"他现在是焦虑的，你要放慢、压低、先接住"），语气才真的会变。
#: 情绪本身是声学感知算出来的（Qwen-Omni 归因 + 韵律 VAD），每轮都不一样。
_TONE = {
    "焦虑": "他现在是紧绷的。语速放慢，句子短，先接住再说事，别一上来就给方案。",
    "沮丧": "他现在情绪很低。声音压低、放软，允许有停顿，别急着安慰也别讲道理。",
    "难过": "他现在难过。轻一点、慢一点，先陪着，别转移话题。",
    "悲伤": "他现在难过。轻一点、慢一点，先陪着，别转移话题。",
    "烦躁": "他现在有点烦。直接说重点，别绕，别追问，也别用哄的语气。",
    "愤怒": "他在气头上。先认下来，语速稳住，别辩解。",
    "开心": "他心情好。跟着热起来，语调扬上去，可以笑出来，别端着。",
    "愉悦": "他心情好。跟着热起来，语调扬上去，可以笑出来，别端着。",
    "兴奋": "他很兴奋。你也兴奋起来，语速快一点、音量抬一点，别泼冷水。",
    "自豪": "他为自己骄傲。替他高兴，说得实在一点，别敷衍地夸。",
    "期待": "他在期待。语气轻快，跟着往前想一步。",
    "紧张": "他紧张。稳住，声音放平放缓，给他确定感。",
    "委屈": "他觉得委屈。先站在他这边，语气软下来，别评理。",
    "平静": "",
}


def _tone_note(emotion: str) -> str:
    return _TONE.get((emotion or "").strip(), "")


#: 说话的基调，每一轮都带。跟 _TONE 拼起来就是这一轮给 TTS 的完整指示。
_SPEAK_BASE = os.environ.get("VOICEMEM_SPEAK_BASE", "像一个老朋友一样讲话。")


def _speak_instruction(emotion: str) -> str:
    """这一轮怎么念，直接给 TTS。

    _TONE 那 13 条本来就全是**发声指示**（"语速放慢""声音压低""语调扬上去"），
    以前拼在文本 prompt 里，等于让文字模型先理解一段发声描述、再指望 TTS 从
    字面上猜出来，中间隔了两层，实测基本没效果。TTS 的 instruction 参数就是
    收这个的（Breeze 有，gpt-4o-mini-tts 也有），直接送过去。
    """
    tone = _tone_note(emotion)
    return f"{_SPEAK_BASE}{tone}" if tone else _SPEAK_BASE


# ── 短期对话历史 ──────────────────────────────────────────────────────────────
#: 回复模型是**无状态**的：reply.py 每轮只发 system + 用户这一句，没有前几轮。
#: 长期记忆管的是"关于这个人的事实"，管不了"我们刚才在聊什么"——于是聊完一个
#: 话题你说一句"ok ok"，它看到的就是一个孤零零的"ok ok"，重新打招呼
#: （"你好呀？有什么事我可以帮你的吗"）。这两种上下文缺一不可。
#: 按空间分开存：切换 Memory Space 等于换一个人，历史不能串。
#: 只在内存里，进程重启就没了——这是**短期**上下文，本来也不该落盘。
_HISTORY: dict = {}
_HISTORY_TURNS = int(os.environ.get("VOICEMEM_HISTORY_TURNS", "6"))
#: 每句最多带这么多字进 prompt。回复有时很长，全塞进去会把记忆挤到后面。
_HISTORY_CHARS = int(os.environ.get("VOICEMEM_HISTORY_CHARS", "200"))


def _history_block(space: str) -> str:
    turns = _HISTORY.get(space) or []
    if not turns:
        return ""
    lines = ["刚才的对话（最后一条离现在最近）："]
    for u, a in turns:
        lines.append(f"用户：{u}")
        lines.append(f"你：{a}")
    return "\n".join(lines)


def _push_history(space: str, user_text: str, reply_text: str) -> None:
    from collections import deque
    dq = _HISTORY.get(space)
    if dq is None:
        dq = _HISTORY[space] = deque(maxlen=_HISTORY_TURNS)
    u = (user_text or "").strip()[:_HISTORY_CHARS]
    a = (reply_text or "").strip()[:_HISTORY_CHARS]
    if u or a:
        dq.append((u, a))


def _realtime_instructions(memory_context: str, stranger: bool = False,
                           replay: bool = False, emotion: str = "",
                           text: str = "") -> str:
    """人设 + 这一轮检索到的记忆。要说清楚这是「你记得的事」，否则模型会把它当成
    背景资料念出来，而不是当成自己对这个用户的记忆自然地用。

    ``text``：用户这一轮说的话。只用来判断"他是不是在找一段录音而我们没找到"——
    那种情况要明说没找到，否则模型会顺口答"马上播放"然后什么都不放。"""
    if stranger:
        out = f"{_RT_PERSONA}\n\n{_STRANGER}"
        return f"{out}\n\n{_lang_note()}" if _lang_note() else out
    parts = [_RT_PERSONA]
    if memory_context:
        parts.append(memory_context)
    else:
        parts.append(_NO_MEMORY_NOTE)      # 一条都没检索到：明说不知道，别编
    if _lang_note():
        parts.append(_lang_note())
    tone = _tone_note(emotion)
    if tone:
        parts.append("他此刻的状态：" + tone)
    if replay:
        parts.append(_REPLAY_NOTE)
    elif _wants_sound(text):
        parts.append(_NO_REPLAY_NOTE)
    return "\n\n".join(parts)


# ══════════════════ 统一配置入口：一个 dict 配齐所有本地/api 模型 ══════════════════
# 打开这个 dict 就知道每个模型走本地还是 api。记忆侧（embedding/slots）走本地 E5
# → 整条 search 0 LLM、0 网络（实测 Search 本体 ~10ms）；reply 段（回复用的
# llm/tts/realtime）也在这一处配，省得分散在各处 env。缺省项走内置默认。
# 想外挂一份自定义 config：--config path.json（或 VOICEMEM_CONFIG）整体覆盖。
CONFIG = {
    "mode": "multi_modal",
    "memory_root": ARGS.memory_root or None,
    "space": ARGS.space,
    "embedding": {"provider": "local"},              # 记忆向量走本地 E5（0 网络）
    "slots":     {"provider": "local"},              # slot 分类走本地 E5（0 LLM）
    # reply：回复用模型（核心不管，web 读）。默认全走 OpenAI api。
    "reply": {
        "llm":      {"provider": "openai", "config": {"model": utils.CHAT_MODEL,
                                                      "system": _RT_PERSONA}},
        "tts":      {"provider": utils.TTS_BACKEND, "config": {"model": utils.TTS_MODEL}},
        "realtime": {"provider": "openai", "config": {"model": utils.RT_MODEL}},
    },
}

# --config / VOICEMEM_CONFIG 指向的 json 整体覆盖上面的 CONFIG（一个文件配齐）。
if ARGS.config:
    CONFIG = json.loads(Path(ARGS.config).read_text(encoding="utf-8"))

REPLY = CONFIG.get("reply")                           # 传给 utils 的回复函数

# 声明式构造：from_config 是现有注入机制之上的糖（VoiceMem(embedding=fn, schema=fn,…)）。
#: 每个 Memory Space 一个 VoiceMem 实例，按需建、建好留着。
#:
#: 同一进程内建第二个实例几乎不花钱：模型是懒加载 + 进程内复用的，实测建实例
#: 0.0s、预热 2.5s（第一个是 6.8s + 4.9s）。所以切换空间不用重启服务。
_SPACES: dict = {}


def space_dir(name: str):
    """这个空间在磁盘上的目录。名字只允许字母数字和 - _，避免路径穿越。"""
    import re as _re
    safe = _re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]", "", (name or "").strip())[:32]
    if not safe:
        raise ValueError("空间名字不能为空")
    return _ROOT / "voicemem_memoryspace" / safe, safe


def get_space(name: str):
    """取（必要时创建）这个空间的 VoiceMem。"""
    _, safe = space_dir(name)
    if safe not in _SPACES:
        cfg = dict(CONFIG)
        cfg["space"] = safe
        t0 = time.monotonic()
        inst = VoiceMem.from_config(cfg)
        inst.warmup(verbose=False)
        _SPACES[safe] = inst
        print(f"[space] 打开「{safe}」用了 {time.monotonic()-t0:.1f}s", flush=True)
    return _SPACES[safe]


def use_space(name: str) -> str:
    """切到这个空间。返回真正用的名字。

    ``vm`` 是模块级全局，下游全部按名字在运行时查找，所以这里重新绑定就够了——
    不用把实例一路传下去。注意 build_app 收的那几个回调必须是 lambda 而不是
    ``vm.classify`` 这种绑定方法，绑定方法会把切换前那个实例焊死。
    """
    global vm, ACTIVE_SPACE
    vm = get_space(name)
    _, ACTIVE_SPACE = space_dir(name)
    return ACTIVE_SPACE


def list_spaces() -> list:
    """磁盘上有哪些 Memory Space，各有多少条记忆。"""
    import sqlite3
    root = _ROOT / "voicemem_memoryspace"
    out = []
    for d in sorted(p for p in root.glob("*") if p.is_dir()):
        n = 0
        try:
            from voicemem.utils.common import space as _sp
            db = _sp.db(d)
            if Path(db).exists():
                c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                n = c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                c.close()
        except Exception:
            n = 0
        out.append({"id": d.name, "name": d.name, "count": n,
                    "active": d.name == ACTIVE_SPACE, "open": d.name in _SPACES})
    return out


def create_space(name: str) -> dict:
    """建一个新的空 Memory Space：磁盘上出现这个名字的文件夹，里面是全新的空库。

    实例这里就建出来（顺带预热），这样点完"创建"立刻就能对话，不用等第一句话
    卡在模型加载上。
    """
    d, safe = space_dir(name)
    if d.exists() and any(d.iterdir()):
        raise FileExistsError(f"「{safe}」已经存在了")
    d.mkdir(parents=True, exist_ok=True)
    get_space(safe)                      # 建库 + 预热
    print(f"[space] 新建「{safe}」→ {d}", flush=True)
    return {"id": safe, "name": safe, "count": 0}


ACTIVE_SPACE = ""
vm = None
use_space(ARGS.space)


#: 语音轮的音频落在这儿。归档表存的是路径，文件本身得真的在。
TURN_AUDIO_DIR = _ROOT / "results" / "turn_audio"


def save_turn_audio(pcm16k) -> str:
    """把这一轮的 PCM 存成 wav，返回路径；存不下就返回 ""（不影响这一轮对话）。

    没有这一步，AudioArchive 里一条记录都不会有——它只在 ingest 收到 audio_path
    时才写。之前 demo 全程走 WS 流、从不落盘，所以"把当时那段原声放回来"做不到。
    """
    if pcm16k is None or not len(pcm16k):
        return ""
    try:
        import numpy as np
        import soundfile as sf
        TURN_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        path = TURN_AUDIO_DIR / f"turn_{uuid.uuid4().hex[:12]}.wav"
        sf.write(path, np.asarray(pcm16k, dtype="float32"), 16000)
        return str(path)
    except Exception as e:
        print(f"[web] 存本轮音频失败（不影响对话）：{e}", flush=True)
        return ""


@dataclass
class Pending:
    """一轮说完时、投机预取早已算好的「预算记忆」——控制流拿来直接回复，不再搜。"""
    text: str
    memory_context: str
    result: object
    spoken: bool = True          # True=语音轮（音频已进 realtime 缓冲），False=打字轮
    audio_path: str = ""         # 这一轮落盘的 wav；ingest 拿它做场景/音乐/声纹感知，
                                 # 并在 audio_archive 里跟记忆绑定，之后能原样放回来
    stranger: bool = False       # 声纹认出说话的不是这个记忆库的主人
    replay: str = ""             # 该把哪条记忆当时那段原声放回来（memory_id），空=不放
    emotion: str = ""            # 上一轮感知到的情绪，用来给这一轮定语气


# ══════════════════ 两条控制流（各 ~10 行，只消费预取好的 Pending）══════════════════

# 分句规则（决定多久出第一声）在 voicemem/tts.py 的 cut_point。
_cut_point = utils.cut_point


def _mentioned(name: str, text: str) -> bool:
    """实体名在这句话里被提到了吗。

    不能直接 `name in text`——图里的实体常带限定词（"Jiaqi的老板"），而人说的是
    "老板"。拆成词块（按"的"和非中文断开）逐个比，任一块出现就算提到。
    单字块不算：一个"我"、一个"歌"太容易误命中。
    """
    t = text or ""
    if not name:
        return False
    if name in t:
        return True
    for chunk in re.split(r"[的\s·、,，]+|[^\u4e00-\u9fffA-Za-z0-9]+", name):
        if len(chunk) >= 2 and chunk in t:
            return True
    return False


#: 声学模型至少要这么有把握才采纳（emotion2vec+ 的 softmax 分数）。
ACOUSTIC_MIN_SCORE = float(os.environ.get("VOICEMEM_ACOUSTIC_MIN", "0.92"))

#: 只有这几类才听声学的。
#:
#: 实测 emotion2vec+ 在真实录音上会**自信地判错**：「哎呀，早上好呀」判悲伤
#: 1.00、「我觉得挺好的」判悲伤 1.00——单靠提高阈值挡不住，它对错的答案给的
#: 就是满分。但它错的时候几乎全错在"悲伤"（对这个说话人的默认倾向）。
#: 高唤起的那几类（真笑出声、真发火）声学特征明显，恰恰是文本看不出来的，
#: 留给它；低唤起的交给语义。
ACOUSTIC_TRUST = set((os.environ.get("VOICEMEM_ACOUSTIC_TRUST")
                      or "开心,委屈,惊讶").split(","))

#: 情绪的语义样例。用本地 E5 把这句话跟这些比相似度——比关键词表宽
#: （"我觉得挺好的"里一个情绪词都没有），比声学准得多。0 网络、9ms。
_EMO_PROTO = {
    "开心": ["我今天特别开心", "太好了我很高兴", "真不错，我挺满意的", "哈哈太有意思了"],
    "悲伤": ["我很难过", "我心里特别难受", "我好失落", "这事让我挺沮丧的"],
    "委屈": ["我好生气", "太气人了", "凭什么这样对我", "我觉得很不公平"],
    "焦虑": ["我压力好大", "我有点紧张", "我很担心做不完", "这事儿让我睡不着"],
    "疲惫": ["我好累啊", "累死了，撑不住了", "一天下来人都空了"],
    # 「平静」要多给几句：日常陈述句和疑问句占了对话的大半，样例太少时它们
    # 撑不出足够的差距，就一路判成"(空)"——实测「我对花生过敏」「我不能吃什么」
    # 都因为这个没标上，加了样例之后 gap 从 0.002 涨到 0.08。
    "平静": ["今天天气不错", "我明天要去开会", "早上好", "我叫小明",
             "这个东西放在桌上", "我对花生过敏", "我在一家公司上班",
             "下周三下午三点有个会", "我不能吃什么", "这个怎么用",
             "帮我看一下", "我住在市中心"],
}
_PROTO = {}


def _emotion_by_meaning(text: str) -> str:
    """语义最近邻。够像**而且**跟第二名拉开差距才给标签，否则空——
    模棱两可时不标，比标错强。"""
    import numpy as np
    if not _PROTO:
        labels, sents = [], []
        for k, vs in _EMO_PROTO.items():
            labels += [k] * len(vs)
            sents += vs
        _PROTO["labels"] = labels
        _PROTO["V"] = np.array(utils.shared_e5().encode(sents, normalize_embeddings=True),
                               dtype=np.float32)
    q = np.array(utils.shared_e5().encode([text], normalize_embeddings=True),
                 dtype=np.float32)[0]
    sims = _PROTO["V"] @ q
    i = int(np.argmax(sims))
    lab, best = _PROTO["labels"][i], float(sims[i])
    other = [float(sims[j]) for j in range(len(sims)) if _PROTO["labels"][j] != lab]
    gap = best - (max(other) if other else 0.0)
    return lab if best >= 0.80 and gap >= 0.02 else ""


#: emotion2vec+ 的英文标签 → 我们用的中文标签
_E2V_MAP = {"happy": "开心", "sad": "悲伤", "angry": "委屈", "fearful": "恐惧",
            "surprised": "惊讶", "disgusted": "厌恶", "neutral": ""}
_E2V = {}


def _acoustic_emotion(audio_path: str):
    """emotion2vec+（专门做语音情绪识别的模型）。返回 (标签, 分数)。"""
    if "m" not in _E2V:
        from funasr import AutoModel
        _E2V["m"] = AutoModel(model=os.environ.get("VOICEMEM_E2V_MODEL",
                                                   "emotion2vec/emotion2vec_plus_base"),
                              hub="hf", disable_update=True)
    r = _E2V["m"].generate(audio_path, granularity="utterance", extract_embedding=False)
    if not r:
        return "", 0.0
    lab, score = max(zip(r[0]["labels"], r[0]["scores"]), key=lambda x: x[1])
    en = str(lab).split("/")[-1].strip().lower()
    return _E2V_MAP.get(en, ""), float(score)


_SV = {}


def _sensevoice():
    """SenseVoiceSmall，一次推理同时出精转写和声学情绪。懒加载、进程内复用。

    注意别用 vm.utils.get("emotion")——那是韵律象限的启发式
    （PaperAlignedEmotionDetector），没有 run_with_emotion，而且判得不准。
    """
    if "t" not in _SV:
        from voicemem.utils.audio.asr import Transcriber, pick_device
        _SV["t"] = Transcriber(pick_device())
    return _SV["t"]


def _kick_acoustic(send, audio_path: str) -> None:
    """把声学情绪扔到后台算，算出可信结果再补一条 tag_update。

    emotion2vec 要跑整段音频，实测 2.3 秒。放在发 memory_hits 之前就等于把这
    2.3 秒加在"用户说完 → 助手开口"中间，而它给的结果十有八九还够不上信任阈值。
    放后台之后热路径一秒都不欠，真判准了 UI 上的情绪标签照样会更新。
    """
    if not audio_path or os.environ.get("VOICEMEM_ACOUSTIC_TAG", "1") == "0":
        return

    async def run():
        try:
            t0 = time.monotonic()
            emo, score = await asyncio.to_thread(_acoustic_emotion, audio_path)
            take = bool(emo) and score >= ACOUSTIC_MIN_SCORE and emo in ACOUSTIC_TRUST
            if BARGE_DEBUG:
                print(f"  [emotion] 声学(后台) {(time.monotonic()-t0)*1000:.0f}ms "
                      f"-> {emo or '-'} {score:.2f}（{'采纳' if take else '不采纳'}）", flush=True)
            if take:
                await send({"type": "tag_update", "emotion": emo, "emotion_from": "acoustic"})
        except Exception as e:
            print(f"[web] 后台声学情绪跳过：{type(e).__name__}: {e}", flush=True)

    asyncio.create_task(run())


def fill_tags(payload: dict, text: str, audio_path: str = "",
              acoustic: bool = True) -> dict:
    """补上标签栏要的 emotion / entities——两样都是 0 LLM、0 网络。

    检索走的是本地 slot 分类器（投机预算内不能联网），它只出 slot，不出实体；
    情绪则要等这一轮 ingest 之后才算得出，而 memory_hits 是在回复之前就发的。
    结果就是标签栏上这两格一直空着。

    · 情绪：用 anchor_router 的关键词表现算一次（纯查表）。
    · 实体：这一轮命中的那几条记忆在认知图里挂了哪些实体，直接读（纯 sqlite）。
    """
    # ⓪ 右脑那几条换成第一人称的人话（只换显示，raw 原文照旧留着给脑图匹配）。
    #    还没改写好的这一轮先显示原文，同时排进后台队列——见 rb_human。
    for h in payload.get("right_brain_hits") or []:
        claim = h.get("claim") or ""
        if not claim:
            continue
        human = rb_human(claim)
        if human and human != claim:
            # 证据那半截（"｜他说过：…"）保留，它才是"你凭什么这么说"的支撑。
            tail = ""
            for sep in ("｜他说过：", " | he said: "):
                if sep in h.get("content", ""):
                    tail = sep + h["content"].split(sep, 1)[1]
                    break
            h["content"] = human + tail

    # ① 人明说了情绪就按他说的（查表，0 网络）——最准
    if not payload.get("emotion") and text.strip():
        try:
            from voicemem.rightbrain.anchor_router import normalize_emotion_strict
            payload["emotion"] = normalize_emotion_strict(text) or ""
        except Exception:
            pass

    # ② 没明说就看这句话的**意思**：本地 E5 跟每种情绪的样例句比语义相似度。
    #    9ms、0 网络，而且比关键词表宽（"我觉得挺好的"里一个情绪词都没有）。
    if not payload.get("emotion") and text.strip():
        try:
            emo = _emotion_by_meaning(text)
            if emo:
                payload["emotion"] = emo
                payload["emotion_from"] = "semantic"
        except Exception as e:
            print(f"[web] 语义情绪跳过：{type(e).__name__}: {e}", flush=True)

    # ③ 声学（emotion2vec+）：只在它**很有把握**时才盖过上面的判断。
    #
    #    为什么不让它当主力：实测在真实录音上它把「哎呀，早上好呀」判成难过
    #    （3.2 秒干净音频，不是静音太长的锅——裁剪过照样如此），SenseVoice 和
    #    韵律启发式也一样。这个麦克风/说话方式下，声学读不准。
    #    所以留着它，但要求 score ≥ ACOUSTIC_MIN_SCORE 才作数——真正带情绪地
    #    说话时它会给 0.95+，平淡说话时给的是 0.6、0.7 那种，正好挡掉。
    if acoustic and audio_path and os.environ.get("VOICEMEM_ACOUSTIC_TAG", "1") != "0":
        try:
            t0 = time.monotonic()
            emo, score = _acoustic_emotion(audio_path)
            take = bool(emo) and score >= ACOUSTIC_MIN_SCORE and emo in ACOUSTIC_TRUST
            if take:
                payload["emotion"] = emo
                payload["emotion_from"] = "acoustic"
            if BARGE_DEBUG:
                why = "采纳" if take else ("把握不够" if score < ACOUSTIC_MIN_SCORE
                                          else f"{emo} 不在信任名单")
                print(f"[emotion] 声学 {(time.monotonic()-t0)*1000:.0f}ms "
                      f"-> {emo or '-'} {score:.2f}（{why}）"
                      f"  最终={payload.get('emotion') or '-'}", flush=True)
        except Exception as e:
            print(f"[web] 声学情绪跳过：{type(e).__name__}: {e}", flush=True)

    # 实体：这里**不猜**。
    #
    # memory_hits 是在回复之前发的，而实体是跟抽事实同一次 LLM 调用出来的
    # （ingest 时，回复之后）。曾经在这儿做过一个"0 成本近似"——从命中的旧记忆
    # 里挑这句话提到过的实体——结果每句话都只剩说话人自己：
    #     说「我对花生过敏」→ 标签栏 ['小林']，真正抽出来的是 ['小林','花生']
    #     说「下周三在国金中心见客户」→ 标签栏空
    # 显示一个错的比先空着更糟。前端会在 ingest 落库后用真结果补上（见
    # voicemem.html 的 loadMemories）。

    rb = payload.get("right_brain_hits") or []
    inner = sum(1 for h in rb if h.get("internal"))
    print(f"[hits] 左脑 {len(payload.get('left_brain') or [])} 条  "
          f"右脑 {len(rb)} 条(内部 {inner}，页面显示 {len(rb)-inner})  "
          f"情绪={payload.get('emotion') or '-'}  "
          f"实体={'、'.join(payload.get('entities') or []) or '-'}", flush=True)
    return payload


async def voicemem_llm_tts(pending, send, send_audio, owner, said=None):
    """记忆已在关键路径外预取好：LLM 流式回复 → TTS 流式语音。

    TTS 跟生成**并行**：LLM 吐满一句就丢进队列，另一条协程取出来合成、发音频。
    等全文生成完再开始合成的话，文本早打完了、音频还没起头（实测 TTS 首帧就要
    ~1.2s，加上生成那几秒，用户看着字干等）。

    ``said``：可选的 dict，边生成边把已说出的文本写进 said["text"]。打断判定要拿
    它挡回声（见 _is_echo）——助手说的话经麦克风绕回 ASR，转出来的字跟真人插话
    在字数上没区别，只能靠内容认。
    """
    await send({"type": "user_transcript", "text": pending.text})
    # 声学情绪**不在这儿算**。它要 2.3 秒（emotion2vec 跑整段音频），而这几行是
    # 用户说完到助手开口之间最要紧的一段——实测这一步就吃掉了 4.5 秒里的一半，
    # 算完还常常因为"把握不够"被丢掉，纯浪费。
    # 先用文本语义那份（毫秒级）把标签发出去，声学放后台跑，可信了再补一条
    # tag_update 覆盖 UI 上的情绪。
    note_hits(pending.result)      # 让脑图快照保证这几条在图上
    await send({"type": "memory_hits",
                **fill_tags(utils.hits_payload(pending.result, has_audio=audio_of,
                                              cluster_of=hit_cluster),
                            pending.text, pending.audio_path or "", acoustic=False)})
    _kick_acoustic(send, pending.audio_path or "")
    if pending.replay:
        _note_replay(pending.replay)
        await send({"type": "play_memory", "memory_id": pending.replay})
    await send({"type": "answer_start"})

    queue: asyncio.Queue = asyncio.Queue()

    # 走注入的那个 TTS（第九个可替换位）。--config 里换 provider、或库用户
    # VoiceMem(tts=lambda: MyTTS()) 传自己的实现，都在这儿生效；没配就是内置默认。
    tts = vm.utils.get("tts")
    # 这一轮怎么念。情绪是逐轮变的，所以按轮传，不写在实例上。
    speak_as = _speak_instruction(pending.emotion)

    def _synth_one(seg):
        """注入的 TTS 可能是用户自己写的、只认 stream(text)——那就退回去，
        少一层语气控制而已，不该因此整条链路报错。"""
        try:
            return tts.stream(seg, speak_as)
        except TypeError:
            return tts.stream(seg)

    # 一段回复被切成好几句、逐句合成，段与段之间会空一拍：合成完这段才发下一段的
    # 请求，中间要等「发请求 → 服务端 prefill → 第一个字节回来」，音频队列在这期间
    # 是空的，听感就是每句开头卡一下。TTS 在远端时（比如 Breeze 跑在 GPU 机器上，
    # 还隔着 SSH 隧道）这一拍尤其明显。
    # 所以拆成两级：拿到一段就**立刻**开合成、各自往自己的小队列里灌；播放那边按
    # 顺序一段段取。这样当前这段还在播的时候，下一段已经在算了。
    # 不限制并发：服务端是不是单并发由它自己决定（Breeze 就是），这边多发几个请求
    # 只是让它排上队，省掉每段一个来回的网络延迟。
    synths: list[asyncio.Task] = []
    streams: asyncio.Queue = asyncio.Queue()      # 每项是一段的 chunk 队列

    async def synth():
        while (seg := await queue.get()) is not None:
            chunks: asyncio.Queue = asyncio.Queue()

            async def run(seg=seg, chunks=chunks):
                try:
                    async for pcm in _synth_one(seg):
                        await chunks.put(pcm)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    print(f"[web] 合成失败：{type(e).__name__}: {e}", flush=True)
                finally:
                    await chunks.put(None)        # 出错也要让播放那边收工

            synths.append(asyncio.create_task(run()))
            await streams.put((seg, chunks))
        await streams.put(None)

    async def speak():
        while (item := await streams.get()) is not None:
            seg, chunks = item
            # 回声判定要拿**用户可能听到的**去比，不是已生成的——生成早跑到几段
            # 之后了。这里是音频真正开始发出去的时刻，最接近"说出口"。
            if said is not None:
                said["text"] = (said.get("text") or "") + seg
            try:
                while (pcm := await chunks.get()) is not None:
                    await send_audio(pcm)
            except asyncio.CancelledError:
                raise
            except Exception as e:                # 多半是听到一半关了页面，不是错误
                print(f"[web] 语音发送中断：{type(e).__name__}", flush=True)
                break

    synther = asyncio.create_task(synth())
    speaker = asyncio.create_task(speak())
    reply, buf, sent = "", "", 0
    interrupted = False
    try:
        # 跟 realtime 用同一份指令：两条路必须表现一致，否则换个 --mode
        # 人设和「右脑不许念出来」的约束就悄悄没了。
        # 走核心回复层（人设在 CONFIG.reply.llm.config.system，见 voicemem/reply.py
        # 的 compose_system：system + memory_context，和 realtime 那条拼出来的一样）。
        ctx = _STRANGER if pending.stranger else pending.memory_context
        if not pending.stranger and not (ctx or "").strip():
            ctx = _NO_MEMORY_NOTE          # 一条都没检索到：明说不知道，别编
        # 情绪不再拼进文本 prompt：那是**发声指示**（"压低、放软、留停顿"），
        # 让文字模型理解一遍再指望 TTS 猜出来，中间隔了两层。TTS 后端的 instruction
        # 参数就是收这个的，该搬过去。搬之前 pending.emotion 这一路暂时没有出口。
        note = (_REPLAY_NOTE if pending.replay
                else (_NO_REPLAY_NOTE if _wants_sound(pending.text) else ""))
        if note:
            ctx = f"{ctx}\n\n{note}" if ctx else note
        hist = _history_block(ACTIVE_SPACE)
        if hist:
            ctx = f"{ctx}\n\n{hist}" if ctx else hist
        if _lang_note():
            ctx = f"{ctx}\n\n{_lang_note()}" if ctx else _lang_note()
        async for d in vm.reply_stream(pending.text, ctx):
            reply += d
            buf += d
            await send({"type": "answer_delta", "text": d})
            if _cut_point(buf, first=sent == 0):
                await queue.put(buf.strip())
                buf, sent = "", sent + 1
        if buf.strip():
            await queue.put(buf.strip())
    except asyncio.CancelledError:
        interrupted = True                      # 用户插话了，这一轮到此为止
    finally:
        await queue.put(None)                   # 生成出错也要让 speak() 收工

    if interrupted:
        # 别把 speak() 留在后台继续往一条已经停播的连接上发音频。
        # 提前起跑的那几段合成也要一起停，否则它们会继续占着远端 TTS 的队列，
        # 下一轮的第一句得排在这些没人要的音频后面——听起来就是打断之后更卡。
        speaker.cancel()
        synther.cancel()
        for t in synths:
            t.cancel()
    else:
        await synther
        await speaker
        await send({"type": "answer_done"})

    # 存这一轮：被打断时存的是用户真正听到的那半句。
    # 先落记忆再收工——ingest 排在音频后面的话，用户一听完就关页面（语音场景很
    # 常见），这一轮就永远存不进去。async_facts=True：抽事实走后台。
    _push_history(ACTIVE_SPACE, pending.text, reply)
    remember_turn(pending, reply, owner)



async def start_realtime_turn(pending, conn, send):
    """把预取好的记忆注入 Realtime session，触发这一轮的原生语音。

    只负责"发起"；收音频/文本和收尾都在常驻的事件泵里（见 realtime_session）——
    OpenAI 的事件流只能有一个消费者，每轮各读各的会串台：上一轮被打断后残留的
    response.done 会被下一轮读到，当成自己说完了。
    """
    await send({"type": "user_transcript", "text": pending.text})
    note_hits(pending.result)      # 让脑图快照保证这几条在图上
    await send({"type": "memory_hits",
                **fill_tags(utils.hits_payload(pending.result, has_audio=audio_of,
                                              cluster_of=hit_cluster),
                            pending.text, pending.audio_path or "", acoustic=False)})
    _kick_acoustic(send, pending.audio_path or "")
    if pending.replay:
        # 前端收下先记着，等这一轮回复播完再放——助手的回复是排队播的，
        # 提前放会跟人声叠在一起。
        _note_replay(pending.replay)
        await send({"type": "play_memory", "memory_id": pending.replay})
    if pending.spoken:
        await conn.input_audio_buffer.commit()
    else:
        await conn.conversation.item.create(item={"type": "message", "role": "user",
                                                  "content": [{"type": "input_text", "text": pending.text}]})
    # 记忆走 response.create 的 per-response instructions，不是 session.update。
    # 后者是会话级设置，实测更新完模型这一轮根本读不到（问"我的猫叫什么"，库里
    # 明明检索到了"叫墨墨"，模型还答"你刚提过但我没听清"）。
    print(f"[lat] 本地判完说完 → 发 response.create", flush=True)
    await conn.response.create(response={
        "instructions": _realtime_instructions(pending.memory_context, pending.stranger,
                                               replay=bool(pending.replay),
                                               emotion=pending.emotion,
                                               text=pending.text),
    })
    await send({"type": "answer_start"})


async def _no_realtime(sock, err):
    """连不上 Realtime 时别让人对着 traceback 猜。

    要分清是**网络**还是**权限**：DNS/连接失败跟 key 没有关系，之前一律说成
    「key 可能没权限」，把人往错的方向指了。
    """
    name, text = type(err).__name__, str(err)
    network = (isinstance(err, (OSError, TimeoutError, ConnectionError))
               or "gaierror" in name.lower()
               or any(k in text.lower() for k in ("nodename", "temporary failure",
                                                  "name or service", "getaddrinfo",
                                                  "connection refused", "timed out")))
    if network:
        why = ("网络连不上 api.openai.com（DNS/代理/VPN 的问题，跟 key 无关）。"
               "确认能上网后重开；离线环境用 `--mode llm_tts` 也一样连不上，"
               "两条路都要访问 OpenAI。")
    elif any(k in text for k in ("401", "403", "invalid_api_key", "insufficient", "model_not_found")):
        why = ("这个 key 没有 Realtime 权限或模型不可用——改用 "
               "`python web/run.py --mode llm_tts`，那条路只要普通 chat + TTS。")
    else:
        why = ("先看这条报错本身；如果只是 Realtime 用不了，可以改用 "
               "`python web/run.py --mode llm_tts`（普通 chat + TTS）。")
    msg = f"连不上 OpenAI Realtime（{name}: {text}）。{why}"
    print(f"[web] {msg}", flush=True)
    try:
        await sock.send_json({"type": "error", "message": msg})
    except Exception:
        pass


# ══════════════════ 驱动 voicemem 核心流式会话（vm.stream()）══════════════════
# ASR + VAD + 投机预取（边说边预取 / 200ms 赌说完 / barge-in / 300ms 确认）
# 全在核心 VoiceStream 里。这里只做 demo 该做的：搬 socket 帧、发 partial、把说完
# 的一轮包成 Pending 交给控制流——demo 就是核心的使用示例，不再平行重写一套。

def remember_turn(pending, reply: str, owner: dict) -> None:
    """存这一轮，并顺手记下说话人是谁。

    说话人不用单独算：ingest 内部本来就要跑一次 preprocess（场景/声纹/情绪），
    返回值里直接带 speaker_id。之前我在热路径上又单独触发了一次完整感知——
    那套一次 424ms（AST 占 361ms），纯属重复劳动，而且挡在读 socket 前面。
    """
    # 没有转写文本的一轮 = 对着麦克风放了段声音（见 stream.py 的 MIN_SOUND_ONLY_S）。
    # 直接 ingest("") 抽不出任何事实，这段音频就白存了——之后问「刚才那首歌帮我
    # 重播」找不到任何记忆。给它一句话，让它能被检索到；音乐是什么由 ingest 内部
    # 的音乐识别打 tune: 标签，这里只负责让它有个记忆载体。
    # 判「有没有真的说话」不能只看空不空：音乐喂进 ASR 会硬转出一两个字母
    # （实测 cafe_song 转成 'i'），非空但毫无意义。要求至少两个中文字或三个
    # 英文字母才算说了话。
    text = pending.text
    meaningful = re.sub(r"[^\w\u4e00-\u9fff]", "", text or "")
    cjk = len(re.findall(r"[\u4e00-\u9fff]", meaningful))
    if pending.audio_path and cjk < 2 and len(meaningful) < 3:
        from voicemem.stream import SOUND_ONLY_TEXT
        text = SOUND_ONLY_TEXT          # 必须用这个常量，核心靠它认出没说话的那轮

    try:
        r = vm.ingest(text, agent_reply=reply, async_facts=True,
                      audio=pending.audio_path or None) or {}
    except Exception as e:
        print(f"[web] 存这一轮失败：{type(e).__name__}: {e}", flush=True)
        return
    # 上一轮的情绪留给下一轮的投机检索用。没有它右脑取不到情感记录，
    # 每轮只会返回同样那几条静态画像（见 voicemem/stream.py 的 emotion 说明）。
    affect = r.get("affect")
    if isinstance(affect, dict):
        affect = affect.get("emotion") or affect.get("label") or ""
    owner["emotion"] = str(affect or "").strip()

    # 这一轮听到音乐了就记住那段录音。tune 识别是同步的，这里拿到的时候后台
    # 入库才刚开始——"刚听完就问"能不能放出来，全靠这一句。
    if r.get("recognized_tune") and pending.audio_path:
        _remember_tune(pending.audio_path)
    elif BARGE_DEBUG and _wants_sound(pending.text or ""):
        print(f"  [replay] 这一轮没记住音乐："
              f"tune={bool(r.get('recognized_tune'))} audio={bool(pending.audio_path)}",
              flush=True)

    sid = r.get("speaker_id") or ""
    if not sid:
        # 这一轮太短，声纹压根没算（见 perceiver 的 VOICEMEM_SPEAKER_MIN_S）。
        # 不知道是谁 ≠ 换了个人，所以连 miss 计数都不动。
        return
    if not owner["id"]:
        owner["id"] = sid                  # 第一个开口的算这场对话的主人
    owner["last"] = sid
    owner["miss"] = 0 if sid == owner["id"] else owner.get("miss", 0) + 1


#: 比到多久以前。原来是 40 字——那是按「生成到哪儿就说到哪儿」估的，可生成比
#: 播放快得多（尤其现在会提前合成下一段），用户此刻听到的往往是好几秒前生成的
#: 内容，早滑出 40 字窗口了。改成按**已经说出口的**文本比，窗口也放宽。
ECHO_WINDOW = int(os.environ.get("VOICEMEM_ECHO_WINDOW", "300"))
#: 模糊匹配门槛：新出的字里，**连续**命中助手原话的那一段最长能占多大比例。
#:
#: 一开始用的是二元组重合率——那是错的：它不看连续性，用户插话只要词汇跟刚才
#: 聊的重合（"压力""GRE"这种，非常常见），散落的二元组就能凑过门槛，真插话被
#: 当回声吞掉，结果就是打断失灵。
#: 回声的特征是**一整段连续的原话**，真插话哪怕用词重合也接不成长串，所以改用
#: 最长公共子串。ASR 差一两个字只会把长串截短一点，仍然远高于真插话。
ECHO_RATIO = float(os.environ.get("VOICEMEM_ECHO_RATIO", "0.6"))
#: 短于这个长度只做精确匹配。两三个字的二元组太少，重合率动不动就是 1.0——
#: 用户跟着复述一个词（"GRE？"）就会被当成回声吞掉，那是真插话。
ECHO_FUZZY_MIN = int(os.environ.get("VOICEMEM_ECHO_FUZZY_MIN", "4"))


def _is_echo(new_chars: str, said: str) -> bool:
    """ASR 新吐出的这几个字，是不是助手自己的声音绕回麦克风了。

    AEC 压不干净时，助手说的话会进 ASR，转出来的字跟真人插话在字数上没有区别。
    但内容上有：回声一定是助手**刚说过的原文**里的片段。

    ``said`` 要传**已经说出口的**文本（不是已生成的），两者能差好几秒。
    大小写要抹平——最早那个漏网的例子就是助手说 "Annie"、ASR 转出小写 "an"。
    """
    s = "".join(ch for ch in (new_chars or "") if ch.strip()).casefold()
    hay = "".join(ch for ch in (said or "")[-ECHO_WINDOW:] if ch.strip()).casefold()
    if not s or not hay:
        return False
    if s in hay:
        return True
    if len(s) < ECHO_FUZZY_MIN:
        return False                       # 太短，只信精确匹配
    return _lcs_len(s, hay) / len(s) >= ECHO_RATIO


def _lcs_len(a: str, b: str) -> int:
    """最长公共**子串**（连续）长度。滚动一行的 DP，串都很短，开销可忽略。"""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for ch in a:
        cur = [0] * (len(b) + 1)
        for j, cj in enumerate(b, 1):
            if ch == cj:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


async def anticipate(sock, on_frame=None, on_speech=None, owner=None, is_busy=None,
                     said=None):
    """驱动核心流式会话，逐个 yield 确认回合的 Pending。
    on_frame(raw24k)：realtime 用它把原始音频平行喂给 OpenAI（方案 A）。
    on_speech()：本地 VAD 一听到人声就叫一次——realtime 拿它做打断（barge-in）。
    is_busy()：助手此刻是不是在说话。助手的声音会经麦克风回到 ASR，转出来的字
    照样会走 partial_transcript——用户就看见自己的输入框里冒出助手刚说的话。
    所以助手说话期间先不发 partial，等转写真的多出几个字（确认是人在插话，
    见 BARGE_MIN_CHARS）之后再放行。"""
    stream = vm.stream(spec_min_chars=SPEC_MIN_CHARS, gamble_s=GAMBLE_S, confirm_s=CONFIRM_S)
    last_partial = ""
    if owner is None:
        owner = {"id": "", "last": "", "miss": 0}   # 主人的声纹 / 上一轮是谁 / 连续认错几轮
    barge_base = 0                        # 上次触发打断时的转写长度
    barged = False                        # 这一轮是否已确认「人在插话」
    while True:
        msg = await sock.receive()
        if msg.get("type") == "websocket.disconnect":         # 关页面/刷新：收工
            return
        if msg.get("text"):                                   # 打字轮
            data = json.loads(msg["text"])
            if data.get("type") == "user_text" and data.get("text", "").strip():
                stream.emotion = owner.get("emotion") or None
                turn = await stream.feed_text(data["text"])
                yield Pending(turn.text, turn.memory_context, turn.result, spoken=False,
                              replay=_replay_id(turn.text, turn.result),
                              emotion=owner.get("emotion", ""))
            continue
        if msg.get("bytes") is None:
            continue
        raw = msg["bytes"]
        if on_frame:
            await on_frame(raw)                               # 方案 A：音频也进 OpenAI 缓冲
        stream.emotion = owner.get("emotion") or None         # 上一轮算出来的情绪
        st = await stream.feed(raw)                           # 核心：ASR + VAD + 投机预取
        # 打断走「ASR 确认制」：不是听到人声就掐，而是等转写真的多出几个字。
        # 助手的回声进了 ASR 也转不出连贯的新字，咳嗽和关门声更不会——这一条
        # 比任何 VAD 阈值都好使，见 BARGE_MIN_CHARS 上面那段。
        cur = st.text.strip()
        grown = len(cur) - barge_base
        if grown >= BARGE_MIN_CHARS:
            # 新出的这几个字是不是助手自己的回声。挡住它，字数门槛才敢降到 2。
            if said is not None and _is_echo(cur[barge_base:], said()):
                if BARGE_DEBUG:
                    print(f"[barge] {cur[barge_base:]!r} 是助手自己的回声，不算插话", flush=True)
                barge_base = len(cur)
            else:
                barge_base = len(cur)
                barged = True
                if on_speech:
                    if BARGE_DEBUG:
                        print(f"[barge] 转写多出 {grown} 个字 → 请求打断：{cur[-12:]!r}", flush=True)
                    await on_speech()
        # 助手正在说话时，ASR 里多半混着它自己的回声，那些字不能显示——用户会看见
        # 自己的输入框冒出助手刚说的话。
        #
        # 但原来的条件是「忙 且 还没确认插话」，等于**助手一忙就全屏蔽**，一直等到
        # 攒够字数、过完回声判定、`barged` 翻真才放行。用户早说完了屏幕还是空的。
        # 现在 _is_echo 够准了（连续子串比对），直接拿它判这句话本身像不像回声：
        # 像就藏，不像就立刻显示，不必再等那个确认。
        echo = (bool(is_busy and is_busy()) and not barged
                and (said is None or _is_echo(cur, said())))
        if st.text.strip() and st.text != last_partial and not echo:
            last_partial = st.text
            await sock.send_json({"type": "partial_transcript", "text": st.text, "replace": True})
        if st.turn:                                           # VAD 确认说完 → 记忆早已预取好
            # 我们正在放录音，而这一轮一个字都没转出来：那是自己的声音绕回来了。
            if _replaying_now() and not (st.turn.text or "").strip():
                if BARGE_DEBUG:
                    print("[replay] 回放期间的空白一轮，是自己的回声，丢掉", flush=True)
                last_partial = ""
                barge_base = 0
                barged = False
                continue
            last_partial = ""
            barge_base = 0                                    # 新一轮，转写从头开始涨
            barged = False
            # 谁在说话。第一个开口的人算这场对话的主人；之后换了另一个声纹，
            # 就是陌生人——不能把主人的记忆讲给他听（"我是谁？"→"你是Jiaqi"
            # 这个 bug 就是因为检索从不看说话人）。
            # 用上一轮算出来的说话人做判断，这一轮的放后台算。
            # 直接读 st.speaker_id 会同步跑声纹+情绪+场景一整套模型——实测 2.1 秒，
            # 而且是在事件循环里，这期间连 socket 都不读，麦克风帧全堆着，
            # 表现就是"ASR 很卡"。代价是换人之后第一句仍按上一个人算。
            stranger = bool(SPEAKER_GATE and owner["id"]
                            and owner.get("miss", 0) >= STRANGER_MIN_TURNS)
            if stranger or SPEAKER_DEBUG:
                # "他怎么突然不认识我了"——看这一行。声纹把同一个人认成两个
                # person_* 时就会这样：记忆被清空，指令换成"就当第一次见面"。
                print(f"[speaker] owner={owner['id'] or '-'} last={owner['last'] or '-'}"
                      f" miss={owner.get('miss', 0)} stranger={stranger}", flush=True)
            yield Pending(st.turn.text,
                          "" if stranger else st.turn.memory_context,
                          st.turn.result, spoken=True,
                          audio_path=await asyncio.to_thread(
                              save_turn_audio, getattr(st, "_pcm", None)),
                          stranger=stranger,
                          replay="" if stranger else _replay_id(st.turn.text, st.turn.result),
                          emotion=owner.get("emotion", ""))



# ══════════════════ 每种 mode 的会话循环 ══════════════════

async def llm_tts_session(sock):
    """llm_tts 这条路的打断。

    原来是 `async for pending in anticipate(sock): await voicemem_llm_tts(...)`——
    两个问题叠在一起，打断在结构上就不可能：
      · on_speech 没传进 anticipate，本地 VAD 听到人声也没人管；
      · voicemem_llm_tts 最后 `await speaker`，要等全部音频发完才返回。async for
        在这期间不会去拉下一个，anticipate 就停在那儿不读 socket 了，麦克风帧全
        堆在缓冲区里。听感就是"说什么都没用，他非要说完"。

    现在回复丢进后台任务，读 socket 的循环一刻不停；听到人声就取消那个任务。
    """
    turn = {"task": None, "t0": 0.0, "reply": {"text": ""}}
    owner = {"id": "", "last": "", "miss": 0}

    async def stop_reply():
        task = turn["task"]
        if task is None or task.done():
            return
        since = (time.monotonic() - turn["t0"]) * 1000
        if since < BARGE_GRACE_MS:
            if BARGE_DEBUG:
                print(f"[barge] 才说了 {since:.0f}ms，还在宽限期内，不打断", flush=True)
            return
        task.cancel()
        turn["task"] = None
        try:
            await sock.send_json({"type": "answer_interrupt"})   # 前端停播已排队的音频
        except Exception:
            pass

    def replying():
        t = turn["task"]
        return t is not None and not t.done()

    async for pending in anticipate(sock, on_speech=stop_reply, owner=owner,
                                    is_busy=replying, said=lambda: turn["reply"]["text"]):
        await stop_reply()                    # 上一轮还没说完就被新的一轮顶掉
        turn["t0"] = time.monotonic()
        turn["reply"]["text"] = ""            # 新一轮，回声比对从空的开始
        turn["task"] = asyncio.create_task(
            voicemem_llm_tts(pending, sock.send_json, sock.send_bytes, owner,
                             said=turn["reply"]))


async def realtime_session(sock):
    """方案 A：整段麦克风音频平行喂给 OpenAI Realtime；本地 ASR+VAD 只负责投机记忆 +
    用 500ms 判回合（关掉 OpenAI 自带 server_vad）。"""
    connected = False
    try:
        async with utils.realtime_connect(REPLY) as conn:
            # 握手成功不代表能用：没权限/模型名不对时，OpenAI 是**连上之后**再发
            # close 4000（invalid_model / 权限错误）。所以要等第一次交互成功才算数。
            # turn_detection 只借 OpenAI 的 VAD 做**打断**，不让它接管回合：
            #   create_response=False    → 什么时候回复仍由我们决定（本地 VAD 判完
            #                              一轮、记忆预取好，才 response.create）
            #   interrupt_response=True  → 用户一开口，服务端直接掐掉正在播的回复
            # 打断判定放在 OpenAI 那侧，是因为它直接对着音频流做；本地 VAD 要等
            # 麦克风帧过完一整条链路（浏览器 AEC → ws → 重采样 → silero），真人
            # 隔着扬声器插话时信号本来就弱，很容易判不出来。
            # turn_detection 在 session.audio.input 下，**不是顶层**——写成顶层
            # 会被静默拒绝（"Unknown parameter: session.turn_detection"，只以 error
            # 事件回来，没人看就以为设上了）。原来那句 {"turn_detection": None} 一直
            # 没生效，于是 server_vad 始终开着、自动抢着回复：它生成的 response 不带
            # 我们注入的记忆，我们自己的 response.create 又撞上"已有 response 在跑"
            # 而失败——语音轮"没用上记忆"就是这么来的。
            #   create_response=False    → 什么时候回复由我们决定（本地 VAD 判完一轮、
            #                              记忆预取好，才 response.create）
            #   interrupt_response=True  → 用户一开口，服务端直接掐掉正在播的回复；
            #                              这条判定在 OpenAI 侧直接对着音频流做，比
            #                              本地 VAD（隔着 AEC + 网络 + 重采样）可靠
            await conn.session.update(session={
                "type": "realtime",
                "audio": {
                    "input": {"turn_detection": _turn_detection()},
                    "output": {"voice": utils.RT_VOICE},
                },
            })
            connected = True

            # 这一轮的状态：谁在说、说了什么、这轮用户的输入是什么。
            # until：前端预计几点才把已发出去的音频播完（见 hearing()）。
            turn = {"live": False, "reply": "", "pending": None,
                    "t0": 0.0, "until": 0.0, "first": False}
            owner = {"id": "", "last": "", "miss": 0}

            def hearing() -> bool:
                """用户此刻还听不听得见助手。

                不能用 turn["live"] 代替：realtime 推音频比实时播放快得多，一段
                十几秒的回复两三秒就推完了，response.done 一到 live 就变 False，
                可前端那边还在播剩下的十几秒。这时候用户插话，on_speech 看到
                live=False 就"忽略"，前端从没收到 answer_interrupt——表现正是
                "打断没反应，它非要念完"。
                所以按**已发出去的音频时长**算：24k PCM16，一个样本 2 字节。
                """
                return turn["live"] or time.monotonic() < turn["until"]

            def close_turn():
                """一轮结束（说完或被打断）：把两半对话一起存。被打断时存的是用户
                真正听到的那部分——跟 llm_tts 那边 capture() 的取舍一致。"""
                p, reply = turn["pending"], turn["reply"]
                turn.update(live=False, reply="", pending=None)
                if p is None:
                    return
                # 存记忆失败不能连累这条会话：close_turn 是在常驻事件泵里调的，
                # 抛出去会打死那个 Task，而 Task 的异常没人 await 就被静默丢弃——
                # 表现是"整个会话突然不响应了，日志里一个字都没有"。
                remember_turn(p, reply, owner)

            async def pump():
                """常驻事件泵：OpenAI 的事件流只有这一个消费者。

                turn["live"] 为假时一律不往前端转发——被打断后 OpenAI 还会吐一会儿
                残余音频，转过去的话前端刚 stopPlayback 又排上新的，打断就不干净了。
                """
                async for ev in conn:
                    t = getattr(ev, "type", "")
                    if t.endswith("output_audio.delta"):
                        if turn["live"]:
                            if not turn["first"]:
                                # 首帧音频延迟：response.create 发出去到 OpenAI 吐第一块
                                # 声音之间的时间。这是"它反应慢"里我们控制不了的那半。
                                turn["first"] = True
                                print(f"[lat] realtime 首帧 "
                                      f"{(time.monotonic()-turn['t0'])*1000:.0f}ms", flush=True)
                            pcm = base64.b64decode(ev.delta)
                            # 前端是排队播的（index.html 的 nextPlay），这里跟着算
                            # 同一条时间线：上一块播完之后再接这一块。
                            turn["until"] = (max(turn["until"], time.monotonic())
                                             + len(pcm) / 2 / 24000)
                            await sock.send_bytes(pcm)
                    elif t.endswith("output_audio_transcript.delta"):
                        if turn["live"]:
                            turn["reply"] += ev.delta
                            await sock.send_json({"type": "answer_delta", "text": ev.delta})
                    elif t == "error" or t.endswith(".error"):
                        err = getattr(ev, "error", None)
                        # server_vad 判完一句会自己 commit 音频缓冲，我们随后那次
                        # commit 就撞上空缓冲。两种情况都得留着手动 commit（说得太短
                        # 时 server_vad 不会自动 commit），所以这条属于预期内，忽略。
                        # response_cancel_not_active：打断有两条路（本地 VAD 的
                        # on_speech + server_vad 的 interrupt_response），互为备份，
                        # 谁先到算谁的，慢的那个扑空是正常的。
                        if getattr(err, "code", "") not in (
                                "input_audio_buffer_commit_empty",
                                "response_cancel_not_active"):
                            print(f"[web] realtime 事件错误：{err or ev}", flush=True)
                    elif t.endswith("input_audio_buffer.speech_stopped"):
                        # OpenAI 判"你说完了"的时刻。跟本地 silero 判完（我们发
                        # response.create 那一刻）比，谁早谁晚——早的那个才是
                        # 真正的 EOU 下限，晚的那部分是白等的。
                        turn["stopped"] = time.monotonic()
                        if BARGE_DEBUG:
                            print("[lat] OpenAI 判说完", flush=True)
                    elif t.endswith("input_audio_buffer.speech_started"):
                        # OpenAI 的 VAD 听到人声：它那侧已经掐了回复，我们同步收尾
                        since = (time.monotonic() - turn["t0"]) * 1000
                        if BARGE_DEBUG:
                            print(f"[barge] OpenAI VAD 听到人声 (live={turn['live']}, "
                                  f"还在播={hearing()}, {since:.0f}ms)", flush=True)
                        # 宽限期，跟本地那条路一样。本地 VAD 判完一轮（静音 500ms）就
                        # 发 response.create，而 OpenAI 的 server_vad 只要 320ms 静音就
                        # 认为下一句开始了——用户的话尾、呼吸声、环境噪声都够触发。
                        # 没有这道门的话，speech_started 会在助手出声之前就到，
                        # 回复被掐在第一个音频块之前：一个字都听不见，还不报错。
                        # 只记录，不再据此打断——见 _turn_detection 里的说明。
                        # 留着这行日志是因为它是判断"回声压没压干净"最直接的证据：
                        # 助手说话期间频繁出现，就说明 AEC 有残留。
                    elif t.endswith("response.done") or t.endswith("response.cancelled"):
                        if turn["live"]:
                            await sock.send_json({"type": "answer_done"})
                            close_turn()

            async def on_frame(raw):
                await conn.input_audio_buffer.append(audio=base64.b64encode(raw).decode())

            async def on_speech():
                """用户在助手说话时开口 → 打断。幂等：hearing() 一变假就不再触发。"""
                if not hearing():
                    if BARGE_DEBUG:
                        print("[barge] 有人声但助手没在说，忽略", flush=True)
                    return
                # 刚开口那一小段不许打断：那时候麦克风里几乎只有助手自己的声音，
                # 回声消除还没跟上，很容易一出声就把自己掐了。
                since = (time.monotonic() - turn["t0"]) * 1000
                if since < BARGE_GRACE_MS:
                    if BARGE_DEBUG:
                        print(f"[barge] 才说了 {since:.0f}ms，还在宽限期内，不打断", flush=True)
                    return
                if BARGE_DEBUG:
                    left = max(0.0, turn["until"] - time.monotonic()) * 1000
                    print(f"[barge] ★ 打断：转写触发（前端还剩 {left:.0f}ms 没播完）",
                          flush=True)
                turn["live"], turn["until"] = False, 0.0
                # 先通知前端停播——这是本地操作，立刻生效；而 response.cancel() 要
                # 等一次 OpenAI 往返。之前顺序反了，人插话后还得听完那一个往返的
                # 时间，听感就是"打断没用，他非要说完"。
                await sock.send_json({"type": "answer_interrupt"})
                close_turn()
                await conn.response.cancel()

            pump_task = asyncio.create_task(pump())
            try:
                async for pending in anticipate(sock, on_frame=on_frame,
                                                on_speech=on_speech, owner=owner,
                                                said=lambda: turn["reply"]):
                    if hearing():                        # 上一轮还没播完就被新的一轮顶掉
                        await on_speech()
                    turn.update(live=True, reply="", pending=pending,
                                t0=time.monotonic(), until=0.0, first=False)
                    await start_realtime_turn(pending, conn, sock.send_json)
            finally:
                pump_task.cancel()
    except Exception as e:
        if connected:
            raise
        await _no_realtime(sock, e)


#: 右脑 slot → 脑图三个簇。
#:
#: 右脑真正的分类单位是 rb_slots 里那 6 个 slot（情绪 / 喜好与厌恶 / 应对方式 /
#: 表达风格 / 思维模式 / 人物地点态度），不是 memory_class——那只有 heartnote /
#: response_experience 两种，分不出东西。脑图上只有三个簇，所以这里把 6 个 slot
#: 收敛成 3 个。
#:
#: 检索命中的内容里，slot 名就写在开头（"情绪：…""喜好与厌恶：…"）；
#: heartnote 是一条条的情绪时刻（"情感记录：…（内心OS：【难过】…）"），归 emotion。
SLOT_TO_CLUSTER = {
    "情绪":         "emotion",
    "情感记录":     "emotion",
    "内心OS":       "emotion",
    "喜好与厌恶":   "preference",
    "思维模式":     "preference",
    "应对方式":     "experiences",
    "表达风格":     "experiences",
    "人物地点态度": "experiences",
    "避免重复":     "experiences",
}
_CALM = ("", "平静", "中性")


def rb_cluster(content: str, memory_class: str = "", emotion: str = "") -> str:
    """一条右脑记忆归到脑图哪个簇。0 LLM，只看 slot 名。"""
    # 先剥掉 "[2026-06-20] " 这种日期前缀，否则它占满取来比对的那一小段，
    # heartnote 的"情感记录"就落到窗口外了。
    text = re.sub(r"^\s*\[[0-9-]{6,12}\]\s*", "", content or "")
    head = text[:14]
    for slot, cluster in SLOT_TO_CLUSTER.items():
        if slot in head:
            return cluster
    if str(memory_class) == "response_experience":
        return "experiences"
    if emotion not in _CALM:
        return "emotion"
    return "experiences"


def audio_of(memory_id: str) -> str:
    """这条记忆当时那段原声在哪；没归档过、或已过保留期被清掉，返回 ""。

    走核心的 GetOriginalAudio——它已经带了"文件还在不在"的检查，不用在这儿重写。
    ``LAST_TUNE_ID`` 是个例外：它指的是刚听过、还没来得及入库的那段（见 _LAST_TUNE）。
    """
    if memory_id == LAST_TUNE_ID:
        return _last_tune_path()
    if memory_id.startswith(GROUP_ID_PREFIX):     # 一组片段，现拼一个完整的
        return _stitch([m for m in memory_id[len(GROUP_ID_PREFIX):].split(",") if m])
    try:
        r = vm._o._audio.GetOriginalAudio(memory_id)
        return r.get("audio_path") or "" if r.get("found") else ""
    except Exception as e:
        print(f"[web] 查存档音频失败：{e}", flush=True)
        return ""


def hit_cluster(content: str, source: str) -> str:
    """给检索命中用：只有 content 和 source，没有 metadata。"""
    return rb_cluster(content, source, "")


def _rb_cluster(m) -> str:
    """给快照用：从 RightBrainMemory 对象取字段。"""
    meta = getattr(m, "metadata", None) or {}
    return rb_cluster(getattr(m, "content", ""),
                      str(getattr(m, "memory_class", "")),
                      meta.get("emotion", ""))


def fact_index(uid: str) -> dict:
    """左脑记忆 id → 事实原文。

    原文在向量库里，认知图的 memories 表只有 id/slot/热度这些，取不到文本——
    一开始用 get_memory_record 取，结果每条 heartnote 的起因都是空的。
    """
    try:
        entries = vm._o._get_repo()._vector_store.list_entries(user_id=uid)
        return {e["id"]: e["text"] for e in entries}
    except Exception as e:
        print(f"[web] 读左脑事实失败：{e}", flush=True)
        return {}


#: 右脑每个 slot 在脑图上最多画几个 entity。
RB_ENTITIES_PER_SLOT = int(os.environ.get("VOICEMEM_RB_GRAPH_PER_SLOT", "6"))
#: 左脑每个 slot 在脑图上最多画几条记忆。
LB_ENTRIES_PER_SLOT = int(os.environ.get("VOICEMEM_LB_GRAPH_PER_SLOT", "7"))


# ── 右脑判断的「人话」版本 ────────────────────────────────────────────────────
# 库里存的 claim 是给**模型**看的：紧凑、第三人称、像标签（"Facing a lot of
# pressure recently"）。那份不能动——prompt 需要的就是这种密度。
# 但页面上是给**人**看的，同一句话摆出来就很像在读档案。这里做一份只用于显示的
# 第一人称改写。
#
# 异步 + 缓存：memory_snapshot 是同步的，不能在里面等一次 LLM 往返。所以第一次
# 显示原文，后台改写完落进缓存，前端下一次轮询（watchMemories 本来就在轮）就换成
# 人话。改写只碰措辞，不新增任何事实。
#: 这个空间的主人叫什么。右脑那些话是**关于他**的，一律写"他"就少了那份认得他的
#: 感觉——"Jiaqi 一紧张就闷声不响"和"他一紧张就闷声不响"，前者才像认识他。
#: 名字来自声纹注册表（自报"我叫X"时绑定的），读不到就回落到"他"。
_OWNER_NAME_CACHE: dict = {}
#: 疑问词。"我叫什么名字？"被当成自我介绍绑进去过，registry 里真的躺着一条
#: name="什么名字"（见 voiceprint/speaker_identity.py 顶上那段）。显示前挡一道。
_BAD_NAME_CHARS = "什谁哪啥吗呢么?？"


def owner_name(space: str = "") -> str:
    """这个空间主人的名字；认不出来就返回 ""（调用方自己回落到"他"）。"""
    space = space or ACTIVE_SPACE
    if space in _OWNER_NAME_CACHE:
        return _OWNER_NAME_CACHE[space]
    name = ""
    try:
        import json
        from voicemem.utils.common import space as _sp
        d, _ = space_dir(space)
        p = _sp.mm(d, "voiceprint_registry.json")
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            cands = []
            for key, v in (data or {}).items():
                if not isinstance(v, dict) or v.get("role") != "user":
                    continue
                n = (v.get("name") or "").strip()
                # 挡掉疑问句绑进来的假名字，也挡掉 "user" 这种占位 key
                if not n or n.lower() == "user" or any(c in n for c in _BAD_NAME_CHARS):
                    continue
                cands.append((bool(v.get("entity_id")), n))
            if cands:
                # 有 entity_id 的更可信（真的在图里落过地）
                cands.sort(key=lambda t: not t[0])
                name = cands[0][1]
    except Exception as e:
        print(f"[rb] 主人姓名读取失败：{type(e).__name__}: {e}", flush=True)
    _OWNER_NAME_CACHE[space] = name
    return name


_RB_HUMAN: dict = {}          # claim 原文 -> 人话版本
_RB_HUMAN_PENDING: set = set()
#: 关掉就一直显示原始 claim（不想为显示花钱时）。
RB_HUMANIZE = os.environ.get("VOICEMEM_RB_HUMANIZE", "1") != "0"

_RB_HUMANIZE_PROMPT = (
    # 四版教训，别再退回去：
    # ① 只写"第一人称"不够——不说清是**谁**在说，模型只做同义替换。
    # ② "不许新增事实"会被读成"不许换说法"，于是退化成往原句前贴个"我注意到"。
    #    要把两件事拆开：事实不能加，措辞必须重说。
    # ③ "我发现/我注意到/在我看来"全是**报告动词**——语法上是第一人称，语气上还是
    #    观察员在汇报。要的是一个懂他的小东西在心疼他，所以这类开头得禁掉。
    # ④ 名字那条第一版写的是"和「他」换着用"，太软，十条全用了"他"——示例里也全是
    #    "他"，模型照着示例走。规则要硬，示例也得点名。
    #    后来改成**一律点名**：页面上每条是独立的一行，不是一段连贯的话，
    #    重复出现名字读起来是"这是关于谁的"，不是啰嗦。
    # ⑤ 显示语言跟**界面**走，不跟原文走。库里中英混着（翻译过一半），跟原文走
    #    页面上就中英混排。显示是给人看的，同一屏里不该夹生。
    "下面每行是一条关于 {who} 的判断，来自一个一直陪着 {who} 的小助手——"
    "像只很懂他的小动物，安静地待在旁边，什么都看在眼里。\n"
    "把每一条改写成这个小助手会说出来的话。\n"
    "\n"
    "语气：\n"
    "· 短。十几个字最好，读起来是一句话，不是一条记录。\n"
    "· 有温度，带一点点护着他的意思——是心疼，不是分析。\n"
    "· **不要用「我发现」「我注意到」「在我看来」开头**，那是汇报的口气。"
    "直接说那件事，或者说你替他觉得怎么样。\n"
    "· 别肉麻、别撒娇、别堆感叹号、别讲道理也别安慰。\n"
    "{name_rule}"
    "\n"
    "内容：\n"
    "· **必须换一种说法**。原句是概括性的词（「简短回应」「寻求认同」），"
    "要还原成人会怎么形容（「话就变少了」「想有人接住他」）。\n"
    "· 但**不许新增任何事实**：不补细节、不猜原因、不加评价。换措辞，不换内容。\n"
    "· 全部用{lang}输出，不管原文是什么语言。\n"
    "\n"
    "逐行输出，行数和顺序跟输入完全一致；不要编号、引号或多余的话。\n"
    "例：\n"
    "{examples}"
)

#: 示例。两件事都靠它带：**语气**和**输出语言**。
#: 模型跟示例走的力度远大于跟规则走——名字那条和语言这条都在这儿栽过：
#: 规则写了"每条点名"但示例用「他」，输出就全是「他」；规则写了"输出英文"
#: 但示例全中文，输出就全是中文。所以示例必须跟着目标语言换，而且都要点名。
_RB_HUMANIZE_EXAMPLES = {
    "zh": (
        "  输入  在焦虑时倾向于简短回应\n"
        "  输出  {who}一紧张就闷声不响\n"
        "  输入  在情绪低落时不喜欢被忽视\n"
        "  输出  {who}难过的时候，最怕没人理\n"
        "  输入  Feels accomplished when recognized\n"
        "  输出  夸{who}一句，他整个人就亮了\n"
        "  输入  Hates being interrupted\n"
        "  输出  打断{who}说话，他立刻就不说了\n"
    ),
    "en": (
        "  输入  在焦虑时倾向于简短回应\n"
        "  输出  {who} goes quiet the moment he tenses up\n"
        "  输入  在情绪低落时不喜欢被忽视\n"
        "  输出  When {who} is down, being ignored is the worst of it\n"
        "  输入  Feels accomplished when recognized\n"
        "  输出  A little praise and {who} lights right up\n"
        "  输入  Hates being interrupted\n"
        "  输出  Cut {who} off mid-sentence and you'll lose him\n"
    ),
}


def _rb_humanize_now(claims: list, name: str = "", lang: str = "zh") -> None:
    """后台线程里跑：一次 LLM 往返改写一批，结果落进 _RB_HUMAN。"""
    lang_name = "英文" if lang == "en" else "中文"
    try:
        from openai import OpenAI
        who = name or "他"
        sysmsg = _RB_HUMANIZE_PROMPT.format(
            who=who,
            lang=lang_name,
            # 规则要硬。写"换着用"的那版，十条全用了"他"。
            name_rule=(f"· **每条都直接叫他「{name}」**，别用「他」代替——"
                       "页面上每条是独立一行，点名是在说「这是关于谁的」。\n"
                       if name else ""),
            examples=_RB_HUMANIZE_EXAMPLES.get(lang, _RB_HUMANIZE_EXAMPLES["zh"])
                     .replace("{who}", who))
        r = OpenAI().chat.completions.create(
            model=utils.CHAT_MODEL, temperature=0.7,
            messages=[{"role": "system", "content": sysmsg},
                      {"role": "user", "content": "\n".join(claims)}],
        )
        lines = [x.strip() for x in (r.choices[0].message.content or "").splitlines() if x.strip()]
        if len(lines) != len(claims):      # 行数对不上就整批丢弃，别错位配对
            print(f"[rb] 改写行数不符（{len(lines)}≠{len(claims)}），这批跳过", flush=True)
            return
        for c, h in zip(claims, lines):
            _RB_HUMAN[(lang, name, c)] = h
    except Exception as e:
        print(f"[rb] 判断改写失败：{type(e).__name__}: {e}", flush=True)
    finally:
        _RB_HUMAN_PENDING.difference_update((lang, name, c) for c in claims)


def rb_human(claim: str) -> str:
    """显示用的那句话。还没改写好就先返回原文，同时排上队。"""
    if not RB_HUMANIZE or not claim:
        return claim
    name, lang = owner_name(), UI_LANG
    hit = _RB_HUMAN.get((lang, name, claim))
    if hit:
        return hit
    if (lang, name, claim) not in _RB_HUMAN_PENDING:
        _RB_HUMAN_PENDING.add((lang, name, claim))
        import threading
        threading.Thread(target=_rb_humanize_now,
                         args=([claim], name, lang), daemon=True).start()
    return claim


def rb_human_batch(claims: list) -> None:
    """一次把缺的都排上队——脑图一屏几十条，一条一个线程太浪费。"""
    if not RB_HUMANIZE:
        return
    name, lang = owner_name(), UI_LANG
    todo = [c for c in dict.fromkeys(claims)
            if c and (lang, name, c) not in _RB_HUMAN
            and (lang, name, c) not in _RB_HUMAN_PENDING]
    if not todo:
        return
    _RB_HUMAN_PENDING.update((lang, name, c) for c in todo)
    import threading
    threading.Thread(target=_rb_humanize_now, args=(todo, name, lang), daemon=True).start()


def right_brain_tree(uid, facts):
    """脑图右半球：slot → 判断 → 证据。

    读右脑的判断表（voicemem/rightbrain/traits_store.py）。旧的
    slot→entity→heartnote 结构已经不再写入，_right_brain_tree_v1 只用来看老数据。
    """
    try:
        store = vm._o._right._traits()
    except Exception as e:
        print(f"[web] 判断表读取失败：{type(e).__name__}: {e}", flush=True)
        return []

    traits = list(store.all(uid, per_slot=RB_ENTITIES_PER_SLOT))
    rb_human_batch([t.claim for t in traits])     # 缺的一次排队，见 rb_human
    out = []
    for t in traits:
        out.append({
            "cluster": t.cluster,
            "slot": t.slot,
            # 节点标题用人话版；还没改写好就是原文，下一次轮询会换上来。
            # raw 留着——前端拿它跟每轮命中做匹配，换成人话就对不上了。
            "raw": t.claim,
            "text": rb_human(t.claim),
            "desc": "",               # 判断本身就是概括，不再垫一行原始事实
            "notes": [{"text": e.quote, "emotion": e.emotion, "cause": e.cause}
                      for e in t.evidence],
        })
    return out


def _right_brain_tree_v1(uid: str, facts: dict) -> list:
    """右脑真实的三层结构：slot → entity → 挂在下面的 heartnote。

    脑图上的节点是 **entity**（"委屈""讨厌坚果和过敏""选择沉默忍耐"），不是
    一条条 heartnote —— entity 才是右脑归纳出来的那个"点"，heartnote 是支撑它
    的证据。每条 heartnote 再带上引发它的左脑事实，这样"为什么委屈"点两下就能看到。

    只读，全部走 graph_store 的公开方法。
    """
    graph = vm._o._right._rb_graph_store()
    repo = vm._o._right._rb_repo()
    notes = {}
    for m in repo.list_all(uid):
        # response_experience 记的是"助手上次怎么答的"，是给回复层看的内部笔记，
        # 不是对用户其人的认识。脑图上不该有它——它长出来的节点写着助手自己
        # 说过的话，看着像"系统把自己的回复当成了对你的了解"。
        if getattr(m, "memory_class", "") == "response_experience":
            continue
        meta = getattr(m, "metadata", None) or {}
        notes[m.id] = {
            "text": m.content,
            "emotion": meta.get("emotion", ""),
            "cause": facts.get(meta.get("left_memory_id", ""), ""),
        }

    out = []
    for slot in graph.list_slots(uid):
        # 没有 description 的 slot 跳过：右脑还没归纳过它，底下挂的就是原始实体
        # 倒进来的一堆东西（实测"人物地点态度"下面是 Jiaqi / 计算机本科 /
        # 新加坡国立大学NUS / 2026年9月 —— 人名、学历、机构、日期，不是态度）。
        # 画在脑图上只是噪声，还占着右半球的位置。
        # 这里原来是「slot 没有 description 就整个跳过」。
        # description 是**长期归因**（session 边界的 _summarize_slot）才生成的，
        # 新建的空间从没跑过 → 六个 slot 全是空描述 → 每个都被跳过 → 脑图右半球
        # 一个节点都没有。而节点画的是**实体**，跟这个 slot 有没有一句画像总结
        # 没关系：实体在、证据在，就该画出来。
        cluster = SLOT_TO_CLUSTER.get(slot.name, "experiences")
        # 每个 slot 只画前几个 entity——库里跑一阵就有 89 个，全画上去右半球糊成
        # 一片（左脑同期才 26 条）。
        #
        # 排序不能只看证据条数。原来是纯按条数倒序，结果「喜好与厌恶」下 24 个
        # 实体里新来的那个只有 1 条证据，永远排最后，**永远进不了图**——用户说
        # 一句全新的话，右脑一个新节点都不长，看着像没记住。
        # 现在留一半席位给最近新增的：一半按证据多少（稳定的画像），一半按新旧
        # （刚说的那句能立刻看见）。
        def rows(newest):
            out = []
            for ent in graph.get_entities_for_slot(uid, slot.id, newest_first=newest):
                mids = [i for i in graph.get_memories_for_entity(ent.id) if i in notes]
                out.append((len(mids), ent, mids))
            return out

        fresh_n = max(1, RB_ENTITIES_PER_SLOT // 2)
        by_recent = rows(True)[:fresh_n]                    # 真·最近新增（rowid 倒序）
        taken = {t[1].id for t in by_recent}
        by_evidence = [t for t in sorted(rows(False), key=lambda t: -t[0])
                       if t[1].id not in taken]
        picked = by_recent + by_evidence[:RB_ENTITIES_PER_SLOT - len(by_recent)]

        for _, ent, mids in picked:
            # 没有任何记忆挂在下面的节点不画。情绪 slot 建空间时会预置 8 个情绪词
            # 实体（悲伤/平静/孤独…），全新的空库打开就摆着六个孤零零的点，看着
            # 像已经记住了什么，其实背后一条证据都没有。
            if not mids:
                continue
            ns = [notes[i] for i in mids]
            # 实体描述只用巩固那一步（_summarize_entity）归纳出来的那句。
            #
            # 曾经在这儿加过兜底：没有描述就拿第一条证据的 cause 顶上。那是错的——
            # cause 是**左脑事实原文**，于是卡片变成
            #     标题：佳琪
            #     正文：用户的名字是佳琪，今年20岁，在NUS读书…   ← 原始事实
            #     note：【委屈】呃我是佳琪啊我今年二十岁了…       ← 原话
            # 而且同一句话抽出的几个实体（佳琪/计算机专业/NUS）cause 相同，三张卡片
            # 正文一模一样。宁可空着——标题本身就该是那句概括，中间不该再垫一行。
            desc = (getattr(ent, "description", "") or "").strip()
            out.append({
                "cluster": cluster,
                "slot": slot.name,
                "text": ent.name,                      # 脑图上显示的就是这个
                "desc": desc,
                "notes": ns,
            })
    return out


#: 最近一轮检索命中的记忆 id。快照要保证它们在图上——脑图上亮起来的必须是
#: 后端真正检索到的那几条，命中了却没画出来，"它记得这件事"就没演出来。
_LAST_HIT_IDS: set = set()


def note_hits(result) -> None:
    """记下这一轮检索命中了哪些左脑记忆。"""
    _LAST_HIT_IDS.clear()
    for h in (getattr(result, "hits", None) or []):
        mid = getattr(h, "memory_id", "")
        if mid:
            _LAST_HIT_IDS.add(str(mid))


def memory_snapshot(limit: int = 48) -> dict:
    """库里已有的记忆，供前端在打开页面时把脑图先铺满。

    只读，不碰模型：左脑走 list_entries + 认知图的 slot 标注，右脑走 list_all。
    库是空的（新用户）就返回空列表，前端照旧从空图开始长。
    """
    from voicemem.leftbrain.cognitive_graph.types import SlotV2

    uid = vm._o._user_id
    left, right = [], []
    try:
        repo = vm._o._get_repo()
        entries = repo._vector_store.list_entries(user_id=uid)
        # slot 标在认知图里，不在记忆条目上——先建 id -> slot 的反查表
        cog = repo._cognitive_store
        slot_of = {}
        for slot in SlotV2:
            for mid in cog.memory_ids_for_slots(uid, [slot]):
                slot_of.setdefault(mid, slot.value)
        # 助手自己说过的话也原样存在库里（见 3ed67f7），但脑图画的是"关于用户的
        # 记忆"——把助手的回复也长成节点，等于把它自己说的话当成对用户的认识。
        entries = [e for e in entries if e.get("role") != "assistant"]
        # 每个 slot 也限量。脑图上一个 slot 就是一块扇形，面积固定——daily_life
        # 攒到二十几条时那块就糊了，而别的 slot 才三四个点。限量之后各簇疏密一致。
        #
        # 但**这一轮检索命中的那几条必须留下**，哪怕它排在限量之外：图上亮起来的
        # 得是后端真检索到的东西，命中了却没画出来，看着就像"检索到 5 条只亮了 3 个"。
        # 命中的先排进去，剩下的名额再按原来的顺序填。
        per_slot, kept = {}, []
        hit_first = ([e for e in entries if str(e["id"]) in _LAST_HIT_IDS] +
                     [e for e in entries if str(e["id"]) not in _LAST_HIT_IDS])
        for e in hit_first:
            sl = slot_of.get(e["id"], "daily_life")
            hit = str(e["id"]) in _LAST_HIT_IDS
            per_slot[sl] = per_slot.get(sl, 0) + 1
            if hit or per_slot[sl] <= LB_ENTRIES_PER_SLOT:
                kept.append(e)
        entries = kept
        # 同理，命中的那几条不能被 limit 截掉
        head = [e for e in entries if str(e["id"]) in _LAST_HIT_IDS]
        rest = [e for e in entries if str(e["id"]) not in _LAST_HIT_IDS]
        for e in (head + rest)[:max(limit, len(head))]:
            # list_entries 的 date 直接截了 time_start 前 10 位，遇到纯时间串会切出
            # "09:20:37" 这种。不像日期就置空，别把垃圾送到前端。
            d = str(e.get("date", ""))
            # 带上这条记忆挂了哪些实体：前端据此在讲同一个人/同一件事的两条记忆
            # 之间连线——脑图上的连线才对应真实关系，而不是随便连。
            # 存的是实体 id（person_jiaqi_5ea413），前端要拿去当标签显示、也拿去
            # 按同名实体连线，所以在这儿换成名字。
            try:
                ents = []
                for eid in cog.entity_ids_for_memory(e["id"]) or []:
                    ent = cog.get_entity(eid)
                    nm = (getattr(ent, "name", "") if ent else "").strip()
                    ents.append(nm or eid)
            except Exception:
                ents = []
            # hit：这条是被这一轮检索命中、因而保底留在图上的。
            # 前端拿"图上多出了哪条"来判断"刚说的这句抽出了什么实体"，而保底进来
            # 的是**旧记忆**——不标出来的话，说一句「啦啦啦」也会让上次那些实体
            # （Jiaqi、室友、打游戏…）冒到标签栏上。
            left.append({"text": e["text"], "date": d if d[:4].isdigit() else "",
                         "slot": slot_of.get(e["id"], "daily_life"),
                         "hit": str(e["id"]) in _LAST_HIT_IDS,
                         "entities": list(ents)[:6]})
    except Exception as e:
        print(f"[web] 左脑快照读取失败：{e}", flush=True)
    try:
        right = right_brain_tree(uid, fact_index(uid))
    except Exception as e:
        print(f"[web] 右脑快照读取失败：{e}", flush=True)
    return {"left": left, "right": right}


# classify 必须包一层：直接传 vm.classify 会把**当前这个**实例焊进去，
# 切换空间之后脑图还在给旧空间分类。
app = utils.build_app(MODE, realtime_session if MODE == "realtime" else llm_tts_session,
                      lambda *a, **k: vm.classify(*a, **k), memory_snapshot, audio_of,
                      spaces=(list_spaces, create_space, use_space, lambda: ACTIVE_SPACE),
                      set_lang=set_lang)


if __name__ == "__main__":
    print(f"[web] mode={MODE} spec≥{SPEC_MIN_CHARS}字 gamble={ARGS.gamble_ms}ms "
          f"confirm={ARGS.confirm_ms}ms -> http://localhost:{ARGS.port}/", flush=True)
    # 全部预热在这儿做完，别让第一句话去等模型加载。ASR(FunASR paraformer)
    # 是懒加载的，等用户开口才拉起来要好几秒——那几秒的音频堆在 socket 缓冲里，
    # 追赶时逐帧喂 VAD，静音会瞬间累计过 confirm_ms，第一句直接被截断（听感就是
    # "第一句又慢又不准"）。
    print("[web] 预热本地模型（embedding / ASR / VAD / 感知）…", flush=True)
    vm.warmup(verbose=True)
    print("[web] 就绪", flush=True)
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)
