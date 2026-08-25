#!/usr/bin/env python3
"""清掉声纹库，保留记忆。演示前跑一次。

为什么需要：``voiceprint_store.identify()`` 的 candidate 分支（分数落在
0.40–0.50 之间）**每次都 fork 一个新 person**。短句的单次声纹分数噪声很大，
于是同一个人会被拆成一堆 ``person_*``——实测一个 demo 库里 12 个，其中 7 个
``obs_count=2, confidence=0.52``。拆散之后"说话的还是不是同一个人"就判不准，
web demo 的陌生人门会突然翻脸说"我不认识你"。

根因已经在 ``perceiver._detect_speaker`` 上了门槛（短于
``VOICEMEM_SPEAKER_MIN_S`` 秒不认人），但**已经脏了的库不会自己变干净**——
那些一次性画像还在里面参与打分。演示前清一次，然后让主人连续说几句
（每句 ≥3 秒）重新建档。

    python scripts/reset_voiceprints.py            # 看看现在有什么，不动
    python scripts/reset_voiceprints.py --apply    # 真的清掉

记忆本身（左脑事实 / 右脑画像 / 音频归档）一条都不动，只清声纹。
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORE = ROOT / "results" / "voice_memory"


def report(store: Path) -> None:
    meta = store / "voiceprints" / "voiceprint_meta.json"
    registry = store / "voiceprint_registry.json"

    if meta.exists():
        d = json.loads(meta.read_text(encoding="utf-8"))
        persons = d.get("persons", {})
        cands = d.get("candidates", {})
        print(f"声纹画像 {len(persons)} 个，候选 {len(cands)} 条：")
        for pid, m in sorted(persons.items(), key=lambda kv: -kv[1].get("obs_count", 0)):
            n, c = m.get("obs_count", 0), m.get("confidence", 0)
            # obs_count 个位数 + confidence 刚过 0.5 的，基本都是 candidate 分支
            # fork 出来的一次性画像，不是真的另一个人。
            flag = "  ← 多半是误拆的" if n <= 3 and c <= 0.6 else ""
            print(f"  {pid}  说过 {n:>3} 次  confidence={c}{flag}")
    else:
        print("没有 voiceprint_meta.json —— 声纹库本来就是空的。")

    if registry.exists():
        print("\n姓名映射：", registry.read_text(encoding="utf-8").strip())


def wipe(store: Path) -> None:
    for path in (store / "voiceprints", store / "voiceprint_registry.json"):
        if not path.exists():
            continue
        shutil.rmtree(path) if path.is_dir() else path.unlink()
        print(f"删掉 {path.relative_to(ROOT)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--store", default=str(DEFAULT_STORE), help="记忆库目录")
    p.add_argument("--apply", action="store_true", help="真的删；不给就只看不动")
    args = p.parse_args()

    store = Path(args.store)
    if not store.exists():
        raise SystemExit(f"记忆库不存在：{store}")

    report(store)

    if not args.apply:
        print("\n（只看不动。加 --apply 才真的清。）")
        return

    print()
    wipe(store)
    print(
        "\n清完了。接下来：\n"
        "  1. 起 demo，让主人连着说 3-4 句，每句说满 3 秒以上\n"
        "     （短于 VOICEMEM_SPEAKER_MIN_S=2.5 秒的一轮直接不认人，不会建档）\n"
        "  2. 日志里 [speaker] owner=... miss=0 一直不变，就说明档建稳了\n"
        "  3. 这时候再让别人开口，才是真的「换了个人」"
    )


if __name__ == "__main__":
    main()
