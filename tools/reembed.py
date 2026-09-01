"""换 embedder 之后，把库里维度作废的向量重新算一遍。

什么时候需要：`_embed_text` 现在跟着注入的 embedder 走，而
`rb_traits` / `graph_entities` 里可能存着上一个 embedder 算的向量。
维度不符的会被跳过（有警告），右脑检索和实体去重因此失效，直到重新 embed。

跑：python3 tools/reembed.py <space> [--apply] [--local]
不加 --apply 只统计；--local 按 web demo 的配置（本地 E5）来算，
不加就是默认 embedder（OpenAI）。**必须跟你实际运行时的配置一致**，
否则算出来的维度不对，等于没修。
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    space = sys.argv[1] if len(sys.argv) > 1 else "demo"
    apply = "--apply" in sys.argv
    db = Path("voicemem_memoryspace") / space / f"{space}.sqlite"
    if not db.is_file():
        print(f"找不到 {db}")
        return

    from voicemem.core import VoiceMem
    cfg = {"mode": "text_mode", "space": space}
    if "--local" in sys.argv:                     # 跟 web demo 的配置对齐
        cfg |= {"embedding": {"provider": "local"}, "slots": {"provider": "local"}}
    vm = VoiceMem.from_config(cfg)
    embed = vm._o._embed_text
    want = len(embed("维度探针"))
    print(f"{space}: 当前 embedder 输出 {want} 维")

    import json
    import numpy as np

    def vec_len(b):
        """两张表的存法不同：rb_traits 是 float32 二进制，graph_entities 是 JSON。"""
        if isinstance(b, (bytes, bytearray)):
            return len(np.frombuffer(b, dtype=np.float32))
        try:
            return len(json.loads(b))
        except Exception:
            return -1

    def pack(table, vec):
        return (np.asarray(vec, dtype=np.float32).tobytes() if table == "rb_traits"
                else json.dumps([float(x) for x in vec]))

    jobs = []          # (表, id 列, 待重算的行)
    con = sqlite3.connect(db)
    for table, idc, txtc in (("rb_traits", "id", "claim"),
                             ("graph_entities", "id", "name")):
        try:
            rows = con.execute(
                f"SELECT {idc}, {txtc}, embedding FROM {table} "
                "WHERE embedding IS NOT NULL").fetchall()
        except sqlite3.OperationalError:
            continue
        stale = [(i, t) for i, t, b in rows if vec_len(b) != want]
        print(f"  {table:16} 共 {len(rows):4} 条，维度作废 {len(stale)}")
        if stale:
            jobs.append((table, idc, stale))

    if not jobs:
        print("没有需要重算的。")
        return
    if not apply:
        print("（未加 --apply，只统计）")
        return

    total = 0
    for table, idc, stale in jobs:
        for n, (mid, text) in enumerate(stale, 1):
            con.execute(f"UPDATE {table} SET embedding=? WHERE {idc}=?",
                        (pack(table, embed(text)), mid))
            total += 1
            if n % 25 == 0:
                con.commit()
                print(f"  {table} …{n}/{len(stale)}")
        con.commit()
    print(f"\n重算完成：{total} 条")


if __name__ == "__main__":
    main()
