"""数据集适配器：每个 benchmark 一个文件，只管两件事——怎么读、怎么判分。

中间那段（逐轮 ingest → 每题 search → 拿记忆作答）对所有数据集完全相同，
写在 run.py 里，这样不同 benchmark 的数字才有可比性。

加一个新 benchmark = 在这个目录加一个文件，实现下面两个函数，再登记到 DATASETS：

    def load(path: str) -> list[Conversation]      # 读成统一结构
    def score(q: Question, answer: str, judge) -> Score   # 判这道题对不对

judge 由 run.py 注入，签名 ``judge(prompt: str) -> str``——各数据集判分口径不同
（有的比对标准答案，有的过 rubric），但用的是同一个裁判模型。
"""
from dataclasses import dataclass, field


@dataclass
class Turn:
    """对话里的一句话。"""
    speaker: str
    text: str
    #: 这句话发生的真实时间（ISO，如 "2023-05-08"）。必须带上——记忆要按时间排序，
    #: 回填历史对话时不传的话，库里全是跑评测那天的时间戳，时序类问题直接废掉。
    observed_at: str = ""


@dataclass
class Question:
    id: str
    text: str
    answer: str = ""                              # 标准答案（有就用来判分）
    rubric: list[str] = field(default_factory=list)   # 评分要点（AudioMC 那类用）
    category: str = ""                            # 题型，用来出分类得分
    meta: dict = field(default_factory=dict)


@dataclass
class Conversation:
    """一段完整对话 + 针对它的问题。

    每段对话在评测时会拿到**独立的记忆库**（见 run.py）——不同对话的记忆串在
    一起，等于把答案偷偷喂给了模型，分数就没意义了。
    """
    id: str
    turns: list[Turn]
    questions: list[Question]


@dataclass
class Score:
    correct: float          # 1/0，或 rubric 满足比例这类小数
    total: float = 1.0      # 这道题的满分（rubric 题就是要点条数）
    note: str = ""          # 判分理由，写进结果文件供人工复核


#: benchmark 名 -> 模块路径。加新的：照着 locomo.py 写个文件，实现 load() 和
#: score()，在这里登记一行。CLI 的 --dataset 直接用这里的键做 choices，
#: 名字写错在参数解析阶段就报错，--help 也会自动列出。
DATASETS = {
    "locomo": "evaluation.datasets.locomo",
}


def names() -> list[str]:
    return sorted(DATASETS)


def display_name(name: str) -> str:
    """打印用的名字，数据集模块里的 NAME；没写就用 CLI 上的键。"""
    return getattr(get(name), "NAME", name)


def get(name: str):
    """按名字拿数据集适配器。"""
    import importlib
    if name not in DATASETS:
        raise SystemExit(
            f"没有这个数据集：{name}。现有：{', '.join(names())}\n"
            f"加一个新的：照着 evaluation/datasets/locomo.py 写个同名文件，"
            f"实现 load() 和 score()，再登记到这里的 DATASETS。")
    return importlib.import_module(DATASETS[name])
