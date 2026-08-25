"""本地 E5 embedder（离线 memory embedding，不碰网络）。

和远程 ``OpenAILocalEmbedder``（``local_memory_store.py``，走 OpenAI Embeddings API）
对称：这里用本地 ``intfloat/multilingual-e5-small`` 出向量，整条 Rank/存取都在
0-300ms 投机预算内不碰网络。

    from voicemem import VoiceMem
    from voicemem.leftbrain.local_e5_embedder import LocalE5Embedder
    vm = VoiceMem(embedding=lambda: LocalE5Embedder())   # 记忆向量走本地 E5，0 网络

``shared_e5()`` 缓存单份 SentenceTransformer，供 embedding 与本地 slot 分类
（``LocalQueryClassifier(model=shared_e5())``）共享，省一份内存。

E5 的 ``"query: "`` / ``"passage: "`` 前缀是必须的（不是装饰）。模型首次用自动下载。
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

# 有离线包（models/embedding/）就用本地，否则用 HF id 首次运行自动下
def _e5_name() -> str:
    from voicemem.utils.common.paths import hf_model
    return hf_model("embedding", "intfloat/multilingual-e5-small", "VOICEMEM_E5_MODEL")


_E5_NAME = _e5_name()


@lru_cache(maxsize=1)
def shared_e5():
    """缓存单份本地 E5（memory embedding + slot 分类共享同一实例，省一份内存）。"""
    import os as _os
    # 每次启动都刷一条 "Loading weights: 100%|███| 199/199"（transformers 用 tqdm
    # 打的）。模型在本地、一瞬间就加载完，这条除了吓人没有信息量。
    # 想看加载细节就设 VOICEMEM_VERBOSE=1。
    if _os.environ.get("VOICEMEM_VERBOSE", "0") == "0":
        try:
            from transformers.utils import logging as _hf_logging
            _hf_logging.disable_progress_bar()
        except Exception:
            pass
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_E5_NAME)


class LocalE5Embedder:
    """注入 VoiceMem(embedding=...)：Rank/存取的向量走本地 E5，0-300ms 预算内不碰网络。"""

    @property
    def model_name(self):
        return f"{_E5_NAME} (local)"

    @property
    def dimensions(self):
        # 新版 sentence-transformers 把它改名成 get_embedding_dimension，旧名还在
        # 但会打 FutureWarning。两个名字都试，跨版本都不吵。
        m = shared_e5()
        fn = getattr(m, "get_embedding_dimension", None) or m.get_sentence_embedding_dimension
        return fn()

    def embed_texts(self, texts):
        if not texts:
            return []
        return np.asarray(shared_e5().encode([f"passage: {t}" for t in texts],
                                             normalize_embeddings=True)).tolist()

    def embed_query_text(self, text):
        return np.asarray(shared_e5().encode([f"query: {text}"], normalize_embeddings=True)[0]).tolist()
