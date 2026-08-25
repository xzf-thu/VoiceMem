"""
封顶累积平均 + 自适应多中心 (最小演示)
======================================
只展示"一个人的画像怎么更新"这一件事, 把两个机制咬合处标清楚:
  · 封顶累积平均: 样本少时是累积平均(稳、降噪), 攒满后退化成 EMA(能适应变化)
  · 自适应多中心: 样本少时单中心; 新样本和已有中心都不够像时, 才裂出新子中心
  · 质量加权: 贯穿始终的地基, 烂样本权重小
依赖: numpy
"""

import numpy as np


def l2norm(v):
    n = np.linalg.norm(v)
    return v / n if n else v


class SubCentroid:
    """一个子中心 = 这个人某种状态/信道下的代表向量。"""
    def __init__(self, vec, w):
        self.vec = l2norm(vec)   # 单位向量
        self.w = w               # 累积权重(= 有效样本量, 会被封顶)


class Profile:
    # ========== 参数 ==========
    def __init__(self, w_max=40.0, t_split=0.55, k_max=8):
        self.w_max = w_max        # 权重上限: 攒满后单中心更新退化成 EMA
        self.t_split = t_split    # 裂中心阈值: 新样本与最近中心 < 此值就另立子中心
        self.k_max = k_max        # 子中心数量上限
        self.subs = []            # 子中心列表(空 = 还没样本)

    # ========== 机制A: 封顶累积平均(单个中心怎么更新) ==========
    def _update_one(self, sub, x, w):
        """质量加权累积平均 + 余弦空间重新归一化 + 权重封顶。
        关键: W 没满时是真累积平均(新样本权重 = w / (W+w), 随 W 增大而变小 -> 稳);
              W 封顶后, 旧权重不再增长, 新样本占比固定 -> 等效 EMA(能跟上变化)。"""
        sub.vec = l2norm(sub.vec * sub.w + x * w)
        sub.w = min(sub.w + w, self.w_max)        # <-- 封顶就在这一行, 它把累积平均变 EMA

    # ========== 机制B: 自适应多中心(该并入老中心, 还是裂个新的) ==========
    def update(self, x, quality=1.0):
        x = l2norm(x)
        w = float(quality)                         # 质量加权: 地基

        # 第一条样本 -> 直接建第一个子中心(此时就是单中心)
        if not self.subs:
            self.subs.append(SubCentroid(x, w))
            return

        # 找最像的子中心
        sims = [float(np.dot(x, s.vec)) for s in self.subs]
        j = int(np.argmax(sims))

        if sims[j] >= self.t_split:
            self._update_one(self.subs[j], x, w)   # 够像 -> 并入(走机制A)
        else:
            self.subs.append(SubCentroid(x, w))    # 都不够像 -> 裂出新子中心(出现了新状态)

        # 子中心过多 -> 合并最近的两个, 控制规模
        if len(self.subs) > self.k_max:
            self._merge_closest()

    def _merge_closest(self):
        best = (-1.0, 0, 1)
        for i in range(len(self.subs)):
            for k in range(i + 1, len(self.subs)):
                s = float(np.dot(self.subs[i].vec, self.subs[k].vec))
                if s > best[0]:
                    best = (s, i, k)
        _, i, k = best
        a, b = self.subs[i], self.subs[k]
        merged = SubCentroid(a.vec * a.w + b.vec * b.w, min(a.w + b.w, self.w_max))
        self.subs = [s for idx, s in enumerate(self.subs) if idx not in (i, k)] + [merged]

    # ========== 机制C: 合并相近子中心(清理早期碎片 / 收敛后自清理) ==========
    def consolidate(self):
        """凡是彼此相似度 >= t_split 的两个子中心,合并。
        单峰: 孤儿中心在主中心收敛后会被并回 -> 收成 1 个。
        双峰: 两个中心接近正交, 不满足条件 -> 保留 2 个。"""
        changed = True
        while changed and len(self.subs) > 1:
            changed = False
            for i in range(len(self.subs)):
                for k in range(i + 1, len(self.subs)):
                    if np.dot(self.subs[i].vec, self.subs[k].vec) >= self.t_split:
                        a, b = self.subs[i], self.subs[k]
                        merged = SubCentroid(a.vec * a.w + b.vec * b.w,
                                             min(a.w + b.w, self.w_max))
                        self.subs = [s for idx, s in enumerate(self.subs)
                                     if idx not in (i, k)] + [merged]
                        changed = True
                        break
                if changed:
                    break

    # ========== 匹配: 对所有子中心取最像的(保留多模态) ==========
    def score(self, x):
        x = l2norm(x)
        return max(float(np.dot(x, s.vec)) for s in self.subs) if self.subs else -1.0


# ============================================================
# 自测: 演示"自适应"——单峰只长一个中心; 双峰自动裂成两个
# ============================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    D = 64
    voiceA = l2norm(rng.standard_normal(D))           # 正常嗓音
    voiceA_cold = l2norm(rng.standard_normal(D))      # 感冒嗓音(另一个模式)

    def noisy(v, s=0.1):
        return l2norm(v + rng.standard_normal(D) * s)

    # 场景1: 只喂正常嗓音(单峰)-> 应该只长出 1 个子中心
    p1 = Profile()
    for _ in range(50):
        p1.update(noisy(voiceA), quality=rng.uniform(0.6, 1.0))
    print(f"单峰场景(清理前): 子中心数 = {len(p1.subs)}")
    p1.consolidate()                                   # 自清理: 把孤儿碎片并回
    fit1 = float(np.dot(p1.subs[0].vec, voiceA))
    print(f"单峰场景(清理后): 子中心数 = {len(p1.subs)} (期望1), "
          f"该中心与真实音色吻合 = {fit1:.3f}, 累积权重 = {p1.subs[0].w:.1f}(已封顶40)")

    # 场景2: 正常 + 感冒两种嗓音混着喂(双峰)-> 应该自动裂成 2 个子中心
    p2 = Profile()
    for _ in range(60):
        v = voiceA if rng.random() < 0.5 else voiceA_cold
        p2.update(noisy(v), quality=rng.uniform(0.6, 1.0))
    p2.consolidate()                                   # 双峰: 两中心接近正交, 不会被误并
    print(f"\n双峰场景: 子中心数 = {len(p2.subs)} (期望2)")
    for s in p2.subs:
        f_norm = float(np.dot(s.vec, voiceA))
        f_cold = float(np.dot(s.vec, voiceA_cold))
        which = "正常嗓音" if f_norm > f_cold else "感冒嗓音"
        print(f"  子中心 -> 更像[{which}] (正常吻合{f_norm:.2f} / 感冒吻合{f_cold:.2f})")

    print("\n结论: 同一套代码, 单峰自动收成1个中心(最准), 双峰自动裂成2个(保留状态)——"
          "不用手动指定该分几类。")
