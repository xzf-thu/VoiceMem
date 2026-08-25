"""声纹画像更新（自适应多中心）+ 3D-Speaker ERes2Net 声纹提取 worker。"""

from voicemem.utils.audio.voiceprint.adaptive_centroid import Profile, SubCentroid, l2norm

__all__ = ["Profile", "SubCentroid", "l2norm"]
