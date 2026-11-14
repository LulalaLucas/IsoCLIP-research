# 核心实现

这个包只负责固定投影、训练描述符适配和指标计算，不导入历史 `reproduction` 脚本。

| 文件 | 职责 |
| --- | --- |
| `projection.py` | 原 IsoCLIP 图像投影、明确命名的混合谱对照、原保留方向内的能量筛选 |
| `adaptation.py` | 训练均值、PCA、白化、局部 QR、软保留、B2/B8；统一 `DescriptorTransform` |
| `spectral.py` | 三参数谱倍率、PK16×8 采样、NCA 损失 |
| `metrics.py` | 完整排序 AP、top10、固定排序 sklearn 交叉核验 |
| `data.py` | 真实特征缓存、固定样本名单与训练／测试边界 |
| `dataset_splits.py` | 从原始标注构建名单、标签与固定训练顺序，并校验PPT划分指纹 |
| `extract.py` | 使用冻结B/32从原图提取投影前特征并保存为作者兼容缓存 |
| `prepare.py` | 从原始标注与已编码缓存生成B/32中间文件，无旧prepared依赖 |
| `train.py` | 复现 PPT 的源训练门控，骨干冻结 |
| `evaluate.py` | 读取一个 JSON 配置，拟合、检索、保存变换与逐查询结果 |

所有投影都使用行向量约定：`features @ projection`。传入 `image_projection` 的 `text` 是原模型的 `text_projection`，内部计算 `image @ text.T`。

```python
import torch
import torch.nn.functional as F
from isoclip import image_projection, fit_transform

# weights来自本地CLIP checkpoint，raw_train/raw_query是投影前真实图像特征。
projection = image_projection(weights['visual.proj'], weights['text_projection'], 150, 50)
projected_train = raw_train @ projection.float()
projected_query = raw_query @ projection.float()
_, _, vh = torch.linalg.svd(projection, full_matrices=False)
output_basis = vh[:312].T
calibration = F.normalize(projected_train[calibration_ids].double(), dim=1)
state, info = fit_transform(calibration, output_basis,
    {'method': 'consensus', 'partitions': 4, 'split_seed': 20261105,
     'threshold': 0.8, 'output_center': True})
gallery = state(projected_train)
query = state(projected_query)
```

均值与方向只从 `calibration` 学习；查询和图库复用同一个变换。`DescriptorTransform` 接收已经经过 Iso 投影、尚未归一化的描述符，内部先归一化，再减均值／投影，最后归一化。保存时用 `state.state_dict()`；从保存的 `matrix`、`center` 构造 `DescriptorTransform(matrix, center)` 即可重新加载。

当前近邻估计和完整实验入口固定使用 CUDA；没有自动切到其他实现。CPU数学检查单列在 `tests/`。
