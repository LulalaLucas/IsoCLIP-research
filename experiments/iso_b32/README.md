# 纯IsoCLIP子空间探索：代码说明

共同起点为OpenAI CLIP ViT-B/32的原Iso投影，截断顶部150、底部50，保留312维有效空间。骨干冻结；无标签几何操作仅在训练图像上拟合，测试查询与训练图库互斥。

## 操作与组合

| 配置 | 操作 |
| --- | --- |
| `iso` | 原Iso直接余弦检索 |
| `center` | 减训练均值 |
| `mean_axis` | 移除均值方向，输出不减均值 |
| `pca90`／`whiten` | 训练协方差PCA方向选择／收缩白化 |
| `energy256` | 仅在原Iso保留轴内，按方差／二阶矩筛方向 |
| `local_qr`／`soft25` | 全局／局部散度的广义谱方向，QR后硬保留／软保留 |
| `centered_neighbors`／`hubness`／`shuffled_graph` | 建图规则对照 |
| `b2`／`b8` | 多半集局部子空间的投影共识 |
| `gate3` | 源训练标签学习三参数平滑谱倍率 |
| `center_*` | 明确开启输出中心化的组合 |
| `gate_center_b8` | 源训练倍率后重新拟合均值与B8 |
| `mixed_*` | 明确命名的混合谱补充配置，不作为其他策略默认入口 |

除仅中心化外，独立策略不默认减输出均值。协方差估计的均值与最终输出减均值分别控制。近邻规则为互近邻10、局部收缩0.3、广义特征值>1、QR；B8为四次互补二分，支持度>0.8。B8指八个子空间，最终维数由训练数据决定。

## 代码

`isoclip/projection.py`构造原Iso与能量筛选；`adaptation.py`拟合几何变换；`spectral.py`和`train.py`训练三参数倍率；`data.py`和`prepare.py`处理缓存；`metrics.py`和`evaluate.py`计算完整排序检索。

## 输入与运行

每个实验一份JSON。`prepared`指定含`spectrum.pt`和各数据集`.pt`的本地目录；每个数据集文件包含查询／图库特征、标签、身份名单、训练校准特征和样本指纹。具体字段见`isoclip/data.py::load_prepared`及`isoclip/prepare.py`。数据与结果不在本仓库内。

- `output`设置为尚不存在的新目录。
- `gate_checkpoint`明确指定源监督倍率权重来源。
- `parameters.input_variant=iso`为纯Iso入口；`output_center`控制最终减均值。
- 训练倍率仅用CUB／Pets／Cars的训练标签，固定2000次更新，原配置见`configs/gate_training.json`。

从仓库根目录运行：

```bash
python -m isoclip.evaluate --config experiments/iso_b32/configs/center_b8.json
python -m isoclip.train --config experiments/iso_b32/configs/gate_training.json
```

重建输入用`python -m isoclip.prepare --config experiments/iso_b32/configs/prepare.json`，需要先提供该配置约定的投影前特征缓存与冻结身份参考。

启动器均为零参数、直接调用Python；已完成命令注释保留，换环境后需明确调整路径和新输出位置。后续按“纯Iso起点→单项操作→机制对照→明确组合”组织实验。
