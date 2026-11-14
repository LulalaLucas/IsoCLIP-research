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

先按[数据准备说明](../../data/README.md)下载原图和固定划分文件，再修改`configs/prepare.json`中的`dataroot`。所有相对路径从仓库根目录解析，默认权重在`cache/clip/`、原始特征在`data/`、中间输入和实验产物在`runs/iso_b32/`。

准备脚本从原始标注生成全部身份名单与标签，按`configs/protocol.json`的固定指纹校验。该JSON只包含样本数与哈希，不包含特征或样本名单。首次提取时自动获取并校验官方B/32权重；无需本项目作者的旧prepared或研究结果。

每个实验一份JSON。`prepared`默认指向`runs/iso_b32/prepared/`，含`spectrum.pt`和各数据集`.pt`。每个数据集文件包含查询／图库特征、标签、身份名单、训练校准特征和样本指纹。具体字段见`isoclip/data.py::load_prepared`及`isoclip/prepare.py`。

- `output`设置为尚不存在的新目录。
- `gate_checkpoint`明确指定源监督倍率权重来源。
- `parameters.input_variant=iso`为纯Iso入口；`output_center`控制最终减均值。
- 训练倍率仅用CUB／Pets／Cars的训练标签，固定2000次更新，原配置见`configs/gate_training.json`。

在激活完整环境、具备CUDA GPU的机器上，从仓库根目录运行全部实验：

```bash
bash experiments/iso_b32/run_all.sh
```

`run_all.sh`依次执行图像编码、准备中间文件、2000步源监督训练和25个评测配置，每条调用都明确列出。骨干始终冻结。所有输出目录都应尚不存在。

只检查基线与主要组合时，分步执行：

```bash
python -m isoclip.extract --config experiments/iso_b32/configs/prepare.json
python -m isoclip.prepare --config experiments/iso_b32/configs/prepare.json
python -m isoclip.evaluate --config experiments/iso_b32/configs/iso.json
python -m isoclip.evaluate --config experiments/iso_b32/configs/center_b8.json
```

只有`gate3`和`gate_center_b8`依赖训练权重；运行它们之前执行：

```bash
python -m isoclip.train --config experiments/iso_b32/configs/gate_training.json
python -m isoclip.evaluate --config experiments/iso_b32/configs/gate3.json
python -m isoclip.evaluate --config experiments/iso_b32/configs/gate_center_b8.json
```

各实验输出目录包含`results.json`、逐查询AP与top10的`*_queries.npz`和`*_transform.pt`；训练输出`gate_training/gate.pt`、`history.json`和`completed.json`。这些文件自动保存在本机，由Git忽略。

如果只想先用一个数据集试跑，复制`prepare.json`及所选评测JSON，将它们的`datasets`都改成同一个名称，并给它们设置对应的新输出目录与`prepared`路径。源监督训练需要CUB、Pets、Cars三个源训练集，不能用单个目标集替代。

已有特征可重复执行`extract`并复用缓存。`prepare`和评测拒绝覆盖已有输出；完成准备后可跳过前两步，直接执行尚未运行的策略配置。换一轮实验时修改对应`output`；换prepared位置时同步修改评测和训练配置的`prepared`，换训练输出时同步修改两个门控评测配置的`gate_checkpoint`。

新的`run_all.sh`使用当前激活环境中的Python，可从任意仓库位置启动。其他旧启动器保留原服务器上的已完成命令注释，作为历史运行记录。后续按“纯Iso起点→单项操作→机制对照→明确组合”组织实验。
