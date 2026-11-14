# IsoCLIP：代码与运行配置

本仓库保存源码、实验配置、完整环境定义与使用README。原始数据集、模型权重、特征缓存、结果表、报告、PPT和历史归档均由使用者在本地准备或生成。

当前代码围绕纯IsoCLIP组织：冻结CLIP骨干，分别实现中心化、去均值方向、PCA、白化、近邻方向、B2/B8共识和源监督三参数谱倍率，并通过明确配置控制策略组合。

## 目录

| 目录 | 内容 |
| --- | --- |
| [isoclip](isoclip/README.md) | 当前封装的投影、几何适配、谱倍率训练、缓存读取与检索指标 |
| [experiments/iso_b32](experiments/iso_b32/README.md) | 单策略、对照和明确组合的JSON与启动器 |
| `src/` | 原作者图像／文本检索、分类、编码与数据集实现，以及已有本地修正 |
| `exp_img-img_retrieval/`、`exp_txt-txt_retrieval/`、`exp_img_classification/` | 作者原始任务启动器 |
| `reproduction/configs/`及子目录中的`configs/` | 固定方法复现配置 |
| `reproduction/programs/run_retrieval.py` | 把JSON参数传给作者检索入口的薄包装 |
| [tests](tests/README.md) | 几何与AP计算的基础检查 |

`src/modifiedclip/`中附带的压缩词表是分词器运行资源，不是实验数据或模型权重。

## 安装与检查

核心包依赖PyTorch、NumPy和scikit-learn；Python要求3.10及以上。

```bash
pip install -e .
python -m unittest discover -s tests -v
```

作者编码器、数据集和完整复现还需要相应CLIP／Dassl等依赖，见`install_requirements.sh`和`requirements.txt`。已有环境可用`pip install -e . --no-deps`安装本包。

## 运行探索

先按[环境说明](environment/README.md)恢复环境，再按[数据说明](data/README.md)准备原图与固定划分文件。修改`experiments/iso_b32/configs/prepare.json`中的`dataroot`为自己的数据根目录；默认`datasets/`表示仓库内的目录。已激活环境后运行：

```bash
bash experiments/iso_b32/run_all.sh
```

完整入口会依次提取并缓存图像特征、从原始划分生成`prepared`、训练源监督谱倍率、运行25个探索配置。首次编码时自动下载并校验官方B/32权重；其余阶段直接复用本地文件。所有探索输入均可重新生成，不需要本项目作者的缓存或结果。

只跑基线或部分策略的分步命令见[实验README](experiments/iso_b32/README.md)。代码要求新生成的`prepared`和各实验输出目录尚不存在；重复提取则复用已有特征。图像缓存使用`data/`，中间数据、训练权重和结果使用`runs/iso_b32/`。旧启动器保留历史执行记录；首次复现使用新的`run_all.sh`或README中的分步命令。

原方法与作者项目：[IsoCLIP](https://github.com/simomagi/IsoCLIP)。原许可证见[LICENSE](LICENSE)。

## 环境与数据

先按[环境恢复说明](environment/README.md)安装实际实验的完整锁定依赖，再按[本地缓存说明](data/README.md)生成并保存特征。数据无需上传GitHub。
