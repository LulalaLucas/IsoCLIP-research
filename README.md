# IsoCLIP：代码与运行配置

本仓库只保存源码、实验配置、依赖说明和使用README。数据集、模型权重、特征缓存、结果表、报告、PPT及历史归档均留在本地研究工作区。

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

阅读[实验README](experiments/iso_b32/README.md)。计算需要在本地准备真实特征缓存与固定训练／测试身份；这些二进制输入不随本仓库上传。修改配置中的路径及新的输出目录后运行：

```bash
python -m isoclip.evaluate --config experiments/iso_b32/configs/center_b8.json
```

代码要求输出目录尚不存在。图像与文本特征缓存使用`data/`，运行产物使用`runs/`。历史启动器保留原服务器的具体路径和已完成命令注释，换环境时需按实际位置调整。

原方法与作者项目：[IsoCLIP](https://github.com/simomagi/IsoCLIP)。原许可证见[LICENSE](LICENSE)。
